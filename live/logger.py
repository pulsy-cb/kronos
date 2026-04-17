"""
SessionLogger - Persistence des evenements de trading en JSONL.
Chaque evenement contient le model_key et model_name pour tracabilite.
Rotation journaliere automatique des fichiers.
"""

import json
import os
from pathlib import Path
from datetime import datetime, date


class SessionLogger:
    """Logger append-only pour une session de trading live."""

    def __init__(self, session_id, symbol, timeframe, model_key, model_name, log_dir="logs", broker="mt5"):
        self.session_id = session_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_key = model_key
        self.model_name = model_name
        self.broker = broker
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._current_date = date.today()
        self._file = None
        self._open_file()

        self._write_header()

    def _get_filepath(self, for_date=None):
        d = for_date or self._current_date
        filename = f"live_{self.symbol}_{self.timeframe}_{d.strftime('%Y-%m-%d')}.jsonl"
        return self.log_dir / filename

    def _open_file(self):
        filepath = self._get_filepath()
        self._file = open(filepath, "a", encoding="utf-8")

    def _check_rotation(self):
        today = date.today()
        if today != self._current_date:
            self.close()
            self._current_date = today
            self._open_file()

    def _write_line(self, obj):
        self._check_rotation()
        line = json.dumps(obj, ensure_ascii=False, default=str)
        self._file.write(line + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def _write_header(self):
        header = {
            "type": "session_start",
            "session_id": self.session_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "model_key": self.model_key,
            "model_name": self.model_name,
            "broker": self.broker,
            "timestamp": datetime.now().isoformat(),
        }
        self._write_line(header)

    def log_trade(self, action, direction, price, volume, sl=None, tp=None,
                  ticket=None, pnl=None, reason=None):
        event = {
            "type": "trade",
            "session_id": self.session_id,
            "model_key": self.model_key,
            "model_name": self.model_name,
            "broker": self.broker,
            "action": action,
            "direction": direction,
            "price": price,
            "volume": volume,
            "timestamp": datetime.now().isoformat(),
        }
        if sl is not None:
            event["sl"] = sl
        if tp is not None:
            event["tp"] = tp
        if ticket is not None:
            event["ticket"] = ticket
        if pnl is not None:
            event["pnl"] = round(pnl, 2)
        if reason is not None:
            event["reason"] = reason
        self._write_line(event)

    def log_signal(self, signal, predicted_return, predicted_close, current_close):
        event = {
            "type": "signal",
            "session_id": self.session_id,
            "model_key": self.model_key,
            "model_name": self.model_name,
            "broker": self.broker,
            "signal": signal,
            "predicted_return": round(predicted_return, 6),
            "predicted_close": round(predicted_close, 4),
            "current_close": current_close,
            "timestamp": datetime.now().isoformat(),
        }
        self._write_line(event)

    def log_equity(self, equity, balance):
        event = {
            "type": "equity",
            "session_id": self.session_id,
            "model_key": self.model_key,
            "model_name": self.model_name,
            "broker": self.broker,
            "equity": round(equity, 2),
            "balance": round(balance, 2),
            "timestamp": datetime.now().isoformat(),
        }
        self._write_line(event)

    def close(self):
        if self._file and not self._file.closed:
            self._write_line({
                "type": "session_stop",
                "session_id": self.session_id,
                "model_key": self.model_key,
                "model_name": self.model_name,
                "broker": self.broker,
                "timestamp": datetime.now().isoformat(),
            })
            self._file.close()
            self._file = None


def write_session_summary(session_id, config, metrics, log_dir="logs"):
    """Ecrit un resume de session dans sessions_summary.jsonl."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    filepath = log_path / "sessions_summary.jsonl"

    summary = {
        "session_id": session_id,
        "symbol": config.get("symbol", "?"),
        "timeframe": config.get("timeframe", "?"),
        "model_key": config.get("model_key", "?"),
        "model_name": config.get("model_name", "?"),
        "direction": config.get("direction", "?"),
        "sizing_method": config.get("sizing_method", "?"),
        "total_trades": metrics.get("total_trades", 0),
        "winning_trades": metrics.get("winning_trades", 0),
        "total_pnl": round(metrics.get("total_pnl", 0), 2),
        "win_rate": metrics.get("win_rate", 0),
        "started_at": metrics.get("started_at", ""),
        "stopped_at": datetime.now().isoformat(),
        "duration_minutes": metrics.get("duration_minutes", 0),
    }

    with open(filepath, "a", encoding="utf-8") as f:
        line = json.dumps(summary, ensure_ascii=False, default=str)
        f.write(line + "\n")
        f.flush()


def read_log_file(filepath, event_type=None, limit=1000):
    """Lit un fichier JSONL et retourne les evenements, optionnellement filtres par type."""
    events = []
    path = Path(filepath)
    if not path.exists():
        return events

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if event_type and obj.get("type") != event_type:
                    continue
                events.append(obj)
            except json.JSONDecodeError:
                continue

    return events[-limit:]


def list_log_files(log_dir="logs"):
    """Liste les fichiers de log disponibles."""
    log_path = Path(log_dir)
    if not log_path.exists():
        return []

    files = []
    for f in sorted(log_path.glob("*.jsonl")):
        files.append({
            "name": f.name,
            "path": str(f),
            "size_kb": round(f.stat().st_size / 1024, 1),
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        })
    return files