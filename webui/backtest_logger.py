"""
BacktestLogger - Persistence des resultats de backtest en JSONL.
Format identique au SessionLogger (live/logger.py) pour compatibilite
avec les scripts d'analyse (analyze_live_logs.py, analyze_temporal_decay.py).
"""

import json
import os
import re
import uuid
from pathlib import Path
from datetime import datetime


# Mapping: backtest exit_reason -> live reason format
_REASON_MAP = {
    "stop_loss": "sl_or_tp",
    "take_profit": "sl_or_tp",
    "max_hold": "max_hold",
    "signal": "signal_exit",
    "forced_close": "max_hold",
}


class BacktestLogger:
    """Logger that writes backtest results as JSONL events compatible with live logs."""

    def __init__(self, model_key, model_name, symbol="?", timeframe="?",
                 log_dir="logs", session_id=None):
        self.model_key = model_key
        self.model_name = model_name
        self.symbol = symbol
        self.timeframe = timeframe
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or f"bt_{uuid.uuid4().hex[:8]}"
        self._file = None

    def _get_filepath(self, date_str):
        filename = f"backtest_{self.model_key}_{self.symbol}_{self.timeframe}_{date_str}.jsonl"
        return self.log_dir / filename

    def open(self, date_str):
        filepath = self._get_filepath(date_str)
        self._file = open(filepath, "a", encoding="utf-8")
        self._write_line({
            "type": "session_start",
            "session_id": self.session_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "model_key": self.model_key,
            "model_name": self.model_name,
            "broker": "backtest",
            "timestamp": datetime.now().isoformat(),
        })
        return filepath

    def log_signal(self, signal, predicted_return, predicted_close, current_close, timestamp):
        event = {
            "type": "signal",
            "session_id": self.session_id,
            "model_key": self.model_key,
            "model_name": self.model_name,
            "broker": "backtest",
            "signal": signal,
            "predicted_return": round(predicted_return, 6),
            "predicted_close": round(predicted_close, 4),
            "current_close": current_close,
            "timestamp": timestamp if isinstance(timestamp, str) else timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
        }
        self._write_line(event)

    def log_trade_open(self, direction, price, volume, timestamp, sl=None, tp=None):
        event = {
            "type": "trade",
            "session_id": self.session_id,
            "model_key": self.model_key,
            "model_name": self.model_name,
            "broker": "backtest",
            "action": "open",
            "direction": direction,
            "price": price,
            "volume": volume,
            "timestamp": timestamp if isinstance(timestamp, str) else timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
        }
        if sl is not None:
            event["sl"] = sl
        if tp is not None:
            event["tp"] = tp
        self._write_line(event)

    def log_trade_close(self, direction, price, volume, pnl, reason, timestamp):
        event = {
            "type": "trade",
            "session_id": self.session_id,
            "model_key": self.model_key,
            "model_name": self.model_name,
            "broker": "backtest",
            "action": "close",
            "direction": direction,
            "price": price,
            "volume": volume,
            "pnl": round(pnl, 2),
            "reason": _REASON_MAP.get(reason, reason),
            "timestamp": timestamp if isinstance(timestamp, str) else timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
        }
        self._write_line(event)

    def log_equity(self, equity, balance, timestamp):
        event = {
            "type": "equity",
            "session_id": self.session_id,
            "model_key": self.model_key,
            "model_name": self.model_name,
            "broker": "backtest",
            "equity": round(equity, 2),
            "balance": round(balance, 2),
            "timestamp": timestamp if isinstance(timestamp, str) else timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
        }
        self._write_line(event)

    def close(self):
        if self._file and not self._file.closed:
            self._write_line({
                "type": "session_stop",
                "session_id": self.session_id,
                "model_key": self.model_key,
                "model_name": self.model_name,
                "broker": "backtest",
                "timestamp": datetime.now().isoformat(),
            })
            self._file.close()
            self._file = None

    def _write_line(self, obj):
        if self._file and not self._file.closed:
            line = json.dumps(obj, ensure_ascii=False, default=str)
            self._file.write(line + "\n")
            self._file.flush()


def detect_symbol_from_path(file_path):
    """Try to extract symbol from data file name (e.g. XAUUSD, EURUSD)."""
    if not file_path:
        return "?"
    name = os.path.basename(file_path).upper()
    patterns = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "BTCUSDT", "ETHUSDT", "XAU", "EUR", "GBP"]
    for p in patterns:
        if p in name:
            return p
    return "?"


def detect_timeframe_from_path(file_path):
    """Try to extract timeframe from data file name."""
    if not file_path:
        return "?"
    name = os.path.basename(file_path)
    # Match M1, M5, M15, H1, H4, D1 etc. — use word boundary to avoid matching inside other words
    m = re.search(r'[_\-\.](S\d+|M\d+|H\d+|D\d?)\.?', name, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return "?"


def extract_date_from_signals(signals):
    """Extract date string from first signal timestamp for log file naming."""
    if not signals:
        return datetime.now().strftime("%Y-%m-%d")
    ts = signals[0].get("timestamp", "")
    if isinstance(ts, str) and len(ts) >= 10:
        return ts[:10]
    return datetime.now().strftime("%Y-%m-%d")