"""
Kronos Polymarket 5-Min Paper Trading — Web Engine
====================================================
Thread-safe session manager for the Flask web UI.
Runs the same trading loop as live_cli.py but exposes
state via shared structures for real-time SSE streaming.
"""

import os
import sys
import json
import time
import threading
import traceback
from datetime import datetime, timedelta, timezone
from collections import deque

import torch
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_cli import (
    AVAILABLE_MODELS,
    TF_SECONDS,
    PRESETS,
    BinancePublicFeed,
    Polymarket5MinTracker,
    load_kronos,
    predict_direction,
    seconds_to_next_window,
)


class PolymarketSession:
    """Single Polymarket paper-trading session, runnable in a background thread."""

    def __init__(self, config: dict):
        self.config = config
        self.symbol = config.get("symbol", "BTCUSDT")
        self.model_key = config.get("model_key", "small")
        self.timeframe = config.get("timeframe", "M5")
        self.bet_amount = config.get("bet_amount", 1.0)
        self.samples = config.get("samples", 5)
        self.device = config.get("device", None)

        self.status = "created"  # created | running | stopped | error
        self.error_message = None
        self.cycle = 0

        # Shared state (thread-safe via self._lock)
        self._lock = threading.Lock()
        self._log_entries = deque(maxlen=500)
        self._open_bets = []
        self._settled_bets = []
        self._pnl_history = []  # [{timestamp, cum_pnl, total_wagered}]
        self._summary = {
            "total_bets": 0, "wins": 0, "accuracy": 0.0,
            "open_bets": 0, "total_wagered_eur": 0.0,
            "total_pnl_eur": 0.0, "roi_pct": 0.0,
        }
        self._current_price = None
        self._last_prediction = None  # {direction, predicted_price, predicted_return, timestamp}
        self._model_info = None
        self._started_at = None
        self._stopped_at = None

        self._stop_event = threading.Event()

    # ─── Public read interface (called from Flask thread) ──────────

    def get_state(self):
        with self._lock:
            return {
                "status": self.status,
                "error_message": self.error_message,
                "cycle": self.cycle,
                "symbol": self.symbol,
                "model_key": self.model_key,
                "model_info": self._model_info,
                "current_price": self._current_price,
                "last_prediction": self._last_prediction,
                "started_at": self._started_at,
                "stopped_at": self._stopped_at,
                "summary": dict(self._summary),
                "open_bets": list(self._open_bets),
                "settled_bets": list(self._settled_bets[-50:]),  # last 50
                "pnl_history": list(self._pnl_history),
                "log_count": len(self._log_entries),
            }

    def get_new_logs(self, since_index: int):
        """Return log entries starting from since_index."""
        with self._lock:
            entries = list(self._log_entries)
            return entries[since_index:]

    # ─── Internal helpers (called from worker thread) ─────────────

    def _log(self, level, message):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
        }
        with self._lock:
            self._log_entries.append(entry)

    def _update_from_tracker(self, tracker: Polymarket5MinTracker):
        with self._lock:
            self._open_bets = [
                self._serialize_position(p) for p in tracker.positions
            ]
            self._summary = tracker.get_summary()
            self._current_price = None  # will be updated separately

    def _add_settled(self, result: dict):
        with self._lock:
            self._settled_bets.append(result)
            # Append P&L history point
            s = self._summary
            self._pnl_history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cum_pnl": s["total_pnl_eur"],
                "total_wagered": s["total_wagered_eur"],
                "accuracy": s["accuracy"],
            })

    def _set_current_price(self, price):
        with self._lock:
            self._current_price = price

    def _set_last_prediction(self, direction, pred_price, pred_return):
        with self._lock:
            self._last_prediction = {
                "direction": direction,
                "predicted_price": round(pred_price, 2) if pred_price else None,
                "predicted_return": round(pred_return, 6) if pred_return else None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    @staticmethod
    def _serialize_position(pos):
        return {
            "direction": pos["direction"],
            "current_price": pos["current_price"],
            "predicted_price": pos.get("predicted_price"),
            "pm_side": pos.get("pm_side"),
            "pm_buy_price": pos.get("pm_buy_price"),
            "bet_amount": pos.get("bet_amount"),
            "potential_profit": pos.get("potential_profit"),
            "window_start_ts": pos.get("window_start_ts"),
            "window_end_ts": pos.get("window_end_ts"),
            "timestamp": pos.get("timestamp"),
        }

    # ─── Main loop ────────────────────────────────────────────────

    def run(self):
        self.status = "running"
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._log("INFO", "=" * 60)
        self._log("INFO", "KRONOS POLYMARKET 5-MIN PAPER TRADER — STARTED")
        self._log("INFO", f"Model: {self.model_key} | Symbol: {self.symbol} | Bet: {self.bet_amount}€")
        self._log("INFO", "=" * 60)

        try:
            self._run_loop()
        except Exception as e:
            self.status = "error"
            self.error_message = str(e)
            self._log("ERROR", f"Fatal error: {e}")
            self._log("ERROR", traceback.format_exc())
        finally:
            self.status = "stopped"
            self._stopped_at = datetime.now(timezone.utc).isoformat()
            self._log("INFO", "Session stopped.")

    def _run_loop(self):
        # Load model
        self._log("INFO", f"Loading model {self.model_key}...")
        predictor = load_kronos(self.model_key, device=self.device)
        cfg = AVAILABLE_MODELS[self.model_key]
        self._model_info = {
            "name": cfg.get("description", self.model_key),
            "params": cfg.get("params", "?"),
            "context_length": cfg.get("context_length", 512),
        }
        self._log("INFO", f"Model loaded: {self._model_info['name']} ({self._model_info['params']})")

        feed = BinancePublicFeed()
        tracker = Polymarket5MinTracker(bet_amount=self.bet_amount)

        tf_secs = TF_SECONDS[self.timeframe]
        preset = PRESETS.get(self.timeframe, PRESETS["M5"])
        lookback = preset["lookback"]
        pred_len = preset["pred_len"]

        self._log("INFO", f"Lookback: {lookback} | Pred len: {pred_len} | Samples: {self.samples}")

        # Fetch initial data
        df, err = feed.get_klines(self.symbol, self.timeframe, limit=500)
        if err:
            raise RuntimeError(f"Failed to fetch data: {err}")
        self._log("INFO", f"Got {len(df)} candles. Last close: {df['close'].iloc[-1]:.2f}")

        while not self._stop_event.is_set():
            self.cycle += 1

            # 1) Settle expired bets
            current_price = feed.get_price(self.symbol)
            if current_price:
                self._set_current_price(current_price)
                settled = tracker.settle_all_expired({self.symbol: current_price})
                for r in settled:
                    icon = "+" if r["won"] else "-"
                    self._log("INFO",
                        f"SETTLED | {icon} {r['side']:4s} | "
                        f"€{r['profit_eur']:+.2f} | "
                        f"Entry {r['entry_price']:.2f} -> Exit {r['exit_price']:.2f} | "
                        f"Actual: {r['actual_direction']}"
                    )
                    self._add_settled(r)

            # 2) Wait until ~15s before next 5-min window
            secs_to_window = seconds_to_next_window()
            if secs_to_window > 20:
                wait_secs = secs_to_window - 15
                next_window = datetime.now(timezone.utc) + timedelta(seconds=wait_secs + 15)
                self._log("INFO",
                    f"[Cycle {self.cycle}] Waiting {wait_secs:.0f}s "
                    f"(window at {next_window.strftime('%H:%M:%S')} UTC)"
                )
                # Sleep in small chunks to check stop_event
                end_time = time.time() + wait_secs
                while time.time() < end_time and not self._stop_event.is_set():
                    time.sleep(min(1, end_time - time.time()))
                if self._stop_event.is_set():
                    break

            # 3) Fetch fresh data
            df, err = feed.get_klines(self.symbol, self.timeframe, limit=500)
            if err:
                self._log("WARNING", f"Data fetch failed: {err}")
                time.sleep(30)
                continue

            current_price = df["close"].iloc[-1]
            self._set_current_price(current_price)
            self._log("INFO", f"[Cycle {self.cycle}] Price: {current_price:.2f}")

            # 4) Run Kronos prediction
            direction, cur_price, pred_price, pred_df = predict_direction(
                predictor, df, lookback, pred_len, tf_secs,
                sample_count=self.samples
            )

            if direction is None:
                self._log("WARNING", "Prediction failed — skipping")
                time.sleep(30)
                continue

            pred_return = (pred_price - cur_price) / cur_price if cur_price else 0
            self._set_last_prediction(direction, pred_price, pred_return)
            self._log("INFO",
                f"KRONOS | {direction:4s} | Predicted: {pred_price:.2f} | Return: {pred_return*100:+.4f}%"
            )

            # 5) Place bet
            if direction == "FLAT":
                self._log("INFO", "No bet (prediction too flat)")
            else:
                result = tracker.place_bet(self.symbol, direction, cur_price, pred_price)
                if result.get("position"):
                    self._log("INFO",
                        f"BET {result['polymarket_side']:4s} | "
                        f"Buy @ {result['polymarket_buy_price']:.4f} | "
                        f"Profit: {result['potential_profit_eur']:+.2f}€ | "
                        f"Window: {result['window_start']}-{result['window_end']} UTC"
                    )
                else:
                    self._log("INFO", f"No bet: {result.get('market', 'unknown')}")

            # Sync tracker state
            self._update_from_tracker(tracker)

            # 6) Wait for window start
            secs_to_window = seconds_to_next_window()
            if secs_to_window > 0:
                end_time = time.time() + secs_to_window + 2
                while time.time() < end_time and not self._stop_event.is_set():
                    time.sleep(min(1, end_time - time.time()))

        # Final sync
        self._update_from_tracker(tracker)
        self._log("INFO", f"Final P&L: {self._summary['total_pnl_eur']:+.2f}€ | ROI: {self._summary['roi_pct']:+.1f}%")

    def stop(self):
        self._stop_event.set()
        self._log("INFO", "Stop requested...")


class PolymarketSessionManager:
    """Manages Polymarket trading sessions for the web UI."""

    def __init__(self):
        self._sessions = {}
        self._lock = threading.Lock()

    def create_and_start(self, config: dict) -> str:
        import uuid
        session_id = str(uuid.uuid4())[:8]
        session = PolymarketSession(config)

        with self._lock:
            self._sessions[session_id] = session

        thread = threading.Thread(target=session.run, daemon=True)
        thread.start()

        return session_id

    def stop(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if not session:
            return False
        session.stop()
        return True

    def get_state(self, session_id: str):
        session = self._sessions.get(session_id)
        if not session:
            return None
        return session.get_state()

    def get_session(self, session_id: str):
        return self._sessions.get(session_id)

    def get_all_states(self):
        with self._lock:
            return {sid: s.get_state() for sid, s in self._sessions.items()}

    def cleanup_old(self, max_age_hours=2):
        """Remove stopped sessions older than max_age_hours."""
        now = datetime.now(timezone.utc)
        with self._lock:
            to_remove = []
            for sid, s in self._sessions.items():
                if s.status in ("stopped", "error") and s._stopped_at:
                    stopped = datetime.fromisoformat(s._stopped_at)
                    if (now - stopped).total_seconds() > max_age_hours * 3600:
                        to_remove.append(sid)
            for sid in to_remove:
                del self._sessions[sid]