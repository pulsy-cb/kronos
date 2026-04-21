"""
LiveTrader - Boucle de trading live avec Kronos.
Genere des signaux de trading en temps reel et execute les ordres via un broker.
"""

import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import traceback

from model.kronos import set_inference_seed
from data.broker_feed import BrokerFeed, TF_SECONDS
from live.broker_executor import BrokerExecutor
from live.config import TradingConfig, DIRECTION_LONG_SHORT, DIRECTION_LONG_ONLY
from live.logger import SessionLogger


class LiveTrader:
    """Boucle de trading live connectee a un broker."""

    def __init__(self, config: TradingConfig, predictor, feed: BrokerFeed, executor: BrokerExecutor, logger: SessionLogger = None, predictor_voter=None):
        self.config = config
        self.predictor = predictor
        self.predictor_voter = predictor_voter
        self.feed = feed
        self.executor = executor
        self.logger = logger

        self.model_key = config.model_key
        self.model_name = config.model_name or config.model_key

        self.running = False
        self.paused = False
        self.thread = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()

        # State
        self.current_position = None
        self.trade_log = []
        self.signal_log = []
        self.last_bar_time = None
        self.last_prediction = None
        self.last_signal = None
        self.equity_history = []
        self.bars_held = 0
        self.error_message = None
        self.status = "idle"
        self.started_at = None

        # Metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

    def start(self):
        if self.running:
            return False, "Deja en cours"

        if not self.feed.is_connected():
            return False, "Broker non connecte"

        self._stop_event.clear()
        self._pause_event.set()
        self.running = True
        self.status = "running"
        self.error_message = None
        self.started_at = datetime.now().isoformat()

        self.thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.thread.start()
        return True, "Trading demarre"

    def stop(self):
        self._stop_event.set()
        self._pause_event.set()
        self.running = False
        if self.logger:
            self.logger.close()

    def pause(self):
        self.paused = True
        self._pause_event.clear()
        self.status = "paused"

    def resume(self):
        self.paused = False
        self._pause_event.set()
        if self.running:
            self.status = "running"

    def get_metrics(self):
        started = datetime.fromisoformat(self.started_at) if self.started_at else datetime.now()
        duration = (datetime.now() - started).total_seconds() / 60
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "total_pnl": self.total_pnl,
            "win_rate": round(self.winning_trades / self.total_trades * 100, 1) if self.total_trades > 0 else 0,
            "started_at": self.started_at or "",
            "duration_minutes": round(duration, 1),
        }

    def get_state(self):
        account = self.feed.get_account_info() if self.feed.is_connected() else None
        positions = self.feed.get_all_positions(symbol=self.config.symbol) if self.feed.is_connected() else []

        return {
            "status": self.status,
            "config": self.config.to_dict(),
            "model_key": self.model_key,
            "model_name": self.model_name,
            "account": account,
            "current_position": self.current_position,
            "open_positions": positions,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "total_pnl": round(self.total_pnl, 2),
            "win_rate": round(self.winning_trades / self.total_trades * 100, 1) if self.total_trades > 0 else 0,
            "last_signal": self.last_signal,
            "last_prediction": self.last_prediction,
            "trade_log": self.trade_log[-50:],
            "signal_log": self.signal_log[-20:],
            "equity_history": self.equity_history[-100:],
            "error_message": self.error_message,
            "timestamp": datetime.now().isoformat(),
        }

    def _trading_loop(self):
        tf_seconds = TF_SECONDS.get(self.config.timeframe, 300)
        poll_interval = min(1.0, tf_seconds / 60.0)
        sync_retries = 0

        while not self._stop_event.is_set():
            try:
                self._pause_event.wait()

                if not self.feed.is_connected():
                    self.status = "error"
                    self.error_message = "Broker deconnecte"
                    time.sleep(5)
                    continue

                new_bar_time, changed = self.feed.wait_for_bar_close(
                    self.config.symbol,
                    self.config.timeframe,
                    self.last_bar_time,
                    poll_interval=poll_interval
                )

                if not changed or new_bar_time is None:
                    continue

                self.last_bar_time = new_bar_time

                df, err = self.feed.get_latest_candles(
                    self.config.symbol,
                    self.config.timeframe,
                    self.config.lookback + 10,
                    as_kronos=True,
                )
                if err or df is None or len(df) < self.config.lookback:
                    self.error_message = f"Donnees insuffisantes: {err}"
                    continue

                df = df.iloc[-self.config.lookback:].reset_index(drop=True)

                last_ts = df["timestamps"].iloc[-1]
                tf_delta = pd.Timedelta(seconds=tf_seconds)
                y_timestamp = pd.Series(
                    [last_ts + tf_delta * (i + 1) for i in range(self.config.pred_len)]
                )

                x_timestamp = df["timestamps"]
                x_df = df[["open", "high", "low", "close", "volume", "amount"]]

                with torch.inference_mode():
                    set_inference_seed()
                    pred_df = self.predictor.predict(
                        df=x_df,
                        x_timestamp=x_timestamp,
                        y_timestamp=y_timestamp,
                        pred_len=self.config.pred_len,
                        T=self.config.temperature,
                        top_p=self.config.top_p,
                        sample_count=self.config.sample_count,
                        sample_logits=False,
                    )

                    # Ensemble voting: if a second predictor is provided, only trade when both agree
                    if self.predictor_voter is not None:
                        set_inference_seed()
                        pred_df_voter = self.predictor_voter.predict(
                            df=x_df,
                            x_timestamp=x_timestamp,
                            y_timestamp=y_timestamp,
                            pred_len=self.config.pred_len,
                            T=self.config.temperature,
                            top_p=self.config.top_p,
                            sample_count=self.config.sample_count,
                            sample_logits=False,
                        )

                current_close = float(df["close"].iloc[-1])
                predicted_end_close = float(pred_df["close"].iloc[-1])
                predicted_return = (predicted_end_close - current_close) / current_close

                # Ensemble voting filter
                if self.predictor_voter is not None:
                    voter_end_close = float(pred_df_voter["close"].iloc[-1])
                    voter_return = (voter_end_close - current_close) / current_close
                    # Only trade when both models agree on direction
                    if np.sign(predicted_return) != np.sign(voter_return):
                        predicted_return = 0.0  # Force neutral

                config = self.config

                if config.direction == DIRECTION_LONG_SHORT:
                    if predicted_return > config.signal_threshold:
                        signal = "long"
                    elif predicted_return < -config.signal_threshold:
                        signal = "short"
                    elif self.current_position is not None:
                        pos_dir = self.current_position["direction"]
                        if pos_dir == "long" and predicted_return < -config.exit_threshold:
                            signal = "exit"
                        elif pos_dir == "short" and predicted_return > config.exit_threshold:
                            signal = "exit"
                        else:
                            signal = "neutral"
                    else:
                        signal = "neutral"
                else:
                    if predicted_return > config.signal_threshold:
                        signal = "long"
                    elif predicted_return < -config.exit_threshold and self.current_position is not None:
                        signal = "exit"
                    else:
                        signal = "neutral"

                self.last_signal = {
                    "timestamp": datetime.now().isoformat(),
                    "signal": signal,
                    "predicted_return": round(predicted_return, 6),
                    "predicted_close": round(predicted_end_close, 4),
                    "current_close": current_close,
                    "bar_time": new_bar_time.isoformat(),
                }
                self.signal_log.append(self.last_signal)

                if self.logger:
                    self.logger.log_signal(
                        signal, predicted_return, predicted_end_close, current_close
                    )

                self._execute_signal(signal, current_close, predicted_return)
                self._check_risk_management()

                acct = self.feed.get_account_info()
                if acct:
                    self.equity_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "equity": acct["equity"],
                        "balance": acct["balance"],
                    })
                    if self.logger:
                        self.logger.log_equity(acct["equity"], acct["balance"])

                sync_retries = 0

            except Exception as e:
                sync_retries += 1
                self.error_message = f"Erreur: {str(e)}\n{traceback.format_exc()}"
                if sync_retries > 5:
                    self.status = "error"
                    self.running = False
                    break
                time.sleep(5)

        self.running = False
        self.status = "stopped"

    def _execute_signal(self, signal, current_close, predicted_return):
        config = self.config

        if signal == "exit" and self.current_position is not None:
            self._close_position("signal_exit")
            return

        if config.direction == DIRECTION_LONG_SHORT and self.current_position is not None:
            pos_dir = self.current_position["direction"]
            if (signal == "long" and pos_dir == "short") or (signal == "short" and pos_dir == "long"):
                self._close_position("reverse")

        if signal in ("long", "short") and self.current_position is None:
            self._open_position(signal, current_close, predicted_return)

    def _open_position(self, direction, current_price, predicted_return=0.0):
        config = self.config
        tick_info = self.feed.get_current_tick(config.symbol)
        if tick_info is None:
            self.error_message = "Pas de tick disponible"
            return

        entry_price = tick_info["ask"] if direction == "long" else tick_info["bid"]

        sl_price, tp_price = self.executor.calculate_sl_tp_prices(
            entry_price, direction,
            config.stop_loss_pct, config.take_profit_pct,
            config.symbol
        )

        if config.sizing_method == "fixed":
            volume = config.fixed_lot
        else:
            sl_distance = abs(entry_price - sl_price) if sl_price else entry_price * config.stop_loss_pct
            account = self.feed.get_account_info()
            balance = account["balance"] if account else 10000
            volume = self.executor.calculate_lot_size(
                config.symbol, balance, config.risk_pct, sl_distance, direction
            )

        volume = min(volume, config.max_lot)

        success, result = self.executor.open_position(
            config.symbol, direction, volume,
            sl_price=sl_price, tp_price=tp_price,
            comment=f"kronos_{direction}"
        )

        if success:
            self.current_position = {
                "ticket": result.get("ticket"),
                "direction": direction,
                "entry_price": result.get("price", entry_price),
                "volume": result.get("volume", volume),
                "entry_time": datetime.now().isoformat(),
                "entry_bar_time": self.last_bar_time.isoformat() if self.last_bar_time else None,
                "sl": sl_price,
                "tp": tp_price,
                "predicted_return": predicted_return,
            }
            self.trade_log.append({
                "type": "open",
                "direction": direction,
                "timestamp": datetime.now().isoformat(),
                "price": result.get("price", entry_price),
                "volume": result.get("volume", volume),
                "sl": sl_price,
                "tp": tp_price,
                "ticket": result.get("ticket"),
            })
            if self.logger:
                self.logger.log_trade(
                    action="open", direction=direction,
                    price=result.get("price", entry_price),
                    volume=result.get("volume", volume),
                    sl=sl_price, tp=tp_price,
                    ticket=result.get("ticket"),
                )
        else:
            self.error_message = f"Erreur ouverture: {result}"

    def _close_position(self, reason="manual"):
        if self.current_position is None:
            return

        ticket = self.current_position.get("ticket")
        if ticket is None:
            self.current_position = None
            return

        success, result = self.executor.close_position(ticket)

        if success:
            pnl = result.get("profit", 0)
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
            self.total_pnl += pnl

            self.trade_log.append({
                "type": "close",
                "timestamp": datetime.now().isoformat(),
                "close_price": result.get("close_price"),
                "pnl": pnl,
                "reason": reason,
                "ticket": ticket,
                "direction": self.current_position["direction"],
                "entry_price": self.current_position["entry_price"],
                "volume": result.get("volume", self.current_position["volume"]),
            })
            if self.logger:
                self.logger.log_trade(
                    action="close", direction=self.current_position["direction"],
                    price=result.get("close_price"),
                    volume=result.get("volume", self.current_position["volume"]),
                    ticket=ticket, pnl=pnl, reason=reason,
                )

        self.current_position = None
        self.bars_held = 0

    def _check_risk_management(self):
        if self.current_position is None:
            return

        config = self.config

        self.bars_held += 1
        if config.max_hold_bars > 0 and self.bars_held >= config.max_hold_bars:
            self._close_position("max_hold")
            return

        positions = self.feed.get_all_positions(symbol=config.symbol)
        ticket = self.current_position.get("ticket")

        our_position = None
        for p in positions:
            if p["ticket"] == ticket:
                our_position = p
                break

        if our_position is None:
            if self.logger:
                self.logger.log_trade(
                    action="close", direction=self.current_position["direction"],
                    price=self.current_position.get("entry_price"),
                    volume=self.current_position["volume"],
                    ticket=ticket, pnl=0, reason="sl_or_tp",
                )
            self.trade_log.append({
                "type": "close",
                "timestamp": datetime.now().isoformat(),
                "close_price": self.current_position.get("entry_price"),
                "pnl": 0,
                "reason": "sl_or_tp",
                "ticket": ticket,
                "direction": self.current_position["direction"],
                "entry_price": self.current_position["entry_price"],
                "volume": self.current_position["volume"],
            })
            self.total_trades += 1
            self.current_position = None
            self.bars_held = 0