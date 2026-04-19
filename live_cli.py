#!/usr/bin/env python3
"""
Kronos CLI Live Trader — Mode Autonome
======================================
Script CLI pour le trading live/paper avec Kronos.
Pas besoin de Web UI — tout se passe en terminal avec logs détaillés.

Usage:
    python live_cli.py --model small --symbol BTCUSDT --timeframe M1 --paper --capital 10000
    python live_cli.py --model base --symbol ETHUSDT --timeframe M1 --paper --capital 10000
    python live_cli.py --model small --symbol BTCUSDT --timeframe M1 --live
"""

import os
import sys
import argparse
import time
import json
import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import torch
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Models Registry ────────────────────────────────────────────────
AVAILABLE_MODELS = {
    "mini": {
        "model_id": "NeoQuasar/Kronos-mini",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
        "context_length": 2048,
        "params": "4.1M",
        "description": "Kronos-mini — lightweight, fast prediction",
    },
    "small": {
        "model_id": "NeoQuasar/Kronos-small",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "context_length": 512,
        "params": "24.7M",
        "description": "Kronos-small — balanced speed/accuracy",
    },
    "base": {
        "model_id": "NeoQuasar/Kronos-base",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "context_length": 512,
        "params": "102.3M",
        "description": "Kronos-base — best accuracy, slower",
    },
    "xaumodel-local": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "xaumodel"),
        "tokenizer_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "xaumodel", "tokenizer"),
        "context_length": 512,
        "params": "~24.7M",
        "description": "XAU Finetuned (local)",
    },
    "xaumodel-mini": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "zmini"),
        "tokenizer_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "zmini", "tokenizer"),
        "context_length": 512,
        "params": "~24.7M",
        "description": "XAU mini (local)",
    },
}

# ─── Timeframe Presets ────────────────────────────────────────────────
TF_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}

PRESETS = {
    "M1": {
        "lookback": 120, "pred_len": 30, "signal_threshold": 0.0008,
        "exit_threshold": 0.0005, "stop_loss_pct": 0.0025,
        "take_profit_pct": 0.005, "max_hold_bars": 120,
    },
    "M5": {
        "lookback": 120, "pred_len": 120, "signal_threshold": 0.0012,
        "exit_threshold": 0.0007, "stop_loss_pct": 0.0035,
        "take_profit_pct": 0.007, "max_hold_bars": 144,
    },
}

# ─── Paper Trading Engine ──────────────────────────────────────────────
class PaperEngine:
    """Moteur de paper trading — simule les ordres sans broker."""

    def __init__(self, capital=10000.0, risk_pct=0.01, commission_pct=0.0004):
        self.initial_capital = capital
        self.capital = capital
        self.risk_pct = risk_pct  # 1% du capital par trade
        self.commission_pct = commission_pct
        self.position = None
        self.trades = []
        self.equity_history = []
        self.signal_log = []
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.total_commission = 0.0

    def get_balance(self):
        return self.capital

    def get_equity(self, current_price=None):
        eq = self.capital
        if self.position and current_price:
            entry = self.position["entry_price"]
            qty = self.position["quantity"]
            direction = self.position["direction"]
            if direction == "long":
                eq += (current_price - entry) * qty
            else:
                eq += (entry - current_price) * qty
        return eq

    def open_position(self, direction, price, predicted_return=0.0):
        if self.position is not None:
            return False, "Déjà en position"

        # Calculate quantity: risk 1% of capital
        risk_amount = self.capital * self.risk_pct
        quantity = risk_amount / price  # Simple: just use 1% of capital worth

        commission = quantity * price * self.commission_pct

        self.position = {
            "direction": direction,
            "entry_price": price,
            "quantity": quantity,
            "commission": commission,
            "entry_time": datetime.now().isoformat(),
            "predicted_return": predicted_return,
            "bars_held": 0,
        }

        self.total_commission += commission
        self.signal_log.append({
            "time": datetime.now().isoformat(),
            "action": "OPEN",
            "direction": direction,
            "price": price,
            "quantity": quantity,
            "predicted_return": round(predicted_return, 6),
        })

        return True, f"Paper {direction} @ {price:.2f}, qty={quantity:.6f}"

    def close_position(self, price, reason="signal"):
        if self.position is None:
            return False, "Pas de position"

        entry = self.position["entry_price"]
        qty = self.position["quantity"]
        direction = self.position["direction"]
        commission = qty * price * self.commission_pct
        self.total_commission += commission

        if direction == "long":
            pnl = (price - entry) * qty - self.position["commission"] - commission
        else:
            pnl = (entry - price) * qty - self.position["commission"] - commission

        self.total_pnl += pnl
        self.capital += pnl
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1

        trade = {
            "direction": direction,
            "entry_price": entry,
            "exit_price": price,
            "quantity": qty,
            "pnl": round(pnl, 4),
            "pnl_pct": round(pnl / (entry * qty) * 100, 4) if entry * qty > 0 else 0,
            "reason": reason,
            "entry_time": self.position["entry_time"],
            "exit_time": datetime.now().isoformat(),
            "bars_held": self.position["bars_held"],
            "predicted_return": self.position.get("predicted_return", 0),
        }
        self.trades.append(trade)
        self.position = None

        self.signal_log.append({
            "time": datetime.now().isoformat(),
            "action": "CLOSE",
            "direction": direction,
            "price": price,
            "pnl": round(pnl, 4),
            "reason": reason,
        })

        return True, f"Paper close {direction} @ {price:.2f}, PnL={pnl:.4f}"

    def check_risk(self, current_price, max_hold_bars, stop_loss_pct, take_profit_pct):
        """Check stop-loss, take-profit, max hold. Returns (should_close, reason)."""
        if self.position is None:
            return False, ""

        self.position["bars_held"] += 1
        entry = self.position["entry_price"]
        direction = self.position["direction"]

        if direction == "long":
            pnl_pct = (current_price - entry) / entry
        else:
            pnl_pct = (entry - current_price) / entry

        if pnl_pct <= -stop_loss_pct:
            return True, f"stop_loss ({pnl_pct:.4f} <= -{stop_loss_pct})"
        if pnl_pct >= take_profit_pct:
            return True, f"take_profit ({pnl_pct:.4f} >= {take_profit_pct})"
        if max_hold_bars > 0 and self.position["bars_held"] >= max_hold_bars:
            return True, f"max_hold ({self.position['bars_held']} bars)"

        return False, ""

    def get_summary(self):
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        return {
            "initial_capital": self.initial_capital,
            "current_capital": round(self.capital, 2),
            "equity": round(self.get_equity(), 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": round(win_rate, 1),
            "total_pnl": round(self.total_pnl, 4),
            "total_pnl_pct": round(self.total_pnl / self.initial_capital * 100, 4),
            "total_commission": round(self.total_commission, 4),
            "position": self.position,
        }


# ─── Binance Public Data Feed (no API key needed) ──────────────────────
class BinancePublicFeed:
    """Récupère les données Binance via API publique — pas de clé nécessaire."""

    TF_MAP = {
        "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
        "H1": "1h", "H4": "4h", "D1": "1d",
    }

    def __init__(self):
        self._session = None

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "Kronos-CLI/1.0"})
        return self._session

    def get_klines(self, symbol, timeframe, limit=500):
        """Get klines from Binance public API. No auth needed."""
        if timeframe not in self.TF_MAP:
            return None, f"Timeframe inconnu: {timeframe}"

        interval = self.TF_MAP[timeframe]
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}

        try:
            resp = self._get_session().get(url, params=params, timeout=10)
            resp.raise_for_status()
            klines = resp.json()

            data = []
            for k in klines:
                data.append({
                    "timestamps": pd.to_datetime(int(k[0]), unit="ms"),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })

            df = pd.DataFrame(data)
            df["amount"] = ((df["open"] + df["high"] + df["low"] + df["close"]) / 4) * df["volume"]

            for col in ["open", "high", "low", "close", "volume", "amount"]:
                df[col] = df[col].astype(np.float64)

            return df, None

        except Exception as e:
            return None, f"Erreur API Binance: {str(e)}"

    def get_price(self, symbol):
        """Get current price."""
        url = "https://api.binance.com/api/v3/ticker/price"
        try:
            resp = self._get_session().get(url, params={"symbol": symbol}, timeout=10)
            resp.raise_for_status()
            return float(resp.json()["price"])
        except Exception:
            return None


# ─── Polymarket Mock Tracker ─────────────────────────────────────────────
class PolymarketMockTracker:
    """
    Mock Polymarket tracker — utilise les predictions Kronos pour simuler
    des positions Polymarket en paper trading.
    
    Pour les marchés crypto sur Polymarket, on peut lier les prédictions
    de prix Kronos aux marchés "Will BTC be above $X at Y time?".
    """

    def __init__(self):
        self.positions = []
        self.results = []

    def evaluate_prediction(self, symbol, current_price, predicted_price, threshold_pct=0.001):
        """
        Simule une prédiction Polymarket: "Le prix va-t-il monter/baisser?"
        
        Returns: dict with market simulation
        """
        predicted_return = (predicted_price - current_price) / current_price
        
        if predicted_return > threshold_pct:
            direction = "UP"
            confidence = min(0.95, 0.5 + abs(predicted_return) * 100)
        elif predicted_return < -threshold_pct:
            direction = "DOWN"
            confidence = min(0.95, 0.5 + abs(predicted_return) * 100)
        else:
            direction = "FLAT"
            confidence = 0.5

        # Simulate a Polymarket "Yes" price for the direction
        yes_price = round(confidence, 3)

        return {
            "symbol": symbol,
            "current_price": current_price,
            "predicted_price": round(predicted_price, 2),
            "predicted_return": round(predicted_return, 6),
            "direction": direction,
            "polymarket_yes_price": yes_price,
            "polymarket_no_price": round(1 - yes_price, 3),
            "timestamp": datetime.now().isoformat(),
        }

    def record_outcome(self, position, actual_price):
        """Record the actual outcome when the prediction period ends."""
        direction = position["direction"]
        entry_price = position["current_price"]
        
        if direction == "UP":
            correct = actual_price > entry_price
        elif direction == "DOWN":
            correct = actual_price < entry_price
        else:
            correct = True  # FLAT always "correct" in this sim

        result = {
            "position": position,
            "actual_price": actual_price,
            "correct": correct,
            "would_have_profit": position["polymarket_yes_price"] - (1 if correct else 0),
        }
        self.results.append(result)
        return result

    def get_summary(self):
        if not self.results:
            return {"total": 0, "correct": 0, "accuracy": 0}
        
        total = len(self.results)
        correct = sum(1 for r in self.results if r["correct"])
        return {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total * 100, 1),
        }


# ─── Logger Setup ──────────────────────────────────────────────────────
def setup_logger(log_dir, model_name, symbol, timeframe):
    """Setup detailed file logger + console logger."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"kronos_{model_name}_{symbol}_{timeframe}_{ts}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(f"kronos_{model_name}_{symbol}")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, str(log_file)


# ─── Main Trading Loop ─────────────────────────────────────────────────
def load_model(model_key, device="cpu"):
    """Load a Kronos model and return (predictor, info)."""
    from model import Kronos, KronosTokenizer, KronosPredictor

    info = AVAILABLE_MODELS[model_key]
    ctx = info["context_length"]

    logger.info(f"Chargement tokenizer: {info.get('tokenizer_id', info.get('tokenizer_path'))}")
    if "tokenizer_id" in info:
        tokenizer = KronosTokenizer.from_pretrained(info["tokenizer_id"])
    else:
        tokenizer = KronosTokenizer.from_pretrained(info["tokenizer_path"])

    logger.info(f"Chargement modèle: {info.get('model_id', info.get('model_path'))}")
    if "model_id" in info:
        model = Kronos.from_pretrained(info["model_id"])
    else:
        model = Kronos.from_pretrained(info["model_path"])

    predictor = KronosPredictor(model, tokenizer, max_context=ctx)
    logger.info(f"Modèle chargé: {model_key} ({info['params']} params, ctx={ctx})")

    return predictor, info


def run_trading_loop(args):
    """Main autonomous trading loop."""
    global logger

    # Setup
    model_key = args.model
    symbol = args.symbol
    timeframe = args.timeframe
    capital = args.capital
    lookback = args.lookback or PRESETS.get(timeframe, {}).get("lookback", 120)
    pred_len = args.pred_len or PRESETS.get(timeframe, {}).get("pred_len", 30)
    max_iterations = args.max_iterations or 0  # 0 = infinite

    log_dir = args.log_dir or os.path.join(os.path.dirname(__file__), "logs")
    logger, log_file = setup_logger(log_dir, model_key, symbol, timeframe)

    logger.info("=" * 60)
    logger.info("KRONOS CLI — Mode Autonome")
    logger.info("=" * 60)
    logger.info(f"Modèle:    {model_key} ({AVAILABLE_MODELS[model_key]['description']})")
    logger.info(f"Symbole:   {symbol}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Capital:   ${capital:,.2f} (paper trading)")
    logger.info(f"Lookback:  {lookback} bougies")
    logger.info(f"Pred_len:  {pred_len}")
    logger.info(f"Log file:  {log_file}")
    logger.info("=" * 60)

    # Load model
    t0 = time.time()
    predictor, model_info = load_model(model_key, args.device)
    load_time = time.time() - t0
    logger.info(f"Modèle chargé en {load_time:.1f}s")

    # Initialize paper trading engine
    engine = PaperEngine(capital=capital, risk_pct=args.risk_pct)

    # Initialize Polymarket mock
    polymarket = PolymarketMockTracker()

    # Initialize data feed
    feed = BinancePublicFeed()

    # Get preset
    preset = PRESETS.get(timeframe, PRESETS["M1"])
    signal_threshold = args.signal_threshold or preset["signal_threshold"]
    exit_threshold = args.exit_threshold or preset["exit_threshold"]
    stop_loss_pct = args.stop_loss or preset["stop_loss_pct"]
    take_profit_pct = args.take_profit or preset["take_profit_pct"]
    max_hold_bars = preset.get("max_hold_bars", 120)

    logger.info(f"Signal threshold: {signal_threshold}")
    logger.info(f"Exit threshold:    {exit_threshold}")
    logger.info(f"Stop loss:         {stop_loss_pct*100:.2f}%")
    logger.info(f"Take profit:       {take_profit_pct*100:.2f}%")
    logger.info(f"Max hold bars:     {max_hold_bars}")
    logger.info("")

    # Main loop
    iteration = 0
    tf_seconds = TF_SECONDS.get(timeframe, 60)
    poll_interval = min(5.0, tf_seconds / 12.0)
    last_bar_time = None

    logger.info("🔄 Démarrage de la boucle de trading...")
    logger.info(f"   Polling toutes les {poll_interval:.1f}s (tf={timeframe})")
    logger.info("")

    # ─── Results file for autonomous monitoring ───
    results_dir = Path(log_dir)
    results_file = results_dir / f"results_{model_key}_{symbol}_{timeframe}.json"

    while True:
        try:
            iteration += 1
            if max_iterations > 0 and iteration > max_iterations:
                logger.info(f"Max iterations atteint ({max_iterations})")
                break

            # Wait for new bar
            df, err = feed.get_klines(symbol, timeframe, limit=lookback + 20)
            if err or df is None or len(df) < lookback:
                logger.warning(f"Données insuffisantes: {err}, retry dans {poll_interval}s")
                time.sleep(poll_interval)
                continue

            # Check if new bar
            current_bar_time = df["timestamps"].iloc[-1]
            if last_bar_time is not None and current_bar_time <= last_bar_time:
                time.sleep(poll_interval)
                continue

            last_bar_time = current_bar_time
            df = df.iloc[-lookback:].reset_index(drop=True)

            # ─── Run prediction ───
            x_df = df[["open", "high", "low", "close", "volume", "amount"]]
            x_timestamp = df["timestamps"]
            last_ts = x_timestamp.iloc[-1]
            tf_delta = pd.Timedelta(seconds=tf_seconds)
            y_timestamp = pd.Series([last_ts + tf_delta * (i + 1) for i in range(pred_len)])

            t_start = time.time()
            with torch.inference_mode():
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=1.0,
                    top_p=0.9,
                    sample_count=1,
                )
            pred_time = time.time() - t_start

            current_close = float(df["close"].iloc[-1])
            predicted_end_close = float(pred_df["close"].iloc[-1])
            predicted_return = (predicted_end_close - current_close) / current_close

            # ─── Signal generation ───
            # Long/short logic
            if predicted_return > signal_threshold:
                signal = "long"
            elif predicted_return < -signal_threshold:
                signal = "short"
            elif engine.position is not None:
                pos_dir = engine.position["direction"]
                if pos_dir == "long" and predicted_return < -exit_threshold:
                    signal = "exit"
                elif pos_dir == "short" and predicted_return > exit_threshold:
                    signal = "exit"
                else:
                    signal = "neutral"
            else:
                signal = "neutral"

            # ─── Polymarket mock ───
            pm = polymarket.evaluate_prediction(symbol, current_close, predicted_end_close)

            # ─── Log everything ───
            logger.info(
                f"[#{iteration}] {current_bar_time.strftime('%H:%M:%S')} | "
                f"Close={current_close:.2f} | Pred={predicted_end_close:.2f} | "
                f"Ret={predicted_return:.6f} ({predicted_return*100:.4f}%) | "
                f"Signal={signal} | PredTime={pred_time:.2f}s | "
                f"PM-{pm['direction']}@{pm['polymarket_yes_price']:.3f}"
            )

            # ─── Execute signal ───
            if signal == "exit" and engine.position is not None:
                success, msg = engine.close_position(current_close, reason="signal_exit")
                logger.info(f"  ↳ CLOSE: {msg}")

            elif signal in ("long", "short") and engine.position is None:
                success, msg = engine.open_position(signal, current_close, predicted_return)
                logger.info(f"  ↳ OPEN: {msg}")

            elif signal in ("long", "short") and engine.position is not None:
                pos_dir = engine.position["direction"]
                if (signal == "long" and pos_dir == "short") or (signal == "short" and pos_dir == "long"):
                    success, msg = engine.close_position(current_close, reason="reverse")
                    logger.info(f"  ↳ REVERSE CLOSE: {msg}")
                    success, msg = engine.open_position(signal, current_close, predicted_return)
                    logger.info(f"  ↳ REVERSE OPEN: {msg}")

            # ─── Risk management ───
            should_close, reason = engine.check_risk(
                current_close, max_hold_bars, stop_loss_pct, take_profit_pct
            )
            if should_close:
                success, msg = engine.close_position(current_close, reason=reason)
                logger.info(f"  ↳ RISK CLOSE: {msg}")

            # ─── Equity tracking ───
            equity = engine.get_equity(current_close)
            engine.equity_history.append({
                "time": datetime.now().isoformat(),
                "equity": round(equity, 2),
                "capital": round(engine.capital, 2),
            })

            # ─── Periodic summary (every 30 iterations) ───
            if iteration % 30 == 0:
                summary = engine.get_summary()
                pm_summary = polymarket.get_summary()
                logger.info("-" * 60)
                logger.info(f"📊 RÉSUMÉ (#{iteration})")
                logger.info(f"   Capital: ${summary['current_capital']:.2f} "
                          f"(PnL: ${summary['total_pnl']:.4f} = {summary['total_pnl_pct']:.4f}%)")
                logger.info(f"   Trades: {summary['total_trades']} | "
                          f"Win rate: {summary['win_rate']:.1f}%")
                logger.info(f"   Polymarket mock accuracy: {pm_summary.get('accuracy', 0)}% "
                          f"({pm_summary.get('correct', 0)}/{pm_summary.get('total', 0)})")
                logger.info("-" * 60)

            # ─── Save results to JSON ───
            if iteration % 10 == 0:
                results = {
                    "model": model_key,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "iteration": iteration,
                    "timestamp": datetime.now().isoformat(),
                    "summary": engine.get_summary(),
                    "polymarket": polymarket.get_summary(),
                    "last_signals": engine.signal_log[-20:],
                    "trades": engine.trades[-50:],
                    "equity_curve": engine.equity_history[-100:],
                }
                with open(str(results_file), "w") as f:
                    json.dump(results, f, indent=2, default=str)

            # Wait for next bar
            time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            break
        except Exception as e:
            logger.error(f"Erreur: {str(e)}\n{traceback.format_exc()}")
            time.sleep(10)

    # ─── Final summary ───
    summary = engine.get_summary()
    pm_summary = polymarket.get_summary()
    logger.info("")
    logger.info("=" * 60)
    logger.info("📋 BILAN FINAL")
    logger.info("=" * 60)
    logger.info(f"Modèle:     {model_key} ({AVAILABLE_MODELS[model_key]['description']})")
    logger.info(f"Symbole:    {symbol}")
    logger.info(f"Timeframe:  {timeframe}")
    logger.info(f"Itérations: {iteration}")
    logger.info(f"Capital:    ${summary['current_capital']:.2f} "
               f"(initial: ${summary['initial_capital']:.2f})")
    logger.info(f"PnL total:  ${summary['total_pnl']:.4f} ({summary['total_pnl_pct']:.4f}%)")
    logger.info(f"Trades:     {summary['total_trades']} | "
               f"Win rate: {summary['win_rate']:.1f}%")
    logger.info(f"Commissions: ${summary['total_commission']:.4f}")
    logger.info(f"PM mock:    accuracy={pm_summary.get('accuracy', 0)}% "
               f"({pm_summary.get('correct', 0)}/{pm_summary.get('total', 0)})")
    logger.info(f"Log file:   {log_file}")
    logger.info("=" * 60)

    # Save final results
    final_results = {
        "model": model_key,
        "symbol": symbol,
        "timeframe": timeframe,
        "iterations": iteration,
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "polymarket": pm_summary,
        "all_trades": engine.trades,
        "equity_curve": engine.equity_history,
    }
    final_file = results_dir / f"final_{model_key}_{symbol}_{timeframe}.json"
    with open(str(final_file), "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    logger.info(f"Résultats finaux sauvegardés: {final_file}")

    return summary


# ─── Benchmark Mode ─────────────────────────────────────────────────────
def benchmark_models(symbols=None, timeframe="M1", capital=10000):
    """Test all models on a single prediction and report timing."""
    if symbols is None:
        symbols = ["BTCUSDT"]

    from model import Kronos, KronosTokenizer, KronosPredictor

    feed = BinancePublicFeed()

    print("\n" + "=" * 60)
    print("🔬 KRONOS MODEL BENCHMARK")
    print("=" * 60)

    results = {}
    for model_key in ["mini", "small", "base"]:
        info = AVAILABLE_MODELS[model_key]
        print(f"\n{'─' * 40}")
        print(f"Modèle: {model_key} ({info['params']} params)")
        print(f"{'─' * 40}")

        # Load model
        t0 = time.time()
        if "tokenizer_id" in info:
            tokenizer = KronosTokenizer.from_pretrained(info["tokenizer_id"])
        else:
            tokenizer = KronosTokenizer.from_pretrained(info["tokenizer_path"])

        if "model_id" in info:
            model = Kronos.from_pretrained(info["model_id"])
        else:
            model = Kronos.from_pretrained(info["model_path"])

        predictor = KronosPredictor(model, tokenizer, max_context=info["context_length"])
        load_time = time.time() - t0
        print(f"  Chargement: {load_time:.1f}s")

        for symbol in symbols:
            # Get data
            df, err = feed.get_klines(symbol, timeframe, limit=200)
            if err:
                print(f"  Erreur données {symbol}: {err}")
                continue

            lookback = min(120, len(df) - 1)
            pred_len = 30

            x_df = df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
            x_timestamp = df.iloc[-lookback:]["timestamps"].reset_index(drop=True)
            last_ts = x_timestamp.iloc[-1]
            tf_delta = pd.Timedelta(seconds=TF_SECONDS[timeframe])
            y_timestamp = pd.Series([last_ts + tf_delta * (i + 1) for i in range(pred_len)])

            # Run 3 predictions for average timing
            times = []
            for i in range(3):
                t0 = time.time()
                with torch.inference_mode():
                    pred_df = predictor.predict(
                        df=x_df,
                        x_timestamp=x_timestamp,
                        y_timestamp=y_timestamp,
                        pred_len=pred_len,
                        T=1.0, top_p=0.9, sample_count=1,
                    )
                times.append(time.time() - t0)

            current_close = float(df["close"].iloc[-1])
            pred_close = float(pred_df["close"].iloc[-1])
            pred_return = (pred_close - current_close) / current_close

            avg_time = sum(times) / len(times)
            print(f"  {symbol}: close={current_close:.2f}, pred={pred_close:.2f}, "
                  f"ret={pred_return*100:.4f}%, avg_pred_time={avg_time:.3f}s")

            results[f"{model_key}_{symbol}"] = {
                "model": model_key,
                "symbol": symbol,
                "load_time": round(load_time, 1),
                "pred_time_avg": round(avg_time, 3),
                "current_close": current_close,
                "predicted_close": round(pred_close, 2),
                "predicted_return": round(pred_return * 100, 4),
            }

    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ BENCHMARK")
    print("=" * 60)
    print(f"{'Modèle':<12} {'Symbole':<10} {'Load':>8} {'Pred':>8} {'Ret%':>10}")
    print("-" * 50)
    for key, r in results.items():
        print(f"{r['model']:<12} {r['symbol']:<10} {r['load_time']:>7.1f}s {r['pred_time_avg']:>7.3f}s {r['predicted_return']:>9.4f}%")

    return results


# ─── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kronos CLI — Mode Autonome")
    parser.add_argument("--model", type=str, default="small",
                       choices=list(AVAILABLE_MODELS.keys()),
                       help="Model to use (mini, small, base, xaumodel-local, xaumodel-mini)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                       help="Trading symbol (BTCUSDT, ETHUSDT, etc.)")
    parser.add_argument("--timeframe", type=str, default="M1",
                       choices=list(TF_SECONDS.keys()),
                       help="Timeframe (M1, M5, M15, etc.)")
    parser.add_argument("--capital", type=float, default=10000.0,
                       help="Initial capital for paper trading")
    parser.add_argument("--risk-pct", type=float, default=0.01,
                       help="Risk per trade as fraction of capital (default: 0.01 = 1%%)")
    parser.add_argument("--lookback", type=int, default=None,
                       help="Lookback window (default: from preset)")
    parser.add_argument("--pred-len", type=int, default=None,
                       help="Prediction length (default: from preset)")
    parser.add_argument("--signal-threshold", type=float, default=None,
                       help="Signal threshold (default: from preset)")
    parser.add_argument("--exit-threshold", type=float, default=None,
                       help="Exit threshold (default: from preset)")
    parser.add_argument("--stop-loss", type=float, default=None,
                       help="Stop loss percentage (default: from preset)")
    parser.add_argument("--take-profit", type=float, default=None,
                       help="Take profit percentage (default: from preset)")
    parser.add_argument("--max-iterations", type=int, default=0,
                       help="Max iterations (0 = infinite)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device (cpu or cuda)")
    parser.add_argument("--log-dir", type=str, default=None,
                       help="Log directory")
    parser.add_argument("--paper", action="store_true", default=True,
                       help="Paper trading mode (default)")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark on all models instead of trading")

    args = parser.parse_args()

    if args.benchmark:
        benchmark_models(symbols=["BTCUSDT", "ETHUSDT"], timeframe=args.timeframe)
    else:
        run_trading_loop(args)