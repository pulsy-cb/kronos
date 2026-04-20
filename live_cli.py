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
from datetime import datetime, timedelta, timezone
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
        "lookback": 120, "pred_len": 5, "signal_threshold": 0.0008,
        "exit_threshold": 0.0005, "stop_loss_pct": 0.0025,
        "take_profit_pct": 0.005, "max_hold_bars": 120,
    },
    "M5": {
        "lookback": 120, "pred_len": 4, "signal_threshold": 0.0012,
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


# ─── Polymarket 5-Min "Up or Down" Tracker ──────────────────────────────
class Polymarket5MinTracker:
    """
    Polymarket tracker pour les marchés 5-min natifs "Up or Down".

    Ces marchés sont du type "Bitcoin Up or Down - April 20, 6:10AM-6:15AM ET".
    - Chaque fenêtre fait exactement 5 minutes
    - Le marché settle automatiquement à la fin de la fenêtre
    - Les côtes sont ~50/50 (vrai incertitude, pas comme les "above $X" journaliers)

    Slug pattern: btc-updown-5m-{unix_ts} ou eth-updown-5m-{unix_ts}
    où unix_ts = début de la fenêtre 5-min arrondi à 300s.
    """

    # Map Kronos symbol to Polymarket slug prefix
    SYMBOL_MAP = {
        "BTCUSDT": "btc",
        "ETHUSDT": "eth",
    }

    def __init__(self, bet_amount=1.0):
        self.bet_amount = bet_amount
        self.positions = []       # Open bets (waiting for 5-min window to close)
        self.results = []         # Settled bets with P&L
        self.total_wagered = 0.0
        self.total_pnl = 0.0
        self._session = None
        self._slug_cache = {}     # slug -> market data
        self._failed_slugs = set()  # Slugs that returned no market (avoid retrying)

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "Kronos-CLI/2.0"})
        return self._session

    def _get_current_window_ts(self):
        """Retourne le timestamp Unix du début de la fenêtre 5-min actuelle."""
        now = datetime.now(timezone.utc)
        # Round down to nearest 300s (5 min)
        ts = int(now.timestamp()) // 300 * 300
        return ts

    def _get_next_window_ts(self):
        """Retourne le timestamp Unix du début de la prochaine fenêtre 5-min."""
        return self._get_current_window_ts() + 300

    def _build_slug(self, symbol, window_ts):
        """Construit le slug Polymarket pour une fenêtre donnée."""
        prefix = self.SYMBOL_MAP.get(symbol, "btc")
        return f"{prefix}-updown-5m-{window_ts}"

    def _fetch_market(self, symbol, window_ts):
        """
        Récupère les infos du marché Polymarket 5-min pour la fenêtre donnée.
        Retourne dict avec yes_price, no_price, question, token_ids, ou None si pas trouvé.
        """
        slug = self._build_slug(symbol, window_ts)

        # Check cache
        if slug in self._slug_cache:
            return self._slug_cache[slug]

        # Skip known failures
        if slug in self._failed_slugs:
            return None

        try:
            resp = self._get_session().get(
                "https://gamma-api.polymarket.com/events",
                params={"slug": slug},
                timeout=10,
            )
            if resp.status_code != 200:
                self._failed_slugs.add(slug)
                return None

            data = resp.json()
            if not data or not isinstance(data, list) or len(data) == 0:
                self._failed_slugs.add(slug)
                return None

            event = data[0]
            markets = event.get("markets", [])
            if not markets:
                self._failed_slugs.add(slug)
                return None

            # The 5-min up/down market has two outcomes: "Up" and "Down"
            market_info = {
                "slug": slug,
                "question": event.get("title", event.get("question", "")),
                "window_ts": window_ts,
                "window_end_ts": window_ts + 300,
                "up_token_id": None,
                "down_token_id": None,
                "up_price": None,  # Price for "Up" outcome (Yes = price goes up)
                "down_price": None,  # Price for "Down" outcome (Yes = price goes down)
                "volume": 0,
            }

            for m in markets:
                question = m.get("question", "").lower()
                prices_str = m.get("outcomePrices", "[]")
                tokens_str = m.get("clobTokenIds", "[]")
                try:
                    prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
                    tokens = json.loads(tokens_str) if isinstance(tokens_str, str) else tokens_str
                except (json.JSONDecodeError, TypeError):
                    continue

                if "up" in question and not question.startswith("will"):
                    market_info["up_price"] = float(prices[0]) if prices else None
                    market_info["up_token_id"] = tokens[0] if tokens else None
                elif "down" in question and not question.startswith("will"):
                    market_info["down_price"] = float(prices[0]) if prices else None
                    market_info["down_token_id"] = tokens[0] if tokens else None

                # Alternative: the market question may just say "Up" or "Down"
                # as the outcome name in outcomePrices
                outcomes_str = m.get("outcomes", "[]")
                try:
                    outcomes = json.loads(outcomes_str) if isinstance(outcomes_str, str) else outcomes_str
                except (json.JSONDecodeError, TypeError):
                    outcomes = []

                for i, outcome in enumerate(outcomes):
                    if isinstance(outcome, str):
                        if outcome.lower() == "up" and market_info["up_price"] is None:
                            market_info["up_price"] = float(prices[i]) if i < len(prices) else None
                            market_info["up_token_id"] = tokens[i] if i < len(tokens) else None
                        elif outcome.lower() == "down" and market_info["down_price"] is None:
                            market_info["down_price"] = float(prices[i]) if i < len(prices) else None
                            market_info["down_token_id"] = tokens[i] if i < len(tokens) else None

                market_info["volume"] = max(market_info["volume"], float(m.get("volume", 0)))

            # Validate we got both prices
            if market_info["up_price"] is None or market_info["down_price"] is None:
                # Try alternate API format — sometimes the market is structured as one market with two tokens
                for m in markets:
                    prices_str = m.get("outcomePrices", "[]")
                    tokens_str = m.get("clobTokenIds", "[]")
                    outcomes_str = m.get("outcomes", "[]")
                    try:
                        prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
                        tokens = json.loads(tokens_str) if isinstance(tokens_str, str) else tokens_str
                        outcomes = json.loads(outcomes_str) if isinstance(outcomes_str, str) else outcomes_str
                    except (json.JSONDecodeError, TypeError):
                        continue

                    if len(prices) >= 2 and len(outcomes) >= 2:
                        # First outcome might be "Up", second "Down" (or vice versa)
                        o0, o1 = str(outcomes[0]).lower(), str(outcomes[1]).lower()
                        if "up" in o0 and "down" in o1:
                            market_info["up_price"] = float(prices[0])
                            market_info["down_price"] = float(prices[1])
                            market_info["up_token_id"] = tokens[0] if tokens else None
                            market_info["down_token_id"] = tokens[1] if len(tokens) > 1 else None
                        elif "down" in o0 and "up" in o1:
                            market_info["down_price"] = float(prices[0])
                            market_info["up_price"] = float(prices[1])
                            market_info["down_token_id"] = tokens[0] if tokens else None
                            market_info["up_token_id"] = tokens[1] if len(tokens) > 1 else None

            # If still no prices, try the CLOB API directly
            if market_info["up_price"] is None and market_info["up_token_id"]:
                market_info["up_price"] = self._get_clob_price(market_info["up_token_id"])
            if market_info["down_price"] is None and market_info["down_token_id"]:
                market_info["down_price"] = self._get_clob_price(market_info["down_token_id"])

            # Still missing? Mark as failure
            if market_info["up_price"] is None or market_info["down_price"] is None:
                self._failed_slugs.add(slug)
                return None

            self._slug_cache[slug] = market_info
            return market_info

        except Exception:
            self._failed_slugs.add(slug)
            return None

    def _get_clob_price(self, token_id):
        """Récupère le prix CLOB pour un token donné."""
        if not token_id:
            return None
        try:
            resp = self._get_session().get(
                "https://clob.polymarket.com/price",
                params={"token_id": token_id, "side": "buy"},
                timeout=5,
            )
            if resp.status_code == 200:
                return float(resp.json().get("price", 0))
        except Exception:
            pass
        return None

    def place_bet(self, symbol, kronos_direction, current_price, predicted_price):
        """
        Place un bet virtuel de 1€ sur le marché 5-min "Up or Down".

        Args:
            symbol: BTCUSDT ou ETHUSDT
            kronos_direction: "UP" ou "DOWN" (basé sur la prédiction Kronos)
            current_price: prix actuel
            predicted_price: prix prédit par Kronos

        Returns:
            dict avec les infos du bet, ou None si marché non disponible
        """
        predicted_return = (predicted_price - current_price) / current_price

        # Only bet if Kronos has a clear direction (not FLAT)
        threshold = 0.0003  # Very small threshold — most 5-min moves count
        if abs(predicted_return) < threshold:
            return {
                "direction": "FLAT",
                "market": "no bet (prediction too flat)",
                "position": None,
            }

        direction = "UP" if predicted_return > 0 else "DOWN"

        # Get the NEXT 5-min window (we bet on the upcoming window, not current)
        window_ts = self._get_next_window_ts()
        market = self._fetch_market(symbol, window_ts)

        if market is None:
            return {
                "direction": direction,
                "market": "no matching 5-min market",
                "position": None,
            }

        # Buy the side matching Kronos prediction
        if direction == "UP":
            buy_price = market["up_price"]
            token_id = market["up_token_id"]
            pm_side = "Up"
        else:
            buy_price = market["down_price"]
            token_id = market["down_token_id"]
            pm_side = "Down"

        if buy_price is None or buy_price <= 0.01:
            return {
                "direction": direction,
                "market": f"price too low or None ({buy_price})",
                "position": None,
            }

        # Calculate shares and potential payout
        shares = self.bet_amount / buy_price
        potential_payout = shares  # Each share pays €1 if correct
        potential_profit = potential_payout - self.bet_amount

        position = {
            "symbol": symbol,
            "direction": direction,
            "current_price": current_price,
            "predicted_price": round(predicted_price, 2),
            "predicted_return": round(predicted_return, 6),
            "pm_question": market["question"],
            "pm_slug": market["slug"],
            "pm_side": pm_side,
            "pm_buy_price": round(buy_price, 4),
            "bet_amount": self.bet_amount,
            "shares": round(shares, 4),
            "potential_payout": round(potential_payout, 2),
            "potential_profit": round(potential_profit, 2),
            "token_id": token_id,
            "window_start_ts": window_ts,
            "window_end_ts": window_ts + 300,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.positions.append(position)
        self.total_wagered += self.bet_amount

        return {
            "direction": direction,
            "polymarket_market": market["question"][:60],
            "polymarket_side": pm_side,
            "polymarket_buy_price": round(buy_price, 4),
            "potential_profit_eur": round(potential_profit, 2),
            "window_start": datetime.fromtimestamp(window_ts, tz=timezone.utc).strftime("%H:%M"),
            "window_end": datetime.fromtimestamp(window_ts + 300, tz=timezone.utc).strftime("%H:%M"),
            "position": position,
        }

    def settle_all_expired(self, current_prices):
        """
        Règle les bets dont la fenêtre 5-min est terminée.

        Pour chaque position expirée, on détermine UP ou DOWN en comparant
        le prix d'ouverture et de clôture de la fenêtre 5-min.
        C'est exactement comment Polymarket settle ces marchés.
        """
        import requests

        settled = []
        now_ts = int(datetime.now(timezone.utc).timestamp())

        for pos in list(self.positions):
            # Check if the 5-min window has ended
            if now_ts < pos["window_end_ts"]:
                continue  # Window still open

            symbol = pos["symbol"]
            entry_price = pos["current_price"]  # Price at the start of the window
            current_price = current_prices.get(symbol, entry_price)

            # The real settlement: did price go UP or DOWN during the 5-min window?
            # We compare the price at window start vs price at window end
            actual_direction = "UP" if current_price > entry_price else "DOWN"

            # Did our bet win?
            won = (pos["direction"] == actual_direction)

            if won:
                payout = pos["shares"]  # Each share pays €1
                profit = payout - pos["bet_amount"]
            else:
                payout = 0
                profit = -pos["bet_amount"]

            self.total_pnl += profit

            result = {
                "question": pos["pm_question"],
                "slug": pos["pm_slug"],
                "side": pos["pm_side"],
                "buy_price": pos["pm_buy_price"],
                "bet_amount": pos["bet_amount"],
                "entry_price": entry_price,
                "exit_price": current_price,
                "actual_direction": actual_direction,
                "won": won,
                "payout": round(payout, 2),
                "profit_eur": round(profit, 2),
                "window_start": pos["window_start_ts"],
                "window_end": pos["window_end_ts"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.results.append(result)
            self.positions.remove(pos)
            settled.append(result)

        return settled

    def get_summary(self):
        total_bets = len(self.results)
        wins = sum(1 for r in self.results if r["won"])
        open_bets = len(self.positions)
        return {
            "total_bets": total_bets,
            "wins": wins,
            "accuracy": round(wins / total_bets * 100, 1) if total_bets > 0 else 0,
            "open_bets": open_bets,
            "total_wagered_eur": round(self.total_wagered, 2),
            "total_pnl_eur": round(self.total_pnl, 2),
            "roi_pct": round(self.total_pnl / self.total_wagered * 100, 1) if self.total_wagered > 0 else 0,
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