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
import random

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
        "lookback": 120, "pred_len": 1, "signal_threshold": 0.0003,
        "exit_threshold": 0.0002, "stop_loss_pct": 0.001,
        "take_profit_pct": 0.002, "max_hold_bars": 5,
    },
    "M5": {
        "lookback": 120, "pred_len": 1, "signal_threshold": 0.0003,
        "exit_threshold": 0.0002, "stop_loss_pct": 0.001,
        "take_profit_pct": 0.002, "max_hold_bars": 5,
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

    def __init__(self, bet_amount=1.0, feed=None, logger=None):
        self.bet_amount = bet_amount
        self.feed = feed
        self.logger = logger
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
        if self.logger:
            self.logger.debug(f"[PM] Fetching market | slug={slug} | symbol={symbol} | window_ts={window_ts}")

        # Check cache
        if slug in self._slug_cache:
            if self.logger:
                self.logger.debug(f"[PM] Cache hit | slug={slug}")
            return self._slug_cache[slug]

        # Skip known failures
        if slug in self._failed_slugs:
            if self.logger:
                self.logger.debug(f"[PM] Skipping known failed slug | slug={slug}")
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
                if self.logger:
                    self.logger.warning(f"[PM] Missing prices after all attempts | slug={slug}")
                self._failed_slugs.add(slug)
                return None

            self._slug_cache[slug] = market_info
            if self.logger:
                self.logger.info(
                    f"[PM] Market found | slug={slug} | "
                    f"up_price={market_info['up_price']:.4f} | "
                    f"down_price={market_info['down_price']:.4f} | "
                    f"volume={market_info['volume']:.2f}"
                )
            return market_info

        except Exception as e:
            if self.logger:
                self.logger.error(f"[PM] Exception fetching market | slug={slug} | error={e}")
            self._failed_slugs.add(slug)
            return None

    def _get_clob_price(self, token_id):
        """Récupère le prix CLOB pour un token donné via l'order book (prix ask = prix d'achat)."""
        if not token_id:
            return None
        try:
            resp = self._get_session().get(
                f"https://clob.polymarket.com/book?token_id={token_id}",
                timeout=5,
            )
            if resp.status_code == 200:
                book = resp.json()
                bids = book.get("bids", [])
                asks = book.get("asks", [])
                best_bid = float(bids[0]["price"]) if bids else None
                best_ask = float(asks[0]["price"]) if asks else None
                # Use ask price as buy price (more conservative / realistic)
                price = best_ask if best_ask is not None else best_bid
                if self.logger:
                    self.logger.debug(
                        f"[PM] CLOB price | token={token_id[:16]}... | "
                        f"ask={best_ask:.4f} | bid={best_bid:.4f} | used={price:.4f}"
                    )
                return price
        except Exception:
            pass
        return None

    def _get_winner_from_api(self, slug):
        """
        Requête l'API Polymarket pour connaître le token vainqueur du marché.
        Retourne 'UP', 'DOWN', ou None si indisponible.
        """
        try:
            resp = self._get_session().get(
                "https://gamma-api.polymarket.com/events",
                params={"slug": slug},
                timeout=10,
            )
            if resp.status_code != 200 or not resp.json():
                return None

            event = resp.json()[0]
            for m in event.get("markets", []):
                winner = m.get("winner", "")
                tokens_str = m.get("clobTokenIds", "[]")
                outcomes_str = m.get("outcomes", "[]")
                try:
                    tokens = json.loads(tokens_str) if isinstance(tokens_str, str) else tokens_str
                    outcomes = json.loads(outcomes_str) if isinstance(outcomes_str, str) else outcomes_str
                except (json.JSONDecodeError, TypeError):
                    continue

                if winner and tokens:
                    # Find which outcome the winning token belongs to
                    for i, tid in enumerate(tokens):
                        if tid.lower() == winner.lower() and i < len(outcomes):
                            outcome = str(outcomes[i]).upper()
                            if "UP" in outcome:
                                return "UP"
                            elif "DOWN" in outcome:
                                return "DOWN"
            return None
        except Exception:
            return None

    def _get_m5_candle(self, symbol, window_ts):
        """
        Récupère la candle M5 exacte pour le timestamp de début de fenêtre.
        Retourne (open_price, close_price) ou (None, None) si non trouvé.
        """
        if self.feed is None:
            return None, None
        try:
            df_m5, err = self.feed.get_klines(symbol, "M5", limit=10)
            if err or df_m5 is None or df_m5.empty:
                return None, None
            # Chercher la candle dont le timestamp correspond au début de la fenêtre
            candle = df_m5[df_m5["timestamps"] == pd.Timestamp(window_ts, unit="s", tz="UTC")]
            if candle.empty:
                return None, None
            return float(candle["open"].iloc[0]), float(candle["close"].iloc[0])
        except Exception:
            return None, None

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
            if self.logger:
                self.logger.info(f"[PM] No bet | FLAT | pred_return={predicted_return:+.6f} < threshold")
            return {
                "direction": "FLAT",
                "market": "no bet (prediction too flat)",
                "position": None,
            }

        direction = "UP" if predicted_return > 0 else "DOWN"

        # Get the NEXT 5-min window (we bet on the upcoming window, not current)
        window_ts = self._get_next_window_ts()
        window_start = datetime.fromtimestamp(window_ts, tz=timezone.utc).strftime("%H:%M")
        window_end = datetime.fromtimestamp(window_ts + 300, tz=timezone.utc).strftime("%H:%M")

        if self.logger:
            self.logger.info(
                f"[PM] Placing bet | symbol={symbol} | direction={direction} | "
                f"window={window_start}-{window_end} | current={current_price:.2f} | predicted={predicted_price:.2f}"
            )

        market = self._fetch_market(symbol, window_ts)

        if market is None:
            if self.logger:
                self.logger.warning(f"[PM] No market found | symbol={symbol} | window={window_start}-{window_end}")
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
            if self.logger:
                self.logger.warning(f"[PM] Invalid buy price | side={pm_side} | price={buy_price}")
            return {
                "direction": direction,
                "market": f"price too low or None ({buy_price})",
                "position": None,
            }

        # Calculate shares and potential payout
        shares = self.bet_amount / buy_price
        potential_payout = shares  # Each share pays €1 if correct
        potential_profit = potential_payout - self.bet_amount

        if self.logger:
            self.logger.info(
                f"[PM] Bet placed | side={pm_side} | buy_price={buy_price:.4f} | "
                f"shares={shares:.4f} | potential_payout={potential_payout:.2f}€ | "
                f"potential_profit={potential_profit:+.2f}€"
            )

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
            "window_start": window_start,
            "window_end": window_end,
            "position": position,
        }

    def settle_all_expired(self, current_prices):
        """
        Règle les bets dont la fenêtre 5-min est terminée.

        Pour chaque position expirée, on détermine UP ou DOWN en comparant
        le prix d'ouverture et de clôture de la candle M5 exacte de la fenêtre.
        C'est exactement comment Polymarket settle ces marchés.
        """
        settled = []
        now_ts = int(datetime.now(timezone.utc).timestamp())

        for pos in list(self.positions):
            # Check if the 5-min window has ended
            if now_ts < pos["window_end_ts"]:
                continue  # Window still open

            symbol = pos["symbol"]
            window_ts = pos["window_start_ts"]
            pm_slug = pos.get("pm_slug", "")

            # 1) Try to get the official winner from Polymarket API
            api_winner = self._get_winner_from_api(pm_slug) if pm_slug else None
            if api_winner:
                actual_direction = api_winner
                settlement_source = "polymarket_api"
                candle_open = candle_close = None  # not needed
                if self.logger:
                    self.logger.info(
                        f"[PM] Settlement (Polymarket API) | symbol={symbol} | "
                        f"slug={pm_slug} | actual={actual_direction}"
                    )
            else:
                # 2) Fallback: get the exact M5 candle for this window
                candle_open, candle_close = self._get_m5_candle(symbol, window_ts)

                if candle_open is not None and candle_close is not None:
                    actual_direction = "UP" if candle_close > candle_open else "DOWN"
                    settlement_source = "m5_candle"
                    if self.logger:
                        self.logger.info(
                            f"[PM] Settlement (candle) | symbol={symbol} | window={window_ts} | "
                            f"open={candle_open:.2f} | close={candle_close:.2f} | "
                            f"actual={actual_direction}"
                        )
                else:
                    # 3) Last fallback: use current price vs entry price (approximate)
                    entry_price = pos["current_price"]
                    current_price = current_prices.get(symbol, entry_price)
                    actual_direction = "UP" if current_price > entry_price else "DOWN"
                    candle_open = candle_close = None  # not available
                    settlement_source = "fallback"
                    if self.logger:
                        self.logger.warning(
                            f"[PM] Settlement (fallback) | symbol={symbol} | window={window_ts} | "
                            f"entry={entry_price:.2f} | current={current_price:.2f} | "
                            f"actual={actual_direction}"
                        )

            # Did our bet win?
            won = (pos["direction"] == actual_direction)

            if won:
                gross_payout = pos["shares"]  # Each share pays €1
                fee = gross_payout * 0.02       # Polymarket ~2% fee on winnings
                net_payout = gross_payout - fee
                profit = net_payout - pos["bet_amount"]
            else:
                gross_payout = 0
                fee = 0
                net_payout = 0
                profit = -pos["bet_amount"]

            self.total_pnl += profit

            result = {
                "question": pos["pm_question"],
                "slug": pos["pm_slug"],
                "side": pos["pm_side"],
                "buy_price": pos["pm_buy_price"],
                "bet_amount": pos["bet_amount"],
                "entry_price": pos["current_price"],
                "exit_price": candle_close if candle_close is not None else current_prices.get(symbol),
                "actual_direction": actual_direction,
                "won": won,
                "gross_payout": round(gross_payout, 2),
                "fee_eur": round(fee, 2),
                "payout": round(net_payout, 2),
                "profit_eur": round(profit, 2),
                "settlement_source": settlement_source,
                "window_start": pos["window_start_ts"],
                "window_end": pos["window_end_ts"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.results.append(result)
            self.positions.remove(pos)
            settled.append(result)

            if self.logger:
                status_icon = "✅" if won else "❌"
                self.logger.info(
                    f"[PM] Settled | {status_icon} {pos['direction']} vs {actual_direction} | "
                    f"profit={profit:+.2f}€ | gross={gross_payout:.2f}€ | fee={fee:.2f}€ | "
                    f"source={settlement_source}"
                )

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


# ─── Colored Formatter ──────────────────────────────────────────────────
class ColoredFormatter(logging.Formatter):
    """Formatter ANSI coloré pour la console."""

    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[35m",  # magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def __init__(self, fmt=None, datefmt=None, use_color=True):
        super().__init__(fmt, datefmt)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in self.COLORS:
            record.levelname = f"{self.BOLD}{self.COLORS[levelname]}{levelname}{self.RESET}"
        formatted = super().format(record)
        record.levelname = levelname  # restore
        return formatted


# ─── Logger Setup ──────────────────────────────────────────────────────
def setup_logger(log_dir, model_name, symbol, timeframe):
    """Setup detailed file logger + console logger with colors."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"kronos_{model_name}_{symbol}_{timeframe}_{ts}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    plain_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    colored_formatter = ColoredFormatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        use_color=True,
    )

    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(plain_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(colored_formatter)

    logger = logging.getLogger(f"kronos_{model_name}_{symbol}")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, str(log_file)


def load_kronos(model_key, device=None):
    """Load Kronos model + tokenizer from registry."""
    from model import Kronos, KronosTokenizer, KronosPredictor

    cfg = AVAILABLE_MODELS[model_key]

    if "model_id" in cfg:
        tokenizer = KronosTokenizer.from_pretrained(cfg["tokenizer_id"])
        model = Kronos.from_pretrained(cfg["model_id"])
    else:
        tokenizer = KronosTokenizer.from_pretrained(cfg["tokenizer_path"])
        model = Kronos.from_pretrained(cfg["model_path"])

    predictor = KronosPredictor(model, tokenizer, max_context=cfg["context_length"], device=device)
    return predictor


def predict_direction(predictor, df, lookback, pred_len, timeframe_secs, sample_count=1):
    """
    Run Kronos prediction and return (direction, current_price, predicted_price, pred_df_or_error).

    direction: "UP", "DOWN", "FLAT", or None on failure
    5th element: error string if prediction failed, None otherwise
    """
    if len(df) < lookback:
        return None, None, None, None, "Not enough data"

    from model.kronos import set_inference_seed
    set_inference_seed()

    x_df = df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]].copy()
    x_timestamp = df.iloc[-lookback:]["timestamps"]

    # Ensure x_timestamp is a Series (not DatetimeIndex) for .dt access
    if isinstance(x_timestamp, pd.DatetimeIndex):
        x_timestamp = pd.Series(x_timestamp, name="timestamps")

    future_times = pd.date_range(
        start=x_timestamp.iloc[-1] + pd.Timedelta(seconds=timeframe_secs),
        periods=pred_len,
        freq=pd.Timedelta(seconds=timeframe_secs),
    )

    # Ensure future_times is a Series (not DatetimeIndex) for .dt access
    if isinstance(future_times, pd.DatetimeIndex):
        future_times = pd.Series(future_times, name="timestamps")

    try:
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=future_times,
            pred_len=pred_len,
            T=0.1,
            top_p=1.0,
            sample_count=1,
            sample_logits=False,
            verbose=False,
        )
    except Exception as e:
        return None, None, None, None, str(e)

    current_close = x_df["close"].iloc[-1]
    predicted_close = pred_df["close"].iloc[-1]

    predicted_return = (predicted_close - current_close) / current_close

    if predicted_return > 0.0003:
        direction = "UP"
    elif predicted_return < -0.0003:
        direction = "DOWN"
    else:
        direction = "FLAT"

    return direction, current_close, predicted_close, pred_df, None


def seconds_to_next_window():
    """Seconds until the next 5-min boundary (Polymarket window start)."""
    now = datetime.now(timezone.utc)
    ts = now.timestamp()
    next_boundary = (int(ts) // 300 + 1) * 300
    return next_boundary - ts


def run_polymarket_loop(args):
    """Main Polymarket 5-min paper trading loop."""
    logger, log_file = setup_logger(
        args.log_dir, args.model, args.symbol, args.timeframe
    )

    logger.info("=" * 70)
    logger.info("KRONOS POLYMARKET 5-MIN PAPER TRADER")
    logger.info("=" * 70)
    logger.info(f"Model:     {args.model} ({AVAILABLE_MODELS[args.model]['params']})")
    logger.info(f"Symbol:    {args.symbol}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Bet size:  {args.bet_amount}€")
    logger.info(f"Log file:  {log_file}")
    logger.info("=" * 70)

    # Load model
    logger.info("Loading Kronos model...")
    predictor = load_kronos(args.model, device=args.device)
    logger.info("Model loaded.")

    # Load ensemble voter model if available (xaumodel-local + xaumodel-mini)
    predictor_voter = None
    voter_key = None
    if args.model == "xaumodel-local" and "xaumodel-mini" in AVAILABLE_MODELS:
        voter_key = "xaumodel-mini"
    elif args.model == "xaumodel-mini" and "xaumodel-local" in AVAILABLE_MODELS:
        voter_key = "xaumodel-local"

    if voter_key:
        logger.info(f"Loading ensemble voter model: {voter_key}...")
        predictor_voter = load_kronos(voter_key, device=args.device)
        logger.info("Voter model loaded. Ensemble voting enabled.")

    # Init feed + tracker
    feed = BinancePublicFeed()
    pm_tracker = Polymarket5MinTracker(bet_amount=args.bet_amount, feed=feed, logger=logger)

    # Load presets
    tf_secs = TF_SECONDS[args.timeframe]
    preset = PRESETS.get(args.timeframe, PRESETS["M5"])
    lookback = preset["lookback"]
    pred_len = preset["pred_len"]

    logger.info(f"Lookback: {lookback} bars | Pred len: {pred_len} | Sample count: {args.samples}")

    # Fetch initial data
    logger.info("Fetching initial market data...")
    df, err = feed.get_klines(args.symbol, args.timeframe, limit=500)
    if err:
        logger.error(f"Failed to fetch data: {err}")
        return
    logger.info(f"Got {len(df)} candles. Last close: {df['close'].iloc[-1]:.2f}")

    last_bar_time = df["timestamps"].iloc[-1]
    cycle = 0

    # ─── Main Loop ────────────────────────────────────────────────────
    logger.info("Starting trading loop. Ctrl+C to stop.")
    logger.info("")

    try:
        while True:
            cycle += 1

            # 1) Settle expired bets (compare real Binance prices)
            current_price = feed.get_price(args.symbol)
            if current_price:
                settled = pm_tracker.settle_all_expired({args.symbol: current_price})
                for r in settled:
                    icon = "+" if r["won"] else "-"
                    logger.info(
                        f"  SETTLED | {icon} {r['side']:4s} | "
                        f"€{r['profit_eur']:+.2f} | "
                        f"Entry {r['entry_price']:.2f} -> Exit {r['exit_price']:.2f} | "
                        f"Actual: {r['actual_direction']}"
                    )

            # 2) Wait until ~15s before next 5-min window
            secs_to_window = seconds_to_next_window()
            if secs_to_window > 20:
                wait_secs = secs_to_window - 15
                next_window = datetime.now(timezone.utc) + timedelta(seconds=wait_secs + 15)
                logger.info(
                    f"[Cycle {cycle}] Waiting {wait_secs:.0f}s for next window "
                    f"(starts at {next_window.strftime('%H:%M:%S')} UTC)..."
                )
                time.sleep(wait_secs)

            # 3) Fetch fresh data right before the window
            df, err = feed.get_klines(args.symbol, args.timeframe, limit=500)
            if err:
                logger.warning(f"Data fetch failed: {err} — skipping cycle")
                time.sleep(60)
                continue

            current_price = df["close"].iloc[-1]
            logger.info(f"[Cycle {cycle}] Price: {current_price:.2f} | Candles: {len(df)}")

            # 4) Run Kronos prediction
            direction, cur_price, pred_price, pred_df, pred_err = predict_direction(
                predictor, df, lookback, pred_len, tf_secs, sample_count=args.samples
            )

            if direction is None:
                logger.warning(f"Prediction failed: {pred_err}")
                time.sleep(60)
                continue

            # Ensemble voting: check voter model and override if disagreement
            if predictor_voter is not None and direction != "FLAT":
                voter_direction, _, voter_pred_price, _, voter_err = predict_direction(
                    predictor_voter, df, lookback, pred_len, tf_secs, sample_count=1
                )
                if voter_err is None and voter_direction != direction:
                    logger.info(f"  VOTE DISAGREE | Main: {direction} | Voter: {voter_direction} -> FLAT")
                    direction = "FLAT"
                elif voter_err is None:
                    logger.info(f"  VOTE CONFIRM  | Both models: {direction}")

            pred_return = (pred_price - cur_price) / cur_price if cur_price else 0
            logger.info(
                f"  KRONOS | {direction:4s} | "
                f"Predicted: {pred_price:.2f} | "
                f"Return: {pred_return*100:+.4f}%"
            )

            # 5) Place Polymarket bet (if not FLAT)
            if direction == "FLAT":
                logger.info("  -> No bet (prediction too flat)")
            else:
                result = pm_tracker.place_bet(args.symbol, direction, cur_price, pred_price)
                if result.get("position"):
                    logger.info(
                        f"  -> BET {result['polymarket_side']:4s} | "
                        f"Buy @ {result['polymarket_buy_price']:.4f} | "
                        f"Profit potentiel: {result['potential_profit_eur']:+.2f}€ | "
                        f"Window: {result['window_start']}-{result['window_end']} UTC"
                    )
                else:
                    logger.info(f"  -> No bet: {result.get('market', 'unknown')}")

            # 6) Print summary every 10 cycles
            if cycle % 10 == 0:
                s = pm_tracker.get_summary()
                logger.info(
                    f"  SUMMARY | Bets: {s['total_bets']} | "
                    f"Wins: {s['wins']} | "
                    f"Accuracy: {s['accuracy']:.1f}% | "
                    f"P&L: {s['total_pnl_eur']:+.2f}€ | "
                    f"ROI: {s['roi_pct']:+.1f}%"
                )

            # 7) Wait for the window to start + a small buffer
            secs_to_window = seconds_to_next_window()
            if secs_to_window > 0:
                time.sleep(secs_to_window + 2)

    except KeyboardInterrupt:
        logger.info("")
        logger.info("Stopped by user.")

    # ─── Final Report ─────────────────────────────────────────────────
    s = pm_tracker.get_summary()
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL REPORT — Polymarket 5-Min Paper Trading")
    logger.info("=" * 70)
    logger.info(f"  Total bets:     {s['total_bets']}")
    logger.info(f"  Wins:           {s['wins']}")
    logger.info(f"  Accuracy:       {s['accuracy']:.1f}%")
    logger.info(f"  Open bets:      {s['open_bets']}")
    logger.info(f"  Total wagered:  {s['total_wagered_eur']:.2f}€")
    logger.info(f"  Total P&L:      {s['total_pnl_eur']:+.2f}€")
    logger.info(f"  ROI:            {s['roi_pct']:+.1f}%")
    logger.info("=" * 70)

    # Save results to JSON
    results_file = Path(args.log_dir) / f"pm_results_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump({
            "summary": s,
            "settled_bets": pm_tracker.results,
            "open_bets": [p for p in pm_tracker.positions],
        }, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")


def parse_args():
    p = argparse.ArgumentParser(description="Kronos Polymarket 5-Min Paper Trader")
    p.add_argument("--model", choices=list(AVAILABLE_MODELS.keys()), default="small",
                   help="Kronos model to use (default: small)")
    p.add_argument("--symbol", default="BTCUSDT",
                   help="Trading pair (default: BTCUSDT)")
    p.add_argument("--timeframe", choices=["M1", "M5"], default="M5",
                   help="Candle timeframe (default: M5 — matches Polymarket 5-min windows)")
    p.add_argument("--bet-amount", type=float, default=1.0,
                   help="Bet amount in EUR per Polymarket bet (default: 1.0)")
    p.add_argument("--samples", type=int, default=5,
                   help="Number of Kronos samples to average (default: 5)")
    p.add_argument("--device", default=None,
                   help="Torch device (default: auto)")
    p.add_argument("--log-dir", default="./logs",
                   help="Directory for log files (default: ./logs)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_polymarket_loop(args)