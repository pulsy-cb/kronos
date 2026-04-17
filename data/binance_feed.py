"""
Flux de donnees en temps reel depuis Binance Futures.
Implementation de BrokerFeed pour Binance USDT-M Futures.
Supporte le mode testnet pour les tests.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading

from data.broker_feed import BrokerFeed

TF_MAP_BINANCE = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d",
}


class BinanceLiveFeed(BrokerFeed):
    """Flux de donnees en temps reel depuis Binance Futures."""

    def __init__(self, api_key=None, api_secret=None, testnet=True):
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._client = None
        self._connected = False
        self._lock = threading.Lock()

    def connect(self, **kwargs):
        """Connecte a Binance Futures (testnet ou live)."""
        from binance.client import Client
        from binance.exceptions import BinanceAPIException

        try:
            if self._testnet:
                self._client = Client(
                    self._api_key,
                    self._api_secret,
                    testnet=True,
                )
                self._client.FUTURES_URL = "https://testnet.binancefuture.com"
            else:
                self._client = Client(
                    self._api_key,
                    self._api_secret,
                )

            self._client.futures_account()

            self._connected = True
            mode = "TESTNET" if self._testnet else "LIVE"
            return True, f"Connecte Binance Futures ({mode})"

        except BinanceAPIException as e:
            return False, f"Binance connexion echouee: {e.message}"
        except Exception as e:
            return False, f"Erreur connexion Binance: {str(e)}"

    def disconnect(self):
        """Deconnecte Binance."""
        with self._lock:
            if self._client:
                try:
                    self._client.close_connection()
                except Exception:
                    pass
                self._client = None
            self._connected = False

    def is_connected(self):
        return self._connected

    def get_account_info(self):
        """Retourne les infos du compte Binance Futures standardisees."""
        if not self._connected or not self._client:
            return None
        try:
            account = self._client.futures_account()
            total_balance = float(account.get("totalWalletBalance", 0))
            total_unrealized = float(account.get("totalUnrealizedProfit", 0))
            available = float(account.get("availableBalance", 0))
            total_margin = float(account.get("totalMarginBalance", 0))
            leverage = int(account.get("maxWithdrawAmount", 0))

            return {
                "login": self._api_key[-8:] if self._api_key and len(self._api_key) > 8 else "binance",
                "server": f"Binance Futures {'Testnet' if self._testnet else 'Live'}",
                "balance": total_balance,
                "equity": total_balance + total_unrealized,
                "margin": total_margin - available,
                "free_margin": available,
                "profit": total_unrealized,
                "currency": "USDT",
                "leverage": leverage if leverage > 0 else 20,
            }
        except Exception as e:
            print(f"Erreur get_account_info Binance: {e}")
            return None

    def get_symbols(self):
        """Liste les symboles Binance Futures disponibles."""
        if not self._connected or not self._client:
            return []
        try:
            info = self._client.futures_exchange_info()
            return [s["symbol"] for s in info.get("symbols", []) if s.get("status") == "TRADING"]
        except Exception:
            return []

    def get_symbol_info(self, symbol):
        """Retourne les infos d'un symbole Binance Futures standardisees."""
        if not self._connected or not self._client:
            return None
        try:
            info = self._client.futures_exchange_info()
            for s in info.get("symbols", []):
                if s["symbol"] == symbol:
                    filters = {f["filterType"]: f for f in s.get("filters", [])}
                    lot_filter = filters.get("LOT_SIZE", {})
                    price_filter = filters.get("PRICE_FILTER", {})
                    tick_size = float(price_filter.get("tickSize", "0.01"))
                    digits = max(0, -int(np.log10(tick_size))) if tick_size > 0 else 2

                    return {
                        "name": s["symbol"],
                        "description": s.get("pair", s["symbol"]),
                        "digits": digits,
                        "point": tick_size,
                        "trade_contract_size": 1.0,
                        "volume_min": float(lot_filter.get("minQty", "0.001")),
                        "volume_max": float(lot_filter.get("maxQty", "10000")),
                        "volume_step": float(lot_filter.get("stepSize", "0.001")),
                        "trade_mode": 4 if s.get("status") == "TRADING" else 0,
                        "spread": 0,
                    }
            return None
        except Exception as e:
            print(f"Erreur get_symbol_info Binance: {e}")
            return None

    def get_current_tick(self, symbol):
        """Retourne le tick actuel pour un symbole Binance Futures."""
        if not self._connected or not self._client:
            return None
        try:
            ticker = self._client.futures_mark_price(symbol=symbol)
            bid = float(ticker.get("markPrice", 0))
            last = float(ticker.get("markPrice", 0))
            return {
                "bid": bid,
                "ask": bid,
                "last": last,
                "volume": 0,
                "time": datetime.fromtimestamp(int(ticker.get("time", 0)) / 1000),
                "spread": 0,
            }
        except Exception as e:
            try:
                book = self._client.futures_orderbook_ticker(symbol=symbol)
                bid = float(book.get("bidPrice", 0))
                ask = float(book.get("askPrice", 0))
                return {
                    "bid": bid,
                    "ask": ask,
                    "last": (bid + ask) / 2,
                    "volume": 0,
                    "time": datetime.now(),
                    "spread": ask - bid,
                }
            except Exception:
                return None

    def get_latest_candles(self, symbol, timeframe, count, as_kronos=True):
        """Recupere les N dernieres bougies depuis Binance Futures."""
        if not self._connected or not self._client:
            return None, "Binance not connected"

        if timeframe not in TF_MAP_BINANCE:
            return None, f"Timeframe inconnu: {timeframe}. Disponibles: {list(TF_MAP_BINANCE.keys())}"

        interval = TF_MAP_BINANCE[timeframe]

        try:
            klines = self._client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=count,
            )

            if not klines or len(klines) == 0:
                return None, f"Aucune donnee pour {symbol} {timeframe}"

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

            if as_kronos:
                cols = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
                df = df[cols]
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col].astype(np.float64)
                df["volume"] = df["volume"].astype(np.float64)
                df["amount"] = df["amount"].astype(np.float64)

            return df, None

        except Exception as e:
            return None, f"Erreur klines Binance: {str(e)}"

    def wait_for_bar_close(self, symbol, timeframe, last_bar_time=None, poll_interval=1.0):
        """Attend la cloture d'une nouvelle bougie Binance."""
        if not self._connected or not self._client:
            return None, False

        if timeframe not in TF_MAP_BINANCE:
            return None, False

        interval = TF_MAP_BINANCE[timeframe]

        while True:
            try:
                klines = self._client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=2,
                )

                if klines and len(klines) >= 2:
                    close_time = int(klines[-2][6])
                    current_bar_time = datetime.fromtimestamp(close_time / 1000)

                    if last_bar_time is None:
                        return current_bar_time, True

                    if current_bar_time > last_bar_time:
                        return current_bar_time, True

            except Exception:
                pass

            time.sleep(poll_interval)

    def get_all_positions(self, symbol=None):
        """Retourne les positions ouvertes Binance Futures standardisees."""
        if not self._connected or not self._client:
            return []
        try:
            positions = self._client.futures_position_information(symbol=symbol)
            result = []
            for p in positions:
                pos_amt = float(p.get("positionAmt", 0))
                if abs(pos_amt) < 1e-8:
                    continue

                direction = "BUY" if pos_amt > 0 else "SELL"
                entry_price = float(p.get("entryPrice", 0))
                mark_price = float(p.get("markPrice", 0))
                unrealized = float(p.get("unrealizedProfit", 0))

                result.append({
                    "ticket": f"{p.get('symbol', '')}_{direction}",
                    "symbol": p.get("symbol", ""),
                    "type": direction,
                    "volume": abs(pos_amt),
                    "price_open": entry_price,
                    "price_current": mark_price,
                    "sl": 0,
                    "tp": 0,
                    "profit": unrealized,
                    "time": datetime.fromtimestamp(int(p.get("updateTime", 0)) / 1000),
                    "comment": "binance_futures",
                })
            return result
        except Exception as e:
            print(f"Erreur get_all_positions Binance: {e}")
            return []

    def get_all_orders(self, symbol=None):
        """Retourne les ordres en attente Binance Futures."""
        if not self._connected or not self._client:
            return []
        try:
            if symbol:
                orders = self._client.futures_get_open_orders(symbol=symbol)
            else:
                orders = self._client.futures_get_open_orders()
            result = []
            for o in orders:
                result.append({
                    "ticket": o.get("orderId", ""),
                    "symbol": o.get("symbol", ""),
                    "type": o.get("type", ""),
                    "volume": float(o.get("origQty", 0)),
                    "price_open": float(o.get("price", 0)),
                    "sl": float(o.get("stopPrice", 0)),
                    "tp": 0,
                    "time_setup": datetime.fromtimestamp(int(o.get("time", 0)) / 1000),
                })
            return result
        except Exception:
            return []

    def get_trading_history(self, symbol=None, days=7):
        """Retourne l'historique des trades Binance Futures."""
        if not self._connected or not self._client:
            return []
        try:
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            params = {"startTime": start_time, "limit": 500}
            if symbol:
                params["symbol"] = symbol

            trades = self._client.futures_account_trades(**params)
            result = []
            for t in trades:
                result.append({
                    "ticket": t.get("id", ""),
                    "order": t.get("orderId", ""),
                    "symbol": t.get("symbol", ""),
                    "type": "BUY" if t.get("side") == "BUY" else "SELL",
                    "volume": float(t.get("qty", 0)),
                    "price": float(t.get("price", 0)),
                    "profit": float(t.get("realizedPnl", 0)),
                    "time": datetime.fromtimestamp(int(t.get("time", 0)) / 1000),
                    "commission": float(t.get("commission", 0)),
                    "swap": 0,
                })
            return result
        except Exception as e:
            print(f"Erreur get_trading_history Binance: {e}")
            return []