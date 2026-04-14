"""
Flux de donnees en temps reel depuis MetaTrader 5.
Gere la connexion MT5, la recuperation de bougies live,
et la detection de cloture de bougie.

Compatible avec plusieurs instances MT5 (login differents).
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import os
import threading

TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

TF_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
}


class MT5LiveFeed:
    """Flux de donnees en temps reel depuis MT5."""

    def __init__(self):
        self._lock = threading.Lock()
        self._connected = False
        self._account_info = None

    def connect(self, login=None, password=None, server=None, path=None):
        """Connecte a une instance MT5. Si pas d'arguments, lit depuis .env."""
        with self._lock:
            if path:
                if not mt5.initialize(path=path):
                    err = mt5.last_error()
                    return False, f"MT5 init failed (path={path}): {err}"
            else:
                if not mt5.initialize():
                    err = mt5.last_error()
                    return False, f"MT5 init failed: {err}"

            if login and password and server:
                login = int(login)
                if not mt5.login(login, password=password, server=server):
                    err = mt5.last_error()
                    mt5.shutdown()
                    return False, f"MT5 login failed: {err}"
            elif login is None:
                try:
                    from dotenv import load_dotenv
                    load_dotenv()
                except ImportError:
                    env_path = Path(__file__).parent.parent / ".env"
                    if env_path.exists():
                        for line in env_path.read_text().strip().splitlines():
                            if "=" in line:
                                k, v = line.split("=", 1)
                                os.environ[k.strip()] = v.strip()

                env_login = os.getenv("MT5_LOGIN", "0")
                env_password = os.getenv("MT5_PASSWORD", "")
                env_server = os.getenv("MT5_SERVER", "")

                if env_login and env_login != "0" and env_password and env_server:
                    if not mt5.login(int(env_login), password=env_password, server=env_server):
                        err = mt5.last_error()
                        mt5.shutdown()
                        return False, f"MT5 login from .env failed: {err}"

            self._account_info = mt5.account_info()
            if self._account_info is None:
                mt5.shutdown()
                return False, "MT5 connected but no account info"

            self._connected = True
            info = self._account_info
            return True, f"Connecte: {info.server} | Compte: {info.login} | Balance: ${info.balance:.2f}"

    def disconnect(self):
        """Deconnecte MT5."""
        with self._lock:
            if self._connected:
                mt5.shutdown()
                self._connected = False
                self._account_info = None

    def is_connected(self):
        return self._connected

    def get_account_info(self):
        """Retourne les infos du compte MT5."""
        if not self._connected:
            return None
        info = mt5.account_info()
        if info is None:
            return None
        return {
            "login": info.login,
            "server": info.server,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "profit": info.profit,
            "currency": info.currency,
            "leverage": info.leverage,
            "trade_mode": info.trade_mode,
            "name": info.name,
        }

    def get_symbols(self):
        """Liste les symboles disponibles."""
        if not self._connected:
            return []
        symbols = mt5.symbols_total()
        if symbols is None:
            return []
        visible = mt5.symbols_get()
        if visible is None:
            return []
        return [s.name for s in visible]

    def get_symbol_info(self, symbol):
        """Retourne les infos d'un symbole."""
        if not self._connected:
            return None
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return {
            "name": info.name,
            "description": info.description,
            "digits": info.digits,
            "point": info.point,
            "trade_contract_size": info.trade_contract_size,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "trade_mode": info.trade_mode,
            "spread": info.spread,
        }

    def get_current_tick(self, symbol):
        """Retourne le tick actuel (bid/ask) pour un symbole."""
        if not self._connected:
            return None
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "volume": tick.volume,
            "time": datetime.fromtimestamp(tick.time),
            "spread": tick.ask - tick.bid if tick.ask and tick.bid else 0,
        }

    def get_latest_candles(self, symbol, timeframe, count, as_kronos=True):
        """
        Recupere les N dernieres bougies depuis MT5.
        Retourne un DataFrame au format Kronos si as_kronos=True.
        """
        if not self._connected:
            return None, "MT5 not connected"

        if timeframe not in TF_MAP:
            return None, f"Timeframe inconnu: {timeframe}. Disponibles: {list(TF_MAP.keys())}"

        tf_mt5 = TF_MAP[timeframe]

        rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, count)
        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            return None, f"Aucune donnee pour {symbol} {timeframe}: {err}"

        df = pd.DataFrame(rates)
        df["timestamps"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"tick_volume": "volume"})
        df["amount"] = ((df["open"] + df["high"] + df["low"] + df["close"]) / 4) * df["volume"]

        if as_kronos:
            cols = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
            df = df[cols]
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(np.float64)
            df["volume"] = df["volume"].astype(np.float64)
            df["amount"] = df["amount"].astype(np.float64)

        return df, None

    def wait_for_bar_close(self, symbol, timeframe, last_bar_time=None, poll_interval=1.0):
        """
        Attend la cloture d'une nouvelle bougie.
        Retourne (new_bar_time, changed) ou (last_bar_time, False) si timeout.
        Bloquant - appeler dans un thread.
        """
        if not self._connected:
            return None, False

        tf_mt5 = TF_MAP.get(timeframe)
        if tf_mt5 is None:
            return None, False

        while True:
            rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, 2)
            if rates is None or len(rates) < 2:
                time.sleep(poll_interval)
                continue

            current_bar_time = datetime.fromtimestamp(rates[-2]["time"])

            if last_bar_time is None:
                return current_bar_time, True

            if current_bar_time > last_bar_time:
                return current_bar_time, True

            time.sleep(poll_interval)

    def get_all_positions(self, symbol=None):
        """Retourne les positions ouvertes."""
        if not self._connected:
            return []
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        if positions is None:
            return []
        result = []
        for p in positions:
            result.append({
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume": p.volume,
                "price_open": p.price_open,
                "price_current": p.price_current,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit,
                "time": datetime.fromtimestamp(p.time),
                "comment": p.comment,
            })
        return result

    def get_all_orders(self, symbol=None):
        """Retourne les ordres en attente."""
        if not self._connected:
            return []
        if symbol:
            orders = mt5.orders_get(symbol=symbol)
        else:
            orders = mt5.orders_get()
        if orders is None:
            return []
        result = []
        for o in orders:
            result.append({
                "ticket": o.ticket,
                "symbol": o.symbol,
                "type": o.type,
                "volume": o.volume_current,
                "price_open": o.price_open,
                "sl": o.sl,
                "tp": o.tp,
                "time_setup": datetime.fromtimestamp(o.time_setup),
            })
        return result

    def get_trading_history(self, symbol=None, days=7):
        """Retourne l'historique des trades."""
        if not self._connected:
            return []
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        deals = mt5.history_deals_get(from_date, to_date)
        if deals is None:
            return []
        result = []
        for d in deals:
            if d.entry != mt5.DEAL_ENTRY_OUT:
                continue
            result.append({
                "ticket": d.ticket,
                "order": d.order,
                "symbol": d.symbol,
                "type": "BUY" if d.type == mt5.DEAL_TYPE_BUY else "SELL",
                "volume": d.volume,
                "price": d.price,
                "profit": d.profit,
                "time": datetime.fromtimestamp(d.time),
                "commission": d.commission,
                "swap": d.swap,
            })
        return result


_live_feed_instance = None
_live_feed_lock = threading.Lock()


def get_live_feed():
    """Retourne l'instance globale du flux MT5 (singleton)."""
    global _live_feed_instance
    with _live_feed_lock:
        if _live_feed_instance is None:
            _live_feed_instance = MT5LiveFeed()
        return _live_feed_instance