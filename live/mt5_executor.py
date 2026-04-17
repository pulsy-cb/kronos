"""
Execution d'ordres sur MetaTrader 5.
Implementation de BrokerExecutor pour MT5.
"""

import MetaTrader5 as mt5
from datetime import datetime
import time

from live.broker_executor import BrokerExecutor


class MT5Executor(BrokerExecutor):
    """Executeur d'ordres MT5."""

    def __init__(self):
        self._slippage_points = 20

    def open_position(self, symbol, direction, volume, sl_price=None, tp_price=None, comment="kronos"):
        """
        Ouvre une position market.
        direction: 'long' ou 'short'
        Retourne (success, detail).
        """
        if direction not in ("long", "short"):
            return False, f"Direction invalide: {direction}"

        order_type = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False, f"Impossible d'obtenir le tick pour {symbol}"

        price = tick.ask if direction == "long" else tick.bid

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return False, f"Symbole {symbol} introuvable"

        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                return False, f"Impossible de selectionner {symbol}"

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": float(price),
            "deviation": self._slippage_points,
            "magic": 777888,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if sl_price is not None and sl_price > 0:
            request["sl"] = float(sl_price)
        if tp_price is not None and tp_price > 0:
            request["tp"] = float(tp_price)

        result = mt5.order_send(request)

        if result is None:
            return False, "order_send retourne None"

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return False, f"Ordre rejete: retcode={result.retcode}, comment={result.comment}"

        return True, {
            "ticket": result.order,
            "price": result.price,
            "volume": result.volume,
            "direction": direction,
            "symbol": symbol,
        }

    def close_position(self, ticket):
        """Ferme une position par ticket."""
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            return False, f"Position {ticket} introuvable"

        pos = position[0]
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return False, f"Impossible d'obtenir le tick pour {pos.symbol}"

        close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": pos.ticket,
            "price": float(close_price),
            "deviation": self._slippage_points,
            "magic": 777888,
            "comment": "kronos_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is None:
            return False, "order_send retourne None"

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return False, f"Fermeture rejetee: retcode={result.retcode}, comment={result.comment}"

        return True, {
            "ticket": pos.ticket,
            "close_price": result.price,
            "profit": pos.profit,
            "volume": pos.volume,
        }

    def close_all_positions(self, symbol=None):
        """Ferme toutes les positions ouvertes, optionnellement filtrees par symbole."""
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if positions is None or len(positions) == 0:
            return []

        results = []
        for pos in positions:
            success, detail = self.close_position(pos.ticket)
            results.append({
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "success": success,
                "detail": detail,
            })
        return results

    def calculate_lot_size(self, symbol, account_balance, risk_pct, sl_distance, direction="long"):
        """
        Calcule la taille de lot en fonction du risque.
        risk_pct: pourcentage du capital a risquer (ex: 0.01 = 1%)
        sl_distance: distance au stop-loss en prix (ex: 5.0 pour $5 de SL)
        """
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return 0.0

        if sl_distance <= 0:
            return symbol_info.volume_min

        risk_amount = account_balance * risk_pct
        point_value = symbol_info.trade_contract_size * symbol_info.point
        lots = risk_amount / (sl_distance / symbol_info.point * point_value) if point_value > 0 else 0

        lots = max(symbol_info.volume_min, min(lots, symbol_info.volume_max))

        step = symbol_info.volume_step
        if step > 0:
            lots = round(lots / step) * step

        return round(lots, 2)

    def calculate_sl_tp_prices(self, entry_price, direction, sl_pct, tp_pct, symbol=None):
        """
        Calcule les prix de stop-loss et take-profit.
        Retourne (sl_price, tp_price).
        """
        if direction == "long":
            sl_price = entry_price * (1 - sl_pct) if sl_pct > 0 else None
            tp_price = entry_price * (1 + tp_pct) if tp_pct > 0 else None
        else:
            sl_price = entry_price * (1 + sl_pct) if sl_pct > 0 else None
            tp_price = entry_price * (1 - tp_pct) if tp_pct > 0 else None

        if symbol:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                digits = symbol_info.digits
                if sl_price is not None:
                    sl_price = round(sl_price, digits)
                if tp_price is not None:
                    tp_price = round(tp_price, digits)

        return sl_price, tp_price


_executor_instance = None


def get_executor():
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = MT5Executor()
    return _executor_instance