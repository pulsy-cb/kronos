"""
Execution d'ordres sur Binance Futures (USDT-M).
Implementation de BrokerExecutor pour Binance.
Supporte le mode testnet et les SL/TP natifs des futures.
"""

import time
from datetime import datetime
import threading

from live.broker_executor import BrokerExecutor


class BinanceExecutor(BrokerExecutor):
    """Executeur d'ordres Binance Futures."""

    def __init__(self, api_key=None, api_secret=None, testnet=True, leverage=10):
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._leverage = leverage
        self._client = None
        self._lock = threading.Lock()

    def _get_client(self):
        """Lazy init du client Binance."""
        if self._client is None:
            from binance.client import Client
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
        return self._client

    def _set_leverage(self, symbol):
        """Definit le levier pour un symbole."""
        client = self._get_client()
        try:
            client.futures_change_leverage(symbol=symbol, leverage=self._leverage)
        except Exception:
            pass

    def _round_quantity(self, symbol, quantity):
        """Arrondit la quantite selon le stepSize du symbole."""
        try:
            client = self._get_client()
            info = client.futures_exchange_info()
            for s in info.get("symbols", []):
                if s["symbol"] == symbol:
                    for f in s.get("filters", []):
                        if f["filterType"] == "LOT_SIZE":
                            step = float(f["stepSize"])
                            precision = len(f["stepSize"].rstrip("0").rstrip(".")) if "." in f["stepSize"] else 0
                            quantity = round(round(quantity / step) * step, precision)
                            return quantity
        except Exception:
            pass
        return round(quantity, 3)

    def _round_price(self, symbol, price):
        """Arrondit le prix selon le tickSize du symbole."""
        try:
            client = self._get_client()
            info = client.futures_exchange_info()
            for s in info.get("symbols", []):
                if s["symbol"] == symbol:
                    for f in s.get("filters", []):
                        if f["filterType"] == "PRICE_FILTER":
                            tick = float(f["tickSize"])
                            precision = len(f["tickSize"].rstrip("0").rstrip(".")) if "." in f["tickSize"] else 0
                            price = round(round(price / tick) * tick, precision)
                            return price
        except Exception:
            pass
        return round(price, 2)

    def open_position(self, symbol, direction, volume, sl_price=None, tp_price=None, comment="kronos"):
        """
        Ouvre une position market sur Binance Futures.
        direction: 'long' ou 'short'
        """
        if direction not in ("long", "short"):
            return False, f"Direction invalide: {direction}"

        client = self._get_client()

        try:
            self._set_leverage(symbol)
        except Exception:
            pass

        side = "BUY" if direction == "long" else "SELL"
        quantity = self._round_quantity(symbol, volume)

        if quantity <= 0:
            return False, f"Quantite trop petite: {quantity}"

        try:
            order = client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
            )

            if not order:
                return False, "Ordre Binance retourne None"

            order_id = order.get("orderId", "")
            fill_price = float(order.get("avgPrice", 0))
            fill_qty = float(order.get("executedQty", quantity))
            status = order.get("status", "")

            if status == "NEW":
                time.sleep(0.5)
                try:
                    check = client.futures_get_order(symbol=symbol, orderId=order_id)
                    if check.get("status") == "FILLED":
                        fill_price = float(check.get("avgPrice", fill_price))
                        fill_qty = float(check.get("executedQty", fill_qty))
                except Exception:
                    pass

            if sl_price is not None and sl_price > 0:
                sl_price_rounded = self._round_price(symbol, sl_price)
                sl_side = "SELL" if direction == "long" else "BUY"
                try:
                    client.futures_create_order(
                        symbol=symbol,
                        side=sl_side,
                        type="STOP_MARKET",
                        stopPrice=str(sl_price_rounded),
                        closePosition=True,
                    )
                except Exception as e:
                    print(f"SL non place: {e}")

            if tp_price is not None and tp_price > 0:
                tp_price_rounded = self._round_price(symbol, tp_price)
                tp_side = "SELL" if direction == "long" else "BUY"
                try:
                    client.futures_create_order(
                        symbol=symbol,
                        side=tp_side,
                        type="TAKE_PROFIT_MARKET",
                        stopPrice=str(tp_price_rounded),
                        closePosition=True,
                    )
                except Exception as e:
                    print(f"TP non place: {e}")

            return True, {
                "ticket": str(order_id),
                "price": fill_price if fill_price > 0 else 0,
                "volume": fill_qty,
                "direction": direction,
                "symbol": symbol,
            }

        except Exception as e:
            return False, f"Erreur ouverture Binance: {str(e)}"

    def close_position(self, ticket):
        """Ferme une position Binance Futures."""
        client = self._get_client()

        try:
            parts = str(ticket).split("_")
            if len(parts) == 2:
                symbol, direction_str = parts[0], parts[1]
                direction = "long" if direction_str == "BUY" else "short"
            else:
                try:
                    positions = client.futures_position_information(symbol=None)
                    for p in positions:
                        if abs(float(p.get("positionAmt", 0))) > 1e-8:
                            symbol = p["symbol"]
                            direction = "long" if float(p["positionAmt"]) > 0 else "short"
                            break
                    else:
                        return False, f"Position {ticket} introuvable"
                except Exception:
                    return False, f"Impossible de trouver la position {ticket}"

            close_side = "SELL" if direction == "long" else "BUY"

            pos_info = None
            try:
                positions = client.futures_position_information(symbol=symbol)
                for p in positions:
                    if p.get("symbol") == symbol and abs(float(p.get("positionAmt", 0))) > 1e-8:
                        pos_info = p
                        break
            except Exception:
                pass

            if pos_info is None:
                return False, f"Position {symbol} introuvable (flat)"

            quantity = abs(float(pos_info.get("positionAmt", 0)))
            quantity = self._round_quantity(symbol, quantity)
            entry_price = float(pos_info.get("entryPrice", 0))
            unrealized = float(pos_info.get("unrealizedProfit", 0))

            try:
                open_orders = client.futures_get_open_orders(symbol=symbol)
                for o in open_orders:
                    if o.get("type") in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
                        try:
                            client.futures_cancel_order(symbol=symbol, orderId=o["orderId"])
                        except Exception:
                            pass
            except Exception:
                pass

            order = client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type="MARKET",
                quantity=quantity,
            )

            if not order:
                return False, "Ordre de fermeture retourne None"

            fill_price = float(order.get("avgPrice", 0))
            order_id = order.get("orderId", "")

            time.sleep(0.5)
            try:
                check = client.futures_get_order(symbol=symbol, orderId=order_id)
                if check.get("status") == "FILLED":
                    fill_price = float(check.get("avgPrice", fill_price))
            except Exception:
                pass

            return True, {
                "ticket": ticket,
                "close_price": fill_price if fill_price > 0 else entry_price,
                "profit": unrealized,
                "volume": quantity,
            }

        except Exception as e:
            return False, f"Erreur fermeture Binance: {str(e)}"

    def close_all_positions(self, symbol=None):
        """Ferme toutes les positions ouvertes Binance Futures."""
        client = self._get_client()
        results = []

        try:
            positions = client.futures_position_information(symbol=symbol)
            for p in positions:
                pos_amt = float(p.get("positionAmt", 0))
                if abs(pos_amt) < 1e-8:
                    continue

                sym = p.get("symbol", "")
                direction = "long" if pos_amt > 0 else "short"
                ticket = f"{sym}_{'BUY' if pos_amt > 0 else 'SELL'}"

                success, detail = self.close_position(ticket)
                results.append({
                    "ticket": ticket,
                    "symbol": sym,
                    "success": success,
                    "detail": detail,
                })

        except Exception as e:
            print(f"Erreur close_all_positions Binance: {e}")

        return results

    def calculate_lot_size(self, symbol, account_balance, risk_pct, sl_distance, direction="long"):
        """
        Calcule la taille de position en quantite pour Binance Futures.
        risk_pct: pourcentage du capital a risquer (ex: 0.01 = 1%)
        sl_distance: distance au stop-loss en prix
        """
        if sl_distance <= 0:
            try:
                client = self._get_client()
                tick = client.futures_mark_price(symbol=symbol)
                price = float(tick.get("markPrice", 0))
                if price > 0:
                    return self._round_quantity(symbol, 10 / price)
            except Exception:
                pass
            return 0.001

        risk_amount = account_balance * risk_pct
        quantity = risk_amount / sl_distance

        try:
            client = self._get_client()
            info = client.futures_exchange_info()
            for s in info.get("symbols", []):
                if s["symbol"] == symbol:
                    for f in s.get("filters", []):
                        if f["filterType"] == "LOT_SIZE":
                            min_qty = float(f["minQty"])
                            max_qty = float(f["maxQty"])
                            quantity = max(min_qty, min(quantity, max_qty))
                            break
                    for f in s.get("filters", []):
                        if f["filterType"] == "MARKET_LOT_SIZE":
                            max_qty = float(f["maxQty"])
                            quantity = min(quantity, max_qty)
                            break
                    break
        except Exception:
            pass

        return self._round_quantity(symbol, quantity)

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
            sl_price = self._round_price(symbol, sl_price) if sl_price else None
            tp_price = self._round_price(symbol, tp_price) if tp_price else None

        return sl_price, tp_price