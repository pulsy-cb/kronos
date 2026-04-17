"""
Interface abstraite BrokerExecutor.
Toutes les implementations d'execution d'ordres (MT5, Binance, etc.)
doivent heriter de cette classe et implementer ses methodes.
"""

from abc import ABC, abstractmethod


class BrokerExecutor(ABC):
    """Interface abstraite pour l'execution d'ordres sur un broker."""

    @abstractmethod
    def open_position(self, symbol, direction, volume, sl_price=None, tp_price=None, comment="kronos"):
        """
        Ouvre une position market.
        direction: 'long' ou 'short'
        Retourne (success, detail). Si success, detail est un dict:
        {
            'ticket': str/int,
            'price': float,
            'volume': float,
            'direction': str,
            'symbol': str,
        }
        """
        pass

    @abstractmethod
    def close_position(self, ticket):
        """
        Ferme une position par ticket/identifiant.
        Retourne (success, detail). Si success, detail est un dict:
        {
            'ticket': str/int,
            'close_price': float,
            'profit': float,
            'volume': float,
        }
        """
        pass

    @abstractmethod
    def close_all_positions(self, symbol=None):
        """
        Ferme toutes les positions ouvertes, optionnellement filtrees par symbole.
        Retourne une liste de resultats.
        """
        pass

    @abstractmethod
    def calculate_lot_size(self, symbol, account_balance, risk_pct, sl_distance, direction="long"):
        """
        Calcule la taille de position en fonction du risque.
        risk_pct: pourcentage du capital a risquer (ex: 0.01 = 1%)
        sl_distance: distance au stop-loss en prix
        Retourne un float (taille de position).
        """
        pass

    @abstractmethod
    def calculate_sl_tp_prices(self, entry_price, direction, sl_pct, tp_pct, symbol=None):
        """
        Calcule les prix de stop-loss et take-profit.
        Retourne (sl_price, tp_price).
        """
        pass