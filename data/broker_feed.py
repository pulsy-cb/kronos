"""
Interface abstraite BrokerFeed.
Toutes les implementations de flux de donnees (MT5, Binance, etc.)
doivent heriter de cette classe et implementer ses methodes.
"""

from abc import ABC, abstractmethod


TF_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
}


class BrokerFeed(ABC):
    """Interface abstraite pour un flux de donnees broker."""

    @abstractmethod
    def connect(self, **kwargs):
        """Connecte au broker. Retourne (success, message)."""
        pass

    @abstractmethod
    def disconnect(self):
        """Deconnecte du broker."""
        pass

    @abstractmethod
    def is_connected(self):
        """Retourne True si connecte."""
        pass

    @abstractmethod
    def get_account_info(self):
        """
        Retourne les infos du compte standardisees:
        {
            'login': str/int,
            'server': str,
            'balance': float,
            'equity': float,
            'margin': float,
            'free_margin': float,
            'profit': float,
            'currency': str,
            'leverage': int,
        }
        """
        pass

    @abstractmethod
    def get_symbols(self):
        """Retourne la liste des symboles disponibles."""
        pass

    @abstractmethod
    def get_symbol_info(self, symbol):
        """
        Retourne les infos d'un symbole standardisees:
        {
            'name': str,
            'description': str,
            'digits': int,
            'point': float,
            'trade_contract_size': float,
            'volume_min': float,
            'volume_max': float,
            'volume_step': float,
            'trade_mode': int,
            'spread': float,
        }
        """
        pass

    @abstractmethod
    def get_current_tick(self, symbol):
        """
        Retourne le tick actuel standardise:
        {
            'bid': float,
            'ask': float,
            'last': float,
            'volume': float,
            'time': datetime,
            'spread': float,
        }
        """
        pass

    @abstractmethod
    def get_latest_candles(self, symbol, timeframe, count, as_kronos=True):
        """
        Retourne (DataFrame, None) ou (None, erreur).
        DataFrame au format Kronos si as_kronos=True.
        """
        pass

    @abstractmethod
    def wait_for_bar_close(self, symbol, timeframe, last_bar_time=None, poll_interval=1.0):
        """
        Attend la cloture d'une nouvelle bougie.
        Retourne (new_bar_time, changed).
        Bloquant - appeler dans un thread.
        """
        pass

    @abstractmethod
    def get_all_positions(self, symbol=None):
        """
        Retourne les positions ouvertes standardisees:
        [
            {
                'ticket': str/int,
                'symbol': str,
                'type': 'BUY' ou 'SELL',
                'volume': float,
                'price_open': float,
                'price_current': float,
                'sl': float,
                'tp': float,
                'profit': float,
                'time': datetime,
                'comment': str,
            },
            ...
        ]
        """
        pass

    @abstractmethod
    def get_all_orders(self, symbol=None):
        """
        Retourne les ordres en attente standardises.
        """
        pass

    @abstractmethod
    def get_trading_history(self, symbol=None, days=7):
        """
        Retourne l'historique des trades standardise.
        """
        pass