"""
Compatibilite descendante: importe depuis mt5_feed.py.
Le code original de ce fichier a ete deplace vers data/mt5_feed.py
pour supporter l'abstraction multi-broker (BrokerFeed).
"""

from data.mt5_feed import MT5LiveFeed, TF_MAP, get_live_feed
from data.broker_feed import TF_SECONDS