"""
BrokerFactory - Cree les instances de BrokerFeed et BrokerExecutor
en fonction du broker selectionne dans la configuration.
"""

from live.config import BROKER_MT5, BROKER_BINANCE


def create_feed(config):
    """Cree un BrokerFeed selon config.broker."""
    broker = config.broker

    if broker == BROKER_MT5:
        from data.mt5_feed import MT5LiveFeed
        return MT5LiveFeed()

    elif broker == BROKER_BINANCE:
        from data.binance_feed import BinanceLiveFeed
        return BinanceLiveFeed(
            api_key=config.binance_api_key,
            api_secret=config.binance_api_secret,
            testnet=config.binance_testnet,
        )

    else:
        raise ValueError(f"Broker inconnu: {broker}. Disponibles: {BROKER_MT5}, {BROKER_BINANCE}")


def create_executor(config):
    """Cree un BrokerExecutor selon config.broker."""
    broker = config.broker

    if broker == BROKER_MT5:
        from live.mt5_executor import MT5Executor
        return MT5Executor()

    elif broker == BROKER_BINANCE:
        from live.binance_executor import BinanceExecutor
        return BinanceExecutor(
            api_key=config.binance_api_key,
            api_secret=config.binance_api_secret,
            testnet=config.binance_testnet,
            leverage=config.binance_leverage,
        )

    else:
        raise ValueError(f"Broker inconnu: {broker}. Disponibles: {BROKER_MT5}, {BROKER_BINANCE}")


def connect_feed(feed, config):
    """Connecte un BrokerFeed avec les bons parametres selon le broker."""
    if config.broker == BROKER_MT5:
        return feed.connect(
            login=config.mt5_login,
            password=config.mt5_password,
            server=config.mt5_server,
            path=config.mt5_path,
        )
    elif config.broker == BROKER_BINANCE:
        return feed.connect()
    else:
        return False, f"Broker inconnu: {config.broker}"