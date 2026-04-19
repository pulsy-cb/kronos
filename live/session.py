"""
TradingSessionManager - Gere plusieurs sessions de trading live
avec differentes instances broker et modeles Kronos.
"""

import os
import sys
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from live.config import TradingConfig, BROKER_MT5, BROKER_BINANCE
from live.trader import LiveTrader
from live.logger import SessionLogger, write_session_summary
from live.broker_factory import create_feed, create_executor, connect_feed

sys.path.append(str(Path(__file__).parent.parent))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

MODEL_NAMES = {
    'xaumodel-local': 'XAU Finetuned (Local)',
    'xaumodel-mini': 'XAU mini (Local)',
    'kronos-mini': 'Kronos-mini (4.1M)',
    'kronos-small': 'Kronos-small (24.7M)',
    'kronos-base': 'Kronos-base (102.3M)',
}


class TradingSession:
    """Une session de trading individuelle."""

    def __init__(self, session_id: str, config: TradingConfig):
        self.session_id = session_id
        self.config = config
        self.trader: Optional[LiveTrader] = None
        self.predictor = None
        self.feed = None
        self.executor = None
        self.logger = None
        self.status = "created"
        self.error_message = None
        self.created_at = datetime.now()

    def to_dict(self):
        d = {
            "session_id": self.session_id,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "config": self.config.to_dict(),
        }
        if self.trader:
            d.update(self.trader.get_state())
        return d


class TradingSessionManager:
    """Gestionnaire de sessions de trading multiples."""

    def __init__(self):
        self.sessions: Dict[str, TradingSession] = {}
        self._lock = threading.Lock()
        self._predictor_cache = {}

    def create_session(self, config_dict):
        """Cree une nouvelle session de trading."""
        config = TradingConfig.from_dict(config_dict) if isinstance(config_dict, dict) else config_dict

        if not config.model_name:
            config.model_name = MODEL_NAMES.get(config.model_key, config.model_key)

        session_id = f"session_{len(self.sessions) + 1}_{datetime.now().strftime('%H%M%S')}"
        session = TradingSession(session_id, config)
        session.status = "created"
        self.sessions[session_id] = session
        return session_id

    def start_session(self, session_id):
        """Connecte le broker, charge le modele, et demarre le trading."""
        session = self.sessions.get(session_id)
        if session is None:
            return False, f"Session {session_id} introuvable"

        config = session.config

        # Step 1: Connect broker
        session.status = "connecting"
        feed = create_feed(config)
        success, msg = connect_feed(feed, config)
        if not success:
            session.status = "error"
            session.error_message = msg
            return False, msg

        session.feed = feed
        session.executor = create_executor(config)

        # Step 2: Load model
        session.status = "loading_model"
        predictor = self._load_model(config.model_key, config.device)
        if predictor is None:
            session.status = "error"
            session.error_message = f"Impossible de charger le modele {config.model_key}"
            feed.disconnect()
            return False, session.error_message

        session.predictor = predictor

        # Step 3: Create logger
        session.logger = SessionLogger(
            session_id=session_id,
            symbol=config.symbol,
            timeframe=config.timeframe,
            model_key=config.model_key,
            model_name=config.model_name,
            broker=config.broker,
        )

        # Step 4: Create trader and start
        trader = LiveTrader(config, predictor, feed, session.executor, logger=session.logger)
        session.trader = trader

        success, msg = trader.start()
        if not success:
            session.status = "error"
            session.error_message = msg
            session.logger.close()
            return False, msg

        session.status = "running"
        return True, msg

    def stop_session(self, session_id):
        """Arrete une session de trading et ecrit le resume."""
        session = self.sessions.get(session_id)
        if session is None:
            return False, f"Session {session_id} introuvable"

        metrics = {}
        if session.trader:
            if session.trader.current_position:
                session.trader._close_position("session_stop")
            metrics = session.trader.get_metrics()
            session.trader.stop()

        if session.logger:
            session.logger.close()

        if session.feed:
            session.feed.disconnect()

        session.status = "stopped"

        # Write session summary
        try:
            write_session_summary(
                session_id=session_id,
                config=session.config.to_dict(),
                metrics=metrics,
            )
        except Exception as e:
            print(f"Erreur ecriture resume session: {e}")

        return True, "Session arretee"

    def pause_session(self, session_id):
        session = self.sessions.get(session_id)
        if session is None:
            return False, f"Session {session_id} introuvable"

        if session.trader:
            session.trader.pause()
            session.status = "paused"
        return True, "Session en pause"

    def resume_session(self, session_id):
        session = self.sessions.get(session_id)
        if session is None:
            return False, f"Session {session_id} introuvable"

        if session.trader:
            session.trader.resume()
            session.status = "running"
        return True, "Session reprise"

    def close_position(self, session_id):
        session = self.sessions.get(session_id)
        if session is None:
            return False, f"Session {session_id} introuvable"

        if session.trader and session.trader.current_position:
            session.trader._close_position("manual_close")
            return True, "Position fermee"
        return False, "Pas de position ouverte"

    def get_session_state(self, session_id):
        session = self.sessions.get(session_id)
        if session is None:
            return None
        return session.to_dict()

    def get_all_sessions_state(self):
        return {sid: s.to_dict() for sid, s in list(self.sessions.items())}

    def delete_session(self, session_id):
        session = self.sessions.get(session_id)
        if session is None:
            return False

        if session.trader and session.trader.running:
            self.stop_session(session_id)

        if session.logger:
            session.logger.close()

        del self.sessions[session_id]
        return True

    def _load_model(self, model_key, device="cuda"):
        """Charge un modele Kronos (avec cache). Fallback CPU si CUDA absent."""
        import torch

        if device != "cpu" and not torch.cuda.is_available():
            device = "cpu"

        cache_key = f"{model_key}_{device}"

        if cache_key in self._predictor_cache:
            return self._predictor_cache[cache_key]

        if not MODEL_AVAILABLE:
            return None

        AVAILABLE_MODELS = {
            'xaumodel-local': {
                'model_path': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'xaumodel'),
                'tokenizer_path': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'xaumodel', 'tokenizer'),
                'context_length': 512,
            },
            'xaumodel-mini': {
                'model_path': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'zmini'),
                'tokenizer_path': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'zmini', 'tokenizer'),
                'context_length': 512,
            },
            'kronos-mini': {
                'model_id': 'NeoQuasar/Kronos-mini',
                'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
                'context_length': 2048,
            },
            'kronos-small': {
                'model_id': 'NeoQuasar/Kronos-small',
                'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
                'context_length': 512,
            },
            'kronos-base': {
                'model_id': 'NeoQuasar/Kronos-base',
                'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
                'context_length': 512,
            },
        }

        if model_key not in AVAILABLE_MODELS:
            return None

        cfg = AVAILABLE_MODELS[model_key]

        try:
            if 'tokenizer_path' in cfg:
                tokenizer = KronosTokenizer.from_pretrained(cfg['tokenizer_path'])
                model = Kronos.from_pretrained(cfg['model_path'])
            else:
                tokenizer = KronosTokenizer.from_pretrained(cfg['tokenizer_id'])
                model = Kronos.from_pretrained(cfg['model_id'])

            tokenizer = tokenizer.to(device)
            model = model.to(device)
            model.eval()
            tokenizer.eval()

            if hasattr(torch, 'compile') and device != 'cpu':
                try:
                    tokenizer = torch.compile(tokenizer)
                    model = torch.compile(model)
                except Exception:
                    pass

            predictor = KronosPredictor(model, tokenizer, device=device, max_context=cfg['context_length'])
            self._predictor_cache[cache_key] = predictor
            return predictor
        except Exception as e:
            print(f"Erreur chargement modele {model_key}: {e}")
            return None

    def cleanup_predictor_cache(self):
        self._predictor_cache.clear()
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


manager = TradingSessionManager()