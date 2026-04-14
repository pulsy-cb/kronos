"""
Configuration pour le trading live.
Presets par timeframe, parametres de risque, etc.
"""

from dataclasses import dataclass, field
from typing import Optional
import yaml
from pathlib import Path

DIRECTION_LONG_ONLY = "long_only"
DIRECTION_LONG_SHORT = "long_short"

SIZING_FIXED = "fixed"
SIZING_PERCENT = "percent"

PRESETS = {
    "M1": {
        "lookback": 480,
        "pred_len": 30,
        "step_size": 30,
        "signal_threshold": 0.0008,
        "exit_threshold": 0.0005,
        "stop_loss_pct": 0.0025,
        "take_profit_pct": 0.005,
        "max_hold_bars": 120,
    },
    "M5": {
        "lookback": 400,
        "pred_len": 120,
        "step_size": 60,
        "signal_threshold": 0.0012,
        "exit_threshold": 0.0007,
        "stop_loss_pct": 0.0035,
        "take_profit_pct": 0.007,
        "max_hold_bars": 144,
    },
    "M15": {
        "lookback": 400,
        "pred_len": 120,
        "step_size": 60,
        "signal_threshold": 0.002,
        "exit_threshold": 0.0012,
        "stop_loss_pct": 0.005,
        "take_profit_pct": 0.01,
        "max_hold_bars": 64,
    },
    "H1": {
        "lookback": 400,
        "pred_len": 48,
        "step_size": 24,
        "signal_threshold": 0.003,
        "exit_threshold": 0.0015,
        "stop_loss_pct": 0.008,
        "take_profit_pct": 0.015,
        "max_hold_bars": 48,
    },
    "H4": {
        "lookback": 200,
        "pred_len": 48,
        "step_size": 24,
        "signal_threshold": 0.005,
        "exit_threshold": 0.0025,
        "stop_loss_pct": 0.012,
        "take_profit_pct": 0.025,
        "max_hold_bars": 24,
    },
    "D1": {
        "lookback": 200,
        "pred_len": 30,
        "step_size": 10,
        "signal_threshold": 0.01,
        "exit_threshold": 0.005,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "max_hold_bars": 15,
    },
}


@dataclass
class TradingConfig:
    name: str = "default"
    symbol: str = "XAUUSD"
    timeframe: str = "M5"
    model_key: str = "xaumodel-local"
    model_name: str = ""
    device: str = "cuda"

    # Prediction parameters
    lookback: int = 400
    pred_len: int = 120
    temperature: float = 1.0
    top_p: float = 0.9
    sample_count: int = 1

    # Signal thresholds
    signal_threshold: float = 0.0012
    exit_threshold: float = 0.0007

    # Risk management
    stop_loss_pct: float = 0.0035
    take_profit_pct: float = 0.007
    max_hold_bars: int = 144
    direction: str = DIRECTION_LONG_SHORT

    # Position sizing
    sizing_method: str = SIZING_PERCENT
    fixed_lot: float = 0.1
    risk_pct: float = 0.01
    max_lot: float = 10.0

    # Commission (MT5 va les gerer, mais pour le suivi)
    commission_per_trade: float = 0.07

    # MT5 connection
    mt5_login: Optional[int] = None
    mt5_password: Optional[str] = None
    mt5_server: Optional[str] = None
    mt5_path: Optional[str] = None

    # Internal
    step_size: int = 60

    def apply_preset(self, preset_name):
        if preset_name in PRESETS:
            for k, v in PRESETS[preset_name].items():
                setattr(self, k, v)
            self.timeframe = preset_name

    def to_dict(self):
        return {
            "name": self.name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "model_key": self.model_key,
            "model_name": self.model_name,
            "device": self.device,
            "lookback": self.lookback,
            "pred_len": self.pred_len,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "sample_count": self.sample_count,
            "signal_threshold": self.signal_threshold,
            "exit_threshold": self.exit_threshold,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "max_hold_bars": self.max_hold_bars,
            "direction": self.direction,
            "sizing_method": self.sizing_method,
            "fixed_lot": self.fixed_lot,
            "risk_pct": self.risk_pct,
            "max_lot": self.max_lot,
        }

    @classmethod
    def from_dict(cls, d):
        c = cls()
        for k, v in d.items():
            if hasattr(c, k) and v is not None:
                if k in ('mt5_login',) and isinstance(v, str) and v.strip() == '':
                    continue
                setattr(c, k, v)
        return c


def load_configs_from_yaml(path):
    """Charge des configurations depuis un fichier YAML."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return []
    configs = data.get("sessions", data if isinstance(data, list) else [data])
    return [TradingConfig.from_dict(c) for c in configs]


def save_configs_to_yaml(configs, path):
    """Sauvegarde des configurations dans un fichier YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {"sessions": [c.to_dict() for c in configs]}
    with open(p, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)