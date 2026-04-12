"""
Chargeur de données RAM-safe.
Charge les parquets mois par mois, jamais tout en mémoire.
Supporte: ticks, M1, M5, M15, H1.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass
class DataChunk:
    """Un mois de données chargées en mémoire."""
    month: str
    signal_timeframe: str          # Timeframe principal pour la détection de signaux
    candles_signal: pd.DataFrame   # Bougies du timeframe signal (M1, M5, M15...)
    candles_m1: Optional[pd.DataFrame]   # Bougies M1 pour résolution fine (None si signal=M1)
    candles_h1: Optional[pd.DataFrame]   # Bougies H1 pour contexte de régime
    ticks_path: Optional[Path]
    _ticks_cache: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    def get_ticks(self, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        """Retourne les ticks d'une fenêtre temporelle. Charge depuis disque une seule fois."""
        if self.ticks_path is None or not self.ticks_path.exists():
            return None

        if self._ticks_cache is None:
            df = pd.read_parquet(self.ticks_path)
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time").sort_index()
            self._ticks_cache = df

        result = self._ticks_cache.loc[start:end].reset_index()
        return result if len(result) > 0 else None

    @property
    def has_ticks(self) -> bool:
        return self.ticks_path is not None and self.ticks_path.exists()

    @property
    def has_m1(self) -> bool:
        return self.candles_m1 is not None and len(self.candles_m1) > 0

    @property
    def has_h1(self) -> bool:
        return self.candles_h1 is not None and len(self.candles_h1) > 0


class DataLoader:
    def __init__(self, data_dir: str = "data/parquet"):
        self.data_dir = Path(data_dir)
        self.ticks_dir = self.data_dir / "ticks"
        self.candles_dir = self.data_dir / "candles"

    def info(self) -> pd.DataFrame:
        """Liste tous les fichiers de données disponibles."""
        rows = []
        for f in sorted(self.data_dir.rglob("*.parquet")):
            try:
                df = pd.read_parquet(f)
                size_mb = f.stat().st_size / 1024 / 1024
                rows.append({
                    "fichier": f.name,
                    "type": "ticks" if "ticks" in str(f) else "candles",
                    "lignes": f"{len(df):,}",
                    "debut": str(df["time"].min())[:10],
                    "fin": str(df["time"].max())[:10],
                    "taille": f"{size_mb:.1f} MB",
                })
            except Exception as e:
                rows.append({
                    "fichier": f.name,
                    "type": "erreur",
                    "lignes": "?",
                    "debut": "?",
                    "fin": "?",
                    "taille": f"ERREUR: {e}",
                })
        return pd.DataFrame(rows)

    def load_candles(self, symbol: str, timeframe: str,
                     start: str, end: str) -> pd.DataFrame:
        """Charge toutes les bougies d'un coup (elles sont légères)."""
        pattern = f"{symbol}_{timeframe}_*.parquet"
        files = sorted(self.candles_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(f"Aucun fichier trouvé: {pattern} dans {self.candles_dir}")

        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)

        # Filtrer par dates
        mask = (df["time"] >= start) & (df["time"] <= end)
        result = df.loc[mask].reset_index(drop=True)

        if len(result) == 0:
            raise FileNotFoundError(
                f"Aucune donnée dans la période {start} → {end} "
                f"(données dispo: {df['time'].min()} → {df['time'].max()})"
            )

        return result

    def has_timeframe(self, symbol: str, timeframe: str) -> bool:
        """Check if a timeframe is available."""
        pattern = f"{symbol}_{timeframe}_*.parquet"
        files = list(self.candles_dir.glob(pattern))
        return len(files) > 0

    def has_ticks(self, symbol: str) -> bool:
        """Check if tick data is available."""
        if not self.ticks_dir.exists():
            return False
        pattern = f"{symbol}_ticks_*.parquet"
        files = list(self.ticks_dir.glob(pattern))
        return len(files) > 0

    def iter_months(self, symbol: str, start: str, end: str,
                    load_h1: bool = True,
                    signal_timeframe: str = "M5") -> Iterator[DataChunk]:
        """
        Itère mois par mois. Charge les bougies du timeframe signal + M1 (résolution fine) + H1 (contexte).
        Supporte: M1, M5, M15, M30, H1 comme timeframe signal.
        Chaque itération = 1 mois en mémoire max.
        """
        # Charger les bougies du timeframe signal (sert au groupement par mois)
        all_signal = self.load_candles(symbol, signal_timeframe, start, end)

        # Charger H1 si disponible et demandé
        all_h1 = None
        if load_h1:
            try:
                all_h1 = self.load_candles(symbol, "H1", start, end)
            except FileNotFoundError:
                all_h1 = None

        # Grouper par mois sur le timeframe signal
        all_signal["month"] = all_signal["time"].dt.to_period("M")

        for month, group in all_signal.groupby("month"):
            month_str = str(month)

            # Chercher le fichier ticks correspondant
            ticks_file = self.ticks_dir / f"{symbol}_ticks_{month_str}.parquet"

            # Charger M1 pour résolution fine — seulement si le signal n'est pas déjà M1
            m1_month = None
            if signal_timeframe != "M1":
                try:
                    m1_month = self.load_candles(
                        symbol, "M1",
                        str(group["time"].min()),
                        str(group["time"].max())
                    )
                except FileNotFoundError:
                    m1_month = None

            # Filtrer H1 pour ce mois
            h1_month = None
            if all_h1 is not None:
                h1_mask = (
                    (all_h1["time"] >= group["time"].min()) &
                    (all_h1["time"] <= group["time"].max())
                )
                h1_filtered = all_h1.loc[h1_mask]
                if len(h1_filtered) > 0:
                    h1_month = h1_filtered.reset_index(drop=True)

            chunk = DataChunk(
                month=month_str,
                signal_timeframe=signal_timeframe,
                candles_signal=group.drop(columns=["month"]).reset_index(drop=True),
                candles_m1=m1_month,
                candles_h1=h1_month,
                ticks_path=ticks_file if ticks_file.exists() else None,
            )

            yield chunk
            # Le garbage collector libère le chunk précédent
