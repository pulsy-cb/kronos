"""
Convert tick-by-tick parquet data into OHLCV candle parquets
for use with Kronos prediction and backtesting.

Outputs one continuous file per symbol+timeframe in data/ (e.g. XAUUSD_M5.parquet).

Usage:
    python data/build_candles.py                    # default: XAUUSD, M5
    python data/build_candles.py --symbol XAUUSD --timeframes M5 M15 H1 H4 D1
    python data/build_candles.py --timeframes all   # build all standard timeframes
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np


TIMEFRAMES = {
    "S30": "30s",
    "M1":  "1min",
    "M5":  "5min",
    "M15": "15min",
    "M30": "30min",
    "H1":  "1h",
    "H4":  "4h",
    "D1":  "1D",
}


def build_candles(symbol: str, timeframe: str, ticks_dir: Path, output_dir: Path):
    """
    Read all monthly tick parquets for a symbol, resample to OHLCV candles,
    and save a single continuous parquet file.
    """
    freq = TIMEFRAMES.get(timeframe)
    if freq is None:
        raise ValueError(f"Unknown timeframe: {timeframe}. Available: {list(TIMEFRAMES.keys())}")

    # Find tick files
    tick_files = sorted(ticks_dir.glob(f"{symbol}_ticks_*.parquet"))
    if not tick_files:
        print(f"  No tick files found for {symbol} in {ticks_dir}")
        return None

    print(f"  Loading {len(tick_files)} tick files for {symbol} {timeframe}...")

    all_dfs = []
    for tf in tick_files:
        # Only load time + bid columns to save RAM
        df = pd.read_parquet(tf, columns=["time", "bid", "ask"])
        all_dfs.append(df)

    print(f"  Concatenating {sum(len(d) for d in all_dfs):,} ticks...")
    ticks = pd.concat(all_dfs, ignore_index=True)
    del all_dfs

    ticks["time"] = pd.to_datetime(ticks["time"])
    ticks = ticks.sort_values("time").drop_duplicates(subset=["time"])

    # Use mid-price for OHLC (average of bid/ask), fallback to bid
    if "ask" in ticks.columns and "bid" in ticks.columns:
        ticks["price"] = (ticks["bid"] + ticks["ask"]) / 2
    elif "bid" in ticks.columns:
        ticks["price"] = ticks["bid"]
    else:
        raise ValueError("No price column found (need bid or last)")

    ticks = ticks.set_index("time")

    print(f"  Resampling to {timeframe} ({freq})...")
    candles = ticks["price"].resample(freq).agg(
        open="first",
        high="max",
        low="min",
        close="last",
    )

    # Volume = tick count per bar
    candles["volume"] = ticks["price"].resample(freq).count()

    # Drop bars with no activity (market closed)
    candles = candles.dropna(subset=["open"])
    candles = candles[candles["volume"] > 0]

    # Amount = approximated as (open+high+low+close)/4 * volume
    candles["amount"] = ((candles["open"] + candles["high"] + candles["low"] + candles["close"]) / 4) * candles["volume"]

    # Reset index to get 'time' as column, then rename to 'timestamps' for Kronos
    candles = candles.reset_index()
    candles = candles.rename(columns={"time": "timestamps"})

    # Ensure correct dtypes
    for col in ["open", "high", "low", "close"]:
        candles[col] = candles[col].astype(np.float64)
    candles["volume"] = candles["volume"].astype(np.float64)
    candles["amount"] = candles["amount"].astype(np.float64)

    # Final column order matching Kronos expectations
    candles = candles[["timestamps", "open", "high", "low", "close", "volume", "amount"]]

    # Save
    output_path = output_dir / f"{symbol}_{timeframe}.parquet"
    candles.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    size_mb = output_path.stat().st_size / 1024 / 1024

    print(f"  Saved {len(candles):,} candles → {output_path.name} ({size_mb:.2f} MB)")
    print(f"  Range: {candles['timestamps'].min()} → {candles['timestamps'].max()}")

    del ticks, candles
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Build OHLCV candle parquets from tick data for Kronos"
    )
    parser.add_argument("--symbol", default="XAUUSD", help="Symbol to process (default: XAUUSD)")
    parser.add_argument(
        "--timeframes", nargs="+", default=["M5"],
        help="Timeframes to build (S30 M1 M5 M15 M30 H1 H4 D1, or 'all')"
    )
    parser.add_argument(
        "--ticks-dir", default="data/parquet/ticks",
        help="Directory containing tick parquets"
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="Output directory for candle parquets (must be visible to webui)"
    )
    args = parser.parse_args()

    if args.timeframes == ["all"]:
        args.timeframes = list(TIMEFRAMES.keys())

    ticks_dir = Path(args.ticks_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify tick data exists
    tick_files = list(ticks_dir.glob(f"{args.symbol}_ticks_*.parquet"))
    if not tick_files:
        print(f"ERROR: No tick files found for {args.symbol} in {ticks_dir}")
        print(f"Expected pattern: {args.symbol}_ticks_YYYY-MM.parquet")
        sys.exit(1)

    print(f"Found {len(tick_files)} tick files for {args.symbol}")
    print(f"Building timeframes: {args.timeframes}")
    print("=" * 60)

    results = []
    for tf in args.timeframes:
        print(f"\nBuilding {args.symbol} {tf}...")
        path = build_candles(args.symbol, tf, ticks_dir, output_dir)
        if path:
            results.append(path)

    print("\n" + "=" * 60)
    print("DONE — Output files:")
    for p in results:
        df = pd.read_parquet(p)
        print(f"  {p.name:20s}  {len(df):>8,} rows  "
              f"{df['timestamps'].min().strftime('%Y-%m-%d')} → {df['timestamps'].max().strftime('%Y-%m-%d')}")

    print(f"\nThese files are in {output_dir}/ and will be visible in the webui.")


if __name__ == "__main__":
    main()