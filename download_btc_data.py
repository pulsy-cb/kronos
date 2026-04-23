#!/usr/bin/env python3
"""
Télécharge les données historiques M1 et M5 depuis Binance
pour la période correspondant aux données Polymarket.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

SYMBOL = "BTCUSDT"
START_TS = 1774208100  # Min from Polymarket data
END_TS = 1776883500    # Max from Polymarket data

OUTPUT_DIR = Path("data/parquet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_klines(symbol, interval, start_ts, end_ts):
    """
    Fetch all klines from Binance between start and end timestamps.
    Binance limits to 1000 candles per request, so we paginate.
    """
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    
    current_start = start_ts * 1000  # Binance uses milliseconds
    end_ms = end_ts * 1000
    
    print(f"Fetching {interval} data from {datetime.fromtimestamp(start_ts, tz=timezone.utc)} to {datetime.fromtimestamp(end_ts, tz=timezone.utc)}")
    
    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(current_start),
            "endTime": int(end_ms),
            "limit": 1000
        }
        
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        klines = resp.json()
        
        if not klines:
            break
            
        all_klines.extend(klines)
        print(f"  Fetched {len(klines)} candles | Total: {len(all_klines)}")
        
        # Move to next batch
        last_ts = int(klines[-1][0])
        current_start = last_ts + 1
    
    return all_klines


def klines_to_df(klines):
    """Convert klines to DataFrame with OHLCV format."""
    if not klines:
        return pd.DataFrame()
    
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "nb_trades", "taker_buy_volume", "taker_buy_quote", "ignore"
    ])
    
    df = df[[
        "timestamp", "open", "high", "low", "close", "volume",
        "quote_volume", "nb_trades"
    ]]
    
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["quote_volume"] = df["quote_volume"].astype(float)
    df["nb_trades"] = df["nb_trades"].astype(int)
    
    # Add amount column (same as in live_cli.py)
    df["amount"] = ((df["open"] + df["high"] + df["low"] + df["close"]) / 4) * df["volume"]
    
    df = df.set_index("timestamp")
    return df


def main():
    print("=" * 70)
    print(f"Downloading Binance data for {SYMBOL}")
    print(f"Period: {datetime.fromtimestamp(START_TS, tz=timezone.utc)} to {datetime.fromtimestamp(END_TS, tz=timezone.utc)}")
    print("=" * 70)
    
    # Fetch M1 data
    print("\n[1/2] Fetching M1 (1-minute) candles...")
    m1_klines = fetch_klines(SYMBOL, "1m", START_TS, END_TS)
    df_m1 = klines_to_df(m1_klines)
    print(f"M1: {len(df_m1)} candles")
    
    # Fetch M5 data
    print("\n[2/2] Fetching M5 (5-minute) candles...")
    m5_klines = fetch_klines(SYMBOL, "5m", START_TS, END_TS)
    df_m5 = klines_to_df(m5_klines)
    print(f"M5: {len(df_m5)} candles")
    
    # Save to parquet
    m1_path = OUTPUT_DIR / "BTCUSDT_M1.parquet"
    m5_path = OUTPUT_DIR / "BTCUSDT_M5.parquet"
    
    df_m1.to_parquet(m1_path)
    print(f"\n[OK] Saved M1: {m1_path}")
    
    df_m5.to_parquet(m5_path)
    print(f"[OK] Saved M5: {m5_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"M1: {len(df_m1):,} candles | {df_m1.index[0]} to {df_m1.index[-1]}")
    print(f"M5: {len(df_m5):,} candles | {df_m5.index[0]} to {df_m5.index[-1]}")
    print(f"Files saved to: {OUTPUT_DIR.absolute()}")
    
    # Verify files
    print(f"\nVerification:")
    print(f"  M1 file exists: {m1_path.exists()}")
    print(f"  M5 file exists: {m5_path.exists()}")


if __name__ == "__main__":
    main()
