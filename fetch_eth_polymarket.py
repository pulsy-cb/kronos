"""
Fetch ETH Polymarket 5min markets from polymarket.com event pages.
Extracts market data from __NEXT_DATA__ JSON embedded in the HTML.
Also determines settlement results from ETHUSDT_M5 price data.
"""
import requests
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

# Load ETHUSDT M5 price data to determine settlement results
print("Loading ETHUSDT_M5 price data...")
df_eth = pd.read_parquet("data/ETHUSDT_M5.parquet")
print(f"Price range: {df_eth.index.min()} to {df_eth.index.max()}")

# Create a fast lookup: timestamp -> closing price (and next candle open)
prices = {}
for ts, row in df_eth.iterrows():
    unix_ts = int(ts.timestamp())
    prices[unix_ts] = {
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
    }

def determine_result(start_ts):
    """Determine if ETH was UP or DOWN during the 5min window.
    Compare window start close vs window end close (or next candle)."""
    if start_ts not in prices:
        return None
    end_ts = start_ts + 300
    start_close = prices[start_ts]["close"]
    if end_ts in prices:
        end_close = prices[end_ts]["close"]
    else:
        return None
    return "UP" if end_close >= start_close else "DOWN"

# ── Scrape ETH event page ──────────────────────────────────────────
url = "https://polymarket.com/event/eth-updown-5m-1776961200"
print(f"\nFetching: {url}")
resp = session.get(url, timeout=20)
print(f"Status: {resp.status_code}, Length: {len(resp.text)}")

# Extract __NEXT_DATA__
next_data_match = re.search(r'<script id="__NEXT_DATA__"[^\u003e]*\u003e(.*?)\u003c/script\u003e', resp.text, re.DOTALL)
if not next_data_match:
    raise RuntimeError("Could not find __NEXT_DATA__ in page")

data = json.loads(next_data_match.group(1).strip())
page_props = data.get("props", {}).get("pageProps", {})
dehydrated = page_props.get("dehydratedState", {})
queries = dehydrated.get("queries", [])

print(f"Queries found: {len(queries)}")

# Extract all markets
markets = []
for q in queries:
    state = q.get("state", {})
    query_data = state.get("data", [])
    if not isinstance(query_data, list):
        continue
    for item in query_data:
        if not isinstance(item, dict):
            continue
        for key in ["markets", "events"]:
            if key not in item:
                continue
            val = item[key]
            if isinstance(val, list):
                for m in val:
                    if isinstance(m, dict) and "eth-updown" in str(m.get("slug", "")):
                        markets.append(m)
            elif isinstance(val, dict):
                for sub_k, sub_v in val.items():
                    if isinstance(sub_v, list):
                        for m in sub_v:
                            if isinstance(m, dict) and "eth-updown" in str(m.get("slug", "")):
                                markets.append(m)

print(f"Total ETH market objects extracted: {len(markets)}")

# ── Build rows ─────────────────────────────────────────────────────
rows = []
for m in markets:
    slug = m.get("slug", "")
    if not slug:
        continue
    # Extract timestamp from slug
    parts = slug.split("-")
    if len(parts) < 4:
        continue
    try:
        event_start_ts = int(parts[3])
    except ValueError:
        continue

    event_end_ts = event_start_ts + 300

    tokens = m.get("tokens", [])
    up_token = None
    down_token = None
    price_open_up = None
    price_open_down = None
    up_token_id = None
    down_token_id = None
    result_up_final_price = None
    result_down_final_price = None

    for t in tokens:
        outcome = t.get("outcome", "")
        if outcome == "Up":
            up_token_id = t.get("token_id") or t.get("id")
            up_token = t
        elif outcome == "Down":
            down_token_id = t.get("token_id") or t.get("id")
            down_token = t

    # Get market mid-point price (neg-risk style)
    mid_point = m.get("midPoint")
    if mid_point is not None:
        price_open_up = float(mid_point)
        price_open_down = 1.0 - price_open_up
    elif up_token:
        price_open_up = float(up_token.get("price", 0.5))
        price_open_down = 1.0 - price_open_up

    # Determine result from ETH price data
    settlement_result = determine_result(event_start_ts)
    if settlement_result:
        if settlement_result == "UP":
            result_up_final_price = 1.0
            result_down_final_price = 0.0
        else:
            result_up_final_price = 0.0
            result_down_final_price = 1.0

    rows.append({
        "slug": slug,
        "condition_id": m.get("conditionId"),
        "question_id": m.get("questionId"),
        "up_token_id": up_token_id,
        "down_token_id": down_token_id,
        "event_start_ts": event_start_ts,
        "event_end_ts": event_end_ts,
        "result": settlement_result,
        "result_up_final_price": result_up_final_price,
        "result_down_final_price": result_down_final_price,
        "price_open_up": price_open_up,
        "price_close_up": price_open_up,  # approximation
        "price_5min_before_up": price_open_up,
        "price_min_up": None,
        "price_max_up": None,
        "price_open_down": price_open_down,
        "price_close_down": price_open_down,
        "price_5min_before_down": price_open_down,
        "price_min_down": None,
        "price_max_down": None,
        "volume_usdc": m.get("volume") or m.get("volumeNum") or 0,
        "liquidity_usdc": m.get("liquidity") or m.get("liquidityNum") or 0,
    })

print(f"Rows built: {len(rows)}")

if not rows:
    print("WARNING: No rows built. Exiting.")
    exit(1)

df_markets = pd.DataFrame(rows).sort_values("event_start_ts").reset_index(drop=True)

# Save
output_path = "data/eth_5min_markets.parquet"
df_markets.to_parquet(output_path)
print(f"\nSaved: {output_path}")
print(f"Total markets: {len(df_markets)}")
print(f"Range: {df_markets['event_start_ts'].min()} to {df_markets['event_start_ts'].max()}")
print(f"Result counts:")
print(df_markets["result"].value_counts().to_string())

# Also save a summary
print(f"\nWith result: {df_markets['result'].notna().sum()} / {len(df_markets)}")
print("Done!")
