import pandas as pd

# Load both files
df_pm = pd.read_parquet('data/btc_5min_markets.parquet')
df_price = pd.read_parquet('data/BTCUSDT_M5.parquet')

print('=== Polymarket Data ===')
print(f'Rows: {len(df_pm)}')
print(f'event_start_ts range: {df_pm["event_start_ts"].min()} to {df_pm["event_start_ts"].max()}')
print(f'First 5 timestamps: {df_pm["event_start_ts"].head().tolist()}')

print('\n=== Price Data ===')
print(f'Rows: {len(df_price)}')
print(f'Index range: {df_price.index[0]} to {df_price.index[-1]}')
print(f'First 5 timestamps (unix): {[int(t.timestamp()) for t in df_price.index[:5]]}')

# Check alignment
pm_ts = set(df_pm['event_start_ts'].astype(int))
price_ts = set([int(t.timestamp()) for t in df_price.index])

common = pm_ts & price_ts
print('\n=== Alignment ===')
print(f'Common timestamps: {len(common)}')
print(f'PM only: {len(pm_ts - price_ts)}')
print(f'Price only: {len(price_ts - pm_ts)}')
if common:
    print(f'First 5 common: {sorted(common)[:5]}')
