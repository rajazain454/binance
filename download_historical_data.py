"""Script to fetch 365 days of Binance 1h historical market data for all symbols."""

import os
import sys
import json
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from binance_client import BinanceClient

def main():
    with open("config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    symbols = cfg.get("symbols", [])
    data_dir = cfg.get("paths", {}).get("data_dir", "data")
    os.makedirs(data_dir, exist_ok=True)

    client = BinanceClient(cfg)
    days = 365
    tf = cfg.get("timeframe", "1h")

    print(f"=== DOWNLOADING {days} DAYS OF BINANCE {tf.upper()} DATA FOR {len(symbols)} SYMBOLS ===")
    total_downloaded = 0

    for idx, sym in enumerate(symbols, 1):
        clean_sym = sym.replace("/", "_")
        cache_file = os.path.join(data_dir, f"binance_{clean_sym}_{tf}.csv")
        print(f"\n[{idx}/{len(symbols)}] Fetching {sym} ({days} days)...")
        t0 = time.time()
        try:
            df = client.fetch_historical_ohlcv(sym, timeframe=tf, days=days)
            if not df.empty:
                df.to_csv(cache_file)
                bars = len(df)
                total_downloaded += bars
                elapsed = time.time() - t0
                print(f"  [OK] Saved {bars:,} bars to {cache_file} ({elapsed:.1f}s)")
                print(f"       Range: {df.index[0]} -> {df.index[-1]}")
            else:
                print(f"  [SKIP] Received empty dataframe for {sym}")
        except Exception as e:
            print(f"  [ERROR] Error fetching {sym}: {e}")

    print("\n" + "=" * 60)
    print(f"DOWNLOAD COMPLETE: {total_downloaded:,} total candles across {len(symbols)} coins.")
    print("=" * 60)

if __name__ == "__main__":
    main()
