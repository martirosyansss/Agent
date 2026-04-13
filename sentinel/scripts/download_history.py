"""
Script to download historical candlesticks (klines) from Binance REST API and save to Sentinel's local SQLite database.
Usage: python download_history.py --symbol BTCUSDT --interval 1h --start "2023-01-01" --end "2023-12-31"
"""

import argparse
import sys
import time
from datetime import datetime, timezone
import requests
from loguru import logger
from pathlib import Path

# Add sentinel root to path so we can import from core and database
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import load_settings
from database.db import Database
from database.repository import Repository
from core.models import Candle

BINANCE_REST_BASE = "https://api.binance.com"

def parse_date(date_str: str) -> int:
    """Parses date string (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS) and returns UNIX timestamp in milliseconds."""
    try:
        if len(date_str) == 10:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        else:
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError as e:
        logger.error(f"Invalid date format: {date_str}. Use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")
        sys.exit(1)

def fetch_klines_chunk(symbol: str, interval: str, start_time: int, end_time: int) -> list:
    """Fetch one chunk of klines (up to 1000) from Binance."""
    url = f"{BINANCE_REST_BASE}/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": 1000
    }
    
    # Simple rate limiting protection
    time.sleep(0.5) 
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        logger.error(f"Failed to fetch klines: HTTP {response.status_code} - {response.text}")
        response.raise_for_status()
        
    return response.json()

def main():
    parser = argparse.ArgumentParser(description="Download Binance historical klines")
    parser.add_argument("--symbol", type=str, required=True, help="Trading pair symbol, e.g., BTCUSDT")
    parser.add_argument("--interval", type=str, default="1h", choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"], help="Kline interval")
    parser.add_argument("--start", type=str, required=True, help="Start date (UTC) in YYYY-MM-DD or 'YYYY-MM-DD HH:MM:SS' format")
    parser.add_argument("--end", type=str, help="End date (UTC). Defaults to current time if omitted.")
    
    args = parser.parse_args()
    
    symbol = args.symbol.upper()
    interval = args.interval
    start_ts = parse_date(args.start)
    end_ts = parse_date(args.end) if args.end else int(time.time() * 1000)
    
    if start_ts >= end_ts:
        logger.error("Start date must be before end date.")
        sys.exit(1)
        
    logger.info(f"Initializing connection to Sentinel database...")
    settings = load_settings()
    db = Database(BASE_DIR / settings.db_path)
    db.connect()
    repo = Repository(db)
    
    logger.info(f"Starting download for {symbol} ({interval})")
    
    total_downloaded = 0
    current_start = start_ts
    
    while current_start < end_ts:
        logger.info(f"Fetching chunk starting from {datetime.fromtimestamp(current_start/1000, tz=timezone.utc)}")
        try:
            raw_klines = fetch_klines_chunk(symbol, interval, current_start, end_ts)
        except Exception as e:
            logger.error(f"Download stopped due to error: {e}")
            break
            
        if not raw_klines:
            logger.info("No more data returned from Binance.")
            break
            
        candles = []
        for k in raw_klines:
            candle_ts = int(k[0])
            candles.append(Candle(
                timestamp=candle_ts,
                symbol=symbol,
                interval=interval,
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                trades_count=int(k[8])
            ))
            
        # Update current_start to the timestamp of the last candle + 1 ms to fetch the next chunk
        last_candle_ts = int(raw_klines[-1][0])
        
        # Stop condition to prevent infinite loops if the API returns old data
        if last_candle_ts < current_start:
            logger.warning("Binance returned data older than requested. Breaking loop.")
            break
            
        current_start = last_candle_ts + 1
        
        # Save to DB in batch
        inserted = repo.upsert_candles_batch(candles)
        total_downloaded += inserted
        logger.info(f"Saved {inserted} candles. Total downloaded so far: {total_downloaded}")

    logger.info(f"Download complete. Total candles saved to the database: {total_downloaded}")

if __name__ == "__main__":
    main()
