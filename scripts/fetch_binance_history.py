#!/usr/bin/env python3
"""
Download **historical** Binance Spot 1m klines + aggregate trades into a student-pack layout:

  DATA_ROOT/crypto-market/Binance_<SYMBOL>_<YEAR>_minute.csv
  DATA_ROOT/crypto-trades/<SYMBOL>_trades.csv

Then run the same detectors on that folder (no more “last 500 minutes only”):

  python3 run_p3.py --data-root DATA_ROOT -o submission_from_binance.csv

Use ``BINANCE_SPOT_API`` if ``api.binance.com`` returns HTTP 451.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p3.config import SYMBOLS
from p3.live.historical import fetch_history_pack


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill Binance history into crypto-market/ + crypto-trades/ CSVs",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "binance-hist",
        help="Output folder (will contain crypto-market/ and crypto-trades/)",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=3.0,
        help="Lookback from now (UTC), in days (default: 3)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated symbols (default: all SYMBOLS from p3.config)",
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        default=250_000,
        metavar="N",
        help="Max aggregate trades to store per symbol (default: 250000)",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.1,
        help="Seconds to sleep between paginated API calls (rate limits)",
    )
    parser.add_argument(
        "--year-suffix",
        type=str,
        default="2026",
        metavar="YYYY",
        help="Filename middle part Binance_SYM_YYYY_minute.csv for io.market_path (default: 2026)",
    )
    args = parser.parse_args()

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(args.days * 86400 * 1000)
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()] or list(SYMBOLS)

    out = args.data_root.resolve()
    print(f"UTC window: {args.days} days ending now")
    print(f"Symbols: {', '.join(syms)}")
    print(f"Output: {out}")

    counts = fetch_history_pack(
        str(out),
        syms,
        start_ms,
        end_ms,
        max_trades_per_symbol=args.max_trades,
        pause_sec=args.pause,
        filename_year=args.year_suffix,
    )
    for sym, (nk, nt) in counts.items():
        print(f"  {sym}: {nk:,} minute bars, {nt:,} agg trades")
    print()
    print("Next:")
    print(f"  python3 run_p3.py --data-root {out} -o submission_from_binance.csv")
    print("Dashboard: set primary CSV to that file, or compare with submission_live.csv.")


if __name__ == "__main__":
    main()
