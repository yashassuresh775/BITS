#!/usr/bin/env python3
"""
Problem 2 — EDGAR-only step: fetch 8-K filings and write a CSV for offline P2.

Use this before touching signals: validate tickers, dates, and API behaviour in ~one command.
Then run the full pipeline without hitting EDGAR:

  python3 run_p2.py --data-root /path/to/equity \\
    --skip-edgar --filings-cache p2_filings_cache.csv -o p2_signals.csv

Needs ``ohlcv.csv`` under ``--data-root`` (and optional ``sec_id_map.csv``); ``trade_data.csv`` is not read.

Usage:
  python3 scripts/fetch_p2_filings_only.py --data-root data/equity-data -o p2_filings_cache.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p2.edgar import build_edgar_search_overrides, fetch_8k_filings

from run_p2 import load_ticker_map


def main() -> None:
    p = argparse.ArgumentParser(description="P2: fetch EDGAR 8-K filings only → CSV cache")
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Folder with ohlcv.csv (optional sec_id_map.csv)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("p2_filings_cache.csv"),
        help="Output filings CSV (default: p2_filings_cache.csv)",
    )
    p.add_argument("--start-date", type=str, default="2026-01-01", help="EDGAR start YYYY-MM-DD")
    p.add_argument("--end-date", type=str, default="2026-03-31", help="EDGAR end YYYY-MM-DD")
    p.add_argument("--edgar-timeout", type=float, default=10.0, help="Per-request timeout (s)")
    p.add_argument("--edgar-sleep", type=float, default=0.3, help="Pause between bursts (s); per-ticker if concurrency=1")
    p.add_argument(
        "--edgar-concurrency",
        type=int,
        default=3,
        help="Tickers per parallel burst (default 3); 1 = sequential starter behavior",
    )
    p.add_argument("--edgar-retries", type=int, default=3, help="Retries on 5xx / network")
    args = p.parse_args()

    root = args.data_root.resolve()
    ohlcv_path = root / "ohlcv.csv"
    if not ohlcv_path.is_file():
        raise SystemExit(f"Missing ohlcv.csv under {root}")

    t0 = time.perf_counter()
    ohlcv = pd.read_csv(ohlcv_path)
    ticker_map = load_ticker_map(root, ohlcv)
    tickers = ticker_map["ticker"].astype(str).str.upper().unique().tolist()
    overrides = build_edgar_search_overrides(ticker_map)

    filings = fetch_8k_filings(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        sleep_s=args.edgar_sleep,
        timeout=args.edgar_timeout,
        max_retries=args.edgar_retries,
        search_overrides=overrides or None,
        batch_concurrency=max(1, args.edgar_concurrency),
    )
    elapsed = time.perf_counter() - t0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if filings.empty:
        filings.to_csv(args.output, index=False)
        print(f"EDGAR returned 0 rows in {elapsed:.3f}s — wrote empty {args.output}")
        return

    # Normalise file_date for CSV (run_p2 parse_dates works either way)
    out = filings.copy()
    out["file_date"] = pd.to_datetime(out["file_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} filing rows to {args.output} in {elapsed:.3f}s ({len(tickers)} tickers)")


if __name__ == "__main__":
    main()
