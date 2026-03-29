#!/usr/bin/env python3
"""Problem 2 — EDGAR 8-K scraper + pre-announcement activity signals → p2_signals.csv."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from p2.edgar import build_edgar_search_overrides, fetch_8k_filings, merge_sec_ids
from p2.insider_signals import build_p2_signals


def load_ticker_map(root: Path, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer sec_id_map.csv if present. Otherwise build from ohlcv.csv columns
    sec_id + ticker (organiser packs often embed the mapping in OHLCV only).
    """
    map_path = root / "sec_id_map.csv"
    if map_path.is_file():
        m = pd.read_csv(map_path)
    elif "ticker" not in ohlcv.columns or "sec_id" not in ohlcv.columns:
        raise SystemExit(
            f"No {map_path.name} and ohlcv.csv has no ticker column — cannot map EDGAR tickers to sec_id."
        )
    else:
        m = (
            ohlcv[["sec_id", "ticker"]]
            .drop_duplicates(subset=["sec_id"])
            .assign(ticker=lambda d: d["ticker"].astype(str).str.upper().str.strip())
        )
        print(f"Using sec_id↔ticker from ohlcv.csv ({len(m)} rows); no separate sec_id_map.csv.")
    if "edgar_query" not in m.columns:
        m = m.assign(edgar_query="")
    else:
        m = m.assign(edgar_query=lambda d: d["edgar_query"].fillna("").astype(str).str.strip())
    m = m.assign(ticker=lambda d: d["ticker"].astype(str).str.upper().str.strip())
    nq = (m["edgar_query"] != "").sum()
    if nq:
        print(f"EDGAR: using non-empty edgar_query overrides for {nq} ticker(s) (legal-name search).")
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Problem 2 insider-trading signal pipeline")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Folder with ohlcv.csv, trade_data.csv (optional sec_id_map.csv if ohlcv has no ticker)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("p2_signals.csv"),
        help="Output CSV (default: p2_signals.csv)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2026-01-01",
        help="EDGAR search start (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2026-03-31",
        help="EDGAR search end (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--skip-edgar",
        action="store_true",
        help="Use pre-fetched filings CSV instead of calling the API (see --filings-cache).",
    )
    parser.add_argument(
        "--filings-cache",
        type=Path,
        default=None,
        help="CSV from a prior EDGAR run (columns: ticker, file_date, headline, filing_url, ...)",
    )
    parser.add_argument(
        "--edgar-timeout",
        type=float,
        default=10.0,
        help="Per-request timeout seconds (organizer starter uses 10).",
    )
    parser.add_argument(
        "--edgar-sleep",
        type=float,
        default=0.3,
        help="Pause between EDGAR bursts (s); with --edgar-concurrency 1, pause after each ticker (starter: 0.3).",
    )
    parser.add_argument(
        "--edgar-concurrency",
        type=int,
        default=3,
        help="Parallel tickers per burst (default 3 for <~60s runs); use 1 for strict sequential + sleep after each.",
    )
    parser.add_argument(
        "--edgar-retries",
        type=int,
        default=3,
        help="Retries on HTTP 5xx / network errors per ticker (not in starter; helps flaky SEC).",
    )
    parser.add_argument(
        "--ma-only",
        action="store_true",
        help="Keep only M&A-classified 8-Ks (headline keywords: merger, acquisition, …).",
    )
    args = parser.parse_args()

    root = args.data_root.resolve()
    ohlcv_path = root / "ohlcv.csv"
    trades_path = root / "trade_data.csv"
    for p, label in [(ohlcv_path, "ohlcv.csv"), (trades_path, "trade_data.csv")]:
        if not p.is_file():
            raise SystemExit(f"Missing {label} under {root}")

    t0 = time.perf_counter()
    ohlcv = pd.read_csv(ohlcv_path)
    trades = pd.read_csv(trades_path)
    ticker_map = load_ticker_map(root, ohlcv)

    tickers = ticker_map["ticker"].astype(str).str.upper().unique().tolist()
    edgar_over = build_edgar_search_overrides(ticker_map)

    if args.skip_edgar:
        if args.filings_cache is None or not args.filings_cache.is_file():
            raise SystemExit("--skip-edgar requires --filings-cache pointing to an existing CSV")
        filings = pd.read_csv(args.filings_cache, parse_dates=["file_date"])
        if "filing_url" not in filings.columns and "source_url" in filings.columns:
            filings = filings.rename(columns={"source_url": "filing_url"})
    else:
        filings = fetch_8k_filings(
            tickers=tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            sleep_s=args.edgar_sleep,
            timeout=args.edgar_timeout,
            max_retries=args.edgar_retries,
            search_overrides=edgar_over or None,
            batch_concurrency=max(1, args.edgar_concurrency),
        )
        if filings.empty:
            print("EDGAR returned no rows — check tickers, dates, or network.")
    if not filings.empty and "event_type" not in filings.columns:
        from p2.edgar import classify_event

        filings["headline"] = filings.get("headline", pd.Series("", index=filings.index)).fillna("")
        filings["event_type"] = filings["headline"].map(classify_event)

    filings = merge_sec_ids(filings, ticker_map)
    filings = filings.dropna(subset=["sec_id"])
    if filings.empty:
        print("No filings joined to sec_id — check ticker_map tickers vs EDGAR entity tickers.")

    elapsed = time.perf_counter() - t0
    out = build_p2_signals(ohlcv, trades, filings, time_to_run_s=elapsed, ma_only=args.ma_only)
    # refresh time_to_run with full pipeline wall time
    out["time_to_run"] = round(time.perf_counter() - t0, 3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output} in {out['time_to_run'].iloc[0] if len(out) else elapsed:.3f}s")
    if not out.empty:
        print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
