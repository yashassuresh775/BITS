#!/usr/bin/env python3
"""Recompute ``source_url`` (and fill ``ticker`` from OHLCV) in ``p2_signals.csv`` without re-hitting EDGAR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p2.insider_signals import coerce_p2_signal_columns

_PREFERRED_ORDER = [
    "sec_id",
    "ticker",
    "event_date",
    "event_type",
    "headline",
    "source_url",
    "pre_drift_flag",
    "suspicious_window_start",
    "remarks",
    "time_to_run",
]


def main() -> None:
    p = argparse.ArgumentParser(description="Refresh P2 source_url + ticker from ohlcv map")
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=ROOT / "p2_signals.csv",
        help="Input CSV (default: repo p2_signals.csv)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: overwrite --input)",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "equity-data",
        help="Folder with ohlcv.csv for sec_id→ticker",
    )
    p.add_argument(
        "--prefer-ohlcv-ticker-for-url",
        action="store_true",
        help=(
            "Use OHLCV ticker in browse-edgar URL (aligns link with sec_id's listing). "
            "Default: URL from headline CIK / Archives only so the link matches the filing text "
            "(EDGAR full-text hits often mention other issuers than the searched ticker)."
        ),
    )
    args = p.parse_args()
    out = args.output or args.input
    df = pd.read_csv(args.input)
    df["sec_id"] = pd.to_numeric(df["sec_id"], errors="coerce")

    ohlcv_path = args.data_root / "ohlcv.csv"
    if ohlcv_path.is_file():
        ohlcv = pd.read_csv(ohlcv_path)
        if "sec_id" in ohlcv.columns and "ticker" in ohlcv.columns:
            tick_map = (
                ohlcv[["sec_id", "ticker"]]
                .dropna(subset=["sec_id"])
                .assign(sec_id=lambda x: pd.to_numeric(x["sec_id"], errors="coerce"))
                .dropna(subset=["sec_id"])
                .drop_duplicates(subset=["sec_id"], keep="first")
            )
            df = df.drop(columns=["ticker"], errors="ignore")
            df = df.merge(tick_map, on="sec_id", how="left")
    if "ticker" not in df.columns:
        df["ticker"] = ""
    df["ticker"] = df["ticker"].fillna("").astype(str).str.upper().str.strip()
    df.loc[df["ticker"].isin(("", "NAN", "NONE", "<NA>")), "ticker"] = ""

    df = coerce_p2_signal_columns(
        df,
        source_url_prefer_listing_ticker=args.prefer_ohlcv_ticker_for_url,
    )

    ordered = [c for c in _PREFERRED_ORDER if c in df.columns]
    ordered += [c for c in df.columns if c not in ordered]
    df = df[ordered]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    n = len(df)
    with_urls = (df["source_url"].astype(str).str.contains("browse-edgar|Archives/edgar", regex=True)).sum()
    print(f"Wrote {n} rows to {out} ({with_urls} rows with browse-edgar or Archives URL)")


if __name__ == "__main__":
    main()
