#!/usr/bin/env python3
"""Problem 1 — order-book concentration + DBSCAN clusters → p1_alerts.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

from p1.io import load_market_data, load_trades_per_minute
from p1.pipeline import build_alerts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Problem 1: per-minute order book anomalies and clustered alerts"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Folder with market_data.csv (and optionally trade_data.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("p1_alerts.csv"),
        help="Output CSV (default: p1_alerts.csv)",
    )
    parser.add_argument(
        "--market-file",
        type=str,
        default="market_data.csv",
        help="Order book CSV filename inside data-root (default: market_data.csv)",
    )
    parser.add_argument(
        "--trades-file",
        type=str,
        default="trade_data.csv",
        help="Optional trades CSV for buy-vs-depth remarks (default: trade_data.csv if present)",
    )
    parser.add_argument(
        "--no-trades",
        action="store_true",
        help="Do not load trade_data.csv even if it exists.",
    )
    args = parser.parse_args()

    root = args.data_root.resolve()
    market_path = root / args.market_file
    if not market_path.is_file():
        raise SystemExit(f"Missing {market_path}")

    md = load_market_data(market_path)
    tpm = None
    if not args.no_trades:
        tp = root / args.trades_file
        if tp.is_file():
            tpm = load_trades_per_minute(str(tp))

    out, elapsed = build_alerts(md, tpm)
    # single time_to_run for whole run (problem asks for seconds to produce output)
    if not out.empty:
        out["time_to_run"] = round(elapsed, 3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output} in {elapsed:.3f}s")
    if not out.empty:
        print(out.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
