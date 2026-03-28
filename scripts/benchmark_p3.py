#!/usr/bin/env python3
"""
Example testing scenario: efficiency (wall-clock + throughput) and smoke checks
on the official student-pack layout — no labels required.

Usage:
  python scripts/benchmark_p3.py
  python scripts/benchmark_p3.py --data-root /path/to/student-pack --runs 3

Default data root: <repo>/data/student-pack (if that directory exists).
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from pathlib import Path

# Repo root = parent of scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from p3.io import discover_symbols, load_trades
from p3.pipeline import hits_to_submission, run_pipeline


def default_data_root() -> Path:
    return ROOT / "data" / "student-pack"


def count_trades(data_root: Path) -> tuple[int, dict[str, int]]:
    per: dict[str, int] = {}
    total = 0
    for sym in discover_symbols(data_root):
        try:
            t = load_trades(data_root, sym)
            n = len(t)
            per[sym] = n
            total += n
        except FileNotFoundError:
            per[sym] = 0
    return total, per


def validate_submission(sub: pd.DataFrame, data_root: Path) -> list[str]:
    """Return list of error strings; empty means all checks passed."""
    errors: list[str] = []
    required = {"symbol", "date", "trade_id"}
    if not required.issubset(sub.columns):
        errors.append(f"Missing columns: {required - set(sub.columns)}")
        return errors

    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    bad_dates = sub.loc[~sub["date"].astype(str).str.match(date_re), "date"]
    if len(bad_dates):
        errors.append(f"Bad date format rows: {len(bad_dates)}")

    # trade_id must exist in the trades file for that symbol
    cache: dict[str, set[str]] = {}
    for sym, g in sub.groupby("symbol"):
        sym = str(sym)
        if sym not in cache:
            try:
                tr = load_trades(data_root, sym)
                cache[sym] = set(tr["trade_id"].astype(str))
            except FileNotFoundError:
                errors.append(f"No trades file for symbol {sym}")
                continue
        known = cache[sym]
        unknown = ~g["trade_id"].astype(str).isin(known)
        if unknown.any():
            errors.append(
                f"{sym}: {unknown.sum()} trade_id(s) not found in {sym}_trades.csv"
            )

    # peg_break rows: price rule on USDCUSDT
    if "violation_type" in sub.columns:
        peg = sub[
            (sub["symbol"].astype(str) == "USDCUSDT")
            & (sub["violation_type"].astype(str) == "peg_break")
        ]
        if len(peg):
            tr = load_trades(data_root, "USDCUSDT")
            prices = tr.set_index("trade_id")["price"]
            for _, row in peg.iterrows():
                tid = str(row["trade_id"])
                if tid not in prices.index:
                    continue
                p = float(prices[tid])
                if abs(p - 1.0) <= 0.005:
                    errors.append(
                        f"peg_break row {tid} has price {p} (expected |p-1|>0.005)"
                    )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 benchmark + smoke test scenario")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="student-pack root (default: ./data/student-pack)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of timed pipeline runs (median reported; default 2)",
    )
    args = parser.parse_args()
    root = args.data_root
    if root is None:
        root = default_data_root()
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(
            f"Not a directory: {root}\n"
            "Copy your student-pack here: data/student-pack/ "
            "(with crypto-market/ and crypto-trades/), or pass --data-root."
        )

    print("=== Problem 3 — example testing scenario ===\n")
    print(f"Data root: {root}\n")

    print("1) Dataset footprint (trade rows per symbol)")
    total_trades, per = count_trades(root)
    for sym, n in sorted(per.items()):
        print(f"   {sym}: {n:,}")
    print(f"   TOTAL trades: {total_trades:,}\n")

    if total_trades == 0:
        raise SystemExit(
            "No trades loaded. Expected:\n"
            f"  {root}/crypto-trades/BTCUSDT_trades.csv  (and siblings)\n"
            "Copy the official student-pack into data/student-pack/ or fix --data-root."
        )

    print(f"2) Timed pipeline runs (n={args.runs}), median wall-clock")
    times: list[float] = []
    last_sub = None
    for i in range(args.runs):
        t0 = time.perf_counter()
        hits = run_pipeline(root)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        last_sub = hits_to_submission(hits)
        rps = total_trades / elapsed if elapsed > 0 else 0
        print(f"   run {i + 1}: {elapsed:.2f}s  (~{rps:,.0f} trade rows/s processed)")

    med = statistics.median(times)
    med_rps = total_trades / med if med > 0 else 0
    print(f"\n   MEDIAN: {med:.2f}s  (~{med_rps:,.0f} trade rows/s)")
    print("   (Compare to hackathon runtime bonus target, e.g. <60s.)\n")

    assert last_sub is not None
    print("3) Smoke checks on submission.csv-shaped output")
    errs = validate_submission(last_sub, root)
    if errs:
        print("   FAILED:")
        for e in errs:
            print(f"   - {e}")
    else:
        print("   OK: columns, dates, trade_id membership, peg_break spot-check")

    print(f"\n4) Output shape: {len(last_sub)} rows, columns {list(last_sub.columns)}")
    if len(last_sub):
        print("\n   Sample:")
        print(last_sub.head(3).to_string(index=False))

    if errs:
        raise SystemExit(1)
    print("\n=== Scenario complete (all checks passed) ===")


if __name__ == "__main__":
    main()
