#!/usr/bin/env python3
"""
Exploratory report for the 8-pair pack — **fast to run**, **rich enough to replace**
a manual ~10-minute slice-and-dice session. All heavy work is vectorised (groupby,
quantiles, transforms — no per-row Python loops).

Sections:
  A) Per-symbol trade table (price/qty/notional + wallets + side split)
  B) USDCUSDT peg audit (off-peg count vs PEG_BREAK_ABS, extremes)
  C) BATUSDT hourly bar volume vs that day’s median hour (hot-hour preview)
  D) BTC/ETH notional profile by UTC hour-of-day (validates major-pair detector logic)
  E) Calendar span + trades per day (density)

Usage:
  python3 scripts/eda_pack_stats.py
  python3 scripts/eda_pack_stats.py --data-root /path/to/student-pack
  python3 scripts/eda_pack_stats.py --out reports/eda_report.txt
  python3 scripts/eda_pack_stats.py --brief    # legacy one-table only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p3.config import BAT_HOUR_VOLUME_MULT, MAJOR_PAIR_HOD_MULT, PEG_BREAK_ABS, PEG_CENTER
from p3.io import discover_symbols, load_market, load_trades


def default_root() -> Path:
    return ROOT / "data" / "student-pack"


def _fmt_num(x: float) -> str:
    if abs(x) >= 1e6 or (abs(x) > 0 and abs(x) < 1e-4):
        return f"{x:,.6g}"
    return f"{x:,.4f}".rstrip("0").rstrip(".")


def load_all_trades(root: Path, symbols: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            out[sym] = load_trades(root, sym)
        except FileNotFoundError:
            print(f"{sym}: MISSING trades file", file=sys.stderr)
    return out


def print_core_table(cache: dict[str, pd.DataFrame], f: TextIO = sys.stdout) -> pd.DataFrame:
    rows: list[dict] = []
    for sym, df in cache.items():
        n = len(df)
        buy_m = df["side"].str.upper().eq("BUY")
        n_buy = int(buy_m.sum())
        notion = df["notional"].astype(float)
        rows.append(
            {
                "symbol": sym,
                "n_trades": n,
                "n_wallets": int(df["wallet_id"].nunique()),
                "buy_pct": 100.0 * n_buy / n if n else 0.0,
                "price_mean": float(df["price"].mean()),
                "price_std": float(df["price"].std(ddof=0)),
                "price_p01": float(df["price"].quantile(0.01)),
                "price_p99": float(df["price"].quantile(0.99)),
                "qty_mean": float(df["quantity"].mean()),
                "qty_p99": float(df["quantity"].quantile(0.99)),
                "notional_mean": float(notion.mean()),
                "notional_p50": float(notion.quantile(0.5)),
                "notional_p90": float(notion.quantile(0.9)),
                "notional_p99": float(notion.quantile(0.99)),
            }
        )
    out = pd.DataFrame(rows).set_index("symbol")
    print("\n=== A) Per-symbol trades (vectorised) ===\n", file=f)
    print(
        out.to_string(
            float_format=_fmt_num,
        ),
        file=f,
    )
    print(f"\nTotal trades: {sum(len(v) for v in cache.values()):,}", file=f)
    return out


def print_usdc_peg(df: pd.DataFrame, f: TextIO = sys.stdout) -> None:
    print("\n=== B) USDCUSDT peg (mental model for peg_break) ===\n", file=f)
    dev = (df["price"].astype(float) - PEG_CENTER).abs()
    off = dev > PEG_BREAK_ABS
    n_off = int(off.sum())
    n = len(df)
    print(f"  Trades: {n:,}", file=f)
    print(
        f"  |price − {PEG_CENTER}| > {PEG_BREAK_ABS}  →  {n_off:,} rows ({100 * n_off / n:.2f}%)",
        file=f,
    )
    print(f"  max |Δ|: {float(dev.max()):.6f}  min price: {float(df['price'].min()):.6f}  max: {float(df['price'].max()):.6f}", file=f)
    # bucket shares (vectorised masks)
    for label, cap in [("≤ 0.1 bp", 1e-5), ("≤ 1 bp", 1e-4), ("≤ 10 bp", 1e-3)]:
        m = dev <= cap
        print(f"  within {label}: {int(m.sum()):,} ({100 * m.mean():.1f}%)", file=f)


def print_bat_hot_preview(market: pd.DataFrame, trades: pd.DataFrame, f: TextIO = sys.stdout) -> None:
    print("\n=== C) BATUSDT hourly bar volume vs daily median hour ===\n", file=f)
    m = market.copy()
    m["day"] = m["Date"].dt.normalize()
    m["hour_bucket"] = m["Date"].dt.floor("h")
    hv = m.groupby(["day", "hour_bucket"], as_index=False)["vol_usdt"].sum()
    med = hv.groupby("day")["vol_usdt"].transform("median").replace(0, np.nan)
    hv["ratio"] = hv["vol_usdt"] / med
    hot = hv[hv["ratio"] >= BAT_HOUR_VOLUME_MULT]
    print(
        f"  Hour-buckets in sample: {len(hv):,}  |  hot (≥{BAT_HOUR_VOLUME_MULT:g}× day median): {len(hot):,}",
        file=f,
    )
    if not hv.empty:
        print(f"  max ratio: {float(hv['ratio'].max()):.2f}x", file=f)
        top = hv.nlargest(5, "ratio")[
            ["day", "hour_bucket", "vol_usdt", "ratio"]
        ]
        print("  top 5 hours by ratio:", file=f)
        print(top.to_string(index=False, float_format=_fmt_num), file=f)
    tt = trades.copy()
    tt["day"] = tt["timestamp"].dt.normalize()
    tt["hour_bucket"] = tt["timestamp"].dt.floor("h")
    if not hot.empty:
        merged = tt.merge(hot[["day", "hour_bucket"]], on=["day", "hour_bucket"], how="inner")
        print(f"  trades landing in those hot hours: {len(merged):,}", file=f)


def print_major_pair_hod(trades: pd.DataFrame, symbol: str, f: TextIO = sys.stdout) -> None:
    t = trades
    hod = t["timestamp"].dt.hour
    med = t.groupby(hod, observed=True)["notional"].transform("median")
    med = med.replace(0, np.nan)
    ratio = t["notional"].astype(float) / med
    t2 = t.assign(_h=hod, _r=ratio)
    summ = (
        t2.groupby("_h", observed=True)
        .agg(
            n=("trade_id", "count"),
            med_n=("notional", "median"),
            p90_n=("notional", lambda s: s.quantile(0.9)),
            max_r=("_r", "max"),
        )
        .reset_index()
        .rename(columns={"_h": "utc_hour"})
    )
    print(f"\n=== D) {symbol} — notional by UTC hour (hod spike detector context) ===\n", file=f)
    print(
        f"  Threshold in pipeline: ≥{MAJOR_PAIR_HOD_MULT:g}× hour median & ≥ min notional.\n",
        file=f,
    )
    # compact: show only hours with highest activity or max ratio
    summ = summ.sort_values("utc_hour")
    with pd.option_context("display.max_rows", 30):
        print(summ.to_string(index=False, float_format=_fmt_num), file=f)
    worst_h = int(summ.loc[summ["max_r"].idxmax(), "utc_hour"])
    print(
        f"\n  Hour with largest max ratio vs hod-median: {worst_h} (max ratio {_fmt_num(float(summ['max_r'].max()))})",
        file=f,
    )


def print_temporal_coverage(cache: dict[str, pd.DataFrame], f: TextIO = sys.stdout) -> None:
    print("\n=== E) Time span & trades/day ===\n", file=f)
    rows = []
    for sym, df in cache.items():
        ts = df["timestamp"]
        lo, hi = ts.min(), ts.max()
        days = (hi.normalize() - lo.normalize()).days + 1
        by_d = df.groupby(df["timestamp"].dt.normalize(), observed=True).size()
        rows.append(
            {
                "symbol": sym,
                "first_utc": str(lo)[:19],
                "last_utc": str(hi)[:19],
                "span_days": days,
                "trades/day_mean": float(by_d.mean()) if len(by_d) else 0,
                "trades/day_max": int(by_d.max()) if len(by_d) else 0,
            }
        )
    print(pd.DataFrame(rows).to_string(index=False), file=f)


def run_report(root: Path, f: TextIO) -> None:
    symbols = discover_symbols(root)
    print(f"Data root: {root}", file=f)
    print(f"Symbols: {', '.join(symbols)}", file=f)

    cache = load_all_trades(root, symbols)
    if not cache:
        raise SystemExit("No trade files loaded.")

    print_core_table(cache, f)

    if "USDCUSDT" in cache:
        print_usdc_peg(cache["USDCUSDT"], f)

    if "BATUSDT" in cache:
        try:
            mbat = load_market(root, "BATUSDT")
            print_bat_hot_preview(mbat, cache["BATUSDT"], f)
        except FileNotFoundError as e:
            print(f"\n(C) Skip BATUSDT bars: {e}\n", file=f)

    for sym in ("BTCUSDT", "ETHUSDT"):
        if sym in cache:
            print_major_pair_hod(cache[sym], sym, f)

    print_temporal_coverage(cache, f)

    print(
        "\n---\nTip: tune `PEG_BREAK_ABS`, `BAT_HOUR_VOLUME_MULT`, `MAJOR_PAIR_HOD_*` in `p3/config.py` from this picture.\n",
        file=f,
    )


def run_brief(root: Path, f: TextIO) -> None:
    symbols = discover_symbols(root)
    print(f"Data root: {root}\nSymbols: {', '.join(symbols)}\n", file=f)
    cache = load_all_trades(root, symbols)
    if not cache:
        raise SystemExit("No trade files loaded.")
    rows = []
    for sym, df in cache.items():
        rows.append(
            {
                "symbol": sym,
                "n_trades": len(df),
                "price_mean": float(df["price"].mean()),
                "price_std": float(df["price"].std(ddof=0)),
                "price_min": float(df["price"].min()),
                "price_max": float(df["price"].max()),
                "qty_mean": float(df["quantity"].mean()),
                "qty_std": float(df["quantity"].std(ddof=0)),
                "qty_min": float(df["quantity"].min()),
                "qty_max": float(df["quantity"].max()),
            }
        )
    out = pd.DataFrame(rows).set_index("symbol")
    print(out.to_string(float_format=_fmt_num), file=f)
    print(f"\nTotal trades: {sum(len(v) for v in cache.values()):,}", file=f)


def main() -> None:
    p = argparse.ArgumentParser(description="EDA mental-model report (vectorised)")
    p.add_argument("--data-root", type=Path, default=None, help="Student-pack root")
    p.add_argument(
        "--brief",
        action="store_true",
        help="Only print the legacy compact price/qty table",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write full report to this file (UTF-8); still prints to stdout if omitted",
    )
    args = p.parse_args()
    root = (args.data_root or default_root()).resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        import io

        bio = io.StringIO()
        if args.brief:
            run_brief(root, bio)
        else:
            run_report(root, bio)
        text = bio.getvalue()
        args.out.write_text(text, encoding="utf-8")
        print(text, end="")
    elif args.brief:
        run_brief(root, sys.stdout)
    else:
        run_report(root, sys.stdout)


if __name__ == "__main__":
    main()
