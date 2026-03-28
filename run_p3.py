#!/usr/bin/env python3
"""Run Problem 3 pipeline and write submission.csv."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from p3.config import SYMBOLS
from p3.live import fetch_live_frames
from p3.pipeline import hits_to_submission, run_pipeline, run_pipeline_from_frames


def _sync_dashboard_sample(repo: Path, sub, *, enabled: bool) -> None:
    """Copy submission into dashboard/ so Streamlit Cloud can load it after commit (see dashboard fallback)."""
    if not enabled:
        return
    dest = repo / "dashboard" / "sample_submission.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(dest, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Problem 3 crypto anomaly hunt")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to student-pack folder (contains crypto-market/, crypto-trades/)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: submission.csv, or submission_offline.csv with --dual)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Fetch 1m bars + agg trades from Binance Spot (public REST), run pipeline, repeat.",
    )
    parser.add_argument(
        "--live-interval",
        type=float,
        default=30.0,
        metavar="SEC",
        help="Seconds between live refresh cycles (default: 30).",
    )
    parser.add_argument(
        "--live-once",
        action="store_true",
        help="With --live: single fetch + run then exit (no loop).",
    )
    parser.add_argument(
        "--live-klines",
        type=int,
        default=1000,
        metavar="N",
        help="1m klines per symbol (max 1000; wider recent window for live/dual).",
    )
    parser.add_argument(
        "--live-trades",
        type=int,
        default=1000,
        metavar="N",
        help="Aggregate trades per symbol (max 1000).",
    )
    parser.add_argument(
        "--dual",
        action="store_true",
        help="Run student-pack pipeline and one live Binance fetch in parallel; write two CSVs.",
    )
    parser.add_argument(
        "--output-live",
        type=Path,
        default=Path("submission_live.csv"),
        metavar="PATH",
        help="With --dual: output path for the live branch (default: submission_live.csv).",
    )
    parser.add_argument(
        "--no-dashboard-sample",
        action="store_true",
        help="Do not write dashboard/sample_submission.csv (default: sync after each successful CSV write).",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    sync_sample = not args.no_dashboard_sample
    default_pack = repo / "data" / "student-pack"

    if args.output is None:
        args.output = Path("submission_offline.csv") if args.dual else Path("submission.csv")

    if args.live and args.dual:
        raise SystemExit("Use either --live or --dual, not both.")

    if args.dual:
        root = args.data_root
        if root is None:
            env = os.environ.get("STUDENT_PACK", "").strip()
            if env:
                root = Path(env).expanduser()
        if root is None or not root.is_dir():
            root = default_pack
        root = root.resolve()
        if not root.is_dir():
            raise SystemExit(
                "--dual needs a student-pack directory (see --data-root / STUDENT_PACK / ./data/student-pack)."
            )

        symbols = list(SYMBOLS)

        def offline_job():
            t0 = time.perf_counter()
            hits = run_pipeline(root)
            sub = hits_to_submission(hits)
            elapsed = time.perf_counter() - t0
            return sub, elapsed

        def live_job():
            t0 = time.perf_counter()
            frames = fetch_live_frames(
                symbols,
                kline_limit=args.live_klines,
                trades_limit=args.live_trades,
            )
            hits = run_pipeline_from_frames(frames)
            sub = hits_to_submission(hits)
            elapsed = time.perf_counter() - t0
            return sub, elapsed

        results: dict[str, tuple[object, float]] = {}
        errors: dict[str, BaseException] = {}
        with ThreadPoolExecutor(max_workers=2) as ex:
            future_map = {ex.submit(offline_job): "offline", ex.submit(live_job): "live"}
            for fut in as_completed(future_map):
                name = future_map[fut]
                try:
                    sub, elapsed = fut.result()
                    results[name] = (sub, elapsed)
                except BaseException as e:
                    errors[name] = e

        args.output.parent.mkdir(parents=True, exist_ok=True)
        if "offline" in results:
            sub_off, elapsed = results["offline"]
            sub_off.to_csv(args.output, index=False)
            _sync_dashboard_sample(repo, sub_off, enabled=sync_sample)
            print(f"[dual/offline] Wrote {len(sub_off)} rows to {args.output} in {elapsed:.2f}s")
            if not sub_off.empty:
                print(sub_off.head(5).to_string(index=False))
        else:
            print(f"[dual/offline] failed: {errors.get('offline', 'unknown')}")

        args.output_live.parent.mkdir(parents=True, exist_ok=True)
        if "live" in results:
            sub_lv, elapsed = results["live"]
            sub_lv.to_csv(args.output_live, index=False)
            if "offline" not in results:
                _sync_dashboard_sample(repo, sub_lv, enabled=sync_sample)
            print(
                f"[dual/live] Wrote {len(sub_lv)} rows to {args.output_live} in {elapsed:.2f}s"
            )
            if not sub_lv.empty:
                print(sub_lv.head(5).to_string(index=False))
        else:
            print(f"[dual/live] failed: {errors.get('live', 'unknown')}")

        if "offline" not in results and "live" not in results:
            raise SystemExit(1)
        if errors:
            import sys

            for name, err in errors.items():
                print(f"[dual] branch {name!r} failed: {err}", file=sys.stderr)
        return

    if args.live:
        symbols = list(SYMBOLS)
        while True:
            t0 = time.perf_counter()
            try:
                frames = fetch_live_frames(
                    symbols,
                    kline_limit=args.live_klines,
                    trades_limit=args.live_trades,
                )
                hits = run_pipeline_from_frames(frames)
                sub = hits_to_submission(hits)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                sub.to_csv(args.output, index=False)
                _sync_dashboard_sample(repo, sub, enabled=sync_sample)
                elapsed = time.perf_counter() - t0
                print(
                    f"[live] Wrote {len(sub)} rows to {args.output} in {elapsed:.2f}s "
                    f"({len(symbols)} symbols)"
                )
                if not sub.empty:
                    print(sub.head(5).to_string(index=False))
            except (OSError, RuntimeError, ValueError) as e:
                print(f"[live] cycle failed: {e}")
            if args.live_once:
                break
            wait = max(0.0, args.live_interval - (time.perf_counter() - t0))
            time.sleep(wait)
        return

    root = args.data_root
    if root is None:
        env = os.environ.get("STUDENT_PACK", "").strip()
        if env:
            root = Path(env).expanduser()
    if root is None or not root.is_dir():
        root = default_pack
    root = root.resolve()
    if not root.is_dir():
        raise SystemExit(
            "Place student-pack under ./data/student-pack/, or use:\n"
            "  python run_p3.py --data-root /path/to/student-pack\n"
            "  export STUDENT_PACK=/path/to/student-pack\n"
            "  python run_p3.py --live   # Binance public API (no local CSVs)"
        )

    t0 = time.perf_counter()
    hits = run_pipeline(root)
    elapsed = time.perf_counter() - t0
    sub = hits_to_submission(hits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.output, index=False)
    _sync_dashboard_sample(repo, sub, enabled=sync_sample)
    print(f"Wrote {len(sub)} rows to {args.output} in {elapsed:.2f}s")
    if not sub.empty:
        print(sub.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
