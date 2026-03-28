#!/usr/bin/env python3
"""Run Problem 3 pipeline and write submission.csv."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from p3.pipeline import hits_to_submission, run_pipeline


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
        default=Path("submission.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    default_pack = repo / "data" / "student-pack"

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
            "  export STUDENT_PACK=/path/to/student-pack"
        )

    t0 = time.perf_counter()
    hits = run_pipeline(root)
    elapsed = time.perf_counter() - t0
    sub = hits_to_submission(hits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.output, index=False)
    print(f"Wrote {len(sub)} rows to {args.output} in {elapsed:.2f}s")
    if not sub.empty:
        print(sub.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
