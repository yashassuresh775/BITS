"""Thresholds and limits for Problem 1 (order-book concentration)."""

from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v.isdigit() else default


# Order book imbalance — sustained extreme
OBI_EXTREME = 0.65
OBI_HIGH = 0.82

# Level-1 share of total depth on one side (concentration)
CONC_HIGH = 0.42

# Minimum consecutive minutes to treat as a "sustained" episode (reduces one-off noise)
MIN_SUSTAINED_MINUTES = 3

# Z-score on spread vs rolling baseline (per ticker, in-sample history)
SPREAD_Z_ALERT = 2.8

# Rolling windows (rows = minutes; US regular session ≈ 390 min/day)
OBI_ROLL_SHORT = 10  # 10-minute rolling mean/std of OBI (spec suggested feature)
SPREAD_BASELINE_MIN = 120  # at least 2h of history before z-score is trusted
# Spread / HHI z baseline (~17 sessions @ 390 min). Use ``P1_SPREAD_ROLL_LONG=11700`` (~30 sessions)
# for a longer in-sample baseline on large histories (slower rolling).
SPREAD_ROLL_LONG = _env_int("P1_SPREAD_ROLL_LONG", 6500)
# Cross-level depth shape: HHI z vs same long window (distribution across L1–L10 vs ticker norm)
HHI_Z_ALERT = 3.0

# DBSCAN (feature-space clustering; extra dims = OBI shock, HHI z — slightly wider eps)
DBSCAN_EPS = 0.72
DBSCAN_MIN_SAMPLES = 4

# Cap rows (FPs are expensive); raise on quiet data if needed
MAX_ALERTS = 120

# Noise cluster label from DBSCAN
NOISE_LABEL = -1
