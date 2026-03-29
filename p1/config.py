"""Thresholds and limits for Problem 1 (order-book concentration)."""

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
# Spread vs ~30 trading sessions when enough rows (390×30); shorter series uses all history
SPREAD_ROLL_LONG = 11700
# Cross-level depth shape: HHI z vs same long window (distribution across L1–L10 vs ticker norm)
HHI_Z_ALERT = 3.0

# DBSCAN (feature-space clustering; extra dims = OBI shock, HHI z — slightly wider eps)
DBSCAN_EPS = 0.72
DBSCAN_MIN_SAMPLES = 4

# Cap rows (FPs are expensive); raise on quiet data if needed
MAX_ALERTS = 120

# Noise cluster label from DBSCAN
NOISE_LABEL = -1
