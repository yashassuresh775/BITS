"""Symbols and default thresholds for Problem 3 detectors."""

SYMBOLS: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "LTCUSDT",
    "BATUSDT",
    "USDCUSDT",
)

# USDC stablecoin peg (problem statement)
PEG_CENTER = 1.0
PEG_BREAK_ABS = 0.005  # 0.5%

# BATUSDT / illiquid: hour volume vs daily median of hours
BAT_HOUR_VOLUME_MULT = 5.0

# BTC/ETH: trade notional vs median for same hour-of-day (UTC) in the loaded sample (vectorised).
MAJOR_PAIR_HOD_MULT = 7.0
MAJOR_PAIR_MIN_NOTIONAL_USDT = 2_500.0

# Wash / round-trip time window (seconds)
WASH_WINDOW_SEC = 180

# Structuring: many trades, similar notionals
STRUCT_MIN_TRADES = 8
STRUCT_CV_MAX = 0.12  # coefficient of variation of notional
STRUCT_MAX_MIN_RATIO = 1.12  # max(notional)/min(notional) for same wallet-day

# Taxonomy bonus: placement_smurfing (first-appearance cluster hour)
PLACEMENT_MIN_WALLETS = 7
PLACEMENT_CV_MAX = 0.14
PLACEMENT_MAX_MIN_RATIO = 1.12

# Taxonomy bonus: coordinated_structuring (multi-wallet same-hour smurfing band)
COORD_STRUCT_MIN_WALLETS = 4
COORD_STRUCT_MIN_TRADES_PER_W = 5

# Taxonomy bonus: manager_consolidation (one dominant leg vs rest of day for wallet)
MANAGER_CONSOL_RATIO = 10.0
MANAGER_MIN_DISTINCT_WALLETS_DAY = 8
MANAGER_MIN_LARGE_NOTIONAL = 12_000.0

# Ramping: consecutive same-side trades with monotone prices
RAMP_MIN_STREAK = 6

# Pump window (minutes) for bar-level heuristic
PUMP_LOOKBACK = 10
PUMP_MIN_RETURN = 0.002
DUMP_MIN_RETURN = -0.0015

# Isolation Forest
IF_CONTAMINATION = 0.03
IF_SYMBOLS = frozenset({"DOGEUSDT", "LTCUSDT", "SOLUSDT", "XRPUSDT"})

# Cap rows: problem has ~50 true violations; overshooting hurts -2 per FP.
MAX_SUBMISSION_ROWS = 220

# Round-trip: require very similar size and short time (reduce benign crosses)
ROUND_TRIP_MAX_SEC = 90
ROUND_TRIP_MIN_QTY_RATIO = 0.92

# --- ML layers ---
# IF + LOF rank fusion (detector name: ensemble_if_lof).
USE_ENSEMBLE_ANOMALY = True
ENSEMBLE_CONTAMINATION = 0.025
ENSEMBLE_SYMBOLS = frozenset(
    {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LTCUSDT", "BATUSDT"}
)

# Pseudo-label re-ranker (HistGradientBoostingClassifier).
USE_ML_RERANKER = True
TRUSTED_DETECTORS = frozenset({"peg_break", "wash_volume_at_peg"})
# When strict labels are missing (typical on **live** spot — USDC stays on peg), widen positives once.
ML_BROAD_TRUSTED_FALLBACK = True
ML_BROAD_TRUSTED_DETECTORS = frozenset(
    {
        "major_pair_hod_spike",
        "bat_hot_hour",
        "pump_dump_bars",
        "cross_btc_div",
    }
)
ML_NEG_MULTIPLIER = 4
ML_NEG_CAP_PER_SYMBOL = 500
ML_MIN_POSITIVES = 3
# Lower floor only for the broad set (still need ≥2 classes after sampling).
ML_MIN_POSITIVES_BROAD = 2
ML_RANDOM_STATE = 42
ML_MAX_ITER = 120
ML_MAX_DEPTH = 8
ML_MIN_SAMPLES_LEAF = 15
