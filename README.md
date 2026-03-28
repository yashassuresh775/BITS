# BITS — Hackathon Problem 3 pipeline

Python pipeline for the **crypto blind anomaly hunt** (Problem 3): load 8 pairs of **1-minute Binance OHLCV** + **synthetic trades** (`trader_id` = wallet), run layered detectors, and write **`submission.csv`** (`symbol`, `date`, `trade_id`, `violation_type`, `remarks`).

## Approach (what we built)

1. **Rule layer (high precision)**  
   - **USDCUSDT `peg_break`**: `abs(price - 1.0) > 0.005`  
   - **`wash_volume_at_peg`**: rapid BUY/SELL flips at ~$1.00 on USDC (heuristic)

2. **Wallet / graph-style patterns**  
   - Same-wallet **wash** within a short window (flat net, matched prices)  
   - **Round-trip** between two wallets (tight time, price, and size ratio)  
   - **Ramping**, **layering_echo**, **AML structuring** (tight notional band), **threshold_testing** (~10k USDT), **coordinated_pump_minute**, **chain_layering** pass-through chain

3. **Market alignment**  
   - **Pump-and-dump** windows from bar paths  
   - **Spoofing** proxy (wide range + quick reversion)  
   - **cross_pair_divergence** vs **BTC** same-minute return z-score

4. **Isolation Forest** (recall) on `DOGE`, `LTC`, `SOL`, `XRP`: `qty_z`, `price_vs_mid_bps`, `wallet_freq`.

5. **Ensemble anomaly (IF + LOF)** — `p3/ml/ensemble_od.py`  
   - On symbols in **`ENSEMBLE_SYMBOLS`** (all pairs except **USDCUSDT**), fuses **IsolationForest** and **LocalOutlierFactor** scores into one **percentile rank**; flags the top **`ENSEMBLE_CONTAMINATION`** fraction.  
   - Detector name: **`ensemble_if_lof`**. Same corroboration rule as IF: dropped unless another detector hits that `trade_id`.

6. **Extra features for ML** — `p3/ml/extra_features.py`  
   - Minute crowding: **`trades_same_minute`**, **`wallets_same_minute`**.  
   - Same-wallet sequence: **`seq_same_side_prev`**, **`seq_price_chg`**.

7. **Pseudo-label re-ranker** — `p3/ml/ranker.py`  
   - **`HistGradientBoostingClassifier`** trained on **trusted** detectors (`TRUSTED_DETECTORS`: e.g. `peg_break`, `wash_volume_at_peg`) as positives and **sampled** normal trades as negatives.  
   - Scores **deduped** candidates, sorts by **`predict_proba` (positive class)**, then **`score`**, then **`head(MAX_SUBMISSION_ROWS)`**.  
   - Appends **`| ml_rank_p=…`** to **`remarks`** (pseudo-labels ≠ organizer labels — tune with care).

8. **Second pass**  
   - Drop uncorroborated **`isolation_forest`** / **`ensemble_if_lof`**.  
   - One row per **`trade_id`** (prefer non-empty **`violation_type`** when **`score`** ties).  
   - Final row cap applied **after** ML re-rank (or by score if **`USE_ML_RERANKER`** is **False**).

## Data layout (student-pack)

**Default location in this repo:** `data/student-pack/`  
Copy the official pack here so paths are:

- `data/student-pack/crypto-market/Binance_<SYMBOL>_2026_minute.csv`
- `data/student-pack/crypto-trades/<SYMBOL>_trades.csv`

Large CSVs under `data/student-pack/` are **gitignored** (only `.gitkeep` is tracked). See `data/README.txt`.

This matches the distributed pack even when filenames differ from `data_schema.md` placeholders.

## Setup

```bash
cd /path/to/BITS
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# If data lives in ./data/student-pack/ (recommended):
python run_p3.py -o submission.csv
# `submission.csv` is gitignored by default (regenerate locally; avoids committing outputs).

# Or set explicitly:
export STUDENT_PACK=/path/to/student-pack
python run_p3.py --data-root /path/to/student-pack -o submission.csv
```

On a typical laptop the full 8-pair run is **~25–35s** with ML layers enabled (extra LOF + boosting fit per run).

## Example testing scenario (efficiency + smoke checks)

This does **not** need labels: it measures **how fast** you process the real CSVs and checks that the output is **internally consistent** with those files.

```bash
cd /path/to/BITS
source .venv/bin/activate   # if you use the venv
python scripts/benchmark_p3.py --runs 3
# or: python scripts/benchmark_p3.py --data-root /path/to/student-pack --runs 3
```

What it does:

1. **Footprint** — Counts trade rows per symbol (same files as production).  
2. **Efficiency** — Runs the full pipeline several times, reports each run’s seconds and **trade rows/s**, and the **median** wall time (compare to your runtime bonus target, e.g. &lt;60s).  
3. **Smoke checks** — `symbol` / `date` / `trade_id` schema; every `trade_id` exists in the matching `*_trades.csv`; **`peg_break`** rows on USDCUSDT satisfy `abs(price - 1.0) > 0.005`.

Exit code **1** if any smoke check fails.

## Tuning before submit

- Edit **`p3/config.py`**: `MAX_SUBMISSION_ROWS`, `PEG_BREAK_ABS`, `ROUND_TRIP_*`, `STRUCT_*`, `IF_CONTAMINATION`, `IF_SYMBOLS`, **`USE_ENSEMBLE_ANOMALY`**, **`USE_ML_RERANKER`**, **`TRUSTED_DETECTORS`**, **`ML_*`**.  
- Re-enable **BATUSDT hot-hour** surfacing by importing `detect_bat_hot_hours` in `p3/pipeline.py` (removed by default — noisy without a confirmation rule).  
- Add **`remarks`**; graders use them for partial credit when `violation_type` is off.

## Layout

| Path | Purpose |
|------|---------|
| `p3/io.py` | Load market/trades; resolve `Volume <BASE>` column names |
| `p3/features.py` | Join trades to bars; rolling qty z-score; wallet frequency |
| `p3/detectors/rules.py` | Peg / USDC wash-at-peg |
| `p3/detectors/wallet_patterns.py` | Wallet-centric manipulation + AML heuristics |
| `p3/detectors/market_patterns.py` | Bar-level pump/dump, BTC divergence, spoof proxy |
| `p3/detectors/isolation.py` | IsolationForest recall layer |
| `p3/ml/extra_features.py` | Minute + wallet-sequence features |
| `p3/ml/ensemble_od.py` | IF + LOF rank fusion |
| `p3/ml/ranker.py` | HistGradientBoosting pseudo-label re-rank |
| `p3/pipeline.py` | Orchestration + corroboration + ML re-rank |
| `run_p3.py` | CLI |
| `scripts/benchmark_p3.py` | Example timed run + submission smoke tests |

## Disclaimer

Detectors are **heuristics** tuned for the hackathon schema — validate on the official held-out scoring and **trim** aggressively if FPs dominate.
