# BITS — Trade surveillance hackathon toolkit

This repo includes **Problem 3** (crypto — main points) plus optional **Problem 1** (order-book alerts) and **Problem 2** (SEC EDGAR + insider-style signals). Below, **Approach** refers to the **Problem 3** pipeline: load 8 pairs of **1-minute OHLCV** + **synthetic trades** (`wallet_id`), run layered detectors, and write **`submission.csv`** (`symbol`, `date`, `trade_id`, `violation_type`, `remarks`).

## Problems 1 & 2 (equity bonus)

Place **`market_data.csv`**, **`ohlcv.csv`**, and **`trade_data.csv`** in one folder (e.g. `data/equity-data/`). A separate **`sec_id_map.csv`** is optional if **`ohlcv.csv`** already has **`ticker`** and **`sec_id`**.

```bash
python run_p1.py --data-root path/to/equity-data -o p1_alerts.csv
python run_p2.py --data-root path/to/equity-data -o p2_signals.csv --start-date 2026-01-01 --end-date 2026-03-31
```

**Problem 1 — speed / baseline:** Spread and cross-level HHI z-scores use a long rolling window (**default 6500** minutes ≈ **17** regular sessions at ~390 min/day). For a **~30-session** baseline when you have enough history, set **`P1_SPREAD_ROLL_LONG=11700`** before running P1 or the dashboard folder mode. The **`time_to_run`** column in **`p1_alerts.csv`** is seconds for **`build_alerts`** on that run (typically **sub-second** on the student equity pack with defaults). Installing **`threadpoolctl`** (optional) lets P1 cap BLAS/OpenMP threads during DBSCAN-heavy work.

**Regenerating CSVs in this repo** (after changing code or data; use **`.venv/bin/python`** if you use a venv):

```bash
cd /path/to/BITS && source .venv/bin/activate
python3 run_p1.py --data-root data/equity-data -o p1_alerts.csv
python3 run_p2.py --data-root data/equity-data -o p2_signals.csv --start-date 2026-01-01 --end-date 2026-03-31
python3 run_p3.py --data-root data/student-pack -o submission.csv
```

Optional copies for demos: **`submission_offline.csv`** (same pack, no dashboard sample sync: add **`--no-dashboard-sample`**), **`submission_from_binance.csv`** with **`--data-root data/binance-hist`**, **`submission_live.csv`** with **`run_p3.py --live --live-once -o submission_live.csv`**. Row counts for **P2** can shift slightly when EDGAR returns change.

**Streamlit:** Run **`python3 -m streamlit run dashboard/app.py`** with the same environment as **`requirements.txt`** so **`p2.insider_signals`** loads the pandas pipeline reliably.

Problem 2 calls the SEC EDGAR API (`requests` in **`requirements.txt`**). Use `make p1 P1_ROOT=...` / `make p2 P2_ROOT=...` if your Makefile defines those targets.

**Run P3 + benchmark + P1 + P2 in one go (correct shell — no `--` after `make`):**

```bash
cd /path/to/BITS
source .venv/bin/activate

# P3 — needs ./data/student-pack/ (crypto CSVs); writes submission.csv (gitignored)
make run
make benchmark

# P1 + P2 — same directory must contain market_data.csv, ohlcv.csv, trade_data.csv
EQUITY_ROOT="$HOME/Downloads/student-pack/equity"   # change if your pack moved
make p1 P1_ROOT="$EQUITY_ROOT"
make p2 P2_ROOT="$EQUITY_ROOT"
```

Or a single Make target (same **`EQUITY_ROOT`** for P1 and P2):

```bash
make run-all EQUITY_ROOT="$HOME/Downloads/student-pack/equity"
```

**Common mistake:** do not append text like `-- comment` on the same line as `make p2 ...` — the shell may treat `--` as arguments to **`make`**, not a comment. Use a new line and `# comment` instead.

**Cursor / sandbox:** if `make run` fails with `PermissionError` on `submission.csv`, run the same commands in your **local terminal** (outside a read-only agent sandbox).

**P2 offline:** cache filings once, then `python3 run_p2.py --data-root "$EQUITY_ROOT" --skip-edgar --filings-cache path/to/filings.csv -o p2_signals.csv`.

**P2 EDGAR-only (scraper first):** `python3 scripts/fetch_p2_filings_only.py --data-root path/to/equity -o p2_filings_cache.csv` then `run_p2.py --skip-edgar --filings-cache p2_filings_cache.csv …`.

**P2 organizer tips (implemented):** **`--edgar-concurrency 3`** (default) runs tickers in small parallel bursts with **`--edgar-sleep 0.3`** between bursts (use **`1`** for strict sequential starter behavior). Dates are forced to `YYYY-MM-DD` strings; optional **`edgar_query`** on the ticker map; **`--ma-only`**; merger rows sorted first; **`trade_data` `trader_id`** hints in **remarks**. Starter doc: `data/student-pack/docs/edgar_starter_snippet.md`.

---

## Problem 3 — Approach (what we built)

1. **Rule layer (high precision)**  
   - **USDCUSDT `peg_break`**: `abs(price - 1.0) > 0.005`  
   - **`wash_volume_at_peg`**: rapid BUY/SELL flips at ~$1.00 on USDC (heuristic)  
   - **`bat_hot_hour`**: BATUSDT trades in hours where bar **USDT volume ≥ 5×** that day’s median hourly volume  
   - **`hod_notional_spike`**: **BTC/ETH** trades with **notional ≥ 7×** the sample’s median notional for the same **UTC hour-of-day** (vectorised)

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
   - If strict peg labels are **too few** (common on **live** spot), a **broad fallback** (`ML_BROAD_TRUSTED_FALLBACK`) adds positives from `ML_BROAD_TRUSTED_DETECTORS` (e.g. HOD spike, BAT hot hour, pump/dump bars, BTC divergence) so **`ml_rank_p`** still trains — disable for student-pack-only parity.  
   - Scores **deduped** candidates, sorts by **`predict_proba` (positive class)**, then **`score`**, then **`head(MAX_SUBMISSION_ROWS)`**.  
   - Appends **`| ml_rank_p=…`** to **`remarks`** (pseudo-labels ≠ organizer labels — tune with care).

8. **Second pass**  
   - Drop uncorroborated **`isolation_forest`** / **`ensemble_if_lof`**.  
   - One row per **`trade_id`** (prefer non-empty **`violation_type`** when **`score`** ties).  
   - Final row cap applied **after** ML re-rank (or by score if **`USE_ML_RERANKER`** is **False**).

9. **BATUSDT hot hours** — bar volume by hour vs that day’s median hourly volume; flag trades in hours **≥5×** median (`bat_hot_hour`).  
10. **BTC / ETH hour-of-day baselines** — vectorised median **notional per UTC hour** in the loaded sample; flag large trades **≥7×** that hour’s median with a minimum notional floor (`hod_notional_spike` / `major_pair_hod_spike`).

### Hackathon workflow alignment (checklist)

| Suggested step | In this repo |
|----------------|--------------|
| 1. Load 8 pairs + basic price/qty stats | `python3 scripts/eda_pack_stats.py` or `make eda-stats` — full **mental-model report** (USDC peg audit, BAT hot hours, BTC/ETH UTC-hour notionals, span/trades-per-day). `--brief` = one table only; `--out reports/eda.txt` saves a copy. |
| 2. USDC peg `abs(price-1)>0.005` | `detect_peg_break` in `p3/detectors/rules.py` |
| 3. BATUSDT hourly volume vs daily median | `detect_bat_hot_hours` (wired in `p3/pipeline.py`) |
| 4. IF on DOGE/LTC/SOL/XRP | `p3/detectors/isolation.py` + `IF_SYMBOLS` |
| 5. BTC/ETH tighter baselines | `detect_major_pair_hod_spike` + ensemble/ wallet stack on those symbols |
| 6. Build `submission.csv` incrementally | Regenerate often with `run_p3.py -o submission.csv`; cap **`MAX_SUBMISSION_ROWS`**. |
| Submission at repo root | Default `-o submission.csv`; file gitignored. |
| README approach | This file + remarks on each row for partial credit. |
| Fast runtime | Prefer **groupby/transform**, rolling windows, matrix ops in `features` / detectors; avoid Python loops on rows where possible. |

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
python -m pip install -r requirements.txt
```

**macOS note:** If the shell says `command not found: python`, use **`python3`** instead, or run **`source .venv/bin/activate`** first (the venv adds a `python` command). From the repo root you can also run **`make run`**, **`make dual`**, or **`make dashboard`** (the Makefile prefers **`.venv/bin/python`** when it exists).

## Run

```bash
# If data lives in ./data/student-pack/ (recommended):
python3 run_p3.py -o submission.csv
# `submission.csv` is gitignored by default (regenerate locally; avoids committing outputs).

# Or set explicitly:
export STUDENT_PACK=/path/to/student-pack
python3 run_p3.py --data-root /path/to/student-pack -o submission.csv
```

### Live (OKX public REST by default, near–real-time)

Pulls **1m klines** + **aggregate trades** for all **`SYMBOLS`**, runs the same detectors, and rewrites **`submission.csv`** on a timer (pair with the Streamlit dashboard **Live reload**).

```bash
# Loop every 30s (Ctrl+C to stop):
python3 run_p3.py --live -o submission.csv

# Single pull + run (cron-friendly):
python3 run_p3.py --live --live-once -o submission.csv

# Tune polling / window size:
python3 run_p3.py --live --live-interval 15 --live-klines 600 --live-trades 1000 -o submission.csv
```

**Live venue (`LIVE_SPOT_VENUE`):**

- **`okx`** (default) — OKX public v5 REST only.
- **`binance`** — one Binance Spot base: **`BINANCE_SPOT_API`** if set, else **`https://api.binance.us/api/v3`** (fails fast on 451; live path does not mirror-hop).
- **`both`** (aliases `okx+binance`, `binance+okx`) — **OKX and Binance in parallel** per symbol: 1m bars are **merged** (wider high/low, summed volumes); trades are **concatenated** with **`okx:`** / **`bn:`** id prefixes. Slower than one venue but richer tape. If one exchange errors, the other is still used when it returned data.

```bash
# Default: OKX
python3 run_p3.py --live --live-once -o submission.csv

# Binance only (single host)
export LIVE_SPOT_VENUE=binance
export BINANCE_SPOT_API=https://api.binance.us/api/v3
python3 run_p3.py --live --live-once -o submission.csv

# OKX + Binance merged (set BINANCE_SPOT_API if .us is wrong for your network)
export LIVE_SPOT_VENUE=both
python3 run_p3.py --live --live-once -o submission.csv
```

**TLS:** the client uses **`certifi`** and respects **`SSL_CERT_FILE`** / **`REQUESTS_CA_BUNDLE`** for corporate proxies. Only if you know the risk: **`BINANCE_INSECURE_SSL=1`** disables certificate verification (debug / broken CA stores only).

**Caveat:** public trades have **no real wallet IDs**; we assign **synthetic** `wallet_id` buckets. Wallet-graph heuristics are weaker than on the official synthetic pack; bar/price/peg/IF-style signals still run.

On a typical laptop the full 8-pair run is **~25–35s** with ML layers enabled (extra LOF + boosting fit per run). A **live** cycle is usually **~15–25s** (network + same detectors on the last **N** minutes of data).

### Historical Binance backfill (not only the live window)

``--live`` / ``--dual`` only pull the **latest** bars and trades (up to **1000** 1m candles and **1000** agg trades per symbol per cycle). To populate **days** of history into the same folder layout as the student pack, run:

```bash
# Optional: export BINANCE_SPOT_API=... to pin a host (auto .com→.us fallback applies if unset)

python3 scripts/fetch_binance_history.py --days 7 --data-root data/binance-hist
# or: make fetch-hist ARGS='--days 7'

python3 run_p3.py --data-root data/binance-hist -o submission_from_binance.csv
```

Use ``--max-trades`` (cap per symbol; default 250000) and ``--pause`` if you hit rate limits. Then in the dashboard, set **Primary** to ``submission_from_binance.csv`` for the long-history run, and optionally **Second** to ``submission_live.csv`` for the short live snapshot.

### Both at once (student pack + Binance in parallel)

One command runs the **offline** pipeline on your CSVs and the **live** fetch+pipeline in **two threads**, then writes **two files** (wall time ≈ max of the two branches, not the sum):

```bash
python3 run_p3.py --dual
# → submission_offline.csv (default) + submission_live.csv

python3 run_p3.py --dual -o my_pack.csv --output-live my_live.csv
```

In the dashboard, set **Primary** to `submission_offline.csv` (or your `-o` path) and **Second CSV** to `submission_live.csv` — you get **two tabs** and auto-refresh can update **both** files each cycle.

To refresh both on a schedule, wrap in a shell loop or use two terminals (`--dual` periodically vs. `--live` + offline on different intervals).

## Example testing scenario (efficiency + smoke checks)

This does **not** need labels: it measures **how fast** you process the real CSVs and checks that the output is **internally consistent** with those files.

```bash
cd /path/to/BITS
source .venv/bin/activate   # if you use the venv
python3 scripts/benchmark_p3.py --runs 3
# or: python3 scripts/benchmark_p3.py --data-root /path/to/student-pack --runs 3
```

What it does:

1. **Footprint** — Counts trade rows per symbol (same files as production).  
2. **Efficiency** — Runs the full pipeline several times, reports each run’s seconds and **trade rows/s**, and the **median** wall time (compare to your runtime bonus target, e.g. &lt;60s).  
3. **Smoke checks** — `symbol` / `date` / `trade_id` schema; every `trade_id` exists in the matching `*_trades.csv`; **`peg_break`** rows on USDCUSDT satisfy `abs(price - 1.0) > 0.005`.

Exit code **1** if any smoke check fails.

## Dashboard (UI) — run locally

One Streamlit app with **three tabs**: **P1** (order-book alerts), **P2** (EDGAR + signals), **P3** (crypto submission). Each tab supports **path / upload / optional Secrets URL**, filters, charts, and optional **auto-refresh** (same overall pattern as the former P3-only UI).

**Tour of the UI (every control + datasets + how to read results):** see **`dashboard/DASHBOARD_GUIDE.md`**.

From the repo root:

```bash
source .venv/bin/activate
pip install -r requirements.txt   # includes streamlit + streamlit-autorefresh
python3 -m streamlit run dashboard/app.py
# or: make dashboard
```

Streamlit opens **`http://127.0.0.1:8501`** (see **`.streamlit/config.toml`**). Generate outputs first, then refresh:

```bash
python3 run_p3.py -o submission.csv
make p1 P1_ROOT=/path/to/equity && make p2 P2_ROOT=/path/to/equity   # optional
```

**P3 tab:** default **Static CSV** → **`submission.csv`**; fallback **`dashboard/sample_submission.csv`**. **Live (API)** = **`run_p3.py --live`**; **`LIVE_SPOT_VENUE`**: **`okx`**, **`binance`**, or **`both`**. Controls live in the **P3 — Data & refresh** expander (not a separate sidebar). Live cache **~90s**; **auto-refresh** defaults **on** for static and **off** for live.

**P1 tab:** static **`p1_alerts.csv`**, **run from folder** (`market_data.csv`, optional `trade_data.csv`), or **Live (poll CSV URLs)** (HTTPS raw CSVs). Secrets **`P1_ALERTS_URL`**, optional **`P1_LIVE_MARKET_URL`** / **`P1_LIVE_TRADE_URL`**. Default equity folder follows **`EQUITY_ROOT`** (or **`EQUITY_DATA_STREAMLIT_DEFAULT`**) if set.

**P2 tab:** static **`p2_signals.csv`** or **run pipeline** (`ohlcv.csv` + `trade_data.csv`). **EDGAR burst size** defaults to **3** parallel tickers (~under **60s** for ~28 names + signal build on a typical connection); set to **1** if you want sequential requests only. With live EDGAR, set a **re-fetch interval** (time-bucketed cache); with **skip EDGAR**, cache tracks local file mtimes. Secret **`P2_SIGNALS_URL`**. Same **`EQUITY_ROOT`** default as P1.

### Streamlit Community Cloud (optional)

**Main file path:** **`dashboard/app.py`**. **Requirements:** root **`requirements.txt`**. Use Secrets for **`PRIMARY_SUBMISSION_URL`**, **`P1_ALERTS_URL`**, **`P2_SIGNALS_URL`**, **`P1_LIVE_*`**, **`LIVE_SPOT_VENUE`**, etc. (see **`.streamlit/secrets.toml.example`**). P2 **Run pipeline** calls the SEC from the cloud host (burst concurrency + sleep still apply). For demos without equity files in the repo, use **P2 → Static CSV** + **`P2_SIGNALS_URL`** or upload **`p2_signals.csv`**.

## Tuning before submit

- Edit **`p3/config.py`**: `MAX_SUBMISSION_ROWS`, `PEG_BREAK_ABS`, **`BAT_HOUR_VOLUME_MULT`**, **`MAJOR_PAIR_HOD_MULT`**, **`MAJOR_PAIR_MIN_NOTIONAL_USDT`**, `ROUND_TRIP_*`, `STRUCT_*`, `IF_CONTAMINATION`, `IF_SYMBOLS`, **`USE_ENSEMBLE_ANOMALY`**, **`USE_ML_RERANKER`**, **`TRUSTED_DETECTORS`**, **`ML_BROAD_TRUSTED_*`**, **`ML_*`**.  
- **BATUSDT hot-hour** and **BTC/ETH HOD spikes** are on by default; raise thresholds if they add too many rows before the ML cap.  
- Add **`remarks`**; graders use them for partial credit when `violation_type` is off.

## Layout

| Path | Purpose |
|------|---------|
| `p3/io.py` | Load market/trades; resolve `Volume <BASE>` column names |
| `p3/features.py` | Join trades to bars; rolling qty z-score; wallet frequency |
| `p3/detectors/rules.py` | Peg, USDC wash-at-peg, BATUSDT hot hours, BTC/ETH HOD notional spike |
| `p3/detectors/wallet_patterns.py` | Wallet-centric manipulation + AML heuristics |
| `p3/detectors/market_patterns.py` | Bar-level pump/dump, BTC divergence, spoof proxy |
| `p3/detectors/isolation.py` | IsolationForest recall layer |
| `p3/ml/extra_features.py` | Minute + wallet-sequence features |
| `p3/ml/ensemble_od.py` | IF + LOF rank fusion |
| `p3/ml/ranker.py` | HistGradientBoosting pseudo-label re-rank |
| `p3/pipeline.py` | Orchestration + corroboration + ML re-rank |
| `p3/live/okx.py` | Public OKX v5 REST → same bar/trade shape as live pipeline |
| `p3/live/binance.py` | Live fetch (OKX or single-host Binance); historical multi-host GET for backfill |
| `run_p3.py` | CLI |
| `Makefile` | `make run` / `make benchmark` / `make p1` / `make p2` / `make run-all EQUITY_ROOT=...` / `make dual` / `make dashboard` / `make fetch-hist` (uses `.venv/bin/python` if present) |
| `run_p1.py` / `run_p2.py` | Equity Problem 1 & 2 CLIs → `p1_alerts.csv`, `p2_signals.csv` |
| `scripts/fetch_binance_history.py` | Paginated historical klines + agg trades → `data/binance-hist/` |
| `scripts/eda_pack_stats.py` | Vectorised EDA report: notionals, peg, BAT hours, major-pair HOD (`make eda-stats`) |
| `scripts/benchmark_p3.py` | Example timed run + submission smoke tests |
| `dashboard/app.py` | Streamlit **P1 \| P2 \| P3** tabs (`tab_p1.py`, `tab_p2.py` + P3 explorer) |
| `.streamlit/secrets.toml.example` | Template for `PRIMARY_SUBMISSION_URL` / `SECOND_SUBMISSION_URL` on Streamlit Cloud |

## Disclaimer

Detectors are **heuristics** tuned for the hackathon schema — validate on the official held-out scoring and **trim** aggressively if FPs dominate.
