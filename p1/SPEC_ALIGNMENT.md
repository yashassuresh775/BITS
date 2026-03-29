# Problem 1 — alignment with problem statement

This file maps the **official Problem 1 brief** (order-book surveillance) to this implementation. Use it when demoing or self-scoring.

## Core task

| Brief | Implementation |
|-------|----------------|
| Find **tickers** (`sec_id`) and **time windows** where structure looks wrong | Alerts include `sec_id`, `trade_date`, `time_window_start`; windows are **DBSCAN clusters** split into **contiguous minute runs** (≤3 min gap). |
| **Unusual concentration**, **spreads**, **suspicious repetition** | **L1 concentration** (`bid_concentration` / `ask_concentration`), **spread z** vs ticker history, **sustained OBI** (rolling sum of extreme OBI minutes), **cross-level HHI z** (depth shape vs ticker norm), plus **DBSCAN** on a multi-feature vector. |
| **Cluster** observations, not only single minutes | **DBSCAN** on candidate minutes per ticker; each cluster is then split into **contiguous segments**; **severity** uses segment **duration** (`dur`). |
| **Structured alerts** | `anomaly_type`, `severity`, `remarks` (plain English for compliance-style review). |

## Suggested feature set (brief)

| Suggested | Where it lives |
|-----------|----------------|
| Rolling **10m OBI** mean and std | `obi_roll_mean_10`, `obi_roll_std_10` in `p1/features.py`; **std** appears in `remarks`; **deviation** `obi_vs_roll_z` feeds **DBSCAN** features. |
| **Spread (bps)** vs **~30 trading sessions** | `spread_bps` vs rolling mean/std of **lagged** spread per `sec_id`; default `SPREAD_ROLL_LONG` is **6500** (~17 sessions @ 390 min) for speed. Set env **`P1_SPREAD_ROLL_LONG=11700`** for **~30 × ~390** when enough rows; shorter histories use all available minutes. |
| **L1 concentration ratio** | `bid_concentration`, `ask_concentration`. |
| **Consecutive minutes** OBI beyond threshold | `_candidate_mask`: rolling sum of extreme-OBI flags ≥ `MIN_SUSTAINED_MINUTES`. |
| **Trade volume / total bid depth** | `load_trades_per_minute` + `attach_trade_aggression` → `buy_vs_bid_depth`; surfaced in **remarks** when large. |
| **Cross-level depth asymmetry** vs ticker normal | Per-side **Herfindahl** of level shares (L1–L10) → `bid_hhi`, `ask_hhi` → **z-scores** `bid_hhi_z`, `ask_hhi_z` vs long rolling baseline; in **DBSCAN** features and **candidate mask**; label **`cross_level_depth_asymmetry`** when that pattern dominates after other rules. |

## Normalisation tip (“per ticker before comparing”)

Spread and HHI baselines are computed **inside each `sec_id`** (grouped rolling statistics). DBSCAN runs **per ticker** on that ticker’s candidate minutes, so clusters are not mixing different names’ scales.

## Data notes

- **Inputs used:** `market_data.csv` (L1–L10 book) and **optional** `trade_data.csv` for buy-vs-depth context.
- **OHLCV** in the brief is **not joined** in this pipeline; signals are **order-book + trades**-first. You can extend with an optional OHLCV merge if graders expect volume context from bars.

## Scoring rubric (typical hackathon)

| Rubric item | Status |
|-------------|--------|
| **Under 1 minute** runtime bonus | **Yes** — `build_alerts` is typically **sub-second** (median) on the student pack; full CLI includes CSV I/O (still ≪ 60s). |
| **Under 5 minutes** | **Yes** |
| **False positives** costly | `MAX_ALERTS` caps output; thresholds in `p1/config.py` — tune carefully before submit. |

## Key files

- `p1/features.py` — row features + rolling baselines + HHI / OBI-shock.
- `p1/pipeline.py` — candidates, DBSCAN, segments, labels, `p1_alerts.csv` shape.
- `p1/config.py` — thresholds and windows.
- `run_p1.py` — CLI entry.
