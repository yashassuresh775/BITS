# BITS surveillance dashboard — what you are looking at

This guide labels every major part of the Streamlit app (`dashboard/app.py`), explains the **datasets** behind each problem, and describes **how to read the results** in plain language so you can demo and evaluate the work yourself.

---

## 1. Overall layout

When you open **`http://127.0.0.1:8501`** (after `streamlit run dashboard/app.py`):

| What you see | What it is |
|--------------|------------|
| **Page title** | “BITS — Trade surveillance” — one app for three hackathon-style problems. |
| **Caption under the title** | Reminder that **P1** = equities order book, **P2** = EDGAR / filings, **P3** = crypto submission. |
| **Three tabs** | **P1 — Order book**, **P2 — EDGAR**, **P3 — Crypto** — each tab is independent (different data, different outputs). |

You only use **one tab at a time** for a coherent story: “equity microstructure alerts,” “equity + SEC filings,” or “crypto violations submission.”

---

## 2. Problem 1 (P1) tab — order-book alerts

### 2.1 What problem P1 solves (simple terms)

**Goal:** Flag **suspicious patterns in the limit order book** (Level 1–10 bid/ask sizes and prices) per stock (`sec_id`), minute by minute. Examples: too much size stacked at the **best bid** or **best ask**, sustained imbalance, or unusual spread vs recent history. Optional **client trades** refine the story (buy vs sell pressure vs depth).

**Output:** A table of **alerts** (rows), similar to `p1_alerts.csv` from `run_p1.py`.

### 2.2 Datasets P1 uses

| File / source | Role |
|---------------|------|
| **`market_data.csv`** | Required. Minute-level (or similar) **order book snapshots**: `sec_id`, timestamp, bid/ask prices and sizes per level. |
| **`trade_data.csv`** | Optional. Client trades aligned in time; used to enrich remarks (e.g. aggression vs book). |
| **`p1_alerts.csv`** | Precomputed alerts — **Static CSV** mode just displays this file. |
| **HTTPS URLs** | **Live (poll CSV URLs)** downloads the same shaped CSVs on a schedule (your hosted copy). |

**Local folder:** Default is `data/equity-data/` unless you set `EQUITY_ROOT` (or related env vars). Large CSVs are usually **not** in git; you copy the student-pack **equity** folder locally.

### 2.3 Labeled UI components (P1 tab, top to bottom)

| Component | Purpose |
|-----------|---------|
| **Subheader** “P1 — Order-book concentration & DBSCAN alerts” | Names the tab. |
| **Caption** under subheader | Short map: Static vs folder vs live URLs + env vars. |
| **Expander: “P1 — Data source & refresh”** | All controls for *where data comes from* and *how often the page reloads*. |
| **Source** (radio) | **Static CSV** = show `p1_alerts.csv`. **Run from equity folder** = run pipeline on `market_data.csv` (+ optional trades). **Live (poll CSV URLs)** = fetch CSVs from URLs on an interval. |
| **Auto-refresh page** | Periodically reruns the app so folder / URL / file updates appear without manual refresh. |
| **Refresh every (s)** | Interval for auto-refresh (only matters if auto-refresh is on). |
| **Upload P1 CSV** | Optional file upload; overrides path/URL for static mode when used. |
| **Path to p1_alerts.csv** | File path for **Static CSV** mode. |
| **Load from** (if `P1_ALERTS_URL` secret exists) | Choose local path vs hosted URL from Streamlit secrets. |
| **Equity data folder** | Directory that must contain **`market_data.csv`** for **Run from equity folder**. |
| **Ignore trade_data.csv** | Run P1 using only the book (no trade enrichment). |
| **Run / refresh pipeline** | Forces a new pipeline run even if the market file mtime unchanged (bypasses part of the cache). |
| **Poll interval / Market URL / Trade URL** | **Live URLs** mode: how often to refetch and where. |
| **Clear P1 live cache** | Drops cached HTTP results for live mode. |
| **Green success line** | Row count + short **source** tag (path, folder, or live tick). |
| **Timing caption** | **Pipeline compute** matches the **`time_to_run`** column and **`run_p1.py`** (``build_alerts`` only). If CSV download/load is extra, the caption also shows **including CSV / network load** for the full refresh. |
| **Metrics: Alerts / sec_id (unique) / HIGH severity** | **Alerts** = rows in the table. **sec_id** = how many different securities appear. **HIGH** = count of high-severity rows (quick risk summary). |
| **Multiselect: Severity / Anomaly type** | Filters the table and charts. |
| **Bar chart: By severity** | How many alerts per severity bucket (e.g. HIGH / MEDIUM). |
| **Bar chart: By anomaly_type** | Which **kinds** of patterns fired (concentration, imbalance, DBSCAN cluster, etc.). |
| **Bar chart: Top sec_id by alert count** | Which tickers/ids generated the most alerts (who is “noisiest” in the book). |
| **Table at bottom** | Full alert rows: read **`remarks`** for the human explanation of each alert. |

### 2.4 How to interpret P1 results

- **`anomaly_type`** — Name of the pattern (e.g. level-one concentration, wide spread, cluster from DBSCAN).
- **`severity`** — Relative strength bucket; **HIGH** is the strongest tier in this pipeline.
- **`time_window_start` / `trade_date`** — When the pattern was observed.
- **`remarks`** — **Most important for grading/demo**: plain-language summary (often includes OBI, window length, optional trade context).

**What you achieved (P1):** A reproducible path from raw **L2-style** equity data to a **prioritized alert list** with typed anomalies and explanations—suitable for surveillance or manual review queues.

---

## 3. Problem 2 (P2) tab — EDGAR 8-K + pre-announcement activity

### 3.1 What problem P2 solves (simple terms)

**Goal:** Combine **public SEC filings** (8-K style events) with **your equity OHLCV and trades** per `sec_id`. For each filing date, the pipeline checks whether **volume or returns** looked unusual **before** the filing (a simple “pre-announcement screen”), and attaches **trade-level hints** in the text.

**Output:** Rows like `p2_signals.csv` from `run_p2.py`.

### 3.2 Datasets P2 uses

| File / source | Role |
|---------------|------|
| **`ohlcv.csv`** | Required. Daily (or bar) prices + volume per `sec_id` (and usually **`ticker`** for EDGAR lookup). |
| **`trade_data.csv`** | Required. Trades for enrichment around the window. |
| **`sec_id_map.csv`** | Optional if `ohlcv.csv` already maps **`ticker`** ↔ **`sec_id`**. |
| **SEC EDGAR** (live) | Fetches 8-K-related filings for those tickers between **start** and **end** dates. |
| **Filings cache CSV** | Used when **Skip EDGAR** is on — same columns as a saved EDGAR pull (`file_date`, `headline`, URL, etc.). |
| **`p2_signals.csv`** | Precomputed table for **Static CSV** mode. |

### 3.3 Labeled UI components (P2 tab)

| Component | Purpose |
|-----------|---------|
| **Subheader** “P2 — EDGAR 8-K + pre-announcement activity” | Names the tab. |
| **Caption** | Static vs pipeline + cache / env hints. |
| **Expander: “P2 — Data source & EDGAR”** | Data source and EDGAR controls. |
| **Source** (radio) | **Static CSV** vs **Run pipeline (EDGAR)**. |
| **Auto-refresh / Refresh every** | Same idea as P1 — useful with live EDGAR + poll bucket. |
| **Upload P2 CSV** | Optional upload for static mode. |
| **Path to p2_signals.csv** | Static output file path. |
| **Equity data folder** | Must contain **`ohlcv.csv`** and **`trade_data.csv`** for the pipeline. |
| **EDGAR start / end** | Date range for SEC search (live mode). |
| **Skip EDGAR** | Use a **local filings cache** instead of hitting SEC every time. |
| **EDGAR re-fetch interval** | With live EDGAR, controls how often cached runs refresh (time buckets). |
| **Filings cache CSV path** | Required path when skipping EDGAR. |
| **Clear P2 pipeline cache / Run pipeline** | Invalidate cache or bump a “force” timestamp so you get a fresh run. |
| **Metrics: Rows / sec_id / pre_drift_flag = 1** | **pre_drift_flag = 1** = rows where the **pre-filing** volume/return screen lit up. |
| **Multiselect: event_type** | Filter by classified headline category. |
| **Multiselect: pre_drift_flag** | Show only suspicious (1) or normal (0) pre-windows. |
| **Charts: By event_type / pre_drift_flag** | Distribution of filing types and how often the pre-screen triggers. |
| **Rows per week** | Time concentration of events. |
| **Table** | Full signal rows — read **`remarks`** for the narrative. |

### 3.4 How to interpret P2 results

- **`event_date`** — Filing (or event) date aligned to the row.
- **`event_type`** — Rule-based label from the filing headline (e.g. earnings, M&A, other).
- **`pre_drift_flag`** — **1** if pre-filing volume z-score or cumulative abnormal return crossed thresholds; **0** otherwise. This is **not** proof of insider trading—it is a **screening flag** for review.
- **`suspicious_window_start`** — Start of the pre-filing window used in the check.
- **`remarks`** — Explains what the screen saw and references **trades** when relevant.

**What you achieved (P2):** An **integrated** equity surveillance slice: **public disclosures** joined to **price/volume/trade** behavior with a transparent **flag** and **text remark** per row.

---

## 4. Problem 3 (P3) tab — crypto submission

### 4.1 What problem P3 solves (simple terms)

**Goal:** Run the **crypto** pipeline (Binance-style minute bars + synthetic trades for eight symbols) through **rule detectors** and optional **ML re-ranking**, and produce a **submission-shaped** table: which **`trade_id`**s look suspicious, with a **`violation_type`** and **`remarks`**.

This matches the main hackathon deliverable: **`submission.csv`**.

### 4.2 Datasets P3 uses

| Source | Role |
|--------|------|
| **`data/student-pack/`** (offline CLI) | Bundled **crypto** OHLCV + trades used by `run_p3.py` (not the same as equity `data/equity-data/`). |
| **`submission.csv`** | Default **primary** file in the dashboard when present. |
| **`dashboard/sample_submission.csv`** | Fallback sample committed in the repo if your `submission.csv` is missing. |
| **Live (API)** | Pulls recent **klines + agg trades** from OKX and/or Binance (see `LIVE_SPOT_VENUE`), runs the **same** pipeline logic on fresh data. |
| **PRIMARY_SUBMISSION_URL / SECOND_SUBMISSION_URL** | Optional hosted CSVs (e.g. on Streamlit Cloud). |
| **Second CSV path / upload** | Enables **side-by-side compare** (e.g. offline vs live). |

### 4.3 Labeled UI components (P3 tab)

| Component | Purpose |
|-----------|---------|
| **Subheader** “P3 — Crypto submission” | Names the tab. |
| **Caption** | Static vs Live + venue env vars. |
| **Expander: “P3 — Data & refresh”** | Main control surface for P3. |
| **Primary source** | **Static CSV** (path/URL/upload/sample) vs **Live (API)**. |
| **Live-only controls** | **Cache live pipeline (~90s)** — avoids redoing heavy fetch+ML every few seconds; **Clear live cache**; **Klines / Agg trades per symbol** — depth of history (larger = slower). |
| **Advanced: CSV upload** | Optional primary/second file upload. |
| **Primary CSV path / Second CSV path** | Local paths; second path feeds the **compare** sub-tab. |
| **Auto-refresh / Refresh every** | Page autoreload interval. **Important:** for Live, intervals shorter than ~30–90s can interrupt the first run before charts appear. |
| **Last run: time** | Clock stamp when that expander rendered (rough “freshness” hint). |
| **Offline / Live CLI hints** | Copy-paste commands for `run_p3.py`, `make dual`, backfill script. |
| **Filters → Reset symbol & type filters when CSV updates** | When a new CSV loads, optionally reset multiselects so you do not accidentally filter to an empty subset. |
| **Blue info (Live mode)** | Warns about first-load duration and auto-refresh pitfalls. |
| **Green success / warnings** | Live row count, venue name, or fallback to bundled sample if live failed. |
| **Buttons: All symbols · primary / second** | Quickly select every symbol in the multiselect for compare views. |
| **Sub-tabs** (if second source loaded) | **Primary CSV** vs **Second CSV** — each runs the same **panel** below. |

**Submission panel** (used once or twice depending on compare):

| Block | Meaning |
|-------|---------|
| **Metrics (4 columns)** | **Flagged trades** = row count. **Symbols** = how many pairs. **With violation_type** = rows with a non-empty type. **ML score (median)** = median of `ml_rank_p` parsed from remarks when present. |
| **Symbol / Violation type / Search** | Filters. **Search** scans **remarks** and **trade_id**. |
| **Filtered rows** | Count after filters. |
| **By symbol** chart | Which pairs have the most flagged rows. |
| **By violation_type** chart | Mix of rule labels (peg_break, wash, etc.). |
| **ML re-rank score histogram** | Distribution of `ml_rank_p` when embedded in remarks. |
| **Flags per day** | Time series of how many flags per calendar day. |
| **Table** | The actual **submission** columns: `symbol`, `date`, `trade_id`, `violation_type`, optional `ml_rank_p`, **`remarks`**. |

### 4.4 How to interpret P3 results

- **`violation_type`** — Which **detector** fired (rules first; ML may reorder or cap rows — see repo README for detector list).
- **`remarks`** — **Primary evidence string**: price vs peg, notional, hot hour, ML score tail, etc.
- **`ml_rank_p`** (if shown) — Higher usually means the ML re-ranker thought the row was **more** worth surfacing (model-specific; use as a sort key, not a legal finding).

**What you achieved (P3):** An end-to-end **crypto trade surveillance** pipeline from raw market+trades to a **graded submission schema**, with optional **live** refresh and **offline vs live** comparison in the UI.

---

## 5. Quick mental model (all three tabs)

```text
┌─────────────────────────────────────────────────────────────────┐
│  BITS — Trade surveillance                                       │
├──────────────┬────────────────────┬───────────────────────────────┤
│  P1 Equity   │  P2 Equity         │  P3 Crypto                     │
│  Order book  │  EDGAR + OHLCV     │  Rules + ML → submission      │
│  → alerts    │  → pre-filing      │  → flagged trade_ids           │
│              │    screen rows     │                                │
└──────────────┴────────────────────┴───────────────────────────────┘
```

---

## 6. How to evaluate your own work with this dashboard

1. **Reproducibility** — Can you switch to **Static CSV** and see the same row counts as the files you submitted (`p1_alerts.csv`, `p2_signals.csv`, `submission.csv`)?
2. **Pipelines** — Does **Run from folder** (P1) / **Run pipeline** (P2) complete on a clean copy of the student-pack equity folder without errors?
3. **Plausibility** — Open **remarks** at random: do they match the **data type** (book depth vs filing vs crypto trade)?
4. **Live (P3 / optional P1 URLs)** — Does the app update when the world changes (new candles/trades or new CSV snapshot)?
5. **Performance** — Note **time_to_run** (P1/P2) and Live P3 latency; compare to any runtime targets in the brief.

---

## 7. Related files in the repo

| Path | Role |
|------|------|
| `dashboard/app.py` | P3 UI + tab wiring + `render_submission_panel`. |
| `dashboard/tab_p1.py` | P1 UI and cached pipeline helpers. |
| `dashboard/tab_p2.py` | P2 UI and cached pipeline helpers. |
| `run_p1.py` / `run_p2.py` / `run_p3.py` | CLI parity with the dashboard. |
| `data/equity-data/README.txt` | What belongs in the equity folder. |
| `README.md` | Full approach, Make targets, Cloud secrets. |
| `.streamlit/secrets.toml.example` | Optional URL and venue keys for hosted deploys. |

If anything in the UI and this guide disagree, treat the **code** as source of truth and open an issue or update this file.
