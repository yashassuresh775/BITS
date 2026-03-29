"""Streamlit tab: Problem 2 EDGAR + insider-style signals (``p2_signals.csv``)."""

from __future__ import annotations

import hashlib
import io
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]

P2_DEFAULT_CSV = REPO_ROOT / "p2_signals.csv"
P2_DEFAULT_DATA = REPO_ROOT / "data" / "equity-data"
P2_MIN_COLS = ("sec_id", "event_date", "remarks")


def default_equity_folder() -> str:
    """Prefer env over bundled ``data/equity-data`` (see ``data/equity-data/README.txt``)."""
    for k in ("EQUITY_ROOT", "EQUITY_DATA_ROOT", "EQUITY_DATA_STREAMLIT_DEFAULT"):
        v = os.environ.get(k, "").strip()
        if v:
            return str(Path(v).expanduser().resolve())
    return str(P2_DEFAULT_DATA.resolve())


def _secret_str(key: str) -> str | None:
    try:
        v = st.secrets[key]
        s = str(v).strip()
        return s or None
    except (KeyError, FileNotFoundError, TypeError, RuntimeError):
        return None


def _mtime(path: str) -> float:
    p = Path(path).expanduser().resolve()
    try:
        return p.stat().st_mtime
    except OSError:
        return -1.0


def _bytes_key(b: bytes, label: str) -> str:
    h = hashlib.sha256(b[: min(len(b), 500_000)]).hexdigest()[:20]
    return f"{label}:{len(b)}:{h}"


def load_ticker_map(root: Path, ohlcv: pd.DataFrame) -> pd.DataFrame:
    map_path = root / "sec_id_map.csv"
    if map_path.is_file():
        return pd.read_csv(map_path)
    if "ticker" not in ohlcv.columns or "sec_id" not in ohlcv.columns:
        raise ValueError(
            f"No {map_path.name} and ohlcv.csv has no ticker column — cannot map EDGAR tickers to sec_id."
        )
    return (
        ohlcv[["sec_id", "ticker"]]
        .drop_duplicates(subset=["sec_id"])
        .assign(ticker=lambda d: d["ticker"].astype(str).str.upper().str.strip())
    )


def _validate_p2(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for c in P2_MIN_COLS:
        if c not in df.columns:
            st.error(f"P2 CSV missing column `{c}`.")
            return pd.DataFrame()
    out = df.copy()
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    if "pre_drift_flag" in out.columns:
        out["pre_drift_flag"] = pd.to_numeric(out["pre_drift_flag"], errors="coerce").fillna(0).astype(int)
    return out


@st.cache_data(show_spinner="Loading P2 signals…")
def load_p2_csv(path: str, _mt: float) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return pd.DataFrame()
    return _validate_p2(pd.read_csv(p))


@st.cache_data(show_spinner="Loading uploaded P2 CSV…")
def load_p2_bytes(content: bytes, _ck: str) -> pd.DataFrame:
    return _validate_p2(pd.read_csv(io.BytesIO(content)))


@st.cache_data(show_spinner="Loading P2 signals from URL…", ttl=120)
def load_p2_url(url: str, _k: str) -> pd.DataFrame:
    return _validate_p2(pd.read_csv(url))


@st.cache_data(
    ttl=3600,
    show_spinner="P2 · EDGAR fetch + signal build (cached; bucket + mtimes + Clear cache)…",
)
def run_p2_pipeline_cached(
    data_root: str,
    start_date: str,
    end_date: str,
    skip_edgar: bool,
    filings_cache_path: str,
    _ohlcv_mt: float,
    _trades_mt: float,
    _filings_mt: float,
    _bust: float,
    _poll_tick: int,
) -> tuple[pd.DataFrame, str | None, float]:
    """
    ``_bust`` bumps cache when user clicks Run (same files, force refresh).
    When EDGAR is on, ``_poll_tick`` is ``int(time.time() // interval)`` so fetches rerun on that interval.
    When ``skip_edgar``, pass ``_poll_tick=0`` so only file mtimes + bust matter.
    """
    try:
        root = Path(data_root).expanduser().resolve()
        ohlcv_path = root / "ohlcv.csv"
        trades_path = root / "trade_data.csv"
        if not ohlcv_path.is_file():
            return pd.DataFrame(), f"Missing ohlcv.csv under {root}", 0.0
        if not trades_path.is_file():
            return pd.DataFrame(), f"Missing trade_data.csv under {root}", 0.0

        t0 = time.perf_counter()
        ohlcv = pd.read_csv(ohlcv_path)
        trades = pd.read_csv(trades_path)
        ticker_map = load_ticker_map(root, ohlcv)
        tickers = ticker_map["ticker"].astype(str).str.upper().unique().tolist()

        from p2.edgar import classify_event, fetch_8k_filings, merge_sec_ids

        if skip_edgar:
            fc = Path(filings_cache_path).expanduser().resolve()
            if not fc.is_file():
                return pd.DataFrame(), f"--skip-edgar: filings cache not found: {fc}", 0.0
            filings = pd.read_csv(fc, parse_dates=["file_date"])
            if "filing_url" not in filings.columns and "source_url" in filings.columns:
                filings = filings.rename(columns={"source_url": "filing_url"})
        else:
            filings = fetch_8k_filings(tickers=tickers, start_date=start_date, end_date=end_date)
            if filings.empty:
                pass

        if not filings.empty and "event_type" not in filings.columns:
            filings["headline"] = filings.get("headline", pd.Series("", index=filings.index)).fillna("")
            filings["event_type"] = filings["headline"].map(classify_event)

        filings = merge_sec_ids(filings, ticker_map)
        filings = filings.dropna(subset=["sec_id"])

        elapsed_mid = time.perf_counter() - t0
        from p2.insider_signals import build_p2_signals

        out = build_p2_signals(ohlcv, trades, filings, time_to_run_s=elapsed_mid)
        out["time_to_run"] = round(time.perf_counter() - t0, 3)
        return _validate_p2(out), None, time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        return pd.DataFrame(), str(e), 0.0


def render_p2_tab() -> None:
    st.subheader("P2 — EDGAR 8-K + pre-announcement activity")
    st.caption(
        "**Static:** `p2_signals.csv` (path, upload, or `P2_SIGNALS_URL`). "
        "**Pipeline:** same as `run_p2.py` — needs `ohlcv.csv` + `trade_data.csv`. "
        "With live EDGAR, set a **re-fetch interval** (cache bucket); with **skip EDGAR**, refresh follows file mtimes. "
        "Set **`EQUITY_ROOT`** (or `EQUITY_DATA_STREAMLIT_DEFAULT`) for the default folder."
    )

    secret_url = _secret_str("P2_SIGNALS_URL")

    with st.expander("P2 — Data source & EDGAR", expanded=True):
        mode = st.radio(
            "Source",
            ["Static CSV", "Run pipeline (EDGAR)"],
            horizontal=True,
            key="p2_mode",
        )
        auto = st.toggle("Auto-refresh page", value=False, key="p2_auto")
        interval = st.slider("Refresh every (s)", 30, 600, 120, 30, disabled=not auto, key="p2_interval")
        upload = st.file_uploader("Upload P2 CSV (optional)", type=["csv"], key="p2_upload")

        df = pd.DataFrame()
        src = ""
        err: str | None = None
        elapsed = 0.0

        if mode == "Static CSV":
            path = st.text_input("Path to p2_signals.csv", value=str(P2_DEFAULT_CSV), key="p2_csv_path")
            if secret_url:
                src_choice = st.radio(
                    "Load from",
                    ["Local path", "Secrets URL (`P2_SIGNALS_URL`)"],
                    horizontal=True,
                    key="p2_src_choice",
                )
            else:
                src_choice = "Local path"
            if upload is not None:
                ck = _bytes_key(upload.getvalue(), upload.name or "p2.csv")
                df = load_p2_bytes(upload.getvalue(), ck)
                src = f"upload:{ck}"
            elif secret_url and src_choice == "Secrets URL (`P2_SIGNALS_URL`)":
                df = load_p2_url(secret_url, secret_url)
                src = f"url:{secret_url}"
            else:
                mt = _mtime(path)
                df = load_p2_csv(path, mt)
                src = f"path:{path}|{mt}"
        else:
            root = st.text_input(
                "Equity data folder (`ohlcv.csv` + `trade_data.csv`)",
                value=default_equity_folder(),
                key="p2_data_root",
            )
            st.caption(
                f"Default folder uses **`EQUITY_ROOT`** / **`EQUITY_DATA_ROOT`** / **`EQUITY_DATA_STREAMLIT_DEFAULT`** "
                f"if set, else `{P2_DEFAULT_DATA}`."
            )
            c1, c2 = st.columns(2)
            start_d = c1.text_input("EDGAR start", value="2026-01-01", key="p2_start")
            end_d = c2.text_input("EDGAR end", value="2026-03-31", key="p2_end")
            skip = st.toggle("Skip EDGAR (use filings cache file)", value=False, key="p2_skip_edgar")
            edgar_poll = st.slider(
                "EDGAR re-fetch interval (seconds, live mode only)",
                60,
                3600,
                300,
                60,
                disabled=skip,
                key="p2_edgar_poll",
                help="While EDGAR is enabled, the pipeline re-runs when this bucket advances (use with auto-refresh).",
            )
            fc_path = st.text_input(
                "Filings cache CSV (required if skip EDGAR)",
                value=str(REPO_ROOT / "p2_filings_cache.csv"),
                key="p2_filings_path",
            )
            if st.button("Clear P2 pipeline cache", key="p2_clear_cache"):
                run_p2_pipeline_cached.clear()
                st.success("Cache cleared — rerun with **Run pipeline**.")
            if st.button("Run pipeline", type="primary", key="p2_run_btn"):
                st.session_state["p2_bust"] = time.time()

            r = Path(root).expanduser().resolve()
            ohlcv_p = r / "ohlcv.csv"
            tr_p = r / "trade_data.csv"
            omt = ohlcv_p.stat().st_mtime if ohlcv_p.is_file() else 0.0
            tmt = tr_p.stat().st_mtime if tr_p.is_file() else 0.0
            fcp = Path(fc_path).expanduser().resolve()
            fmt = fcp.stat().st_mtime if fcp.is_file() else 0.0
            bust = float(st.session_state.get("p2_bust", 0))
            poll_sec = max(int(edgar_poll), 60)
            poll_tick = 0 if skip else int(time.time() // poll_sec)

            df, err, elapsed = run_p2_pipeline_cached(
                str(r),
                start_d,
                end_d,
                skip,
                str(fc_path),
                omt,
                tmt,
                fmt,
                bust,
                poll_tick,
            )
            src = f"pipeline:{root}|skip={skip}|bust={bust}|tick={poll_tick}"

    try:
        from streamlit_autorefresh import st_autorefresh
    except ImportError:
        st_autorefresh = None  # type: ignore[misc, assignment]
    if auto and st_autorefresh is not None:
        st_autorefresh(interval=int(interval * 1000), limit=None, key="p2_autoref")
    elif auto:
        st.warning("Install `streamlit-autorefresh` for auto-refresh.")

    if err:
        st.error(f"P2 pipeline error: {err}")
    if df.empty:
        st.info("No P2 rows. Load a CSV, use secrets URL, or run the pipeline on a valid equity folder.")
        return

    st.success(f"**{len(df)}** rows · `{src[:100]}…`" if len(src) > 100 else f"**{len(df)}** rows · `{src}`")
    if elapsed > 0:
        st.caption(f"Last pipeline wall time: **{elapsed:.2f}s**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("sec_id (unique)", int(df["sec_id"].nunique()))
    if "pre_drift_flag" in df.columns:
        c3.metric("pre_drift_flag = 1", int((df["pre_drift_flag"] == 1).sum()))
    else:
        c3.metric("Columns", len(df.columns))

    et = sorted(df["event_type"].dropna().astype(str).unique().tolist()) if "event_type" in df.columns else []
    e_sel = st.multiselect("event_type", et, default=et, key="p2_f_et") if et else []
    view = df[df["event_type"].astype(str).isin(e_sel)] if e_sel else df
    if "pre_drift_flag" in df.columns:
        fl = st.multiselect("pre_drift_flag", [0, 1], default=[0, 1], key="p2_f_fl")
        view = view[view["pre_drift_flag"].isin(fl)]
    chart_df = view if len(view) > 0 else df
    if len(view) == 0 and len(df) > 0:
        st.info("Filters excluded all rows — charts use full dataset.")

    cc1, cc2 = st.columns(2)
    with cc1:
        if "event_type" in chart_df.columns:
            st.markdown("**By event_type**")
            st.bar_chart(chart_df["event_type"].astype(str).value_counts().head(15))
    with cc2:
        if "pre_drift_flag" in chart_df.columns:
            st.markdown("**pre_drift_flag**")
            st.bar_chart(chart_df["pre_drift_flag"].value_counts())

    if "event_date" in chart_df.columns and chart_df["event_date"].notna().any():
        st.markdown("**Rows per week (event_date)**")
        w = chart_df.dropna(subset=["event_date"]).copy()
        w["_w"] = w["event_date"].dt.to_period("W").astype(str)
        st.bar_chart(w.groupby("_w").size())

    cols = [c for c in df.columns]
    st.dataframe(view[cols] if len(view) else chart_df[cols], use_container_width=True, height=420)
