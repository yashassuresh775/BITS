"""Streamlit tab: Problem 1 order-book alerts (``p1_alerts.csv``)."""

from __future__ import annotations

import hashlib
import io
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]

P1_DEFAULT_CSV = REPO_ROOT / "p1_alerts.csv"
P1_DEFAULT_DATA = REPO_ROOT / "data" / "equity-data"
P1_COLS_MIN = ("sec_id", "anomaly_type", "severity")


def default_equity_folder() -> str:
    """Prefer env over bundled ``data/equity-data`` (see ``data/equity-data/README.txt``)."""
    for k in ("EQUITY_ROOT", "EQUITY_DATA_ROOT", "EQUITY_DATA_STREAMLIT_DEFAULT"):
        v = os.environ.get(k, "").strip()
        if v:
            return str(Path(v).expanduser().resolve())
    return str(P1_DEFAULT_DATA.resolve())


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


def _validate_p1(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for c in P1_COLS_MIN:
        if c not in df.columns:
            st.error(f"P1 CSV missing column `{c}`.")
            return pd.DataFrame()
    out = df.copy()
    if "trade_date" in out.columns:
        out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    if "time_window_start" in out.columns:
        out["time_window_start"] = out["time_window_start"].astype(str)
    return out


@st.cache_data(show_spinner="Loading P1 alerts…")
def load_p1_csv(path: str, _mt: float) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return pd.DataFrame()
    return _validate_p1(pd.read_csv(p))


@st.cache_data(show_spinner="Loading uploaded P1 CSV…")
def load_p1_bytes(content: bytes, _ck: str) -> pd.DataFrame:
    return _validate_p1(pd.read_csv(io.BytesIO(content)))


@st.cache_data(show_spinner="Loading P1 alerts from URL…", ttl=120)
def load_p1_url(url: str, _k: str) -> pd.DataFrame:
    return _validate_p1(pd.read_csv(url))


def _bytes_key(b: bytes, label: str) -> str:
    h = hashlib.sha256(b[: min(len(b), 500_000)]).hexdigest()[:20]
    return f"{label}:{len(b)}:{h}"


@st.cache_data(show_spinner="Running P1 pipeline…")
def run_p1_from_folder(
    data_root: str, _mkt_mtime: float, no_trades: bool
) -> tuple[pd.DataFrame, str | None, float, float]:
    """
    Returns ``(df, err, pipeline_seconds, total_seconds)``.
    ``pipeline_seconds`` matches ``run_p1.py`` / column ``time_to_run`` (``build_alerts`` only).
    ``total_seconds`` includes CSV load + pipeline (what the browser refresh actually waits for).
    """
    try:
        root = Path(data_root).expanduser().resolve()
        mp = root / "market_data.csv"
        if not mp.is_file():
            return pd.DataFrame(), f"Missing market_data.csv under {root}", 0.0, 0.0
        from p1.io import load_market_data, load_trades_per_minute
        from p1.pipeline import build_alerts

        t0 = time.perf_counter()
        md = load_market_data(mp)
        tpm = None
        if not no_trades:
            tp = root / "trade_data.csv"
            if tp.is_file():
                tpm = load_trades_per_minute(str(tp))
        t1 = time.perf_counter()
        out, pipe_elapsed = build_alerts(md, tpm)
        total_elapsed = time.perf_counter() - t0
        if not out.empty:
            out["time_to_run"] = round(pipe_elapsed, 3)
        return _validate_p1(out), None, pipe_elapsed, total_elapsed
    except Exception as e:  # noqa: BLE001
        return pd.DataFrame(), str(e), 0.0, 0.0


@st.cache_data(show_spinner="P1 live · HTTP GET + pipeline…", ttl=3600)
def run_p1_live_urls(
    market_url: str,
    trades_url: str,
    no_trades: bool,
    _poll_tick: int,
) -> tuple[pd.DataFrame, str | None, float, float]:
    """
    Re-download CSVs when ``_poll_tick`` changes (``int(time.time() // poll_interval)``).
    There is no public “order-book WebSocket” in-repo — this is **HTTP polling** of raw CSV URLs (your feed / CDN / gist).
    """
    try:
        import requests

        mu = (market_url or "").strip()
        if not mu:
            return pd.DataFrame(), "Set **Market data URL** (or secret `P1_LIVE_MARKET_URL`).", 0.0, 0.0

        t0 = time.perf_counter()
        r = requests.get(mu, timeout=120, headers={"User-Agent": "BITS-p1-live/1.0"})
        r.raise_for_status()
        md = pd.read_csv(io.BytesIO(r.content))
        from p1.io import load_market_data, load_trades_per_minute
        from p1.pipeline import build_alerts

        mdf = load_market_data(md)
        tpm = None
        if not no_trades:
            tu = (trades_url or "").strip()
            if tu:
                rt = requests.get(tu, timeout=120, headers={"User-Agent": "BITS-p1-live/1.0"})
                rt.raise_for_status()
                tdf = pd.read_csv(io.BytesIO(rt.content))
                tpm = load_trades_per_minute(tdf)

        out, pipe_elapsed = build_alerts(mdf, tpm)
        total_elapsed = time.perf_counter() - t0
        if not out.empty:
            out["time_to_run"] = round(pipe_elapsed, 3)
        return _validate_p1(out), None, pipe_elapsed, total_elapsed
    except Exception as e:  # noqa: BLE001
        return pd.DataFrame(), str(e), 0.0, 0.0


def render_p1_tab() -> None:
    st.subheader("P1 — Order-book concentration & DBSCAN alerts")
    st.caption(
        "**Static:** `p1_alerts.csv`. **Folder:** `run_p1.py`-style on `market_data.csv` (+ optional `trade_data.csv`). "
        "**Live (URLs):** poll **HTTPS CSV** endpoints (your hosted feed). "
        "Default folder: **`EQUITY_ROOT`** / **`EQUITY_DATA_STREAMLIT_DEFAULT`** (see `data/equity-data/README.txt`)."
    )

    secret_alerts = _secret_str("P1_ALERTS_URL")
    secret_mkt = _secret_str("P1_LIVE_MARKET_URL")
    secret_tr = _secret_str("P1_LIVE_TRADE_URL")

    with st.expander("P1 — Data source & refresh", expanded=True):
        mode = st.radio(
            "Source",
            ["Static CSV", "Run from equity folder", "Live (poll CSV URLs)"],
            horizontal=True,
            key="p1_mode",
        )
        auto = st.toggle(
            "Auto-refresh page",
            value=(mode == "Live (poll CSV URLs)"),
            key="p1_auto",
            help="Rerun the app on an interval (folder mtime, live poll tick, or static file reload).",
        )
        interval = st.slider(
            "Refresh every (s)",
            15,
            600,
            120 if mode == "Live (poll CSV URLs)" else 60,
            15,
            disabled=not auto,
            key="p1_interval",
        )
        upload = st.file_uploader("Upload P1 CSV (optional)", type=["csv"], key="p1_upload")

        df = pd.DataFrame()
        src = ""
        err: str | None = None
        elapsed_pipe = 0.0
        elapsed_total = 0.0

        if mode == "Static CSV":
            path = st.text_input("Path to p1_alerts.csv", value=str(P1_DEFAULT_CSV), key="p1_csv_path")
            if secret_alerts:
                src_opts = st.radio(
                    "Load from",
                    ["Local path", "Secrets URL (`P1_ALERTS_URL`)"],
                    horizontal=True,
                    key="p1_src_choice",
                )
            else:
                src_opts = "Local path"
            if upload is not None:
                ck = _bytes_key(upload.getvalue(), upload.name or "p1.csv")
                df = load_p1_bytes(upload.getvalue(), ck)
                src = f"upload:{ck}"
            elif secret_alerts and src_opts == "Secrets URL (`P1_ALERTS_URL`)":
                df = load_p1_url(secret_alerts, secret_alerts)
                src = f"url:{secret_alerts}"
            else:
                mt = _mtime(path)
                df = load_p1_csv(path, mt)
                src = f"path:{path}|{mt}"

        elif mode == "Run from equity folder":
            root = st.text_input(
                "Equity data folder (`market_data.csv` required)",
                value=default_equity_folder(),
                key="p1_data_root",
            )
            st.caption(
                f"Default folder: **`EQUITY_ROOT`** / **`EQUITY_DATA_ROOT`** / **`EQUITY_DATA_STREAMLIT_DEFAULT`**, "
                f"else `{P1_DEFAULT_DATA}`. See `data/equity-data/README.txt`."
            )
            no_tr = st.toggle("Ignore trade_data.csv", value=False, key="p1_no_trades")
            mkt = Path(root).expanduser() / "market_data.csv"
            mkt_mt = mkt.stat().st_mtime if mkt.is_file() else 0.0
            if st.button("Run / refresh pipeline", key="p1_run_btn", type="primary"):
                st.session_state["p1_force_run"] = time.time()
            force = st.session_state.get("p1_force_run", 0)
            df, err, elapsed_pipe, elapsed_total = run_p1_from_folder(root, mkt_mt + force * 1e-9, no_tr)
            src = f"folder:{root}|mkt={mkt_mt}|f={force}"

        else:
            st.markdown(
                "Serve **`market_data.csv`** (and optionally **`trade_data.csv`**) at stable HTTPS URLs "
                "(S3, CloudFront, private API, etc.). **Poll interval** controls how often the app re-downloads and re-runs the pipeline."
            )
            poll = st.slider("Poll interval (seconds)", 30, 600, 120, 15, key="p1_live_poll")
            murl = st.text_input(
                "Market data CSV URL",
                value=secret_mkt or "",
                key="p1_live_murl",
                help="Or set Streamlit secret `P1_LIVE_MARKET_URL` as default hint.",
            )
            no_tr = st.toggle("Ignore trades URL", value=False, key="p1_live_no_tr")
            turl = ""
            if not no_tr:
                turl = st.text_input(
                    "Trade data CSV URL (optional)",
                    value=secret_tr or "",
                    key="p1_live_turl",
                    help="Secret `P1_LIVE_TRADE_URL` pre-fills when set.",
                )
            if st.button("Clear P1 live cache", key="p1_live_clear"):
                run_p1_live_urls.clear()
                st.success("Cache cleared.")
            tick = int(time.time() // max(int(poll), 15))
            df, err, elapsed_pipe, elapsed_total = run_p1_live_urls(murl, turl, no_tr, tick)
            src = f"live_urls|tick={tick}|poll={poll}"

    try:
        from streamlit_autorefresh import st_autorefresh
    except ImportError:
        st_autorefresh = None  # type: ignore[misc, assignment]
    if auto and st_autorefresh is not None:
        st_autorefresh(interval=int(interval * 1000), limit=None, key="p1_autoref")
    elif auto:
        st.warning("Install `streamlit-autorefresh` for auto-refresh.")

    if err:
        st.error(f"P1 pipeline error: {err}")
    if df.empty:
        st.info(
            "No P1 rows. Use **Static CSV**, a folder with **`market_data.csv`**, or **Live URLs**. "
            "Set **`export EQUITY_ROOT=/path/to/equity`** so **Run from folder** defaults correctly."
        )
        return

    st.success(f"**{len(df)}** alerts · source `{src[:80]}…`" if len(src) > 80 else f"**{len(df)}** alerts · `{src}`")
    if elapsed_pipe > 0 or elapsed_total > 0:
        cap = f"**Pipeline compute** (`time_to_run`, same as CLI): **{elapsed_pipe:.2f}s**"
        if elapsed_total > elapsed_pipe + 0.02:
            cap += f" · **including CSV / network load:** **{elapsed_total:.2f}s**"
        st.caption(cap)

    c1, c2, c3 = st.columns(3)
    c1.metric("Alerts", len(df))
    c2.metric("sec_id (unique)", int(df["sec_id"].nunique()))
    c3.metric("HIGH severity", int((df["severity"].astype(str).str.upper() == "HIGH").sum()))

    severities = sorted(df["severity"].dropna().astype(str).unique().tolist())
    types = sorted(df["anomaly_type"].dropna().astype(str).unique().tolist())
    s_sel = st.multiselect("Severity", severities, default=severities, key="p1_f_sev")
    t_sel = st.multiselect("Anomaly type", types, default=types, key="p1_f_type")
    view = df[df["severity"].astype(str).isin(s_sel) & df["anomaly_type"].astype(str).isin(t_sel)]
    chart_df = view if len(view) > 0 else df
    if len(view) == 0 and len(df) > 0:
        st.info("Filters excluded all rows — charts use full dataset.")

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**By severity**")
        st.bar_chart(chart_df["severity"].astype(str).value_counts())
    with cc2:
        st.markdown("**By anomaly_type**")
        st.bar_chart(chart_df["anomaly_type"].astype(str).value_counts().head(20))

    st.markdown("**Top sec_id by alert count**")
    st.bar_chart(chart_df.groupby("sec_id", observed=True).size().sort_values(ascending=False).head(25))

    show_cols = list(df.columns)
    st.dataframe(view[show_cols] if len(view) else chart_df[show_cols], use_container_width=True, height=400)
