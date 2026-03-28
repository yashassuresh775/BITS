"""
Problem 3 submission explorer — run from repo root:

  streamlit run dashboard/app.py

**Local default:** primary is **Static CSV** (``submission.csv`` or bundled ``dashboard/sample_submission.csv``).
Switch to **Live (API)** for the same pipeline as ``run_p3.py --live`` (OKX / Binance / both via ``LIVE_SPOT_VENUE``).
Optional CSV upload lives under **Advanced** in the sidebar.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None  # type: ignore[misc, assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from p3.live.binance import live_spot_venue

DEFAULT_CSV = REPO_ROOT / "submission.csv"
BUNDLED_SAMPLE_CSV = REPO_ROOT / "dashboard" / "sample_submission.csv"
ML_P_RE = re.compile(r"ml_rank_p=([\d.]+)")
REQUIRED_COLS = ("symbol", "date", "trade_id", "violation_type", "remarks")


def _secret_str(key: str) -> str | None:
    try:
        v = st.secrets[key]
        s = str(v).strip()
        return s or None
    except (KeyError, FileNotFoundError, TypeError, RuntimeError):
        return None


def submission_file_mtime(path: str) -> float:
    p = Path(path).expanduser().resolve()
    try:
        return p.stat().st_mtime
    except OSError:
        return -1.0


def _validate_and_enrich_submission(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    for col in REQUIRED_COLS:
        if col not in df.columns:
            st.error(f"Missing column `{col}` in CSV.")
            return pd.DataFrame()
    out = df.copy()
    out["remarks"] = out["remarks"].fillna("").astype(str)
    out["violation_type"] = out["violation_type"].fillna("").astype(str)
    ml = out["remarks"].str.extract(ML_P_RE, expand=False)
    out["ml_rank_p"] = pd.to_numeric(ml, errors="coerce")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


@st.cache_data(show_spinner="Loading submission…")
def load_submission(path: str, _file_mtime: float) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p)
    return _validate_and_enrich_submission(df)


def _bytes_cache_key(content: bytes, label: str) -> str:
    h = hashlib.sha256(content[: min(len(content), 500_000)]).hexdigest()[:20]
    return f"{label}:{len(content)}:{h}"


@st.cache_data(show_spinner="Loading uploaded CSV…")
def load_submission_bytes(content: bytes, _cache_key: str) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content))
    return _validate_and_enrich_submission(df)


@st.cache_data(show_spinner="Loading submission from URL…", ttl=120)
def load_submission_url(url: str, _url_key: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    return _validate_and_enrich_submission(df)


def _apply_binance_env_from_secrets() -> None:
    """Map optional Streamlit secrets into ``os.environ`` so ``p3.live.binance`` sees them on Cloud."""
    for key in (
        "BINANCE_SPOT_API",
        "BINANCE_INSECURE_SSL",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "LIVE_SPOT_VENUE",
    ):
        if os.environ.get(key, "").strip():
            continue
        try:
            v = st.secrets[key]
            s = str(v).strip()
            if s:
                os.environ[key] = s
        except (KeyError, FileNotFoundError, TypeError, RuntimeError):
            pass


def _live_env_fingerprint() -> str:
    return "|".join(
        [
            os.environ.get("LIVE_SPOT_VENUE", "").strip(),
            os.environ.get("BINANCE_SPOT_API", "").strip()[:120],
        ]
    )


def _run_live_binance_submission_impl(kline_limit: int, trades_limit: int) -> tuple[pd.DataFrame, str | None]:
    try:
        _apply_binance_env_from_secrets()
        from p3.config import SYMBOLS
        from p3.live import fetch_live_frames
        from p3.pipeline import hits_to_submission, run_pipeline_from_frames

        frames = fetch_live_frames(
            list(SYMBOLS),
            kline_limit=int(kline_limit),
            trades_limit=int(trades_limit),
        )
        hits = run_pipeline_from_frames(frames)
        sub = hits_to_submission(hits)
        return _validate_and_enrich_submission(sub), None
    except Exception as e:  # noqa: BLE001 — show any API / pipeline error in UI
        return pd.DataFrame(), str(e)


@st.cache_data(ttl=90, show_spinner="Live fetch + pipeline (first run; cached ~90s if enabled)…")
def run_live_binance_submission_cached(
    kline_limit: int,
    trades_limit: int,
    _env_fp: str,
) -> tuple[pd.DataFrame, str | None]:
    """Same as uncached path; TTL avoids re-running ML on every autorefresh tick."""
    return _run_live_binance_submission_impl(kline_limit, trades_limit)


def render_submission_panel(
    df: pd.DataFrame,
    *,
    key_prefix: str,
    heading: str | None = None,
    data_signature: str | None = None,
    reset_filters_on_csv_change: bool = True,
) -> None:
    if heading:
        st.subheader(heading)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flagged trades", len(df))
    c2.metric("Symbols", df["symbol"].nunique())
    vt_nonempty = (df["violation_type"].str.len() > 0).sum()
    c3.metric("With violation_type", int(vt_nonempty))
    if df["ml_rank_p"].notna().any():
        c4.metric("ML score (median)", f"{df['ml_rank_p'].median():.3f}")
    else:
        c4.metric("ML score", "—")

    st.divider()

    symbols = sorted(df["symbol"].dropna().unique().tolist())
    type_opts = ["(empty)"] + sorted(
        {t for t in df["violation_type"].unique() if str(t).strip()}
    )

    sym_key = f"{key_prefix}sym"
    vt_key = f"{key_prefix}vt"
    search_key = f"{key_prefix}search"
    sig_key = f"{key_prefix}file_sig"

    if data_signature is not None and reset_filters_on_csv_change:
        prev = st.session_state.get(sig_key)
        if prev != data_signature:
            st.session_state[sig_key] = data_signature
            st.session_state[sym_key] = list(symbols)
            st.session_state[vt_key] = list(type_opts)
        elif sym_key not in st.session_state:
            st.session_state[sym_key] = list(symbols)
            st.session_state[vt_key] = list(type_opts)
        else:
            ok_s = [s for s in st.session_state[sym_key] if s in symbols]
            st.session_state[sym_key] = ok_s if ok_s else list(symbols)
            ok_t = [t for t in st.session_state[vt_key] if t in type_opts]
            st.session_state[vt_key] = ok_t if ok_t else list(type_opts)
    elif sym_key not in st.session_state:
        st.session_state[sym_key] = list(symbols)
        st.session_state[vt_key] = list(type_opts)
    else:
        ok = [s for s in st.session_state[sym_key] if s in symbols]
        st.session_state[sym_key] = ok if ok else list(symbols)
        ok_t = [t for t in st.session_state[vt_key] if t in type_opts]
        st.session_state[vt_key] = ok_t if ok_t else list(type_opts)

    if sym_key in st.session_state and not st.session_state[sym_key] and symbols:
        st.session_state[sym_key] = list(symbols)
    if vt_key in st.session_state and not st.session_state[vt_key] and type_opts:
        st.session_state[vt_key] = list(type_opts)

    fc1, fc2, fc3 = st.columns(3)
    sel_sym = fc1.multiselect("Symbol", symbols, key=sym_key)
    sel_type = fc2.multiselect("Violation type", type_opts, key=vt_key)
    search = fc3.text_input("Search remarks / trade_id", key=search_key)

    view = df[df["symbol"].isin(sel_sym)]
    if sel_type:
        parts = []
        if "(empty)" in sel_type:
            parts.append(view["violation_type"].str.len() == 0)
        real = [t for t in sel_type if t != "(empty)"]
        if real:
            parts.append(view["violation_type"].isin(real))
        if parts:
            view = view[np.logical_or.reduce(parts) if len(parts) > 1 else parts[0]]

    if search.strip():
        q = search.strip().lower()
        view = view[
            view["remarks"].str.lower().str.contains(q, na=False)
            | view["trade_id"].astype(str).str.lower().str.contains(q, na=False)
        ]

    st.markdown(f"**Filtered rows:** {len(view)}")

    # Charts need non-empty frames; empty filters should not blank the whole dashboard.
    chart_df = view if len(view) > 0 else df
    if len(view) == 0 and len(df) > 0:
        st.info(
            "No rows match **Symbol** / **Violation type** / **Search**. "
            "Widen sidebar filters — charts below show the **full** dataset until then."
        )

    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("**By symbol**")
        sym_counts = chart_df.groupby("symbol", observed=True).size().sort_values(ascending=False)
        if len(sym_counts) > 0:
            st.bar_chart(sym_counts)
        else:
            st.caption("No data for this chart.")
    with ch2:
        st.markdown("**By violation_type**")
        vt = chart_df["violation_type"].replace("", "(empty)")
        vc = vt.value_counts()
        if len(vc) > 0:
            st.bar_chart(vc)
        else:
            st.caption("No data for this chart.")

    ml = chart_df["ml_rank_p"].dropna()
    if len(ml) > 1:
        st.markdown("**ML re-rank score (`ml_rank_p`) — bin counts**")
        hist, edges = np.histogram(ml.to_numpy(dtype=float), bins=16)
        mids = (edges[:-1] + edges[1:]) / 2.0
        st.bar_chart(pd.Series(hist, index=mids))

    if chart_df["date"].notna().any():
        st.markdown("**Flags per day**")
        daily = chart_df.dropna(subset=["date"]).copy()
        daily["_d"] = daily["date"].dt.date
        lc = daily.groupby("_d").size()
        if len(lc) > 0:
            st.line_chart(lc)

    st.divider()
    st.markdown("**Table**")
    disp = view.copy()
    if disp["date"].notna().any():
        disp["date"] = disp["date"].dt.strftime("%Y-%m-%d")
    cols = [
        c
        for c in ("symbol", "date", "trade_id", "violation_type", "ml_rank_p", "remarks")
        if c in disp.columns
    ]
    st.dataframe(disp[cols], use_container_width=True, height=420)


def main() -> None:
    st.set_page_config(
        page_title="BITS — Problem 3 dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_binance_env_from_secrets()
    st.title("BITS — Problem 3 submission explorer")
    st.caption(
        "Default **Static CSV** = `submission.csv` or bundled sample. **Live (API)** = public REST + pipeline. "
        "Venue: **`LIVE_SPOT_VENUE`** = `okx` (default), `binance`, or `both` (OKX+Binance merged; optional **`BINANCE_SPOT_API`**)."
    )

    secret_primary = _secret_str("PRIMARY_SUBMISSION_URL")
    secret_second = _secret_str("SECOND_SUBMISSION_URL")

    with st.sidebar:
        st.header("Data")
        primary_mode = st.radio(
            "Primary source",
            ["Static CSV", "Live (API)"],
            horizontal=True,
            key="primary_source_v4",
            help="Static = local path, URL, upload, or bundled sample. Live = `run_p3.py --live` pipeline; venue from `LIVE_SPOT_VENUE` (okx / binance / both).",
        )
        live_klines = 400
        live_trades = 400
        use_live_cache = True
        if primary_mode == "Live (API)":
            st.markdown(
                "Same detectors as `run_p3.py --live`. **Venue** via env / Secrets **`LIVE_SPOT_VENUE`**: "
                "**`okx`** (default), **`binance`** (single host; optional **`BINANCE_SPOT_API`**), or **`both`** "
                "(OKX + Binance in parallel, merged bars + combined trades)."
            )
            use_live_cache = st.toggle(
                "Cache live pipeline (~90s)",
                value=True,
                help="Reuses the last successful run for 90s so auto-refresh does not redo fetch+ML every few seconds.",
            )
            if use_live_cache and st.button("Clear live cache", help="Force a full refetch on next run."):
                run_live_binance_submission_cached.clear()
                st.rerun()
            live_klines = st.number_input(
                "Klines / symbol",
                min_value=100,
                max_value=1000,
                value=400,
                step=50,
                help="Smaller = faster load (default 400). Max 1000.",
            )
            live_trades = st.number_input(
                "Agg trades / symbol",
                min_value=100,
                max_value=1000,
                value=400,
                step=50,
                help="Smaller = faster (default 400).",
            )
            st.caption(
                "First live run is often **30–90s**. Keep **Auto-refresh off** until you see charts, or set **Refresh every ≥ 120s**."
            )
        else:
            st.markdown(
                "Uses **path** below, then `PRIMARY_SUBMISSION_URL` secret, then optional **upload** "
                "in Advanced, then `dashboard/sample_submission.csv` if the path file is missing."
            )
        upload_primary = None
        upload_second = None
        with st.expander("Advanced: CSV upload (optional)", expanded=False):
            st.caption("Overrides static path/URL when a file is selected. Ignored when primary is **Live**.")
            upload_primary = st.file_uploader(
                "Upload primary CSV",
                type=["csv"],
                key="upload_primary",
                help="Only used for **Static CSV** primary mode.",
            )
            upload_second = st.file_uploader(
                "Upload second CSV (compare tab)",
                type=["csv"],
                key="upload_second",
            )
        if secret_primary:
            st.caption("Secrets: `PRIMARY_SUBMISSION_URL` is set (used in Static CSV mode).")
        if secret_second:
            st.caption("Secrets: `SECOND_SUBMISSION_URL` is set.")
        csv_path = st.text_input(
            "Primary CSV path (Static mode)",
            value=str(DEFAULT_CSV),
            disabled=(primary_mode == "Live (API)"),
            help="Ignored when primary is Live.",
        )
        csv_path_b = st.text_input(
            "Second CSV path (optional — compare tab)",
            value="",
            placeholder="submission_live.csv",
            help="Local path when not using second upload or SECOND_SUBMISSION_URL.",
        )
        st.divider()
        _is_live = primary_mode == "Live (API)"
        live = st.toggle(
            "Auto-refresh (no manual browser refresh)",
            value=not _is_live,
            help="**Live:** default **off** so the browser does not reload while the 30–90s pipeline runs (reloads cancel the run → no charts). Turn on after the first load or set Refresh ≥ 120s. **Static:** on by default.",
        )
        interval_s = st.slider(
            "Refresh every (seconds)",
            min_value=30,
            max_value=300,
            value=120,
            step=15,
            disabled=not live,
            help="Must be **longer** than one live pipeline run (~30–90s) or refreshes interrupt loading and graphs never appear.",
        )
        st.caption(f"Last run: **{datetime.now().strftime('%H:%M:%S')}**")
        st.divider()
        st.markdown("Offline: `python3 run_p3.py -o submission.csv`")
        st.markdown("Live only: `python3 run_p3.py --live -o submission_live.csv`")
        st.markdown("**Both at once:** `make dual` or `python3 run_p3.py --dual`")
        st.markdown("**History backfill:** `python3 scripts/fetch_binance_history.py --days 7` → `data/binance-hist/`")
        st.divider()
        st.subheader("Filters")
        reset_on_change = st.toggle(
            "Reset symbol & type filters when CSV updates",
            value=True,
            help="After path/mtime/row-count changes (e.g. new `run_p3` with all 8 pairs), selections go back to **all** symbols and types so you do not stay stuck on ETH/USDC/XRP only.",
        )

    if live:
        if st_autorefresh is not None:
            st_autorefresh(interval=int(interval_s * 1000), limit=None, key="p3_submission_live")
        else:
            st.sidebar.warning("Install `streamlit-autorefresh` for live mode (`pip install -r requirements.txt`).")

    # Primary: Live (API) | upload > secret URL > local path > bundled sample
    df_a = pd.DataFrame()
    primary_src = ""
    if primary_mode == "Live (API)":
        st.info(
            "**Live** loads in **~30–90s** the first time. **Metrics and charts show below** when the run finishes. "
            "If the main area stays empty, turn **Auto-refresh** **off** in the sidebar — short refresh intervals reload the page and **cancel** the run before charts render."
        )
        if use_live_cache:
            df_a, live_err = run_live_binance_submission_cached(
                int(live_klines),
                int(live_trades),
                _live_env_fingerprint(),
            )
        else:
            df_a, live_err = _run_live_binance_submission_impl(int(live_klines), int(live_trades))
        primary_src = f"live_binance:t={time.time():.3f}|n={len(df_a)}"
        if live_err:
            st.error(f"Live pipeline failed: {live_err}")
        if df_a.empty and BUNDLED_SAMPLE_CSV.is_file():
            sp = str(BUNDLED_SAMPLE_CSV)
            mtime_fb = submission_file_mtime(sp)
            df_fb = load_submission(sp, mtime_fb)
            if not df_fb.empty:
                df_a = df_fb
                primary_src = f"bundled_snapshot:{BUNDLED_SAMPLE_CSV.resolve()}|{mtime_fb}"
                st.warning(
                    "Showing **bundled** `dashboard/sample_submission.csv` from the repo (offline snapshot, not live). "
                    "Set **`LIVE_SPOT_VENUE`** (`okx`, `binance`, or `both`) if live fetch fails. Reboot the app after deploy if you still see old behavior."
                )
    elif upload_primary is not None:
        raw_p = upload_primary.getvalue()
        ck_p = _bytes_cache_key(raw_p, upload_primary.name or "primary.csv")
        df_a = load_submission_bytes(raw_p, ck_p)
        primary_src = f"upload:{ck_p}"
    elif secret_primary:
        df_a = load_submission_url(secret_primary, secret_primary)
        primary_src = f"url:{secret_primary}"
    else:
        mtime_a = submission_file_mtime(csv_path)
        df_a = load_submission(csv_path, mtime_a)
        primary_src = f"path:{Path(csv_path).expanduser().resolve()}|{mtime_a}"
        if df_a.empty and BUNDLED_SAMPLE_CSV.is_file():
            sp = str(BUNDLED_SAMPLE_CSV)
            mtime_s = submission_file_mtime(sp)
            df_a = load_submission(sp, mtime_s)
            primary_src = f"path:{BUNDLED_SAMPLE_CSV.resolve()}|{mtime_s}"

    path_b = csv_path_b.strip()
    df_b = pd.DataFrame()
    second_src = ""
    if upload_second is not None:
        raw_s = upload_second.getvalue()
        ck_s = _bytes_cache_key(raw_s, upload_second.name or "second.csv")
        df_b = load_submission_bytes(raw_s, ck_s)
        second_src = f"upload:{ck_s}"
    elif secret_second:
        df_b = load_submission_url(secret_second, secret_second)
        second_src = f"url:{secret_second}"
    elif path_b:
        mtime_b = submission_file_mtime(path_b)
        df_b = load_submission(path_b, mtime_b)
        second_src = f"path:{Path(path_b).expanduser().resolve()}|{mtime_b}"

    if df_a.empty:
        if primary_mode == "Live (API)":
            st.warning(
                "No rows from live run (pipeline returned empty or fetch failed). "
                "Check network / symbols on your venue, set `BINANCE_SPOT_API` if needed, or switch **Primary source** to **Static CSV**."
            )
        else:
            st.warning(
                "No primary data loaded. Commit `dashboard/sample_submission.csv`, set **Static CSV** path, "
                "add `PRIMARY_SUBMISSION_URL` in Secrets, or use **Advanced → upload**. "
                f"Default path: `{DEFAULT_CSV}`"
            )
        return

    if primary_mode == "Live (API)":
        if str(primary_src).startswith("bundled_snapshot"):
            st.info(
                f"**Bundled snapshot** — {len(df_a)} rows (saved CSV from the repo; switch to **Static CSV** or fix live fetch)."
            )
        else:
            cache_note = " (90s cache on)" if use_live_cache else " (cache off — every refresh reruns ML)"
            st.success(
                f"**Live pipeline** — {len(df_a)} rows · refresh **{interval_s}s**{cache_note} · venue **{live_spot_venue()}**."
            )
    elif live and st_autorefresh is not None:
        st.success(
            f"**Auto-refresh:** every **{interval_s}s** — no browser reload needed. "
            "Symbol filters reset when the file changes if the sidebar toggle is on."
        )

    sig_a = f"{primary_src}|n={len(df_a)}"

    cbtn1, cbtn2 = st.sidebar.columns(2)
    with cbtn1:
        if st.button("All sym · P1", help="Primary tab: select every symbol in the CSV", key="btn_all_sym_p1"):
            st.session_state["p1_sym"] = sorted(df_a["symbol"].dropna().unique().tolist())
            st.rerun()
    with cbtn2:
        if not df_b.empty and st.button(
            "All sym · P2", help="Second tab: select every symbol", key="btn_all_sym_p2"
        ):
            st.session_state["p2_sym"] = sorted(df_b["symbol"].dropna().unique().tolist())
            st.rerun()

    if (path_b or upload_second is not None or secret_second) and df_b.empty:
        hint = path_b or "(upload or SECOND_SUBMISSION_URL)"
        st.info(f"Second source set but no valid rows loaded: `{hint}`")

    if df_b.empty:
        render_submission_panel(
            df_a,
            key_prefix="p1_",
            data_signature=sig_a,
            reset_filters_on_csv_change=reset_on_change,
        )
    else:
        sig_b = f"{second_src}|n={len(df_b)}"
        tab_a, tab_b = st.tabs(["Primary (e.g. student pack)", "Second (e.g. Binance live)"])
        with tab_a:
            render_submission_panel(
                df_a,
                key_prefix="p1_",
                data_signature=sig_a,
                reset_filters_on_csv_change=reset_on_change,
            )
        with tab_b:
            render_submission_panel(
                df_b,
                key_prefix="p2_",
                data_signature=sig_b,
                reset_filters_on_csv_change=reset_on_change,
            )


if __name__ == "__main__":
    main()
