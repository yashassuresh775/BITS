"""
Problem 3 submission explorer — run from repo root:

  streamlit run dashboard/app.py
"""

from __future__ import annotations

import re
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
DEFAULT_CSV = REPO_ROOT / "submission.csv"
ML_P_RE = re.compile(r"ml_rank_p=([\d.]+)")


def submission_file_mtime(path: str) -> float:
    p = Path(path).expanduser().resolve()
    try:
        return p.stat().st_mtime
    except OSError:
        return -1.0


@st.cache_data(show_spinner="Loading submission…")
def load_submission(path: str, _file_mtime: float) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for col in ("symbol", "date", "trade_id", "violation_type", "remarks"):
        if col not in df.columns:
            st.error(f"Missing column `{col}` in CSV.")
            return pd.DataFrame()
    df["remarks"] = df["remarks"].fillna("").astype(str)
    df["violation_type"] = df["violation_type"].fillna("").astype(str)
    ml = df["remarks"].str.extract(ML_P_RE, expand=False)
    df["ml_rank_p"] = pd.to_numeric(ml, errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


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

    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("**By symbol**")
        sym_counts = view.groupby("symbol", observed=True).size().sort_values(ascending=False)
        st.bar_chart(sym_counts)
    with ch2:
        st.markdown("**By violation_type**")
        vt = view["violation_type"].replace("", "(empty)")
        st.bar_chart(vt.value_counts())

    ml = view["ml_rank_p"].dropna()
    if len(ml) > 1:
        st.markdown("**ML re-rank score (`ml_rank_p`) — bin counts**")
        hist, edges = np.histogram(ml.to_numpy(dtype=float), bins=16)
        mids = (edges[:-1] + edges[1:]) / 2.0
        st.bar_chart(pd.Series(hist, index=mids))

    if view["date"].notna().any():
        st.markdown("**Flags per day**")
        daily = view.dropna(subset=["date"]).copy()
        daily["_d"] = daily["date"].dt.date
        st.line_chart(daily.groupby("_d").size())

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
    st.title("BITS — Problem 3 submission explorer")
    st.caption("Visualize `submission.csv` from the crypto anomaly pipeline.")

    with st.sidebar:
        st.header("Data")
        csv_path = st.text_input(
            "Primary submission CSV",
            value=str(DEFAULT_CSV),
            help="e.g. submission.csv or submission_offline.csv",
        )
        csv_path_b = st.text_input(
            "Second CSV (optional — compare with live)",
            value="",
            placeholder="submission_live.csv",
            help="After `python3 run_p3.py --dual` (or `make dual`), set to submission_live.csv for two tabs.",
        )
        st.divider()
        live = st.toggle(
            "Auto-refresh (no manual browser refresh)",
            value=True,
            help="Reloads this page on an interval so charts pick up updated CSVs. Turn off for a static file.",
        )
        interval_s = st.slider(
            "Refresh every (seconds)",
            min_value=2,
            max_value=120,
            value=5,
            step=1,
            disabled=not live,
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

    mtime_a = submission_file_mtime(csv_path)
    df_a = load_submission(csv_path, mtime_a)

    path_b = csv_path_b.strip()
    df_b = pd.DataFrame()
    if path_b:
        mtime_b = submission_file_mtime(path_b)
        df_b = load_submission(path_b, mtime_b)

    if df_a.empty:
        st.warning(
            "No data loaded for the primary CSV. Run the pipeline first or fix the path. "
            f"Default expects: `{DEFAULT_CSV}`"
        )
        return

    if live and st_autorefresh is not None:
        st.success(
            f"**Auto-refresh:** every **{interval_s}s** — no browser reload needed. "
            "Symbol filters reset when the file changes if the sidebar toggle is on."
        )

    rp_a = str(Path(csv_path).expanduser().resolve())
    sig_a = f"{rp_a}|{mtime_a}|{len(df_a)}"

    cbtn1, cbtn2 = st.sidebar.columns(2)
    with cbtn1:
        if st.button("All sym · P1", help="Primary tab: select every symbol in the CSV", key="btn_all_sym_p1"):
            st.session_state["p1_sym"] = sorted(df_a["symbol"].dropna().unique().tolist())
            st.rerun()
    with cbtn2:
        if path_b and not df_b.empty and st.button(
            "All sym · P2", help="Second tab: select every symbol", key="btn_all_sym_p2"
        ):
            st.session_state["p2_sym"] = sorted(df_b["symbol"].dropna().unique().tolist())
            st.rerun()

    if path_b and df_b.empty:
        st.info(f"Second CSV path set but no valid rows loaded: `{path_b}`")

    if not path_b:
        render_submission_panel(
            df_a,
            key_prefix="p1_",
            data_signature=sig_a,
            reset_filters_on_csv_change=reset_on_change,
        )
    elif df_b.empty:
        render_submission_panel(
            df_a,
            key_prefix="p1_",
            data_signature=sig_a,
            reset_filters_on_csv_change=reset_on_change,
        )
    else:
        rp_b = str(Path(path_b).expanduser().resolve())
        sig_b = f"{rp_b}|{mtime_b}|{len(df_b)}"
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
