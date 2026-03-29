"""P2 signal pipeline (pandas). Import from ``p2.insider_signals`` in application code."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --- column normalisation ---

_OHLCV_ALIASES = {
    "trade_date": ["trade_date", "TradeDate", "timestamp", "Timestamp", "date", "Date"],
    "sec_id": ["sec_id", "SecId", "SEC_ID"],
    "close": ["close", "Close", "adj_close", "AdjClose"],
    "volume": ["volume", "Volume", "vol"],
}

_TRADE_ALIASES = {
    "trade_date": ["trade_date", "TradeDate", "timestamp", "Timestamp", "date", "Date"],
    "sec_id": ["sec_id", "SecId", "SEC_ID"],
    "trade_id": ["trade_id", "TradeId", "id"],
    "quantity": ["quantity", "Quantity", "qty", "size"],
    "price": ["price", "Price"],
    "side": ["side", "Side"],
    "trader_id": ["trader_id", "TraderId", "client_id", "account_id"],
}


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    lower = {x.lower(): x for x in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


from p2.sec_source_url import (
    SEC_SEARCH_FALLBACK_URL,
    extract_cik_from_text,
    keep_precomputed_source_url,
    resolve_p2_source_url,
    sec_edgar_browse_8k_url,
)


def coerce_p2_signal_columns(
    df: pd.DataFrame,
    *,
    source_url_prefer_listing_ticker: bool = False,
) -> pd.DataFrame:
    """
    Ensure ``source_url``, ``suspicious_window_start``, and ``pre_drift_flag`` are always
    populated for CSV round-trip and Streamlit ``st.dataframe`` (NaN becomes ``None`` in UI).
    """
    if df.empty:
        return df
    out = df.copy()
    fb = SEC_SEARCH_FALLBACK_URL
    if "source_url" not in out.columns:
        out["source_url"] = fb
    else:
        raw = out["source_url"]
        strv = raw.fillna("").astype(str).str.strip()
        bad = raw.isna() | strv.isin(("", "nan", "None", "NaN", "<NA>", "null"))
        out["source_url"] = strv.mask(bad, fb)
    if "headline" in out.columns:
        hl = out["headline"].fillna("").astype(str)
        _tk = out["ticker"].fillna("").astype(str) if "ticker" in out.columns else pd.Series("", index=out.index)
        _su = out["source_url"].fillna("").astype(str)
        out["source_url"] = [
            su
            if keep_precomputed_source_url(su)
            else resolve_p2_source_url(
                su, h, "", str(tk), prefer_listing_ticker=source_url_prefer_listing_ticker
            )
            for su, h, tk in zip(_su, hl, _tk)
        ]
    ed = pd.to_datetime(out["event_date"], errors="coerce")
    fb_sw = (ed - pd.Timedelta(days=7)).dt.strftime("%Y-%m-%d").fillna("n/a")
    if "suspicious_window_start" not in out.columns:
        out["suspicious_window_start"] = fb_sw
    else:
        sw = pd.to_datetime(out["suspicious_window_start"], errors="coerce")
        sw_str = sw.dt.strftime("%Y-%m-%d")
        out["suspicious_window_start"] = sw_str.where(sw.notna(), fb_sw).fillna("n/a")
    if "pre_drift_flag" in out.columns:
        out["pre_drift_flag"] = pd.to_numeric(out["pre_drift_flag"], errors="coerce").fillna(0).astype(int)
    return out


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    td = _pick_col(out, _OHLCV_ALIASES["trade_date"])
    sid = _pick_col(out, _OHLCV_ALIASES["sec_id"])
    cl = _pick_col(out, _OHLCV_ALIASES["close"])
    vol = _pick_col(out, _OHLCV_ALIASES["volume"])
    if not all([td, sid, cl, vol]):
        raise ValueError(
            "ohlcv.csv needs trade_date, sec_id, close, volume (case-insensitive aliases ok). "
            f"Got columns: {list(df.columns)}"
        )
    out = out.rename(columns={td: "trade_date", sid: "sec_id", cl: "close", vol: "volume"})
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    out["sec_id"] = pd.to_numeric(out["sec_id"], errors="coerce")
    out = out.dropna(subset=["trade_date", "sec_id"])
    return out.sort_values(["sec_id", "trade_date"])


def normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    td = _pick_col(out, _TRADE_ALIASES["trade_date"])
    sid = _pick_col(out, _TRADE_ALIASES["sec_id"])
    tid = _pick_col(out, _TRADE_ALIASES["trade_id"])
    qty = _pick_col(out, _TRADE_ALIASES["quantity"])
    pr = _pick_col(out, _TRADE_ALIASES["price"])
    side = _pick_col(out, _TRADE_ALIASES["side"])
    tr = _pick_col(out, _TRADE_ALIASES["trader_id"])
    if not all([td, sid, qty, pr, side]):
        raise ValueError(
            "trade_data.csv needs timestamp/trade_date, sec_id, quantity, price, side. "
            f"Got columns: {list(df.columns)}"
        )
    ren = {td: "trade_date", sid: "sec_id", qty: "quantity", pr: "price", side: "side"}
    if tid:
        ren[tid] = "trade_id"
    out = out.rename(columns=ren)
    if not tid:
        out["trade_id"] = "synth_" + out.index.astype(str)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    out["sec_id"] = pd.to_numeric(out["sec_id"], errors="coerce")
    if tr:
        out = out.rename(columns={tr: "trader_id"})
    else:
        out["trader_id"] = np.nan
    out["notional"] = out["quantity"].astype(float) * out["price"].astype(float)
    return out.dropna(subset=["trade_date", "sec_id"])


def _prep_ohlcv_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Per sec_id: daily return, 15d rolling mean/std volume and return (shifted, no leak)."""
    pieces: list[pd.DataFrame] = []
    for _, x in ohlcv.groupby("sec_id", sort=False):
        x = x.sort_values("trade_date").copy()
        x["daily_ret"] = x["close"].pct_change()
        x["vol_15d_mean"] = x["volume"].shift(1).rolling(15, min_periods=5).mean()
        x["vol_15d_std"] = x["volume"].shift(1).rolling(15, min_periods=5).std()
        x["ret_15d_mean"] = x["daily_ret"].shift(1).rolling(15, min_periods=5).mean()
        x["ret_15d_std"] = x["daily_ret"].shift(1).rolling(15, min_periods=5).std()
        x["volume_z"] = (x["volume"] - x["vol_15d_mean"]) / x["vol_15d_std"].replace(0, np.nan)
        pieces.append(x)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _last_n_trading_dates_before(
    dates: np.ndarray, cutoff: pd.Timestamp, n: int
) -> list[pd.Timestamp]:
    """dates sorted ascending; return last n dates strictly before cutoff."""
    ts = pd.Timestamp(cutoff).normalize()
    below = dates[dates < np.datetime64(ts)]
    if len(below) < n:
        return []
    return [pd.Timestamp(x) for x in below[-n:]]


def _trading_dates_before(dates: np.ndarray, cutoff: pd.Timestamp) -> list[pd.Timestamp]:
    """All OHLCV dates strictly before filing (ascending)."""
    ts = pd.Timestamp(cutoff).normalize()
    below = dates[dates < np.datetime64(ts)]
    return [pd.Timestamp(x) for x in below]


def _calendar_pre_window(file_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Fallback T-7..T-1 calendar (not trading) when OHLCV is missing or empty."""
    fd = pd.Timestamp(file_date).normalize()
    return fd - pd.Timedelta(days=7), fd - pd.Timedelta(days=1)


def compute_pre_drift_flags(
    ohlcv_feat: pd.DataFrame,
    filings: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each filing row (sec_id, file_date), compute:
    - volume_z on T-1, T-2
    - cumulative abnormal return over T-5..T-1 vs 15d mean daily return
    - pre_drift_flag: vol z > 3 on T-1 or T-2, or CAR > 2 * sqrt(5) * ret_15d_std at T-1
    """
    rows: list[dict] = []
    o = ohlcv_feat.sort_values(["sec_id", "trade_date"])
    for sec in filings["sec_id"].dropna().unique():
        fo = o[o["sec_id"] == sec]
        fsub = filings[filings["sec_id"] == sec]
        if fo.empty:
            for _, frow in fsub.iterrows():
                fd = pd.Timestamp(frow["file_date"]).normalize()
                ws, we = _calendar_pre_window(fd)
                rows.append(
                    {
                        **frow.to_dict(),
                        "pre_drift_flag": 0,
                        "suspicious_window_start": ws,
                        "_pre_window_end": we,
                        "remarks": f"No OHLCV rows for sec_id={sec}; window is calendar T-7..T-1 for trade scan.",
                        "volume_z_T-1": np.nan,
                        "volume_z_T-2": np.nan,
                        "car_5d": np.nan,
                    }
                )
            continue
        dates = fo["trade_date"].values
        for _, frow in fsub.iterrows():
            fd = pd.Timestamp(frow["file_date"]).normalize()
            last5 = _last_n_trading_dates_before(dates, fd, 5)
            if len(last5) < 5:
                prior = _trading_dates_before(dates, fd)
                if len(prior) == 0:
                    ws, we = _calendar_pre_window(fd)
                    rows.append(
                        {
                            **frow.to_dict(),
                            "pre_drift_flag": 0,
                            "suspicious_window_start": ws,
                            "_pre_window_end": we,
                            "remarks": (
                                f"sec_id {sec}: no trading days before filing {fd.date()} in OHLCV; "
                                "window is calendar T-7..T-1 for trade scan."
                            ),
                            "volume_z_T-1": np.nan,
                            "volume_z_T-2": np.nan,
                            "car_5d": np.nan,
                        }
                    )
                    continue
                lastk = prior[-min(5, len(prior)) :]
                suspicious_start = lastk[0]
                pre_end = lastk[-1]
                sub = fo.set_index("trade_date")
                d_t1, d_t2 = lastk[-1], lastk[-2] if len(lastk) >= 2 else lastk[-1]
                vz1 = float(sub.loc[d_t1, "volume_z"]) if d_t1 in sub.index else np.nan
                vz2 = float(sub.loc[d_t2, "volume_z"]) if d_t2 in sub.index else np.nan
                car = 0.0
                for d in lastk:
                    if d not in sub.index:
                        continue
                    r = float(sub.loc[d, "daily_ret"])
                    if np.isnan(r):
                        continue
                    mu = float(sub.loc[d, "ret_15d_mean"])
                    if np.isnan(mu):
                        mu = 0.0
                    car += r - mu
                sig = float(sub.loc[d_t1, "ret_15d_std"]) if d_t1 in sub.index else np.nan
                car_thresh = 2.0 * (np.sqrt(5.0) * sig) if sig and not np.isnan(sig) else np.inf
                vol_hit = (not np.isnan(vz1) and vz1 > 3) or (not np.isnan(vz2) and vz2 > 3)
                car_hit = bool(not np.isnan(car) and not np.isnan(sig) and car > car_thresh)
                pre_flag = int(vol_hit or car_hit)
                rows.append(
                    {
                        **frow.to_dict(),
                        "pre_drift_flag": pre_flag,
                        "suspicious_window_start": suspicious_start,
                        "_pre_window_end": pre_end,
                        "remarks": (
                            f"Partial history ({len(lastk)} trading days before filing vs 5 ideal). "
                        ),
                        "volume_z_T-1": vz1,
                        "volume_z_T-2": vz2,
                        "car_5d": car,
                    }
                )
                continue
            d_t1, d_t2 = last5[-1], last5[-2]
            sub = fo.set_index("trade_date")
            vz1 = float(sub.loc[d_t1, "volume_z"]) if d_t1 in sub.index else np.nan
            vz2 = float(sub.loc[d_t2, "volume_z"]) if d_t2 in sub.index else np.nan
            car = 0.0
            for d in last5:
                if d not in sub.index:
                    continue
                r = float(sub.loc[d, "daily_ret"])
                if np.isnan(r):
                    continue
                mu = float(sub.loc[d, "ret_15d_mean"])
                if np.isnan(mu):
                    mu = 0.0
                car += r - mu
            sig = float(sub.loc[d_t1, "ret_15d_std"]) if d_t1 in sub.index else np.nan
            car_thresh = 2.0 * (np.sqrt(5.0) * sig) if sig and not np.isnan(sig) else np.inf
            vol_hit = (vz1 > 3) or (vz2 > 3)
            car_hit = bool(not np.isnan(car) and not np.isnan(sig) and car > car_thresh)
            pre_flag = int(vol_hit or car_hit)
            suspicious_start = last5[0]
            pre_end = last5[-1]
            rows.append(
                {
                    **frow.to_dict(),
                    "pre_drift_flag": pre_flag,
                    "suspicious_window_start": suspicious_start,
                    "_pre_window_end": pre_end,
                    "remarks": "",  # filled in enrich_remarks_with_trades
                    "volume_z_T-1": vz1,
                    "volume_z_T-2": vz2,
                    "car_5d": car,
                }
            )
    return pd.DataFrame(rows)


def enrich_remarks_with_trades(
    flagged: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.Series:
    """Use ``trade_data`` ``trader_id`` + notional in the pre-filing window vs prior history (organizer tip)."""
    remarks = []
    for _, row in flagged.iterrows():
        base = str(row.get("remarks", "") or "")
        sec = row["sec_id"]
        ws = row.get("suspicious_window_start")
        pre_end = row.get("_pre_window_end")
        if pd.isna(ws) or pd.isna(pre_end):
            remarks.append(base.strip())
            continue
        ws = pd.Timestamp(ws).normalize()
        pre_end = pd.Timestamp(pre_end).normalize()
        tsub = trades[
            (trades["sec_id"] == sec)
            & (trades["trade_date"] >= ws)
            & (trades["trade_date"] <= pre_end)
        ]
        hist = trades[(trades["sec_id"] == sec) & (trades["trade_date"] < ws)]
        hints: list[str] = []
        if "trader_id" in tsub.columns and tsub["trader_id"].notna().any():
            for tr, g in tsub.groupby("trader_id"):
                if pd.isna(tr):
                    continue
                h = hist[hist["trader_id"] == tr]
                med = h["notional"].median() if len(h) else 0.0
                mx = g["notional"].max()
                if med > 0 and mx > 8 * med and mx > 5000:
                    hints.append(
                        f"trader {tr} max notional {mx:,.0f} vs prior median {med:,.0f} in pre-window"
                    )
                elif med == 0 and mx > 25_000:
                    hints.append(f"trader {tr} large first activity notional {mx:,.0f} before filing")
        tip = "; ".join(hints[:3])
        vz1 = row.get("volume_z_T-1")
        vz2 = row.get("volume_z_T-2")
        car = row.get("car_5d")
        parts = []
        if row.get("pre_drift_flag") == 1:
            if vz1 is not None and not np.isnan(vz1) and vz1 > 3:
                parts.append(f"volume z-score T-1={vz1:.2f}")
            if vz2 is not None and not np.isnan(vz2) and vz2 > 3:
                parts.append(f"volume z-score T-2={vz2:.2f}")
            if car is not None and not np.isnan(car):
                parts.append(f"5d cumulative excess return≈{car:.4f}")
        narrative = "Pre-announcement screen: " + (
            "; ".join(parts) if parts else "no strong volume/return spike vs 15d baseline"
        )
        if tip:
            narrative += f". Trade-level: {tip}."
        narrative += f" Event: {row.get('event_type', '')} — {str(row.get('headline', ''))[:200]}"
        if base.strip():
            narrative = base.strip() + " " + narrative
        remarks.append(narrative)
    return pd.Series(remarks, index=flagged.index)


def build_p2_signals(
    ohlcv: pd.DataFrame,
    trades: pd.DataFrame,
    filings: pd.DataFrame,
    time_to_run_s: float,
    *,
    ma_only: bool = False,
) -> pd.DataFrame:
    empty_cols = [
        "sec_id",
        "ticker",
        "event_date",
        "event_type",
        "headline",
        "source_url",
        "pre_drift_flag",
        "suspicious_window_start",
        "remarks",
        "time_to_run",
    ]
    if filings is None or filings.empty:
        return pd.DataFrame(columns=empty_cols)

    o = normalize_ohlcv(ohlcv)
    t = normalize_trades(trades)
    f = filings.copy()
    if "file_date" in f.columns:
        f["file_date"] = pd.to_datetime(f["file_date"], errors="coerce")
    f = f.dropna(subset=["sec_id", "file_date"])
    if f.empty:
        return pd.DataFrame(columns=empty_cols)
    if ma_only and "event_type" in f.columns:
        f = f[f["event_type"].astype(str) == "merger"].copy()
        if f.empty:
            return pd.DataFrame(columns=empty_cols)
    o_feat = _prep_ohlcv_features(o)
    merged = compute_pre_drift_flags(o_feat, f).reset_index(drop=True)
    if merged.empty:
        return pd.DataFrame(columns=empty_cols)
    merged["remarks"] = enrich_remarks_with_trades(merged, t)

    # source_url: direct Archives link if present; else CIK from headline/entity → browse-edgar 8-K
    _fu = (
        merged["filing_url"].fillna("").astype(str).str.strip()
        if "filing_url" in merged.columns
        else pd.Series("", index=merged.index)
    )
    if "source_url" in merged.columns:
        _su = merged["source_url"].fillna("").astype(str).str.strip()
        _fu = _fu.where(_fu.ne(""), _su)
    _ent = (
        merged["entity_name"].fillna("").astype(str)
        if "entity_name" in merged.columns
        else pd.Series("", index=merged.index)
    )
    _hl = merged["headline"].fillna("").astype(str)
    _tk = (
        merged["ticker"].fillna("").astype(str)
        if "ticker" in merged.columns
        else pd.Series("", index=merged.index)
    )
    source_urls = pd.Series(
        [
            resolve_p2_source_url(fu, hl, en, tk, prefer_listing_ticker=False)
            for fu, hl, en, tk in zip(_fu, _hl, _ent, _tk)
        ],
        index=merged.index,
    )

    sw = pd.to_datetime(merged["suspicious_window_start"], errors="coerce")
    ev = merged["file_date"]
    sw = sw.fillna(ev - pd.Timedelta(days=7))
    sw_str = sw.dt.strftime("%Y-%m-%d")
    fb_sw = (ev - pd.Timedelta(days=7)).dt.strftime("%Y-%m-%d")
    suspicious_out = sw_str.where(sw.notna(), fb_sw).fillna("n/a")

    pflag = pd.to_numeric(merged["pre_drift_flag"], errors="coerce").fillna(0).astype(int)

    out = pd.DataFrame(
        {
            "sec_id": pd.to_numeric(merged["sec_id"], errors="coerce").astype("Int64"),
            "ticker": _tk.str.upper().str.strip(),
            "event_date": merged["file_date"].dt.strftime("%Y-%m-%d"),
            "event_type": merged["event_type"].fillna("other"),
            "headline": merged["headline"].fillna("").map(lambda x: re.sub(r"[\r\n]+", " ", str(x))[:300]),
            "source_url": source_urls,
            "pre_drift_flag": pflag,
            "suspicious_window_start": suspicious_out,
            "remarks": merged["remarks"],
            "time_to_run": round(time_to_run_s, 3),
        }
    )
    # M&A-related rows first (organizer: clearest pre-announcement drift)
    pri = out["event_type"].astype(str).map(lambda e: 0 if e == "merger" else 1)
    out = out.assign(_pri=pri).sort_values(["_pri", "event_date", "sec_id"]).drop(columns="_pri")
    return coerce_p2_signal_columns(out)
