from __future__ import annotations

import numpy as np
import pandas as pd

from p3.config import DUMP_MIN_RETURN, PUMP_LOOKBACK, PUMP_MIN_RETURN


def detect_pump_dump_trades(
    trades: pd.DataFrame, market: pd.DataFrame, symbol: str
) -> pd.DataFrame:
    """Mark trades in minutes that fall inside pump-then-dump bar pattern."""
    m = market.sort_values("Date").reset_index(drop=True)
    c = m["Close"].values
    v = m["vol_usdt"].values.astype(float)
    tidx = []
    for i in range(PUMP_LOOKBACK, len(m) - 2):
        window = c[i - PUMP_LOOKBACK : i + 1]
        ret = (window[-1] - window[0]) / max(window[0], 1e-12)
        vol_seg = v[i - PUMP_LOOKBACK : i + 1]
        vol_ok = vol_seg[-1] > np.median(vol_seg[:-1]) * 1.5 if len(vol_seg) > 2 else True
        if ret < PUMP_MIN_RETURN * 5:
            continue
        if not vol_ok:
            continue
        dump = (c[i + 2] - c[i]) / max(c[i], 1e-12)
        if dump > DUMP_MIN_RETURN:
            continue
        start = m["Date"].iloc[i - PUMP_LOOKBACK]
        end = m["Date"].iloc[i + 2]
        tidx.append((start, end))
    if not tidx:
        return pd.DataFrame()
    hit_parts = []
    for start, end in tidx:
        mask = (trades["timestamp"] >= start) & (trades["timestamp"] <= end)
        hit_parts.append(trades.loc[mask])
    hit = pd.concat(hit_parts, ignore_index=True)
    if hit.empty:
        return pd.DataFrame()
    hit["violation_type"] = "pump_and_dump"
    hit["detector"] = "pump_dump_bars"
    hit["score"] = 2
    hit["remarks"] = (
        f"{symbol}: trade falls in window with sustained rise then sharp drop in "
        f"{PUMP_LOOKBACK}m+ bars (pump-and-dump footprint)."
    )
    return hit


def detect_cross_pair_divergence(
    trades: pd.DataFrame,
    market: pd.DataFrame,
    btc_market: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """Alt pair return diverges from BTC return same minute (heuristic)."""
    if symbol in ("BTCUSDT", "USDCUSDT"):
        return pd.DataFrame()
    m = market.set_index("Date").sort_index()
    b = btc_market.set_index("Date").sort_index()
    common = m.index.intersection(b.index)
    if len(common) < 50:
        return pd.DataFrame()
    m = m.loc[common]
    b = b.loc[common]
    rc = m["Close"].pct_change()
    rb = b["Close"].pct_change()
    div = rc - rb
    z = (div - div.rolling(120, min_periods=30).mean()) / (
        div.rolling(120, min_periods=30).std().replace(0, np.nan)
    )
    bad = z[z.abs() > 3.5].index
    if len(bad) == 0:
        return pd.DataFrame()
    hit_parts = []
    for minute in bad:
        hit_parts.append(trades[trades["minute"] == minute])
    hit = pd.concat(hit_parts, ignore_index=True)
    if hit.empty:
        return pd.DataFrame()
    hit["violation_type"] = "cross_pair_divergence"
    hit["detector"] = "cross_btc_div"
    hit["score"] = 2
    hit["remarks"] = (
        f"{symbol}: minute return vs BTC z>3.5 — idiosyncratic move vs correlated leader."
    )
    return hit


def detect_spoofing_proxy(trades: pd.DataFrame, market: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Large bar range + quick reversion next bar; trades in spike minute."""
    m = market.sort_values("Date").reset_index(drop=True)
    rng = (m["High"] - m["Low"]) / m["Close"].replace(0, np.nan)
    med = rng.rolling(60, min_periods=20).median()
    spike = rng > med * 4
    rev = m["Close"].diff(-1)  # next bar - current? use shift
    fwd = m["Close"].shift(-1) - m["Close"]
    reversion = (fwd / m["Close"].replace(0, np.nan)).abs() > 0.0015
    flag = spike & reversion
    minutes = m.loc[flag, "Date"]
    if minutes.empty:
        return pd.DataFrame()
    hit = trades[trades["minute"].isin(minutes)].copy()
    if hit.empty:
        return pd.DataFrame()
    hit["violation_type"] = "spoofing"
    hit["detector"] = "spoofing_proxy"
    hit["score"] = 2
    hit["remarks"] = (
        f"{symbol}: wide 1m range vs local median with quick mean reversion — spoofing-like proxy."
    )
    return hit
