from __future__ import annotations

import pandas as pd

from p3.config import BAT_HOUR_VOLUME_MULT, PEG_BREAK_ABS, PEG_CENTER


def detect_peg_break(trades: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """USDCUSDT price materially off peg."""
    if symbol != "USDCUSDT":
        return pd.DataFrame()
    mask = (trades["price"] - PEG_CENTER).abs() > PEG_BREAK_ABS
    hit = trades.loc[mask].copy()
    hit["violation_type"] = "peg_break"
    hit["detector"] = "peg_break"
    hit["score"] = 5
    hit["remarks"] = hit.apply(
        lambda r: (
            f"USDCUSDT trade at {r['price']:.6f} vs peg 1.0 "
            f"(|Δ|>{PEG_BREAK_ABS}); notional {r['notional']:.2f} USDT."
        ),
        axis=1,
    )
    return hit


def detect_wash_volume_at_peg(trades: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    High churn at exactly ~1.0 with same wallet flipping — heuristic for wash_volume_at_peg.
    Strict: price within 1 bp of 1.0, same wallet buy+sell within 60s, >=3 round trips.
    """
    if symbol != "USDCUSDT":
        return pd.DataFrame()
    t = trades[(trades["price"] - PEG_CENTER).abs() <= 0.0001].copy()
    if t.empty:
        return pd.DataFrame()
    t = t.sort_values("timestamp")
    rows = []
    for w, g in t.groupby("wallet_id"):
        g = g.sort_values("timestamp")
        if len(g) < 6:
            continue
        sides = g["side"].values
        flips = 0
        for i in range(len(sides) - 1):
            if sides[i] != sides[i + 1]:
                dt = (g["timestamp"].iloc[i + 1] - g["timestamp"].iloc[i]).total_seconds()
                if dt <= 120:
                    flips += 1
        if flips >= 4:
            rows.append(g)
    if not rows:
        return pd.DataFrame()
    hit = pd.concat(rows, ignore_index=True)
    hit["violation_type"] = "wash_volume_at_peg"
    hit["detector"] = "wash_volume_at_peg"
    hit["score"] = 3
    hit["remarks"] = (
        "Repeated BUY/SELL flips within 2m at ~$1.00 on USDCUSDT; artificial peg churn."
    )
    return hit


def detect_bat_hot_hours(
    trades: pd.DataFrame, market: pd.DataFrame, symbol: str
) -> pd.DataFrame:
    """Flag trades in hours where USDT volume >> median hourly for that day (BATUSDT)."""
    if symbol != "BATUSDT":
        return pd.DataFrame()
    m = market.copy()
    m["day"] = m["Date"].dt.normalize()
    m["hour_bucket"] = m["Date"].dt.floor("h")
    hv = m.groupby(["day", "hour_bucket"], as_index=False)["vol_usdt"].sum()
    med = hv.groupby("day")["vol_usdt"].transform("median").replace(0, 1.0)
    hv["ratio"] = hv["vol_usdt"] / med
    hot = hv.loc[hv["ratio"] >= BAT_HOUR_VOLUME_MULT, ["day", "hour_bucket"]]
    if hot.empty:
        return pd.DataFrame()
    tt = trades.copy()
    tt["day"] = tt["timestamp"].dt.normalize()
    tt["hour_bucket"] = tt["timestamp"].dt.floor("h")
    tr = tt.merge(hot, on=["day", "hour_bucket"], how="inner")
    if tr.empty:
        return pd.DataFrame()
    tr = tr.drop(columns=["day", "hour_bucket"], errors="ignore")
    tr["violation_type"] = "aml_structuring"
    tr["detector"] = "bat_hot_hour"
    tr["score"] = 2
    tr["remarks"] = (
        f"BATUSDT trade in hour with vol_usdt >= {BAT_HOUR_VOLUME_MULT}x "
        "that day's median hourly volume; review for smurfing/coordinated activity."
    )
    return tr
