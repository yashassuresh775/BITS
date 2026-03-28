"""Per-minute order book features and rolling baselines (Problem 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from p1.config import OBI_ROLL_SHORT, SPREAD_BASELINE_MIN, SPREAD_ROLL_LONG


def compute_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add OBI, spread_bps, L1 concentration, depth_ratio."""
    out = df.copy()
    bid_cols = [f"bid_size_level{i:02d}" for i in range(1, 11)]
    ask_cols = [f"ask_size_level{i:02d}" for i in range(1, 11)]
    tb = out[bid_cols].sum(axis=1).astype(float)
    ta = out[ask_cols].sum(axis=1).astype(float)
    denom = tb + ta
    out["total_bid"] = tb
    out["total_ask"] = ta
    out["obi"] = np.where(denom > 0, (tb - ta) / denom, 0.0)
    bp = out["bid_price_level01"].astype(float)
    ap = out["ask_price_level01"].astype(float)
    out["spread"] = (ap - bp).clip(lower=0)
    out["spread_bps"] = np.where(bp > 0, (ap - bp) / bp * 10000.0, 0.0)
    b1 = out["bid_size_level01"].astype(float)
    a1 = out["ask_size_level01"].astype(float)
    out["bid_concentration"] = np.where(tb > 0, b1 / tb, 0.0)
    out["ask_concentration"] = np.where(ta > 0, a1 / ta, 0.0)
    out["depth_ratio"] = np.where(a1 > 0, b1 / a1, np.nan)
    out["trade_date"] = out["minute"].dt.date
    return out


def add_rolling_baselines(g: pd.DataFrame) -> pd.DataFrame:
    """Per sec_id: rolling OBI stats, long-window spread baseline + z-score."""
    g = g.sort_values("minute")
    g["obi_roll_mean_10"] = g["obi"].rolling(OBI_ROLL_SHORT, min_periods=3).mean()
    g["obi_roll_std_10"] = g["obi"].rolling(OBI_ROLL_SHORT, min_periods=3).std()
    # Spread: expanding then shift(1) for baseline without lookahead
    mu = g["spread_bps"].shift(1).rolling(SPREAD_ROLL_LONG, min_periods=SPREAD_BASELINE_MIN).mean()
    sd = g["spread_bps"].shift(1).rolling(SPREAD_ROLL_LONG, min_periods=SPREAD_BASELINE_MIN).std()
    g["spread_bps_baseline"] = mu
    g["spread_bps_z"] = (g["spread_bps"] - mu) / sd.replace(0, np.nan)
    g["spread_bps_z"] = g["spread_bps_z"].fillna(0.0)
    return g


def enrich_all(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for sec, g in df.groupby("sec_id", sort=False):
        parts.append(add_rolling_baselines(g))
    return pd.concat(parts, ignore_index=True)


def attach_trade_aggression(fe: pd.DataFrame, tpm: pd.DataFrame | None) -> pd.DataFrame:
    """buy_qty_minute / total_bid — aggressive buying into a thick bid book."""
    if tpm is None or tpm.empty:
        out = fe.copy()
        out["buy_vs_bid_depth"] = np.nan
        return out
    m = fe.merge(tpm, on=["sec_id", "minute"], how="left")
    m["buy_vs_bid_depth"] = np.where(
        m["total_bid"] > 0, m["buy_qty_minute"].fillna(0) / m["total_bid"], np.nan
    )
    return m
