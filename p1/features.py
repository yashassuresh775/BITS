"""Per-minute order book features and rolling baselines (Problem 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from p1.config import OBI_ROLL_SHORT, SPREAD_BASELINE_MIN, SPREAD_ROLL_LONG

_EPS = 1e-9


def _z_vs_rolling_lag(out: pd.DataFrame, col: str, lag_col: str) -> np.ndarray:
    """Z-score of ``col`` vs long rolling mean/std of prior values (per sec_id)."""
    g = out.groupby("sec_id", sort=False)
    out[lag_col] = g[col].shift(1)
    g2 = out.groupby("sec_id", sort=False)
    rlg = g2.rolling(SPREAD_ROLL_LONG, min_periods=SPREAD_BASELINE_MIN)
    mu = rlg[lag_col].mean().to_numpy()
    sd = rlg[lag_col].std().to_numpy()
    v = out[col].to_numpy(dtype=np.float64)
    z = (v - mu) / np.where(sd == 0, np.nan, sd)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def compute_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add OBI, spread_bps, L1 concentration, depth_ratio, cross-level HHI per side."""
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
    # Herfindahl of depth shares across levels 1–10 (high = stacked in few levels; layering-style signal)
    bmat = out[bid_cols].to_numpy(dtype=np.float64)
    amat = out[ask_cols].to_numpy(dtype=np.float64)
    tbn = np.maximum(tb.to_numpy(dtype=np.float64), 1e-12)
    tan = np.maximum(ta.to_numpy(dtype=np.float64), 1e-12)
    pb = bmat / tbn[:, None]
    pa = amat / tan[:, None]
    out["bid_hhi"] = (pb * pb).sum(axis=1)
    out["ask_hhi"] = (pa * pa).sum(axis=1)
    out["trade_date"] = out["minute"].dt.date
    return out


def enrich_all(df: pd.DataFrame) -> pd.DataFrame:
    """Per sec_id: rolling OBI stats, spread z, HHI z, OBI shock vs 10m regime (vectorized)."""
    out = df.sort_values(["sec_id", "minute"], kind="mergesort").reset_index(drop=True)
    g = out.groupby("sec_id", sort=False)
    r10 = g.rolling(OBI_ROLL_SHORT, min_periods=3)
    out = out.copy()
    out["obi_roll_mean_10"] = r10["obi"].mean().to_numpy()
    out["obi_roll_std_10"] = r10["obi"].std().to_numpy()
    out["obi_vs_roll_z"] = (out["obi"] - out["obi_roll_mean_10"]) / (
        out["obi_roll_std_10"].replace(0, np.nan) + _EPS
    )
    out["obi_vs_roll_z"] = np.nan_to_num(
        out["obi_vs_roll_z"].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )

    out["_sp_lag"] = g["spread_bps"].shift(1)
    g2 = out.groupby("sec_id", sort=False)
    rlg = g2.rolling(SPREAD_ROLL_LONG, min_periods=SPREAD_BASELINE_MIN)
    mu = rlg["_sp_lag"].mean().to_numpy()
    sd = rlg["_sp_lag"].std().to_numpy()
    out["spread_bps_baseline"] = mu
    sp = out["spread_bps"].to_numpy(dtype=np.float64)
    z = (sp - mu) / np.where(sd == 0, np.nan, sd)
    out["spread_bps_z"] = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    out["bid_hhi_z"] = _z_vs_rolling_lag(out, "bid_hhi", "_bh_lag")
    out["ask_hhi_z"] = _z_vs_rolling_lag(out, "ask_hhi", "_ah_lag")

    return out.drop(columns=["_sp_lag", "_bh_lag", "_ah_lag"], errors="ignore")


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
