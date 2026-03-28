from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from p3.config import ENSEMBLE_CONTAMINATION

ENSEMBLE_FEATURE_COLS = [
    "qty_z",
    "price_vs_mid_bps",
    "wallet_freq",
    "trades_same_minute",
    "wallets_same_minute",
    "seq_same_side_prev",
    "seq_price_chg",
]


def detect_ensemble_if_lof(enriched: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Rank-fuse IsolationForest + LocalOutlierFactor; flag top contamination fraction.
    """
    t = enriched.copy()
    for c in ENSEMBLE_FEATURE_COLS:
        if c not in t.columns:
            t[c] = 0.0
    t[ENSEMBLE_FEATURE_COLS] = (
        t[ENSEMBLE_FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    if len(t) < 250:
        return pd.DataFrame()

    X = t[ENSEMBLE_FEATURE_COLS].astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    cont = float(min(ENSEMBLE_CONTAMINATION, 0.08))
    iforest = IsolationForest(
        n_estimators=180,
        contamination=cont,
        random_state=42,
        n_jobs=-1,
    )
    iforest.fit(Xs)
    s_if = iforest.score_samples(Xs)
    r_if = 1.0 - pd.Series(s_if, index=t.index).rank(pct=True).values

    n_neighbors = min(35, max(10, len(t) // 50))
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=cont,
        novelty=False,
        n_jobs=-1,
    )
    lof.fit_predict(Xs)
    raw_lof = -lof.negative_outlier_factor_
    r_lof = pd.Series(raw_lof, index=t.index).rank(pct=True).values

    fusion = (r_if + r_lof) / 2.0
    fusion_ser = pd.Series(fusion, index=t.index)
    thresh = fusion_ser.quantile(1.0 - cont)
    hit = t.loc[fusion_ser >= thresh].copy()
    if hit.empty:
        return pd.DataFrame()

    hit["violation_type"] = ""
    hit["detector"] = "ensemble_if_lof"
    hit["score"] = 1
    fv = fusion_ser.loc[hit.index].values
    hit["remarks"] = [
        f"{symbol}: ensemble IF+LOF fused outlier strength={float(v):.3f}."
        for v in fv
    ]
    return hit
