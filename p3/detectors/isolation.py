from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from p3.config import IF_CONTAMINATION


def isolation_candidates(trades_enriched: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """IsolationForest on engineered trade features (recall layer)."""
    t = trades_enriched.dropna(
        subset=["qty_z", "price_vs_mid_bps", "wallet_freq"]
    ).copy()
    if len(t) < 200:
        return pd.DataFrame()
    X = t[["qty_z", "price_vs_mid_bps", "wallet_freq"]].astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    if len(X) < 200:
        return pd.DataFrame()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = IsolationForest(
        n_estimators=200,
        contamination=min(IF_CONTAMINATION, 0.1),
        random_state=42,
        n_jobs=-1,
    )
    pred = clf.fit_predict(Xs)
    scores = clf.score_samples(Xs)
    t2 = t.loc[X.index]
    t2 = t2.assign(_if_pred=pred, _if_score=scores)
    hit = t2[t2["_if_pred"] == -1].copy()
    if hit.empty:
        return pd.DataFrame()
    hit["violation_type"] = ""
    hit["detector"] = "isolation_forest"
    hit["score"] = 1
    hit["remarks"] = hit["_if_score"].apply(
        lambda s: (
            f"{symbol}: IF outlier on qty_z, price_vs_mid_bps, wallet_freq "
            f"(score={float(s):.3f})."
        )
    )
    hit = hit.drop(columns=["_if_pred", "_if_score"], errors="ignore")
    return hit
