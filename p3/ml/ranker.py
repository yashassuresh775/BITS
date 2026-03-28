from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from p3.config import (
    MAX_SUBMISSION_ROWS,
    ML_MAX_DEPTH,
    ML_MAX_ITER,
    ML_MIN_POSITIVES,
    ML_MIN_SAMPLES_LEAF,
    ML_NEG_CAP_PER_SYMBOL,
    ML_NEG_MULTIPLIER,
    ML_RANDOM_STATE,
    TRUSTED_DETECTORS,
)
from p3.ml.ensemble_od import ENSEMBLE_FEATURE_COLS

RANKER_NUMERIC = list(ENSEMBLE_FEATURE_COLS) + ["log1p_notional", "log1p_qty"]
FEATURE_NAMES = RANKER_NUMERIC + ["symbol_code"]


def _prepare_enriched(en: pd.DataFrame) -> pd.DataFrame:
    e = en.copy()
    for c in ENSEMBLE_FEATURE_COLS:
        if c not in e.columns:
            e[c] = 0.0
    cols = [c for c in RANKER_NUMERIC if c in e.columns]
    e[cols] = e[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "notional" not in e.columns:
        e["notional"] = e["price"] * e["quantity"]
    e["log1p_notional"] = np.log1p(e["notional"].clip(lower=0).astype(float))
    e["log1p_qty"] = np.log1p(e["quantity"].clip(lower=0).astype(float))
    return e


def _rows_for_training(
    enriched_map: dict[str, pd.DataFrame],
    hits_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray] | None:
    trusted = hits_raw[hits_raw["detector"].isin(TRUSTED_DETECTORS)]
    if trusted.empty:
        return None
    pos_keys = trusted[["symbol", "trade_id"]].drop_duplicates()
    if len(pos_keys) < ML_MIN_POSITIVES:
        return None

    sym_to_code = {s: i for i, s in enumerate(sorted(enriched_map.keys()))}

    feat_rows: list[dict] = []
    labels: list[int] = []

    for _, r in pos_keys.iterrows():
        sym, tid = str(r["symbol"]), str(r["trade_id"])
        en = enriched_map.get(sym)
        if en is None:
            continue
        enp = _prepare_enriched(en)
        m = enp[enp["trade_id"].astype(str) == tid]
        if m.empty:
            continue
        row = m.iloc[0]
        d = {c: float(row.get(c, 0.0)) for c in RANKER_NUMERIC}
        d["symbol_code"] = float(sym_to_code.get(sym, 0))
        feat_rows.append(d)
        labels.append(1)

    if len(feat_rows) < ML_MIN_POSITIVES:
        return None

    pos_ids_by_sym: dict[str, set[str]] = {}
    for _, r in pos_keys.iterrows():
        sym, tid = str(r["symbol"]), str(r["trade_id"])
        pos_ids_by_sym.setdefault(sym, set()).add(tid)

    rng = np.random.default_rng(ML_RANDOM_STATE)
    for sym, en in enriched_map.items():
        enp = _prepare_enriched(en)
        pos_ids = pos_ids_by_sym.get(sym, set())
        n_pos_sym = len(pos_ids)
        neg_target = min(
            ML_NEG_CAP_PER_SYMBOL,
            max(50, ML_NEG_MULTIPLIER * max(n_pos_sym, 1)),
        )
        pool = enp[~enp["trade_id"].astype(str).isin(pos_ids)]
        if pool.empty:
            continue
        take = min(neg_target, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        samp = pool.iloc[idx]
        for _, row in samp.iterrows():
            d = {c: float(row.get(c, 0.0)) for c in RANKER_NUMERIC}
            d["symbol_code"] = float(sym_to_code.get(sym, 0))
            feat_rows.append(d)
            labels.append(0)

    X = pd.DataFrame(feat_rows)
    y = np.array(labels, dtype=np.int8)
    return X, y


def _fallback_cap(hits: pd.DataFrame) -> pd.DataFrame:
    return hits.sort_values("score", ascending=False).head(MAX_SUBMISSION_ROWS)


def ml_rerank(
    hits_deduped: pd.DataFrame,
    enriched_map: dict[str, pd.DataFrame],
    hits_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Train HistGradientBoosting on pseudo-labels; score deduped hits; cap rows.
    """
    if hits_deduped.empty:
        return hits_deduped

    train = _rows_for_training(enriched_map, hits_raw)
    if train is None:
        return _fallback_cap(hits_deduped)

    X_train, y_train = train
    if len(np.unique(y_train)) < 2:
        return _fallback_cap(hits_deduped)

    X_train = X_train[FEATURE_NAMES].astype(float)

    clf = HistGradientBoostingClassifier(
        max_iter=ML_MAX_ITER,
        max_depth=ML_MAX_DEPTH,
        min_samples_leaf=ML_MIN_SAMPLES_LEAF,
        random_state=ML_RANDOM_STATE,
        class_weight="balanced",
    )
    try:
        clf.fit(X_train, y_train)
    except ValueError:
        return _fallback_cap(hits_deduped)

    h = hits_deduped.reset_index(drop=True)
    sym_codes = {s: j for j, s in enumerate(sorted(enriched_map.keys()))}
    infer_rows: list[dict] = []
    for k in range(len(h)):
        r = h.iloc[k]
        sym, tid = str(r["symbol"]), str(r["trade_id"])
        en = enriched_map.get(sym)
        if en is None:
            return _fallback_cap(hits_deduped)
        enp = _prepare_enriched(en)
        m = enp[enp["trade_id"].astype(str) == tid]
        if m.empty:
            return _fallback_cap(hits_deduped)
        row = m.iloc[0]
        d = {c: float(row.get(c, 0.0)) for c in RANKER_NUMERIC}
        d["symbol_code"] = float(sym_codes.get(sym, 0))
        infer_rows.append(d)

    X_hit = pd.DataFrame(infer_rows)[FEATURE_NAMES].astype(float)
    proba = clf.predict_proba(X_hit)[:, 1]

    out = h.copy()
    out["_ml_p"] = proba
    out["remarks"] = (
        out["remarks"].astype(str) + " | ml_rank_p=" + out["_ml_p"].map(lambda x: f"{x:.3f}")
    )
    out = out.sort_values(["_ml_p", "score"], ascending=[False, False])
    out = out.drop(columns=["_ml_p"], errors="ignore")
    return out.head(MAX_SUBMISSION_ROWS)
