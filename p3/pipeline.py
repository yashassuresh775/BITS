from __future__ import annotations

from pathlib import Path

import pandas as pd

from p3.config import (
    ENSEMBLE_SYMBOLS,
    IF_SYMBOLS,
    MAX_SUBMISSION_ROWS,
    USE_ENSEMBLE_ANOMALY,
    USE_ML_RERANKER,
)
from p3.detectors.isolation import isolation_candidates
from p3.detectors.market_patterns import (
    detect_cross_pair_divergence,
    detect_pump_dump_trades,
    detect_spoofing_proxy,
)
from p3.detectors.rules import detect_peg_break, detect_wash_volume_at_peg
from p3.detectors.wallet_patterns import (
    detect_aml_structuring,
    detect_chain_pass_through,
    detect_coordinated_pump_minute,
    detect_layering_echo,
    detect_ramping,
    detect_round_trip_pair,
    detect_threshold_testing,
    detect_wash_same_wallet,
)
from p3.features import (
    attach_market_to_trades,
    symbol_quantity_zscore,
    wallet_frequency,
)
from p3.io import discover_symbols, load_btc_market, load_market, load_trades
from p3.ml.extra_features import augment_graph_and_sequence


def _prepare_symbol(
    data_root: Path,
    symbol: str,
    btc_market: pd.DataFrame,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    market = load_market(data_root, symbol)
    trades = load_trades(data_root, symbol)
    enriched = attach_market_to_trades(trades, market)
    enriched["qty_z"] = symbol_quantity_zscore(enriched)
    enriched["wallet_freq"] = wallet_frequency(enriched)
    enriched = augment_graph_and_sequence(enriched)

    parts: list[pd.DataFrame] = []

    parts.append(detect_peg_break(trades, symbol))
    parts.append(detect_wash_volume_at_peg(trades, symbol))

    parts.append(detect_wash_same_wallet(trades, symbol))
    parts.append(detect_round_trip_pair(trades, symbol))
    parts.append(detect_ramping(trades, symbol))
    parts.append(detect_layering_echo(trades, symbol))
    parts.append(detect_aml_structuring(trades, symbol))
    parts.append(detect_threshold_testing(trades, symbol))
    parts.append(detect_coordinated_pump_minute(trades, symbol))
    parts.append(detect_chain_pass_through(trades, symbol))

    parts.append(detect_pump_dump_trades(trades, market, symbol))
    parts.append(detect_spoofing_proxy(trades, market, symbol))
    parts.append(detect_cross_pair_divergence(trades, market, btc_market, symbol))

    if USE_ENSEMBLE_ANOMALY and symbol in ENSEMBLE_SYMBOLS:
        from p3.ml.ensemble_od import detect_ensemble_if_lof

        e = detect_ensemble_if_lof(enriched, symbol)
        if not e.empty:
            parts.append(e)

    if symbol in IF_SYMBOLS:
        parts.append(isolation_candidates(enriched, symbol))

    out_parts = [p.assign(symbol=symbol) for p in parts if not p.empty]
    return out_parts, enriched


def _corroborate_and_dedupe(hits: pd.DataFrame) -> pd.DataFrame:
    """Drop uncorroborated IF / ensemble_if_lof; one row per trade_id (no row cap)."""
    if hits.empty:
        return hits
    h = hits.copy()
    weak = h["detector"].isin(["isolation_forest", "ensemble_if_lof"])
    corroborated = set(h.loc[~weak, "trade_id"])
    h = h[~(weak & ~h["trade_id"].isin(corroborated))]

    if h.empty:
        return h

    h["_has_vt"] = h["violation_type"].fillna("").astype(str).str.len() > 0
    h = h.sort_values(["score", "_has_vt"], ascending=[False, False])
    h = h.drop(columns=["_has_vt"], errors="ignore")
    h = h.drop_duplicates(subset=["trade_id"], keep="first")
    return h


def run_pipeline(data_root: str | Path) -> pd.DataFrame:
    root = Path(data_root)
    btc = load_btc_market(root)
    symbols = discover_symbols(root)
    chunks: list[pd.DataFrame] = []
    enriched_map: dict[str, pd.DataFrame] = {}

    for sym in symbols:
        try:
            parts, enr = _prepare_symbol(root, sym, btc)
            enriched_map[sym] = enr
            chunks.extend(parts)
        except FileNotFoundError:
            continue

    empty_cols = [
        "symbol",
        "date",
        "trade_id",
        "violation_type",
        "remarks",
        "score",
        "detector",
    ]
    if not chunks:
        return pd.DataFrame(columns=empty_cols)

    hits_raw = pd.concat(chunks, ignore_index=True)
    h = _corroborate_and_dedupe(hits_raw)

    if h.empty:
        return pd.DataFrame(columns=empty_cols)

    if USE_ML_RERANKER:
        from p3.ml.ranker import ml_rerank

        h = ml_rerank(h, enriched_map, hits_raw)
    else:
        h = h.sort_values("score", ascending=False).head(MAX_SUBMISSION_ROWS)

    return h


def hits_to_submission(hits: pd.DataFrame) -> pd.DataFrame:
    if hits.empty:
        return pd.DataFrame(
            columns=["symbol", "date", "trade_id", "violation_type", "remarks"]
        )
    out = pd.DataFrame(
        {
            "symbol": hits["symbol"],
            "date": hits["timestamp"].dt.strftime("%Y-%m-%d"),
            "trade_id": hits["trade_id"],
            "violation_type": hits["violation_type"].fillna(""),
            "remarks": hits["remarks"],
        }
    )
    return out
