from __future__ import annotations

import numpy as np
import pandas as pd


def augment_graph_and_sequence(enriched: pd.DataFrame) -> pd.DataFrame:
    """
    Minute-level crowding + same-wallet sequence cues (no GNN / torch).
    """
    e = enriched.sort_values(["wallet_id", "timestamp"]).copy()

    e["trades_same_minute"] = e.groupby("minute")["trade_id"].transform("count")
    e["wallets_same_minute"] = e.groupby("minute")["wallet_id"].transform("nunique")

    e["_prev_side"] = e.groupby("wallet_id")["side"].shift(1)
    e["seq_same_side_prev"] = (e["side"] == e["_prev_side"]).astype(np.float64).fillna(
        0.0
    )

    e["_prev_price"] = e.groupby("wallet_id")["price"].shift(1)
    pp = e["_prev_price"].replace(0, np.nan)
    e["seq_price_chg"] = ((e["price"] - e["_prev_price"]) / pp).fillna(0.0)
    e["seq_price_chg"] = e["seq_price_chg"].replace([np.inf, -np.inf], 0.0)

    return e.drop(columns=["_prev_side", "_prev_price"], errors="ignore")
