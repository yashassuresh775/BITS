from __future__ import annotations

import numpy as np
import pandas as pd


def attach_market_to_trades(trades: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Left-join each trade to its 1-minute OHLCV bar on ``minute``."""
    m = market.rename(columns={"Date": "minute"})[
        [
            "minute",
            "Open",
            "High",
            "Low",
            "Close",
            "mid",
            "vol_base",
            "vol_usdt",
            "tradecount",
        ]
    ].copy()
    out = trades.merge(m, on="minute", how="left", suffixes=("", "_m"))
    mid = out["mid"].replace(0, np.nan)
    out["price_vs_mid_bps"] = ((out["price"] - mid) / mid * 10000.0).fillna(0.0)
    return out


def wallet_frequency(trades: pd.DataFrame) -> pd.Series:
    cnt = trades["wallet_id"].value_counts()
    return trades["wallet_id"].map(cnt).astype(int)


def symbol_quantity_zscore(trades: pd.DataFrame, window: int = 200) -> pd.Series:
    def roll_z(s: pd.Series) -> pd.Series:
        m = s.rolling(window, min_periods=30).mean().shift(1)
        v = s.rolling(window, min_periods=30).std().shift(1).replace(0, np.nan)
        return (s - m) / v

    return (
        trades.groupby("symbol", group_keys=False)["quantity"]
        .apply(roll_z)
        .fillna(0.0)
    )


def hourly_usdt_volume(market: pd.DataFrame) -> pd.DataFrame:
    m = market.copy()
    m["day"] = m["Date"].dt.normalize()
    m["hour_bucket"] = m["Date"].dt.floor("h")
    sym = m["symbol"].iloc[0] if "symbol" in m.columns else None
    g = m.groupby(["day", "hour_bucket"], as_index=False)["vol_usdt"].sum()
    if sym is not None:
        g.insert(0, "symbol", sym)
    return g


def first_trade_hour_per_wallet(trades: pd.DataFrame) -> pd.DataFrame:
    """First timestamp per wallet (for placement_smurfing)."""
    idx = trades.groupby("wallet_id")["timestamp"].idxmin()
    first = trades.loc[idx, ["wallet_id", "timestamp"]].rename(
        columns={"timestamp": "first_ts"}
    )
    first["first_hour"] = first["first_ts"].dt.floor("h")
    return first
