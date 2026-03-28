from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from p3.config import SYMBOLS


def _base_volume_col(columns: list[str]) -> str:
    for c in columns:
        if c.startswith("Volume ") and "USDT" not in c:
            return c
    raise ValueError(f"No base volume column in {columns}")


def market_path(data_root: Path, symbol: str) -> Path:
    p = data_root / "crypto-market" / f"Binance_{symbol}_2026_minute.csv"
    if p.exists():
        return p
    alt = data_root / "crypto-market" / f"{symbol}_market.csv"
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Market file for {symbol} not under {data_root}")


def trades_path(data_root: Path, symbol: str) -> Path:
    p = data_root / "crypto-trades" / f"{symbol}_trades.csv"
    if p.exists():
        return p
    raise FileNotFoundError(f"Trades file for {symbol} not under {data_root}")


def load_market(data_root: Path, symbol: str) -> pd.DataFrame:
    path = market_path(data_root, symbol)
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    base_vol = _base_volume_col(list(df.columns))
    df = df.rename(
        columns={
            "Symbol": "symbol",
            base_vol: "vol_base",
            "Volume USDT": "vol_usdt",
            "tradecount": "tradecount",
        }
    )
    df["symbol"] = df["symbol"].fillna(symbol)
    for c in ("Open", "High", "Low", "Close"):
        if c not in df.columns:
            raise ValueError(f"Missing {c} in market CSV")
    df["mid"] = (df["High"] + df["Low"]) / 2.0
    return df


def load_trades(data_root: Path, symbol: str) -> pd.DataFrame:
    path = trades_path(data_root, symbol)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["symbol"] = symbol
    if "wallet_id" not in df.columns and "trader_id" in df.columns:
        df = df.rename(columns={"trader_id": "wallet_id"})
    need = {"trade_id", "timestamp", "price", "quantity", "side", "wallet_id"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Trades missing columns {missing}: {path}")
    df["side"] = df["side"].str.upper()
    df["notional"] = df["price"] * df["quantity"]
    df["date"] = df["timestamp"].dt.normalize()
    df["minute"] = df["timestamp"].dt.floor("min")
    return df


def load_btc_market(data_root: Path) -> pd.DataFrame:
    return load_market(data_root, "BTCUSDT")


def discover_symbols(data_root: Path) -> list[str]:
    """Return symbols found in crypto-trades/*.csv (intersect known list)."""
    root = Path(data_root) / "crypto-trades"
    if not root.is_dir():
        return list(SYMBOLS)
    found: set[str] = set()
    for p in root.glob("*_trades.csv"):
        m = re.match(r"^(.+USDT)_trades\.csv$", p.name)
        if m:
            found.add(m.group(1))
    ordered = [s for s in SYMBOLS if s in found]
    if ordered:
        return ordered
    return list(SYMBOLS)
