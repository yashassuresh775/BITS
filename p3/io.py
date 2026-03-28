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


def normalize_market_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Same transforms as ``load_market`` after reading CSV (for API-fed frames)."""
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").reset_index(drop=True)
    base_vol = _base_volume_col(list(out.columns))
    out = out.rename(
        columns={
            "Symbol": "symbol",
            base_vol: "vol_base",
            "Volume USDT": "vol_usdt",
            "tradecount": "tradecount",
        }
    )
    out["symbol"] = out["symbol"].fillna(symbol)
    for c in ("Open", "High", "Low", "Close"):
        if c not in out.columns:
            raise ValueError(f"Missing {c} in market frame")
    out["mid"] = (out["High"] + out["Low"]) / 2.0
    return out


def load_market(data_root: Path, symbol: str) -> pd.DataFrame:
    path = market_path(data_root, symbol)
    df = pd.read_csv(path, parse_dates=["Date"])
    return normalize_market_dataframe(df, symbol)


def normalize_trades_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Same transforms as ``load_trades`` after reading CSV (for API-fed frames)."""
    out = df.copy()
    out["symbol"] = symbol
    if "wallet_id" not in out.columns and "trader_id" in out.columns:
        out = out.rename(columns={"trader_id": "wallet_id"})
    need = {"trade_id", "timestamp", "price", "quantity", "side", "wallet_id"}
    missing = need - set(out.columns)
    if missing:
        raise ValueError(f"Trades missing columns {missing}")
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["side"] = out["side"].astype(str).str.upper()
    out["notional"] = out["price"] * out["quantity"]
    out["date"] = out["timestamp"].dt.normalize()
    out["minute"] = out["timestamp"].dt.floor("min")
    return out


def load_trades(data_root: Path, symbol: str) -> pd.DataFrame:
    path = trades_path(data_root, symbol)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return normalize_trades_dataframe(df, symbol)


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
