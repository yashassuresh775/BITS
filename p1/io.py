"""Load and normalise equity market_data.csv (L1–L10 order book)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _find_sec_id(df: pd.DataFrame) -> str:
    for c in ("sec_id", "SecId", "SEC_ID", "security_id"):
        if c in df.columns:
            return c
    lower = {x.lower(): x for x in df.columns}
    if "sec_id" in lower:
        return lower["sec_id"]
    raise ValueError("market_data.csv needs a sec_id column")


def _level_cols(prefix: str, df: pd.DataFrame) -> list[str]:
    """Return columns for levels 1..10 (bid_size_level01 style or regex match)."""
    cols: list[str | None] = [None] * 10
    lower_map = {x.lower(): x for x in df.columns}
    for i in range(1, 11):
        for pat in (
            f"{prefix}_level{i:02d}",
            f"{prefix}_level{i}",
            f"{prefix}{i:02d}",
            f"{prefix.lower()}_level{i:02d}",
        ):
            if pat in df.columns:
                cols[i - 1] = pat
                break
            if pat.lower() in lower_map:
                cols[i - 1] = lower_map[pat.lower()]
                break
    # Regex fallback: BidSizeLevel01, etc.
    if any(c is None for c in cols):
        pat_re = re.compile(rf"{re.escape(prefix)}.*?0?(\d+)$", re.I)
        for c in df.columns:
            m = pat_re.match(str(c).replace(" ", "_"))
            if m:
                idx = int(m.group(1))
                if 1 <= idx <= 10:
                    cols[idx - 1] = c
    missing = [i + 1 for i, c in enumerate(cols) if c is None]
    if missing:
        raise ValueError(
            f"Could not find all 10 {prefix} level columns; missing levels {missing}. "
            f"Sample columns: {list(df.columns)[:40]}"
        )
    return [c for c in cols if c is not None]


def _parse_timestamp(df: pd.DataFrame) -> pd.Series:
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
    if "DateTime" in df.columns:
        return pd.to_datetime(df["DateTime"], errors="coerce")
    if "datetime" in df.columns:
        return pd.to_datetime(df["datetime"], errors="coerce")
    if "Date" in df.columns and "Time" in df.columns:
        return pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce"
        )
    if "trade_date" in df.columns and "time" in df.columns:
        return pd.to_datetime(
            df["trade_date"].astype(str) + " " + df["time"].astype(str), errors="coerce"
        )
    if "Date" in df.columns:
        return pd.to_datetime(df["Date"], errors="coerce")
    raise ValueError(
        "Need a timestamp: timestamp, DateTime, or Date+Time / trade_date+time. Got: "
        + str(list(df.columns))
    )


def load_market_data(path: str | pd.DataFrame) -> pd.DataFrame:
    """
    Load per-minute order book CSV.

    Required structure:
    - sec_id
    - time column (see _parse_timestamp)
    - bid_price_level01, ask_price_level01 (or aliases)
    - bid_size_level01..10, ask_size_level01..10
    """
    if isinstance(path, pd.DataFrame):
        df = path.copy()
    else:
        df = pd.read_csv(path)

    sid = _find_sec_id(df)
    df = df.rename(columns={sid: "sec_id"})
    df["sec_id"] = pd.to_numeric(df["sec_id"], errors="coerce")

    df["minute"] = _parse_timestamp(df)
    df = df.dropna(subset=["minute", "sec_id"])

    # L1 prices
    bp1 = _pick(df, ["bid_price_level01", "BidPriceLevel01", "bid_price_01"])
    ap1 = _pick(df, ["ask_price_level01", "AskPriceLevel01", "ask_price_01"])
    df["bid_price_level01"] = pd.to_numeric(df[bp1], errors="coerce")
    df["ask_price_level01"] = pd.to_numeric(df[ap1], errors="coerce")

    bid_sz = _level_cols("bid_size", df)
    ask_sz = _level_cols("ask_size", df)
    rename_b = {c: f"bid_size_level{i:02d}" for i, c in enumerate(bid_sz, start=1)}
    rename_a = {c: f"ask_size_level{i:02d}" for i, c in enumerate(ask_sz, start=1)}
    df = df.rename(columns={**rename_b, **rename_a})

    keep = ["sec_id", "minute"] + [f"bid_size_level{i:02d}" for i in range(1, 11)]
    keep += [f"ask_size_level{i:02d}" for i in range(1, 11)] + ["bid_price_level01", "ask_price_level01"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.sort_values(["sec_id", "minute"]).reset_index(drop=True)
    return df


def _pick(df: pd.DataFrame, names: list[str]) -> str:
    for n in names:
        if n in df.columns:
            return n
    low = {x.lower(): x for x in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    raise ValueError(f"Missing column; tried {names}")


def load_trades_per_minute(trades_path: str | pd.DataFrame | None) -> pd.DataFrame | None:
    """Optional: aggregate trade quantity per (sec_id, minute) for buy-aggression remarks."""
    if trades_path is None:
        return None
    if isinstance(trades_path, pd.DataFrame):
        t = trades_path.copy()
    else:
        t = pd.read_csv(trades_path)
    td = None
    for c in ("trade_date", "TradeDate", "timestamp", "datetime"):
        if c in t.columns:
            td = c
            break
    if td is None:
        return None
    t["_m"] = pd.to_datetime(t[td], errors="coerce").dt.floor("min")
    sid = None
    for c in ("sec_id", "SecId"):
        if c in t.columns:
            sid = c
            break
    if sid is None:
        return None
    t["sec_id"] = pd.to_numeric(t[sid], errors="coerce")
    q = None
    for c in ("quantity", "Quantity", "qty"):
        if c in t.columns:
            q = c
            break
    if q is None:
        return None
    side = None
    for c in ("side", "Side"):
        if c in t.columns:
            side = c
            break
    t["qty"] = pd.to_numeric(t[q], errors="coerce")
    if side:
        buy = t[t[side].astype(str).str.upper() == "BUY"]
        g = buy.groupby(["sec_id", "_m"], as_index=False)["qty"].sum()
    else:
        g = t.groupby(["sec_id", "_m"], as_index=False)["qty"].sum()
    g = g.rename(columns={"_m": "minute", "qty": "buy_qty_minute"})
    return g
