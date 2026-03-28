"""
Paginated Binance Spot **historical** 1m klines + aggregate trades (public REST).

Writes the same column layout the pipeline expects from CSV (see ``scripts/fetch_binance_history.py``).
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Iterable

from p3.live.binance import (
    BINANCE_SPOT,
    _http_get_json,
    agg_trades_to_trades_dataframe,
    klines_to_market_dataframe,
)

_MS_MIN = 60_000


def fetch_klines_historical(
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    pause_sec: float = 0.08,
) -> list[list]:
    """Walk forward in 1m chunks (max 1000 bars per request)."""
    if end_ms < start_ms:
        return []
    all_rows: list[list] = []
    seen_opens: set[int] = set()
    cur = start_ms
    while cur <= end_ms:
        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": str(cur),
            "endTime": str(end_ms),
            "limit": "1000",
        }
        q = urllib.parse.urlencode(params)
        url = f"{BINANCE_SPOT}/klines?{q}"
        chunk = _http_get_json(url)
        if not isinstance(chunk, list) or not chunk:
            break
        for k in chunk:
            ot = int(k[0])
            if ot not in seen_opens:
                seen_opens.add(ot)
                all_rows.append(k)
        last_open = int(chunk[-1][0])
        nxt = last_open + _MS_MIN
        if nxt <= cur:
            break
        cur = nxt
        if pause_sec > 0:
            time.sleep(pause_sec)
    all_rows.sort(key=lambda x: int(x[0]))
    return all_rows


def fetch_agg_trades_historical(
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    max_trades: int = 400_000,
    pause_sec: float = 0.05,
) -> list[dict]:
    """Page forward using ``startTime`` + ``limit=1000`` until window exhausted or cap hit."""
    if end_ms < start_ms:
        return []
    out: list[dict] = []
    cur_start = start_ms
    while cur_start <= end_ms and len(out) < max_trades:
        params = {
            "symbol": symbol,
            "startTime": str(cur_start),
            "endTime": str(end_ms),
            "limit": "1000",
        }
        q = urllib.parse.urlencode(params)
        url = f"{BINANCE_SPOT}/aggTrades?{q}"
        chunk = _http_get_json(url)
        if not isinstance(chunk, list) or not chunk:
            break
        out.extend(chunk)
        last_t = int(chunk[-1]["T"])
        nxt = last_t + 1
        if nxt <= cur_start:
            break
        cur_start = nxt
        if len(chunk) < 1000:
            break
        if pause_sec > 0:
            time.sleep(pause_sec)
    return out[:max_trades]


def historical_symbol_to_csvs(
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    market_csv: str | None,
    trades_csv: str | None,
    max_trades: int = 400_000,
    pause_sec: float = 0.08,
) -> tuple[int, int]:
    """
    Fetch history for one symbol and write CSV files. Pass ``None`` to skip a file.
    Returns (n_klines_written, n_trades_written).
    """
    k_raw = fetch_klines_historical(symbol, start_ms, end_ms, pause_sec=pause_sec)
    mdf = klines_to_market_dataframe(symbol, k_raw)
    if market_csv and not mdf.empty:
        Path(market_csv).parent.mkdir(parents=True, exist_ok=True)
        mdf.to_csv(market_csv, index=False)

    t_raw = fetch_agg_trades_historical(
        symbol, start_ms, end_ms, max_trades=max_trades, pause_sec=pause_sec * 0.6
    )
    tdf = agg_trades_to_trades_dataframe(symbol, t_raw)
    if trades_csv and not tdf.empty:
        Path(trades_csv).parent.mkdir(parents=True, exist_ok=True)
        export = tdf[
            ["trade_id", "timestamp", "price", "quantity", "side", "wallet_id"]
        ].copy()
        export.to_csv(trades_csv, index=False)

    return len(mdf), len(tdf)


def fetch_history_pack(
    data_root: str,
    symbols: Iterable[str],
    start_ms: int,
    end_ms: int,
    *,
    max_trades_per_symbol: int = 400_000,
    pause_sec: float = 0.08,
    filename_year: str = "2026",
) -> dict[str, tuple[int, int]]:
    """
    Populate ``data_root/crypto-market`` and ``data_root/crypto-trades`` with Binance history.
    Market files: ``Binance_{SYMBOL}_{filename_year}_minute.csv`` (matches ``p3.io.market_path``).
    Trades files: ``{SYMBOL}_trades.csv``.
    """
    root = Path(data_root)
    (root / "crypto-market").mkdir(parents=True, exist_ok=True)
    (root / "crypto-trades").mkdir(parents=True, exist_ok=True)
    counts: dict[str, tuple[int, int]] = {}
    for i, sym in enumerate(symbols):
        if i > 0 and pause_sec > 0:
            time.sleep(pause_sec)
        mpath = root / "crypto-market" / f"Binance_{sym}_{filename_year}_minute.csv"
        tpath = root / "crypto-trades" / f"{sym}_trades.csv"
        try:
            nk, nt = historical_symbol_to_csvs(
                sym,
                start_ms,
                end_ms,
                market_csv=str(mpath),
                trades_csv=str(tpath),
                max_trades=max_trades_per_symbol,
                pause_sec=pause_sec,
            )
            counts[sym] = (nk, nt)
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError, OSError) as e:
            raise RuntimeError(f"Historical fetch failed for {sym}: {e}") from e
    return counts
