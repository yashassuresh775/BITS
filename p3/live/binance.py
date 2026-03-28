"""
Binance **public** Spot REST — no API key.

- `GET /api/v3/klines` → 1m OHLCV + quote volume + trade count (matches student-pack-style bars).
- `GET /api/v3/aggTrades` → recent aggregate trades.

``wallet_id`` is **synthetic** (bucketed from first trade id in the aggregate). Live public data
has no wallet labels; wallet-heavy detectors are weaker than on the official synthetic pack.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd

from p3.io import normalize_market_dataframe, normalize_trades_dataframe

# Shown for docs / legacy imports; live requests use ``binance_spot_get`` (451-aware fallback).
BINANCE_SPOT = os.environ.get(
    "BINANCE_SPOT_API",
    "https://api.binance.com/api/v3",
).rstrip("/")
DEFAULT_UA = "BITS-p3-live/1.0"

_resolved_spot_base: str | None = None

_BINANCE_US = "https://api.binance.us/api/v3"
_BINANCE_COM = "https://api.binance.com/api/v3"


def _spot_bases() -> list[str]:
    """
    Hosts to try in order (HTTP 451 / connection errors move to the next).

    - **Unset** ``BINANCE_SPOT_API``: **.us first**, then .com (Streamlit Cloud / many US IPs get 451 on .com only).
    - **Pinned** to ``api.binance.com``: same URL plus **.us** so secrets that point at .com still auto-fallback.
    - **Pinned** to anything else: that URL only (no extra hops).
    """
    env = os.environ.get("BINANCE_SPOT_API", "").strip().rstrip("/")
    if not env:
        return [_BINANCE_US, _BINANCE_COM]
    bases = [env]
    if "api.binance.com" in env and "binance.us" not in env.lower():
        bases.append(_BINANCE_US)
    return bases


def _should_try_next_host(err: urllib.error.HTTPError) -> bool:
    return err.code in (403, 451)


def binance_spot_get(path_with_query: str, *, timeout: float = 45.0) -> object:
    """
    GET ``{base}/{path}`` with fallback across `_spot_bases()` on HTTP **403/451** or connection errors.
    """
    global _resolved_spot_base
    rel = path_with_query.lstrip("/")
    bases = _spot_bases()
    multi = len(bases) > 1

    if _resolved_spot_base and _resolved_spot_base in bases:
        url = f"{_resolved_spot_base}/{rel}"
        try:
            return _http_get_json(url, timeout=timeout)
        except urllib.error.HTTPError as e:
            if _should_try_next_host(e) and multi:
                _resolved_spot_base = None
            else:
                raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if multi:
                _resolved_spot_base = None
            else:
                raise

    last_err: BaseException | None = None
    for base in bases:
        url = f"{base}/{rel}"
        try:
            out = _http_get_json(url, timeout=timeout)
            _resolved_spot_base = base
            return out
        except urllib.error.HTTPError as e:
            last_err = e
            if _should_try_next_host(e) and multi and base != bases[-1]:
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            if multi and base != bases[-1]:
                continue
            raise
    assert last_err is not None
    raise last_err


def _ssl_context() -> ssl.SSLContext:
    """Prefer ``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE``, then certifi (fixes many macOS Pythons)."""
    if os.environ.get("BINANCE_INSECURE_SSL", "").strip().lower() in ("1", "true", "yes"):
        return ssl._create_unverified_context()
    for env_key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        ca = os.environ.get(env_key, "").strip()
        if ca and Path(ca).is_file():
            return ssl.create_default_context(cafile=ca)
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _http_get_json(url: str, *, timeout: float = 45.0) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_UA})
    ctx = _ssl_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read().decode())


def _volume_base_column(symbol: str) -> str:
    base = symbol.removesuffix("USDT")
    return f"Volume {base}"


def fetch_klines_raw(symbol: str, *, limit: int = 500) -> list[list]:
    if limit > 1000:
        limit = 1000
    q = f"symbol={symbol}&interval=1m&limit={limit}"
    raw = binance_spot_get(f"klines?{q}")
    if not isinstance(raw, list):
        raise ValueError(f"Unexpected klines response for {symbol}")
    return raw


def fetch_agg_trades_raw(symbol: str, *, limit: int = 1000) -> list[dict]:
    if limit > 1000:
        limit = 1000
    q = f"symbol={symbol}&limit={limit}"
    raw = binance_spot_get(f"aggTrades?{q}")
    if not isinstance(raw, list):
        raise ValueError(f"Unexpected aggTrades response for {symbol}")
    return raw


def klines_to_market_dataframe(symbol: str, raw: list[list]) -> pd.DataFrame:
    vol_col = _volume_base_column(symbol)
    rows: list[dict] = []
    for k in raw:
        rows.append(
            {
                "Date": pd.to_datetime(int(k[0]), unit="ms", utc=True).tz_convert(None),
                "Symbol": symbol,
                "Open": float(k[1]),
                "High": float(k[2]),
                "Low": float(k[3]),
                "Close": float(k[4]),
                vol_col: float(k[5]),
                "Volume USDT": float(k[7]),
                "tradecount": int(k[8]),
            }
        )
    return pd.DataFrame(rows)


def agg_trades_to_trades_dataframe(symbol: str, raw: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for t in raw:
        is_buyer_maker = bool(t["m"])
        side = "SELL" if is_buyer_maker else "BUY"
        f_id = int(t["f"])
        rows.append(
            {
                "trade_id": str(t["a"]),
                "timestamp": pd.to_datetime(int(t["T"]), unit="ms", utc=True).tz_convert(None),
                "price": float(t["p"]),
                "quantity": float(t["q"]),
                "side": side,
                "wallet_id": str(f_id % 8192),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_symbol_frames(
    symbol: str,
    *,
    kline_limit: int = 1000,
    trades_limit: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    k_raw = fetch_klines_raw(symbol, limit=kline_limit)
    t_raw = fetch_agg_trades_raw(symbol, limit=trades_limit)
    m_raw = klines_to_market_dataframe(symbol, k_raw)
    tr_raw = agg_trades_to_trades_dataframe(symbol, t_raw)
    market = normalize_market_dataframe(m_raw, symbol)
    trades = normalize_trades_dataframe(tr_raw, symbol)
    return market, trades


def fetch_live_frames(
    symbols: Iterable[str],
    *,
    kline_limit: int = 1000,
    trades_limit: int = 1000,
    pause_sec: float = 0.12,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Pull all symbols from Binance Spot (public). Small pause between symbols to stay under
    lightweight rate limits.
    """
    out: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    sym_list = list(symbols)
    for i, sym in enumerate(sym_list):
        if i > 0 and pause_sec > 0:
            time.sleep(pause_sec)
        try:
            out[sym] = fetch_symbol_frames(
                sym, kline_limit=kline_limit, trades_limit=trades_limit
            )
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError) as e:
            raise RuntimeError(f"Binance fetch failed for {sym}: {e}") from e
    return out
