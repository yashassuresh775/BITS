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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import pandas as pd

from p3.io import normalize_market_dataframe, normalize_trades_dataframe

# Shown for docs / legacy imports. **Historical** backfill uses ``binance_spot_get`` (multi-host).
BINANCE_SPOT = os.environ.get(
    "BINANCE_SPOT_API",
    "https://api.binance.com/api/v3",
).rstrip("/")
DEFAULT_UA = "BITS-p3-live/1.0"

_resolved_spot_base: str | None = None

_BINANCE_US = "https://api.binance.us/api/v3"
_BINANCE_COM = "https://api.binance.com/api/v3"


def live_spot_venue() -> str:
    """
    Live ``run_p3 --live`` / dashboard:

    - ``okx`` (default) — OKX public v5 only.
    - ``binance`` — one Binance Spot base only (``BINANCE_SPOT_API`` or default .us).
    - ``both`` (aliases: ``okx+binance``, ``binance+okx``) — fetch **both** in parallel per symbol,
      merge 1m bars and concatenate trades (``okx:`` / ``bn:`` trade_id prefixes).
    """
    v = os.environ.get("LIVE_SPOT_VENUE", "okx").strip().lower()
    if v in ("both", "okx+binance", "binance+okx", "okx_binance"):
        return "both"
    if v == "binance":
        return "binance"
    return "okx"


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


def _binance_live_base() -> str:
    """Single Binance Spot REST base for **live** fetches (no host fallback)."""
    b = os.environ.get("BINANCE_SPOT_API", "").strip().rstrip("/")
    return b if b else _BINANCE_US


def binance_live_get_json(path_with_query: str, *, timeout: float = 45.0) -> object:
    """One GET to ``BINANCE_SPOT_API`` or default **api.binance.us** — fails fast on 451."""
    rel = path_with_query.lstrip("/")
    url = f"{_binance_live_base()}/{rel}"
    return _http_get_json(url, timeout=timeout)


def _volume_base_column(symbol: str) -> str:
    base = symbol.removesuffix("USDT")
    return f"Volume {base}"


def fetch_klines_raw(symbol: str, *, limit: int = 500) -> list[list]:
    if live_spot_venue() == "both":
        raise ValueError(
            "LIVE_SPOT_VENUE=both: use fetch_symbol_frames / fetch_live_frames (not fetch_klines_raw)."
        )
    if limit > 1000:
        limit = 1000
    q = f"symbol={symbol}&interval=1m&limit={limit}"
    if live_spot_venue() == "okx":
        from p3.live import okx

        raw = okx.fetch_klines_normalized(symbol, limit=limit)
    else:
        raw = binance_live_get_json(f"klines?{q}")
    if not isinstance(raw, list):
        raise ValueError(f"Unexpected klines response for {symbol}")
    return raw


def fetch_agg_trades_raw(symbol: str, *, limit: int = 1000) -> list[dict]:
    if live_spot_venue() == "both":
        raise ValueError(
            "LIVE_SPOT_VENUE=both: use fetch_symbol_frames / fetch_live_frames (not fetch_agg_trades_raw)."
        )
    if limit > 1000:
        limit = 1000
    q = f"symbol={symbol}&limit={limit}"
    if live_spot_venue() == "okx":
        from p3.live import okx

        raw = okx.fetch_trades_normalized(symbol, limit=limit)
    else:
        raw = binance_live_get_json(f"aggTrades?{q}")
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


def _fetch_okx_symbol_frames(
    symbol: str, *, kline_limit: int, trades_limit: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from p3.live import okx

    kl = min(int(kline_limit), okx.MAX_KLINES)
    tl = min(int(trades_limit), okx.MAX_TRADES)
    k_raw = okx.fetch_klines_normalized(symbol, limit=kl)
    t_raw = okx.fetch_trades_normalized(symbol, limit=tl)
    m_raw = klines_to_market_dataframe(symbol, k_raw)
    tr_raw = agg_trades_to_trades_dataframe(symbol, t_raw)
    return (
        normalize_market_dataframe(m_raw, symbol),
        normalize_trades_dataframe(tr_raw, symbol),
    )


def _fetch_binance_symbol_frames(
    symbol: str, *, kline_limit: int, trades_limit: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    klim = min(int(kline_limit), 1000)
    tlim = min(int(trades_limit), 1000)
    qk = f"symbol={symbol}&interval=1m&limit={klim}"
    qt = f"symbol={symbol}&limit={tlim}"
    k_raw = binance_live_get_json(f"klines?{qk}")
    t_raw = binance_live_get_json(f"aggTrades?{qt}")
    if not isinstance(k_raw, list):
        raise ValueError(f"Unexpected klines response for {symbol}")
    if not isinstance(t_raw, list):
        raise ValueError(f"Unexpected aggTrades response for {symbol}")
    m_raw = klines_to_market_dataframe(symbol, k_raw)
    tr_raw = agg_trades_to_trades_dataframe(symbol, t_raw)
    return (
        normalize_market_dataframe(m_raw, symbol),
        normalize_trades_dataframe(tr_raw, symbol),
    )


def _merge_market_bars(m_okx: pd.DataFrame, m_bn: pd.DataFrame) -> pd.DataFrame:
    """One row per UTC minute: combine highs/lows/volumes from OKX + Binance."""
    if m_okx.empty:
        return m_bn.copy()
    if m_bn.empty:
        return m_okx.copy()
    sym = str(m_okx["symbol"].iloc[0])
    a = m_okx.copy()
    b = m_bn.copy()
    a["Date"] = pd.to_datetime(a["Date"]).dt.floor("min")
    b["Date"] = pd.to_datetime(b["Date"]).dt.floor("min")
    combo = pd.concat([a.assign(_src=0), b.assign(_src=1)], ignore_index=True)
    combo = combo.sort_values(["Date", "_src"])
    rows: list[dict] = []
    for d, g in combo.groupby("Date", sort=True):
        g = g.sort_values("_src")
        rows.append(
            {
                "Date": d,
                "Open": float(g["Open"].iloc[0]),
                "High": float(g["High"].max()),
                "Low": float(g["Low"].min()),
                "Close": float(g["Close"].iloc[-1]),
                "vol_base": float(g["vol_base"].sum()),
                "vol_usdt": float(g["vol_usdt"].sum()),
                "tradecount": int(g["tradecount"].sum()),
                "symbol": sym,
            }
        )
    out = pd.DataFrame(rows)
    out["mid"] = (out["High"] + out["Low"]) / 2.0
    return out.sort_values("Date").reset_index(drop=True)


def _merge_dual_trades(t_okx: pd.DataFrame, t_bn: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    if not t_okx.empty:
        o = t_okx.copy()
        o["trade_id"] = "okx:" + o["trade_id"].astype(str)
        parts.append(o)
    if not t_bn.empty:
        b = t_bn.copy()
        b["trade_id"] = "bn:" + b["trade_id"].astype(str)
        parts.append(b)
    if not parts:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "timestamp",
                "price",
                "quantity",
                "side",
                "wallet_id",
                "symbol",
                "notional",
                "date",
                "minute",
            ]
        )
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values("timestamp").reset_index(drop=True)
    out["notional"] = out["price"] * out["quantity"]
    out["date"] = out["timestamp"].dt.normalize()
    out["minute"] = out["timestamp"].dt.floor("min")
    return out


def _fetch_symbol_frames_dual(
    symbol: str, *, kline_limit: int, trades_limit: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    err_okx: BaseException | None = None
    err_bn: BaseException | None = None
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_okx = ex.submit(
            _fetch_okx_symbol_frames,
            symbol,
            kline_limit=kline_limit,
            trades_limit=trades_limit,
        )
        f_bn = ex.submit(
            _fetch_binance_symbol_frames,
            symbol,
            kline_limit=kline_limit,
            trades_limit=trades_limit,
        )
        try:
            m_o, tr_o = f_okx.result()
        except BaseException as e:
            err_okx = e
            m_o, tr_o = pd.DataFrame(), pd.DataFrame()
        try:
            m_b, tr_b = f_bn.result()
        except BaseException as e:
            err_bn = e
            m_b, tr_b = pd.DataFrame(), pd.DataFrame()
    if m_o.empty and m_b.empty:
        msg = f"Live fetch failed for {symbol}"
        if err_okx and err_bn:
            raise RuntimeError(f"{msg}: OKX ({err_okx}); Binance ({err_bn})") from err_okx
        raise RuntimeError(msg + ".") from (err_okx or err_bn)

    m_out = _merge_market_bars(m_o, m_b)
    tr_out = _merge_dual_trades(tr_o, tr_b)
    return m_out, tr_out


def fetch_symbol_frames(
    symbol: str,
    *,
    kline_limit: int = 1000,
    trades_limit: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    v = live_spot_venue()
    if v == "both":
        return _fetch_symbol_frames_dual(
            symbol, kline_limit=kline_limit, trades_limit=trades_limit
        )
    if v == "okx":
        return _fetch_okx_symbol_frames(
            symbol, kline_limit=kline_limit, trades_limit=trades_limit
        )
    return _fetch_binance_symbol_frames(
        symbol, kline_limit=kline_limit, trades_limit=trades_limit
    )


def fetch_live_frames(
    symbols: Iterable[str],
    *,
    kline_limit: int = 1000,
    trades_limit: int = 1000,
    pause_sec: float = 0.12,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Pull spot OHLCV + recent trades per symbol.

    - **OKX** (default): ``LIVE_SPOT_VENUE`` unset / ``okx``.
    - **Binance** only: ``LIVE_SPOT_VENUE=binance`` (optional ``BINANCE_SPOT_API``).
    - **Both** (parallel per symbol): ``LIVE_SPOT_VENUE=both`` — merged 1m bars, trades with
      ``okx:`` / ``bn:`` id prefixes. If one venue fails, the other is still used when it returned data.
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
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            ValueError,
            KeyError,
            RuntimeError,
            OSError,
        ) as e:
            raise RuntimeError(f"Live market fetch failed for {sym}: {e}") from e
    return out
