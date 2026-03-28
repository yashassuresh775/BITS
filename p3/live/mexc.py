"""
MEXC **public** Spot REST (Binance-compatible ``/api/v3`` paths).

Used when Binance Spot returns **451/403** or connection errors from cloud/datacenter IPs.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.request

MEXC_SPOT = "https://api.mexc.com/api/v3"
DEFAULT_UA = "BITS-p3-live/1.0"


def _ssl_context() -> ssl.SSLContext:
    if os.environ.get("BINANCE_INSECURE_SSL", "").strip().lower() in ("1", "true", "yes"):
        return ssl._create_unverified_context()
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def mexc_get_json(path_with_query: str, *, timeout: float = 45.0) -> object:
    """GET ``https://api.mexc.com/api/v3/{path}``."""
    rel = path_with_query.lstrip("/")
    url = f"{MEXC_SPOT}/{rel}"
    req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_UA})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
        return json.loads(resp.read().decode())


def normalize_mexc_klines_to_binance_shape(raw: list) -> list[list]:
    """MEXC rows are 8 fields; expand to Binance 12-field shape for ``klines_to_market_dataframe``."""
    out: list[list] = []
    for k in raw:
        if not isinstance(k, list) or len(k) < 8:
            continue
        out.append(
            [
                k[0],
                k[1],
                k[2],
                k[3],
                k[4],
                k[5],
                k[6],
                k[7],
                0,
                "0",
                "0",
                "0",
            ]
        )
    return out


def normalize_mexc_agg_to_binance_shape(raw: list) -> list[dict]:
    """MEXC may omit ``a``/``f``; synthesize for ``agg_trades_to_trades_dataframe``."""
    out: list[dict] = []
    for i, t in enumerate(raw):
        if not isinstance(t, dict):
            continue
        row = dict(t)
        if row.get("a") is None or row.get("f") is None:
            ts = int(row["T"])
            synthetic = (ts % 900_000_000) * 1000 + (i % 1000)
            row["a"] = synthetic
            row["f"] = synthetic
        out.append(row)
    return out


def fetch_klines_normalized(symbol: str, *, limit: int) -> list[list]:
    q = f"symbol={symbol}&interval=1m&limit={limit}"
    raw = mexc_get_json(f"klines?{q}")
    if not isinstance(raw, list):
        raise ValueError(f"Unexpected MEXC klines for {symbol}")
    shaped = normalize_mexc_klines_to_binance_shape(raw)
    if not shaped:
        raise ValueError(f"Empty MEXC klines for {symbol}")
    return shaped


def fetch_agg_trades_normalized(symbol: str, *, limit: int) -> list[dict]:
    q = f"symbol={symbol}&limit={limit}"
    raw = mexc_get_json(f"aggTrades?{q}")
    if not isinstance(raw, list):
        raise ValueError(f"Unexpected MEXC aggTrades for {symbol}")
    return normalize_mexc_agg_to_binance_shape(raw)
