"""
OKX **public** v5 REST (no API key) — extra live venue when Binance / MEXC are blocked.

- ``GET /api/v5/market/candles`` — 1m spot candles (max **300** per call).
- ``GET /api/v5/market/trades`` — recent trades (max **500** per call).

Responses are normalized to the same shapes as ``p3.live.binance`` klines / aggTrades.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.parse
import urllib.request

OKX_BASE = "https://www.okx.com/api/v5"
DEFAULT_UA = "BITS-p3-live/1.0"
MAX_KLINES = 300
MAX_TRADES = 500


def symbol_to_inst_id(symbol: str) -> str:
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}-USDT"
    return symbol.replace("USDT", "-USDT")


def _ssl_context() -> ssl.SSLContext:
    if os.environ.get("BINANCE_INSECURE_SSL", "").strip().lower() in ("1", "true", "yes"):
        return ssl._create_unverified_context()
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def okx_get_json(path: str, *, timeout: float = 45.0) -> dict:
    """``path`` is e.g. ``market/candles?instId=BTC-USDT&bar=1m&limit=10``."""
    path = path.lstrip("/")
    url = f"{OKX_BASE}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_UA})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
        return json.loads(resp.read().decode())


def _okx_candle_to_binance_row(r: list) -> list:
    """OKX: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm] → Binance 12-field row."""
    ts = int(r[0])
    close_ms = ts + 60_000 - 1
    base_v = float(r[5])
    quote_v = float(r[6]) if len(r) > 6 else 0.0
    return [
        ts,
        r[1],
        r[2],
        r[3],
        r[4],
        base_v,
        close_ms,
        quote_v,
        0,
        "0",
        "0",
        "0",
    ]


def fetch_klines_normalized(symbol: str, *, limit: int) -> list[list]:
    inst = symbol_to_inst_id(symbol)
    lim = min(max(1, int(limit)), MAX_KLINES)
    q = urllib.parse.urlencode({"instId": inst, "bar": "1m", "limit": str(lim)})
    data = okx_get_json(f"market/candles?{q}")
    if str(data.get("code", "")) != "0":
        raise ValueError(f"OKX candles error for {symbol}: {data}")
    rows = data.get("data") or []
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Empty OKX candles for {symbol}")
    shaped = [_okx_candle_to_binance_row(r) for r in rows if isinstance(r, list) and len(r) >= 7]
    shaped.sort(key=lambda x: int(x[0]))
    return shaped


def fetch_trades_normalized(symbol: str, *, limit: int) -> list[dict]:
    inst = symbol_to_inst_id(symbol)
    lim = min(max(1, int(limit)), MAX_TRADES)
    q = urllib.parse.urlencode({"instId": inst, "limit": str(lim)})
    data = okx_get_json(f"market/trades?{q}")
    if str(data.get("code", "")) != "0":
        raise ValueError(f"OKX trades error for {symbol}: {data}")
    rows = data.get("data") or []
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected OKX trades for {symbol}")
    out: list[dict] = []
    for t in rows:
        if not isinstance(t, dict):
            continue
        tid = int(t["tradeId"])
        side = str(t.get("side", "")).lower()
        out.append(
            {
                "a": tid,
                "f": tid,
                "p": t["px"],
                "q": t["sz"],
                "T": int(t["ts"]),
                "m": side == "sell",
            }
        )
    out.sort(key=lambda x: int(x["T"]))
    return out
