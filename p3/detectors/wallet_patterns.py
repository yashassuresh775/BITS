from __future__ import annotations

import numpy as np
import pandas as pd

from p3.config import (
    RAMP_MIN_STREAK,
    ROUND_TRIP_MAX_SEC,
    ROUND_TRIP_MIN_QTY_RATIO,
    STRUCT_CV_MAX,
    STRUCT_MAX_MIN_RATIO,
    STRUCT_MIN_TRADES,
    WASH_WINDOW_SEC,
)


def _trade_date_series(ts: pd.Series) -> pd.Series:
    return ts.dt.strftime("%Y-%m-%d")


def detect_wash_same_wallet(trades: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Same wallet BUY and SELL within window, similar price, near-flat net base qty."""
    t = trades.sort_values("timestamp").reset_index(drop=True)
    hits = []
    for w, g in t.groupby("wallet_id"):
        g = g.sort_values("timestamp")
        if len(g) < 2:
            continue
        ts = g["timestamp"].values
        sides = g["side"].values
        prices = g["price"].values
        qtys = g["quantity"].values
        idxs = g.index.values
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                dt = (pd.Timestamp(ts[j]) - pd.Timestamp(ts[i])).total_seconds()
                if dt > WASH_WINDOW_SEC:
                    break
                if sides[i] == sides[j]:
                    continue
                if abs(prices[i] - prices[j]) / max(prices[i], 1e-12) > 0.002:
                    continue
                net = (
                    qtys[i] if sides[i] == "BUY" else -qtys[i]
                ) + (qtys[j] if sides[j] == "BUY" else -qtys[j])
                if abs(net) < 0.25 * max(qtys[i], qtys[j], 1e-12):
                    hits.extend([idxs[i], idxs[j]])
    if not hits:
        return pd.DataFrame()
    hit = t.loc[sorted(set(hits))].copy()
    hit["violation_type"] = "wash_trading"
    hit["detector"] = "wash_same_wallet"
    hit["score"] = 4
    hit["remarks"] = (
        f"{symbol}: same wallet BUY/SELL within {WASH_WINDOW_SEC}s at similar prices; "
        "net base position ~flat — classic wash."
    )
    return hit


def detect_round_trip_pair(trades: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Two wallets alternating opposite sides within a short window at matched size/price."""
    t = trades.sort_values("timestamp").reset_index(drop=True)
    hits: set[int] = set()
    arr = t[
        ["timestamp", "wallet_id", "side", "price", "quantity", "trade_id"]
    ].to_numpy()
    n = len(arr)
    for i in range(n):
        wi, si, pi, qi = arr[i][1], arr[i][2], float(arr[i][3]), float(arr[i][4])
        for j in range(i + 1, min(i + 40, n)):
            wj, sj, pj, qj = arr[j][1], arr[j][2], float(arr[j][3]), float(arr[j][4])
            dt = (pd.Timestamp(arr[j][0]) - pd.Timestamp(arr[i][0])).total_seconds()
            if dt > ROUND_TRIP_MAX_SEC:
                break
            if wi == wj or si == sj:
                continue
            if abs(pi - pj) / max(pi, 1e-12) > 0.0015:
                continue
            rq = min(qi, qj) / max(qi, qj)
            if rq >= ROUND_TRIP_MIN_QTY_RATIO:
                hits.add(t.index[i])
                hits.add(t.index[j])
    if not hits:
        return pd.DataFrame()
    hit = t.loc[sorted(hits)].copy()
    hit["violation_type"] = "round_trip_wash"
    hit["detector"] = "round_trip_pair"
    hit["score"] = 4
    hit["remarks"] = (
        f"{symbol}: two wallets trade opposite sides within {ROUND_TRIP_MAX_SEC}s "
        "at tight price/size match — reciprocal wash pattern."
    )
    return hit


def detect_ramping(trades: pd.DataFrame, symbol: str) -> pd.DataFrame:
    streak_rows: list[int] = []
    for _, g in trades.groupby("wallet_id"):
        g = g.sort_values("timestamp")
        run_idx: list[int] = []
        last_buy_price = None
        for idx_row, row in g.iterrows():
            if row.side != "BUY":
                if len(run_idx) >= RAMP_MIN_STREAK:
                    streak_rows.extend(run_idx)
                run_idx = []
                last_buy_price = None
                continue
            p = float(row.price)
            if last_buy_price is None:
                run_idx = [idx_row]
                last_buy_price = p
            elif p > last_buy_price * 1.00005:
                run_idx.append(idx_row)
                last_buy_price = p
            else:
                if len(run_idx) >= RAMP_MIN_STREAK:
                    streak_rows.extend(run_idx)
                run_idx = [idx_row]
                last_buy_price = p
        if len(run_idx) >= RAMP_MIN_STREAK:
            streak_rows.extend(run_idx)
    if not streak_rows:
        return pd.DataFrame()
    hit = trades.loc[sorted(set(streak_rows))].copy()
    hit["violation_type"] = "ramping"
    hit["detector"] = "ramping"
    hit["score"] = 3
    hit["remarks"] = (
        f"{symbol}: monotonically rising BUY prices from same wallet "
        f"(streak>={RAMP_MIN_STREAK}) — advancing the bid / ramping."
    )
    return hit


def detect_layering_echo(trades: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Buys pushing up then sells reversing within short window (same wallet)."""
    hits: set[int] = set()
    for w, g in trades.groupby("wallet_id"):
        g = g.sort_values("timestamp")
        if len(g) < 6:
            continue
        sides = g["side"].values
        prices = g["price"].values
        ts = g["timestamp"]
        idx = g.index.values
        i = 0
        while i < len(g) - 3:
            if sides[i] != "BUY":
                i += 1
                continue
            buy_run = [i]
            j = i + 1
            while j < len(g) and sides[j] == "BUY":
                buy_run.append(j)
                j += 1
            if len(buy_run) < 3:
                i += 1
                continue
            sell_run = []
            k = j
            while k < len(g) and sides[k] == "SELL":
                sell_run.append(k)
                k += 1
            if len(sell_run) < 3:
                i += 1
                continue
            t0 = ts.iloc[buy_run[0]]
            t1 = ts.iloc[sell_run[-1]]
            if (t1 - t0).total_seconds() > 600:
                i += 1
                continue
            p_buy_max = max(prices[b] for b in buy_run)
            p_sell_min = min(prices[s] for s in sell_run)
            if p_buy_max > p_sell_min * 1.0002:
                for b in buy_run:
                    hits.add(idx[b])
                for s in sell_run:
                    hits.add(idx[s])
            i = k
    if not hits:
        return pd.DataFrame()
    hit = trades.loc[sorted(hits)].copy()
    hit["violation_type"] = "layering_echo"
    hit["detector"] = "layering_echo"
    hit["score"] = 3
    hit["remarks"] = (
        f"{symbol}: wallet runs BUYs then SELLs within 10m with price extension "
        "and reversion — layering / echo pattern."
    )
    return hit


def detect_aml_structuring(trades: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Tight band of notionals, many trades same wallet-day."""
    t = trades.copy()
    t["_d"] = _trade_date_series(t["timestamp"])
    rows = []
    for (w, d), g in t.groupby(["wallet_id", "_d"]):
        if len(g) < STRUCT_MIN_TRADES:
            continue
        n = g["notional"]
        if n.min() <= 0:
            continue
        cv = n.std() / n.mean() if n.mean() else 1.0
        ratio = n.max() / n.min()
        if cv <= STRUCT_CV_MAX and ratio <= STRUCT_MAX_MIN_RATIO:
            rows.append(g)
    if not rows:
        return pd.DataFrame()
    hit = pd.concat(rows, ignore_index=True)
    hit = hit.drop(columns=["_d"], errors="ignore")
    hit["violation_type"] = "aml_structuring"
    hit["detector"] = "aml_structuring"
    hit["score"] = 4
    hit["remarks"] = hit.apply(
        lambda r: (
            f"{symbol}: {STRUCT_MIN_TRADES}+ trades same wallet/day with near-identical "
            f"notional (~USDT); CV low — classic smurfing band."
        ),
        axis=1,
    )
    return hit


def detect_threshold_testing(trades: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """One trade at round USDT notional then cluster just below (e.g. 10k)."""
    t = trades.copy()
    t["_d"] = _trade_date_series(t["timestamp"])
    hits = []
    for (w, d), g in t.groupby(["wallet_id", "_d"]):
        if len(g) < 5:
            continue
        n = g["notional"].values
        at_10k = np.any((n >= 9950) & (n <= 10050))
        below = n[n < 9950]
        if not at_10k or len(below) < 4:
            continue
        if np.max(below) < 8500:
            continue
        if np.std(below) / (np.mean(below) + 1e-9) > 0.08:
            continue
        hits.append(g)
    if not hits:
        return pd.DataFrame()
    hit = pd.concat(hits, ignore_index=True).drop(columns=["_d"], errors="ignore")
    hit["violation_type"] = "threshold_testing"
    hit["detector"] = "threshold_testing"
    hit["score"] = 4
    hit["remarks"] = (
        f"{symbol}: wallet hits ~10k USDT notional then repeated trades just below — "
        "threshold probing + structuring."
    )
    return hit


def detect_coordinated_pump_minute(trades: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Many distinct wallets buying same minute with large bar tradecount."""
    t = trades.copy()
    t["_m"] = t["timestamp"].dt.floor("min")
    buy = t[t["side"] == "BUY"]
    g = buy.groupby("_m").agg(
        n_wallets=("wallet_id", "nunique"),
        trade_id=("trade_id", "count"),
    )
    if g.empty:
        return pd.DataFrame()
    hot_minutes = g.loc[(g["n_wallets"] >= 14) & (g["trade_id"] >= 22)].index
    if len(hot_minutes) == 0:
        return pd.DataFrame()
    hit = buy[buy["_m"].isin(hot_minutes)].copy()
    hit = hit.drop(columns=["_m"], errors="ignore")
    hit["violation_type"] = "coordinated_pump"
    hit["detector"] = "coordinated_pump_minute"
    hit["score"] = 2
    hit["remarks"] = (
        f"{symbol}: many distinct wallets BUY same minute — coordinated entry / pump footprint."
    )
    return hit


def detect_chain_pass_through(trades: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """W1 SELL -> W2 BUY -> W2 SELL -> W3 BUY within short window (chain_layering hint)."""
    t = trades.sort_values("timestamp").reset_index(drop=True)
    hits: set[int] = set()
    arr = t[["timestamp", "wallet_id", "side", "quantity"]].to_numpy()
    n = len(arr)
    for i in range(n - 3):
        w0, s0, q0 = arr[i][1], arr[i][2], float(arr[i][3])
        if s0 != "SELL":
            continue
        for j in range(i + 1, min(i + 15, n)):
            if (pd.Timestamp(arr[j][0]) - pd.Timestamp(arr[i][0])).total_seconds() > 120:
                break
            if arr[j][2] != "BUY" or arr[j][1] == w0:
                continue
            w1, q1 = arr[j][1], float(arr[j][3])
            if abs(q1 - q0) / max(q0, 1e-12) > 0.35:
                continue
            for k in range(j + 1, min(j + 15, n)):
                if (pd.Timestamp(arr[k][0]) - pd.Timestamp(arr[j][0])).total_seconds() > 120:
                    break
                if arr[k][1] != w1 or arr[k][2] != "SELL":
                    continue
                q2 = float(arr[k][3])
                for m in range(k + 1, min(k + 15, n)):
                    if (pd.Timestamp(arr[m][0]) - pd.Timestamp(arr[k][0])).total_seconds() > 120:
                        break
                    if arr[m][2] != "BUY" or arr[m][1] in (w0, w1):
                        continue
                    q3 = float(arr[m][3])
                    if abs(q3 - q2) / max(q2, 1e-12) <= 0.4:
                        hits.update([t.index[i], t.index[j], t.index[k], t.index[m]])
    if not hits:
        return pd.DataFrame()
    hit = t.loc[sorted(hits)].copy()
    hit["violation_type"] = "chain_layering"
    hit["detector"] = "chain_pass_through"
    hit["score"] = 3
    hit["remarks"] = (
        f"{symbol}: sequential SELL/BUY/SELL/BUY across wallets with matched sizes — "
        "pass-through / layering chain."
    )
    return hit
