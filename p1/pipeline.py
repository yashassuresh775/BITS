"""Cluster order-book anomalies and build p1_alerts.csv."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from p1.config import (
    CONC_HIGH,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    MAX_ALERTS,
    MIN_SUSTAINED_MINUTES,
    OBI_EXTREME,
    OBI_HIGH,
    SPREAD_Z_ALERT,
)
from p1.features import attach_trade_aggression, compute_row_features, enrich_all


def _candidate_mask(df: pd.DataFrame) -> pd.Series:
    """Flag minutes worth clustering (sustained imbalance, concentration, or wide spread)."""
    eo = (df["obi"].abs() >= OBI_EXTREME).astype(int)
    sustained = (
        eo.groupby(df["sec_id"])
        .rolling(MIN_SUSTAINED_MINUTES, min_periods=MIN_SUSTAINED_MINUTES)
        .sum()
        .reset_index(level=0, drop=True)
        >= MIN_SUSTAINED_MINUTES
    )
    hi_conc = (df["bid_concentration"] >= CONC_HIGH) | (df["ask_concentration"] >= CONC_HIGH)
    wide_sp = df["spread_bps_z"].abs() >= SPREAD_Z_ALERT
    return sustained | hi_conc | wide_sp


def _feature_matrix(sub: pd.DataFrame) -> np.ndarray:
    depth = sub["depth_ratio"].replace([np.inf, -np.inf], np.nan)
    med = float(depth.median()) if depth.notna().any() else 1.0
    depth = depth.fillna(med)
    X = np.column_stack(
        [
            sub["obi"].values,
            np.clip(sub["spread_bps_z"].values, -12, 12),
            sub["bid_concentration"].values,
            sub["ask_concentration"].values,
            np.log1p(depth.clip(1e-6, 1e6).values),
        ]
    )
    return X


def _label_cluster(sub: pd.DataFrame) -> tuple[str, str]:
    """Return (anomaly_type, short remark fragment) from cluster centroid."""
    mobi = float(sub["obi"].mean())
    msp = float(sub["spread_bps_z"].mean())
    mbc = float(sub["bid_concentration"].mean())
    mac = float(sub["ask_concentration"].mean())
    if msp >= SPREAD_Z_ALERT * 0.85:
        return (
            "wide_spread_microstructure",
            f"spread z≈{msp:.1f}σ vs ticker’s rolling baseline; top-of-book gap widened vs recent history",
        )
    if mbc >= CONC_HIGH and mbc >= mac:
        return (
            "level_one_bid_concentration",
            f"~{mbc*100:.0f}% of bid depth stacked at best bid — unusual depth asymmetry",
        )
    if mac >= CONC_HIGH and mac > mbc:
        return (
            "level_one_ask_concentration",
            f"~{mac*100:.0f}% of ask depth at best ask — thin ladder behind the touch",
        )
    if mobi >= OBI_EXTREME:
        return (
            "order_book_imbalance_bid_heavy",
            f"sustained OBI≈{mobi:.2f} (bid depth dominates total book)",
        )
    if mobi <= -OBI_EXTREME:
        return (
            "order_book_imbalance_ask_heavy",
            f"sustained OBI≈{mobi:.2f} (ask depth dominates total book)",
        )
    return (
        "order_book_microstructure_cluster",
        f"multi-feature outlier cluster (OBI {mobi:.2f}, bid L1 share {mbc:.2f}, ask L1 share {mac:.2f})",
    )


def _severity(sub: pd.DataFrame) -> str:
    dur = len(sub)
    max_obi = float(sub["obi"].abs().max())
    max_sp = float(sub["spread_bps_z"].abs().max())
    if max_obi >= OBI_HIGH or max_sp >= SPREAD_Z_ALERT + 1.5 or dur >= 18:
        return "HIGH"
    if max_obi >= OBI_EXTREME or max_sp >= SPREAD_Z_ALERT or dur >= 8:
        return "MEDIUM"
    return "LOW"


def _split_contiguous(seg: pd.DataFrame) -> list[pd.DataFrame]:
    """Split on gaps longer than 3 minutes."""
    seg = seg.sort_values("minute")
    if seg.empty:
        return []
    gap = seg["minute"].diff().dt.total_seconds().fillna(0) > 180
    gid = gap.cumsum()
    return [g for _, g in seg.groupby(gid)]


def _cluster_one_ticker(sec: float, g: pd.DataFrame, trades_note: str) -> list[dict[str, Any]]:
    """Returns list of alert dicts for one sec_id."""
    alerts: list[dict[str, Any]] = []
    mask = _candidate_mask(g)
    cand = g.loc[mask].copy()
    if cand.empty:
        return alerts

    n = len(cand)
    ms = min(DBSCAN_MIN_SAMPLES, max(2, n // 3))
    if n < ms:
        cand["_cluster"] = 0
    else:
        X = _feature_matrix(cand)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        lab = DBSCAN(eps=DBSCAN_EPS, min_samples=ms).fit_predict(Xs)
        cand["_cluster"] = lab

    for lab in sorted(cand["_cluster"].unique()):
        part = cand[cand["_cluster"] == lab].sort_values("minute")
        for seg in _split_contiguous(part):
            if seg.empty:
                continue
            atype, frag = _label_cluster(seg)
            t0 = seg["minute"].min()
            trade_date = t0.strftime("%Y-%m-%d")
            tw = t0.strftime("%H:%M:%S")
            sev = _severity(seg)
            dur = len(seg)
            buy_note = ""
            if "buy_vs_bid_depth" in seg.columns and seg["buy_vs_bid_depth"].notna().any():
                mx = float(seg["buy_vs_bid_depth"].max())
                if mx > 0.35:
                    buy_note = (
                        f" Trade prints show aggressive buying ≈{mx*100:.0f}% of visible bid depth "
                        "in the same minute(s)."
                    )
            remarks = (
                f"{frag} Window spans {dur} minute(s) starting {tw}. "
                f"10m OBI std≈{float(seg['obi_roll_std_10'].mean()):.3f}."
                f"{buy_note}{trades_note}"
            )
            alerts.append(
                {
                    "sec_id": int(sec) if pd.notna(sec) else sec,
                    "trade_date": trade_date,
                    "time_window_start": tw,
                    "anomaly_type": atype,
                    "severity": sev,
                    "remarks": remarks.replace("\n", " ")[:1800],
                    "_dur": dur,
                    "_sev": {"HIGH": 3, "MEDIUM": 2, "LOW": 1}[sev],
                }
            )
    return alerts


def build_alerts(
    market_df: pd.DataFrame,
    trades_per_min: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, float]:
    t0 = time.perf_counter()
    fe = enrich_all(compute_row_features(market_df))
    fe = attach_trade_aggression(fe, trades_per_min)

    all_rows: list[dict[str, Any]] = []
    trades_note = ""
    if trades_per_min is not None and not trades_per_min.empty:
        trades_note = " Client trades (optional) used only for buy-vs-depth ratio in remarks."

    for sec, g in fe.groupby("sec_id", sort=False):
        all_rows.extend(_cluster_one_ticker(sec, g, trades_note))

    elapsed = time.perf_counter() - t0
    if not all_rows:
        out = pd.DataFrame(
            columns=[
                "alert_id",
                "sec_id",
                "trade_date",
                "time_window_start",
                "anomaly_type",
                "severity",
                "remarks",
                "time_to_run",
            ]
        )
        return out, elapsed

    out = pd.DataFrame(all_rows)
    out = out.sort_values(["_sev", "_dur"], ascending=[False, False])
    out = out.drop(columns=["_dur", "_sev"], errors="ignore")
    out = out.head(MAX_ALERTS).reset_index(drop=True)
    out.insert(0, "alert_id", range(1, len(out) + 1))
    out["time_to_run"] = round(elapsed, 3)
    return out, elapsed
