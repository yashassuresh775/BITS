"""Cluster order-book anomalies and build p1_alerts.csv."""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any

import numpy as np

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None  # type: ignore[misc, assignment]
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from p1.config import (
    CONC_HIGH,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    HHI_Z_ALERT,
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
    hhi_shape = (df["bid_hhi_z"].abs() >= HHI_Z_ALERT) | (df["ask_hhi_z"].abs() >= HHI_Z_ALERT)
    return sustained | hi_conc | wide_sp | hhi_shape


def _feature_matrix(sub: pd.DataFrame) -> np.ndarray:
    depth = sub["depth_ratio"].replace([np.inf, -np.inf], np.nan)
    med = float(depth.median()) if depth.notna().any() else 1.0
    depth = depth.fillna(med)
    X = np.column_stack(
        [
            sub["obi"].values,
            np.clip(sub["obi_vs_roll_z"].to_numpy(dtype=np.float64, copy=False), -8, 8),
            np.clip(sub["spread_bps_z"].values, -12, 12),
            sub["bid_concentration"].values,
            sub["ask_concentration"].values,
            np.log1p(depth.clip(1e-6, 1e6).values),
            np.clip(sub["bid_hhi_z"].to_numpy(dtype=np.float64, copy=False), -8, 8),
            np.clip(sub["ask_hhi_z"].to_numpy(dtype=np.float64, copy=False), -8, 8),
        ]
    )
    return X


def _severity_np(dur: int, obi: np.ndarray, spread_z: np.ndarray) -> str:
    max_obi = float(np.max(np.abs(obi))) if obi.size else 0.0
    max_sp = float(np.max(np.abs(spread_z))) if spread_z.size else 0.0
    if max_obi >= OBI_HIGH or max_sp >= SPREAD_Z_ALERT + 1.5 or dur >= 18:
        return "HIGH"
    if max_obi >= OBI_EXTREME or max_sp >= SPREAD_Z_ALERT or dur >= 8:
        return "MEDIUM"
    return "LOW"


def _label_cluster_np(
    obi: np.ndarray,
    sp: np.ndarray,
    mbc: np.ndarray,
    mac: np.ndarray,
    bhz: np.ndarray,
    ahz: np.ndarray,
) -> tuple[str, str]:
    """Label segment from centroid features (spread, L1 conc, OBI, cross-level HHI z)."""
    mobi = float(np.mean(obi))
    msp = float(np.mean(sp))
    mbc_m = float(np.mean(mbc))
    mac_m = float(np.mean(mac))
    mbhz = float(np.mean(bhz)) if bhz.size else 0.0
    mahz = float(np.mean(ahz)) if ahz.size else 0.0
    if msp >= SPREAD_Z_ALERT * 0.85:
        return (
            "wide_spread_microstructure",
            f"spread z≈{msp:.1f}σ vs ticker’s rolling baseline; top-of-book gap widened vs recent history",
        )
    if mbc_m >= CONC_HIGH and mbc_m >= mac_m:
        return (
            "level_one_bid_concentration",
            f"~{mbc_m*100:.0f}% of bid depth stacked at best bid — unusual depth asymmetry",
        )
    if mac_m >= CONC_HIGH and mac_m > mbc_m:
        return (
            "level_one_ask_concentration",
            f"~{mac_m*100:.0f}% of ask depth at best ask — thin ladder behind the touch",
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
    if max(abs(mbhz), abs(mahz)) >= HHI_Z_ALERT * 0.85:
        return (
            "cross_level_depth_asymmetry",
            f"depth spread across L1–L10 unusual vs this ticker’s history "
            f"(bid HHI z≈{mbhz:.1f}, ask HHI z≈{mahz:.1f} — layering-style shape signal)",
        )
    return (
        "order_book_microstructure_cluster",
        f"multi-feature outlier cluster (OBI {mobi:.2f}, bid L1 share {mbc_m:.2f}, ask L1 share {mac_m:.2f})",
    )


_GAP_NS = 180 * 1_000_000_000


def _contiguous_row_groups(sorted_positions: np.ndarray, minute_ns: np.ndarray) -> list[np.ndarray]:
    """``sorted_positions`` = row indices into candidate frame, ordered by time."""
    if sorted_positions.size == 0:
        return []
    m = minute_ns[sorted_positions.astype(np.intp)]
    if sorted_positions.size == 1:
        return [sorted_positions]
    d = m[1:].astype(np.int64) - m[:-1].astype(np.int64)
    splits = np.flatnonzero(d > _GAP_NS) + 1
    starts = np.r_[0, splits]
    ends = np.r_[splits, sorted_positions.size]
    return [sorted_positions[starts[i] : ends[i]] for i in range(len(starts))]


def _cluster_one_ticker(sec: float, g: pd.DataFrame, trades_note: str) -> list[dict[str, Any]]:
    """Returns list of alert dicts for one sec_id."""
    alerts: list[dict[str, Any]] = []
    mask = _candidate_mask(g)
    cand = g.loc[mask]
    if cand.empty:
        return alerts

    n = len(cand)
    ms = min(DBSCAN_MIN_SAMPLES, max(2, n // 3))
    if n < ms:
        cl = np.zeros(n, dtype=np.int32)
    else:
        X = _feature_matrix(cand)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X).astype(np.float32, copy=False)
        # n_jobs=1 avoids per-ticker joblib pool startup (dominant vs fit time at this scale).
        # ball_tree is slightly faster than auto on typical candidate counts in this pack.
        cl = DBSCAN(
            eps=DBSCAN_EPS, min_samples=ms, n_jobs=1, algorithm="ball_tree"
        ).fit_predict(Xs).astype(np.int32, copy=False)

    # Numpy path: avoid thousands of small DataFrame slices in inner loops.
    minute_ns = cand["minute"].values.astype("datetime64[ns]").astype(np.int64)
    obi_a = cand["obi"].to_numpy(dtype=np.float64, copy=False)
    sp_a = cand["spread_bps_z"].to_numpy(dtype=np.float64, copy=False)
    mbc_a = cand["bid_concentration"].to_numpy(dtype=np.float64, copy=False)
    mac_a = cand["ask_concentration"].to_numpy(dtype=np.float64, copy=False)
    bhz_a = cand["bid_hhi_z"].to_numpy(dtype=np.float64, copy=False)
    ahz_a = cand["ask_hhi_z"].to_numpy(dtype=np.float64, copy=False)
    roll_a = cand["obi_roll_std_10"].to_numpy(dtype=np.float64, copy=False)
    if "buy_vs_bid_depth" in cand.columns:
        buy_a = cand["buy_vs_bid_depth"].to_numpy(dtype=np.float64, copy=False)
    else:
        buy_a = np.full(n, np.nan, dtype=np.float64)
    minute_dt = cand["minute"].values

    sid_out = int(sec) if pd.notna(sec) else sec
    for lab in np.unique(cl):
        row_idx = np.flatnonzero(cl == lab)
        order = row_idx[np.argsort(minute_ns[row_idx], kind="mergesort")]
        for grp in _contiguous_row_groups(order, minute_ns):
            if grp.size == 0:
                continue
            go = obi_a[grp]
            gs = sp_a[grp]
            gb = mbc_a[grp]
            ga = mac_a[grp]
            atype, frag = _label_cluster_np(go, gs, gb, ga, bhz_a[grp], ahz_a[grp])
            t0 = pd.Timestamp(minute_dt[grp[0]])
            trade_date = t0.strftime("%Y-%m-%d")
            tw = t0.strftime("%H:%M:%S")
            dur = int(grp.size)
            sev = _severity_np(dur, go, gs)
            buy_note = ""
            bsub = buy_a[grp]
            if np.isfinite(bsub).any():
                bmx = float(np.nanmax(bsub))
                if bmx > 0.35:
                    buy_note = (
                        f" Trade prints show aggressive buying ≈{bmx*100:.0f}% of visible bid depth "
                        "in the same minute(s)."
                    )
            rsub = roll_a[grp]
            obi_std_m = float(np.nanmean(rsub)) if np.isfinite(rsub).any() else 0.0
            remarks = (
                f"{frag} Window spans {dur} minute(s) starting {tw}. "
                f"10m OBI std≈{obi_std_m:.3f}."
                f"{buy_note}{trades_note}"
            )
            alerts.append(
                {
                    "sec_id": sid_out,
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
    # One BLAS thread per process avoids oversubscription across many small DBSCAN fits.
    tp_ctx = (
        threadpool_limits(limits=1)
        if threadpool_limits is not None
        else nullcontext()
    )
    with tp_ctx:
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
