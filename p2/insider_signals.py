"""Cross-reference EDGAR filing dates with OHLCV + trade activity (Problem 2 signal rules).

Lightweight URL/CIK helpers load from ``p2.sec_source_url`` without pandas. Pipeline functions
normally load from ``p2._insider_signals_impl`` via explicit import so ``from p2.insider_signals
import coerce_p2_signal_columns`` works under Streamlit and plain importers (not only ``__getattr__``).
"""

from __future__ import annotations

import importlib
from typing import Any

from p2.sec_source_url import (
    SEC_SEARCH_FALLBACK_URL,
    extract_cik_from_text,
    resolve_p2_source_url,
    sec_edgar_browse_8k_url,
)

__all__ = [
    "SEC_SEARCH_FALLBACK_URL",
    "extract_cik_from_text",
    "resolve_p2_source_url",
    "sec_edgar_browse_8k_url",
    "coerce_p2_signal_columns",
    "normalize_ohlcv",
    "normalize_trades",
    "compute_pre_drift_flags",
    "enrich_remarks_with_trades",
    "build_p2_signals",
]

_PIPELINE_NAMES = frozenset(
    {
        "coerce_p2_signal_columns",
        "normalize_ohlcv",
        "normalize_trades",
        "compute_pre_drift_flags",
        "enrich_remarks_with_trades",
        "build_p2_signals",
    }
)

def _deferred_impl_callable(name: str):
    """Bind a real function object so ``from p2.insider_signals import …`` always works."""

    def _call(*args: Any, **kwargs: Any) -> Any:
        mod = importlib.import_module("p2._insider_signals_impl")
        return getattr(mod, name)(*args, **kwargs)

    _call.__name__ = _call.__qualname__ = name
    return _call


try:
    from p2._insider_signals_impl import (
        build_p2_signals,
        coerce_p2_signal_columns,
        compute_pre_drift_flags,
        enrich_remarks_with_trades,
        normalize_ohlcv,
        normalize_trades,
    )
except ImportError:
    # Eager import failed (e.g. no pandas in this interpreter). ``from … import coerce_…`` does
    # not use module ``__getattr__`` reliably; keep real callables in ``__dict__`` that load impl on use.
    coerce_p2_signal_columns = _deferred_impl_callable("coerce_p2_signal_columns")
    normalize_ohlcv = _deferred_impl_callable("normalize_ohlcv")
    normalize_trades = _deferred_impl_callable("normalize_trades")
    compute_pre_drift_flags = _deferred_impl_callable("compute_pre_drift_flags")
    enrich_remarks_with_trades = _deferred_impl_callable("enrich_remarks_with_trades")
    build_p2_signals = _deferred_impl_callable("build_p2_signals")


def __getattr__(name: str) -> Any:
    if name == "__path__":
        raise AttributeError(name)
    if name not in _PIPELINE_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        mod = importlib.import_module("p2._insider_signals_impl")
    except ImportError as e:
        raise ImportError(
            "Problem 2 pipeline requires project dependencies (pandas, numpy, …). "
            "From the repo root: python3 -m pip install -r requirements.txt "
            "or python3 -m venv .venv && .venv/bin/pip install -r requirements.txt "
            "then use .venv/bin/python."
        ) from e
    return getattr(mod, name)


def __dir__() -> list[str]:
    return sorted(__all__)
