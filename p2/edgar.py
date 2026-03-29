"""Fetch 8-K filings from SEC EDGAR full-text search (organizer starter + extras).

Follows ``data/student-pack/docs/edgar_starter_snippet.md`` (same URL, params, ``sleep(0.3)``,
``timeout=10`` defaults). This module adds:

- Descriptive ``User-Agent`` / ``Accept`` (SEC guidance)
- Optional ``headline`` text for ``classify_event``
- ``merge_sec_ids`` for ``sec_id`` join
- Short retries on HTTP 5xx (transient SEC errors)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from time import sleep

import pandas as pd
import requests

EDGAR_URL = "https://efts.sec.gov/LATEST/search-index"

# Starter defaults (see edgar_starter_snippet.md)
DEFAULT_SLEEP_S = 0.3
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_RETRIES = 3
# Small parallel bursts (separate Session per ticker) shorten dashboard/CLI wall time vs strict sequential.
DEFAULT_BATCH_CONCURRENCY = 3

# SEC asks for a descriptive User-Agent; replace contact if you ship this.
DEFAULT_HEADERS = {
    "User-Agent": "BITS-Hackathon/1.0 (contact: research@localhost; educational)",
    "Accept": "application/json",
}


def normalize_edgar_ymd(value: str | date | datetime | pd.Timestamp) -> str:
    """
    EDGAR ``startdt`` / ``enddt`` must be ``YYYY-MM-DD`` strings in the request params
    (not datetime objects — see organizer common issues).
    """
    if hasattr(value, "strftime") and callable(getattr(value, "strftime")):
        return value.strftime("%Y-%m-%d")  # type: ignore[union-attr]
    s = str(value).strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Not a valid EDGAR date (need YYYY-MM-DD): {value!r}")
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def build_edgar_search_overrides(ticker_map: pd.DataFrame) -> dict[str, str]:
    """
    Optional column ``edgar_query`` on ``sec_id_map.csv`` / ticker map: legal name or
    phrase to put inside the quoted ``q`` param when the ticker alone returns no hits
    (EDGAR search is name-oriented).
    """
    if ticker_map.empty or "edgar_query" not in ticker_map.columns:
        return {}
    out: dict[str, str] = {}
    for _, r in ticker_map.iterrows():
        q = str(r.get("edgar_query", "") or "").strip()
        if not q:
            continue
        t = str(r.get("ticker", "") or "").upper().strip()
        if t:
            out[t] = q
    return out


def classify_event(headline: str) -> str:
    headline_lower = (headline or "").lower()
    event_keywords = {
        "merger": [
            "merger",
            "acquisition",
            "acquired",
            "takeover",
            "combine",
            "business combination",
        ],
        "earnings": [
            "earnings",
            "revenue",
            "quarterly results",
            "guidance",
            "eps",
        ],
        "leadership": [
            "ceo",
            "chief executive",
            "resign",
            "appoint",
            "board",
            "officer",
        ],
        "restatement": [
            "restate",
            "restatement",
            "correction",
            "material weakness",
        ],
    }
    for event_type, keywords in event_keywords.items():
        if any(kw in headline_lower for kw in keywords):
            return event_type
    return "other"


def _get_json_with_retries(
    sess: requests.Session,
    params: dict[str, str],
    *,
    timeout: float,
    max_retries: int,
) -> dict:
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = sess.get(EDGAR_URL, params=params, headers=DEFAULT_HEADERS, timeout=timeout)
            if r.status_code >= 500 and attempt < max_retries - 1:
                sleep(1.0 + attempt * 0.5)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_err = e
            if attempt < max_retries - 1:
                sleep(1.0 + attempt * 0.5)
    assert last_err is not None
    raise last_err


def _fetch_8k_rows_for_ticker(
    ticker: str,
    phrase: str,
    start_ymd: str,
    end_ymd: str,
    *,
    timeout: float,
    max_retries: int,
) -> list[dict[str, str]]:
    """One EDGAR request; uses its own Session (thread-safe for parallel bursts)."""
    sess = requests.Session()
    params = {
        "q": f'"{phrase}"',
        "forms": "8-K",
        "dateRange": "custom",
        "startdt": start_ymd,
        "enddt": end_ymd,
    }
    data = _get_json_with_retries(sess, params, timeout=timeout, max_retries=max_retries)
    hits = data.get("hits", {}).get("hits", [])
    rows: list[dict[str, str]] = []
    for hit in hits:
        src = hit.get("_source", {})
        path = src.get("file_path", "") or ""
        headline = src.get("display_names", "")
        if isinstance(headline, list):
            headline = " ".join(str(x) for x in headline)
        if not headline:
            headline = src.get("file_name", "") or entity_snippet(src)
        rows.append(
            {
                "entity_name": src.get("entity_name", ""),
                "ticker": ticker,
                "file_date": src.get("file_date", ""),
                "form_type": src.get("form_type", ""),
                "filing_url": "https://www.sec.gov" + path if path else "",
                "headline": str(headline)[:500],
            }
        )
    return rows


def fetch_8k_filings(
    tickers: list[str],
    start_date: str,
    end_date: str,
    sleep_s: float = DEFAULT_SLEEP_S,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
    session: requests.Session | None = None,
    search_overrides: dict[str, str] | None = None,
    batch_concurrency: int = DEFAULT_BATCH_CONCURRENCY,
) -> pd.DataFrame:
    """
    Fetch 8-K filings for tickers (quoted search per ticker).

    Same request shape as the organizer starter: ``q`` = ``"…"``, ``forms=8-K``,
    ``dateRange=custom``, ``startdt`` / ``enddt`` as ``YYYY-MM-DD`` strings.

    Pass ``search_overrides`` (ticker → phrase) for legal-name queries when the
    ticker string alone is a poor match.

    ``batch_concurrency``: requests per burst; ``sleep_s`` pauses between bursts
    (``1`` = sequential + sleep after every ticker, matching the starter literally).

    When ``batch_concurrency > 1``, ``session`` is ignored (each call uses a fresh Session).

    Returns columns: entity_name, ticker, file_date, form_type, filing_url, headline
    """
    start_ymd = normalize_edgar_ymd(start_date)
    end_ymd = normalize_edgar_ymd(end_date)
    ov = {k.upper().strip(): v for k, v in (search_overrides or {}).items()}
    tickers_list = [str(t).upper().strip() for t in tickers]
    results: list[dict[str, str]] = []
    bc = max(1, int(batch_concurrency))

    if bc <= 1:
        sess = session or requests.Session()
        for ticker in tickers_list:
            phrase = ov.get(ticker, ticker)
            params = {
                "q": f'"{phrase}"',
                "forms": "8-K",
                "dateRange": "custom",
                "startdt": start_ymd,
                "enddt": end_ymd,
            }
            data = _get_json_with_retries(sess, params, timeout=timeout, max_retries=max_retries)
            hits = data.get("hits", {}).get("hits", [])
            for hit in hits:
                src = hit.get("_source", {})
                path = src.get("file_path", "") or ""
                headline = src.get("display_names", "")
                if isinstance(headline, list):
                    headline = " ".join(str(x) for x in headline)
                if not headline:
                    headline = src.get("file_name", "") or entity_snippet(src)
                results.append(
                    {
                        "entity_name": src.get("entity_name", ""),
                        "ticker": ticker,
                        "file_date": src.get("file_date", ""),
                        "form_type": src.get("form_type", ""),
                        "filing_url": "https://www.sec.gov" + path if path else "",
                        "headline": str(headline)[:500],
                    }
                )
            sleep(sleep_s)
    else:
        for i in range(0, len(tickers_list), bc):
            if i > 0:
                sleep(sleep_s)
            chunk = tickers_list[i : i + bc]
            with ThreadPoolExecutor(max_workers=len(chunk)) as pool:
                futs = [
                    pool.submit(
                        _fetch_8k_rows_for_ticker,
                        t,
                        ov.get(t, t),
                        start_ymd,
                        end_ymd,
                        timeout=timeout,
                        max_retries=max_retries,
                    )
                    for t in chunk
                ]
                for fut in as_completed(futs):
                    results.extend(fut.result())

    df = pd.DataFrame(results)
    if df.empty:
        return df
    df["file_date"] = pd.to_datetime(df["file_date"], errors="coerce")
    df = df.dropna(subset=["file_date"])
    df["event_type"] = df["headline"].map(classify_event)
    return df.sort_values("file_date").reset_index(drop=True)


def entity_snippet(src: dict) -> str:
    for key in ("biz_locations", "items", "file_name"):
        v = src.get(key)
        if v:
            return str(v)[:300]
    return "8-K filing"


def merge_sec_ids(filings: pd.DataFrame, ticker_map: pd.DataFrame) -> pd.DataFrame:
    """Join filings to numeric sec_id via ticker map (columns: ticker, sec_id)."""
    m = ticker_map.copy()
    m["ticker"] = m["ticker"].astype(str).str.upper().str.strip()
    f = filings.copy()
    f["ticker"] = f["ticker"].astype(str).str.upper().str.strip()
    out = f.merge(m, on="ticker", how="left")
    return out
