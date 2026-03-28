"""Fetch 8-K filings from SEC EDGAR full-text search (public API)."""

from __future__ import annotations

from time import sleep

import pandas as pd
import requests

EDGAR_URL = "https://efts.sec.gov/LATEST/search-index"

# SEC asks for a descriptive User-Agent; replace contact if you ship this.
DEFAULT_HEADERS = {
    "User-Agent": "BITS-Hackathon/1.0 (contact: research@localhost; educational)",
    "Accept": "application/json",
}


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


def fetch_8k_filings(
    tickers: list[str],
    start_date: str,
    end_date: str,
    sleep_s: float = 0.31,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """
    Fetch 8-K filings for tickers (quoted search per ticker).

    Returns columns: entity_name, ticker, file_date, form_type, filing_url, headline
    """
    sess = session or requests.Session()
    results: list[dict[str, str]] = []
    for ticker in tickers:
        params = {
            "q": f'"{ticker}"',
            "forms": "8-K",
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
        }
        r = sess.get(EDGAR_URL, params=params, headers=DEFAULT_HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
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
