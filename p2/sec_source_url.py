"""SEC EDGAR source URLs from filing paths and CIK text (stdlib only — no pandas)."""

from __future__ import annotations

import re
from urllib.parse import quote

SEC_SEARCH_FALLBACK_URL = "https://www.sec.gov/edgar/search/"


def keep_precomputed_source_url(u: str) -> bool:
    """
    If a CSV already has a concrete SEC link, do not overwrite it in ``coerce_p2_signal_columns``
    (Streamlit reload would otherwise re-run ``resolve_p2_source_url`` and ticker-first logic can
    replace headline-CIK URLs from ``scripts/refresh_p2_source_urls.py``).
    """
    s = (u or "").strip()
    if not s:
        return False
    sl = s.lower()
    if sl.rstrip("/") == SEC_SEARCH_FALLBACK_URL.rstrip("/").lower():
        return False
    return "cgi-bin/browse-edgar" in sl or "/archives/edgar/" in sl


# EDGAR full-text hits often embed ``(CIK 0002037431)`` — SEC company pages use 10-digit CIK.
_CIK_PATTERN = re.compile(r"(?:CIK|C\.I\.K\.)\s*[:\s]*(\d{7,10})\b", re.IGNORECASE)


def extract_cik_from_text(text: str) -> str | None:
    """Return CIK zero-padded to 10 digits, or None if not found."""
    if not text or not isinstance(text, str):
        return None
    m = _CIK_PATTERN.search(text)
    if not m:
        return None
    digits = m.group(1).strip()
    if not digits.isdigit():
        return None
    return digits.zfill(10)[-10:]


def sec_edgar_browse_8k_url(cik_or_ticker: str) -> str:
    """
    Company 8-K list on EDGAR. The ``CIK`` query field accepts a 10-digit CIK or an exchange ticker
    (SEC resolves the issuer).
    """
    s = (cik_or_ticker or "").strip()
    if not s:
        return SEC_SEARCH_FALLBACK_URL
    q = quote(s.upper(), safe="")
    return (
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
        f"&CIK={q}&type=8-K&owner=exclude&count=40&hidefilings=0"
    )


_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,14}$")


def _normalize_equity_ticker(raw: str) -> str:
    """Ticker from our map / EDGAR join (uppercase, strip junk); empty if not usable for SEC lookup."""
    t = (raw or "").strip().upper()
    if not t or t in ("NAN", "NONE", "NAT", "<NA>", "NULL"):
        return ""
    if not _TICKER_RE.match(t):
        return ""
    return t


def resolve_p2_source_url(
    filing_url: str,
    headline: str,
    entity_name: str = "",
    ticker: str = "",
    *,
    prefer_listing_ticker: bool = False,
) -> str:
    """
    Prefer the direct Archives document URL.

    Otherwise build browse-edgar 8-K:

    - Default (``prefer_listing_ticker=False``): **CIK from headline/entity first**, then listing
      **ticker**. EDGAR full-text hits are searched by ticker but the hit text often names another
      filer — matching the link to the headline/remarks avoids apparent row mismatches in the UI.
    - ``prefer_listing_ticker=True``: **ticker** first (align URL with ``sec_id``'s map symbol), then
      headline CIK (used by ``scripts/refresh_p2_source_urls.py --prefer-ohlcv-ticker-for-url``).
    """
    u = (filing_url or "").strip()
    if u.startswith("https://www.sec.gov/Archives/edgar/"):
        return u
    tk = _normalize_equity_ticker(ticker)
    cik = extract_cik_from_text(headline or "") or extract_cik_from_text(entity_name or "")
    if prefer_listing_ticker:
        if tk:
            return sec_edgar_browse_8k_url(tk)
        if cik:
            return sec_edgar_browse_8k_url(cik)
    else:
        if cik:
            return sec_edgar_browse_8k_url(cik)
        if tk:
            return sec_edgar_browse_8k_url(tk)
    return SEC_SEARCH_FALLBACK_URL
