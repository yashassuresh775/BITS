# EDGAR API Starter — Problem 2

This snippet fetches 8-K filings from SEC EDGAR for a list of tickers and a date range. Test this before the event — confirm it returns JSON on your machine.

---

## Quick test (run this now)

```bash
curl "https://efts.sec.gov/LATEST/search-index?q=%22acquisition%22&forms=8-K&dateRange=custom&startdt=2026-01-01&enddt=2026-02-28" | python3 -m json.tool | head -60
```

You should see a JSON blob with a `hits` key. If you do, the API is working.

---

## Working Python starter

```python
import requests
import pandas as pd
from time import sleep

EDGAR_URL = "https://efts.sec.gov/LATEST/search-index"

def fetch_8k_filings(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch 8-K filings from SEC EDGAR for a list of ticker symbols.

    Parameters
    ----------
    tickers    : list of ticker strings, e.g. ["AAPL", "MSFT"]
    start_date : "YYYY-MM-DD"
    end_date   : "YYYY-MM-DD"

    Returns
    -------
    DataFrame with columns: entity_name, ticker, file_date, form_type, filing_url
    """
    results = []

    for ticker in tickers:
        params = {
            "q": f'"{ticker}"',
            "forms": "8-K",
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
        }

        response = requests.get(EDGAR_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        hits = data.get("hits", {}).get("hits", [])
        for hit in hits:
            src = hit.get("_source", {})
            results.append({
                "entity_name": src.get("entity_name", ""),
                "ticker":      ticker,
                "file_date":   src.get("file_date", ""),
                "form_type":   src.get("form_type", ""),
                "filing_url":  "https://www.sec.gov" + src.get("file_path", ""),
            })

        sleep(0.3)  # be polite to the API

    df = pd.DataFrame(results)
    if not df.empty:
        df["file_date"] = pd.to_datetime(df["file_date"])
        df = df.sort_values("file_date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    # Replace with the tickers in your ohlcv.csv
    sample_tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "JPM"]

    filings = fetch_8k_filings(
        tickers=sample_tickers,
        start_date="2026-01-01",
        end_date="2026-02-28",
    )

    print(f"Found {len(filings)} 8-K filings\n")
    print(filings[["entity_name", "ticker", "file_date", "form_type"]].to_string(index=False))
```

---

## Connecting EDGAR tickers to your trade data

The `sec_id` in `ohlcv.csv` and `trade_data.csv` is an internal numeric identifier. You will need a mapping from `sec_id` to ticker symbol to join the EDGAR filings back to your trade data.

The organisers will provide this mapping at kickoff as a small reference file:

```
sec_id,ticker
10042,AAPL
10081,MSFT
...
```

Load it and join:

```python
ticker_map = pd.read_csv("sec_id_map.csv")
filings = filings.merge(ticker_map, on="ticker", how="left")
# now filings has sec_id — join to ohlcv on (sec_id, file_date)
```

---

## What to look for after you have the dates

Once you have `(sec_id, file_date)` pairs:

```python
# Compute rolling 15-day volume baseline per ticker
ohlcv = pd.read_csv("ohlcv.csv", parse_dates=["trade_date"])
ohlcv = ohlcv.sort_values(["sec_id", "trade_date"])
ohlcv["vol_15d_mean"] = ohlcv.groupby("sec_id")["volume"].transform(
    lambda x: x.shift(1).rolling(15).mean()
)
ohlcv["vol_15d_std"] = ohlcv.groupby("sec_id")["volume"].transform(
    lambda x: x.shift(1).rolling(15).std()
)
ohlcv["volume_z"] = (ohlcv["volume"] - ohlcv["vol_15d_mean"]) / ohlcv["vol_15d_std"]

# For each filing, check the T-2 and T-1 days
for _, filing in filings.iterrows():
    window = ohlcv[
        (ohlcv["sec_id"] == filing["sec_id"]) &
        (ohlcv["trade_date"] >= filing["file_date"] - pd.Timedelta(days=5)) &
        (ohlcv["trade_date"] < filing["file_date"])
    ]
    if (window["volume_z"] > 3).any():
        print(f"FLAGGED: {filing['ticker']} before {filing['file_date'].date()} — volume z-score {window['volume_z'].max():.1f}")
```

---

## Common issues

- **Rate limiting:** Add `sleep(0.3)` between requests. The API is public but will throttle aggressive crawlers.
- **Company name vs ticker:** EDGAR indexes by company name, not ticker. Some tickers will return no results if the company name differs significantly — try the company's legal name if a ticker returns nothing.
- **Date format:** EDGAR expects `YYYY-MM-DD`. Pass strings, not datetime objects, in the `params` dict.
