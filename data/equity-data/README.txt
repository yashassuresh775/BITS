Place equity student-pack files here (same folder):

  market_data.csv
  ohlcv.csv
  trade_data.csv
  (optional) sec_id_map.csv

Or set environment variable EQUITY_ROOT (or EQUITY_DATA_STREAMLIT_DEFAULT) to that folder
so the dashboard defaults to the correct path.

Example (macOS):
  export EQUITY_ROOT="$HOME/Downloads/student-pack/equity"

Regenerate equity outputs from repo root (venv recommended):
  python3 run_p1.py --data-root data/equity-data -o p1_alerts.csv
  python3 run_p2.py --data-root data/equity-data -o p2_signals.csv --start-date 2026-01-01 --end-date 2026-03-31
Optional P1 env: P1_SPREAD_ROLL_LONG=11700 for a longer spread/HHI baseline (see README.md).
