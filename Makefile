# macOS often has no `python` on PATH — use this Makefile or `python3` / `.venv/bin/python`.
PY ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python3)

.PHONY: run p1 p2 dual live-once dashboard benchmark fetch-hist eda-stats

run:
	$(PY) run_p3.py -o submission.csv

# Problem 1 — equity: market_data.csv (+ optional trade_data.csv)
# Example: make p1 P1_ROOT=./data/equity-data
p1:
	@test -n "$(P1_ROOT)" || (echo "Set P1_ROOT=/path/to/equity-data" && exit 1)
	$(PY) run_p1.py --data-root "$(P1_ROOT)" -o p1_alerts.csv

# Problem 2 — ohlcv.csv, trade_data.csv (ticker map in ohlcv if no sec_id_map.csv)
p2:
	@test -n "$(P2_ROOT)" || (echo "Set P2_ROOT=/path/to/equity-pack" && exit 1)
	$(PY) run_p2.py --data-root "$(P2_ROOT)" -o p2_signals.csv

dual:
	$(PY) run_p3.py --dual

live-once:
	$(PY) run_p3.py --live --live-once -o submission_live.csv

dashboard:
	$(PY) -m streamlit run dashboard/app.py

benchmark:
	$(PY) scripts/benchmark_p3.py --runs 1

# Historical Binance → data/binance-hist/ (pass args: make fetch-hist ARGS='--days 7')
fetch-hist:
	$(PY) scripts/fetch_binance_history.py $(ARGS)

eda-stats:
	$(PY) scripts/eda_pack_stats.py $(ARGS)
