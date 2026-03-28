# macOS often has no `python` on PATH — use this Makefile or `python3` / `.venv/bin/python`.
PY ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python3)

.PHONY: run dual live-once dashboard benchmark fetch-hist eda-stats

run:
	$(PY) run_p3.py -o submission.csv

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
