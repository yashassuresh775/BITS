Place the distributed student-pack here:

  data/student-pack/crypto-market/
  data/student-pack/crypto-trades/
  data/student-pack/docs/          (optional)

Then run from repo root:

  python run_p3.py -o submission.csv
  python scripts/benchmark_p3.py --runs 2

CSV files under data/student-pack/ are gitignored (see .gitignore).

Root-level outputs (p1_alerts.csv, p2_signals.csv, submission*.csv) are regenerated
from the repo root; see the "Regenerating CSVs" section in README.md.
