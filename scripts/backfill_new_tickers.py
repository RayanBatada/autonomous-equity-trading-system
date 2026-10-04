"""One-off: backfill 2023+ yfinance history for tickers added 2026-06-12.

New universe names have no training history (the nightly fetch only spans 45
days). INSERT OR REPLACE under writer_lock; source='yfinance'.
"""
import sys
from datetime import date

import duckdb
import pandas as pd
import yfinance as yf

from sma.ingest.universe import load_universe_membership
from sma.locks import writer_lock

membership = load_universe_membership("src/sma/universe.yaml")
new = sorted(t for t, added in membership.items() if added == date(2026, 6, 12))
print(f"backfilling {len(new)} tickers from 2023-01-01")

df = yf.download(
    [t.replace(".", "-") for t in new], start="2023-01-01",
    group_by="ticker", auto_adjust=False, progress=False, threads=True,
)
rows = []
for t in new:
    yt = t.replace(".", "-")
    try:
        sub = df[yt].dropna(subset=["Close"])
    except KeyError:
        print(f"  {t}: NO DATA — prune from universe")
        continue
    for dt, r in sub.iterrows():
        # Close is already NaN-guarded by dropna(subset=["Close"]) above; Adj
        # Close alone can still be NaN -- store NULL, not NaN (2026-09-18
        # incident audit).
        adj_close = None if pd.isna(r["Adj Close"]) else float(r["Adj Close"])
        rows.append((t, dt.date(), float(r["Open"]), float(r["High"]), float(r["Low"]),
                     float(r["Close"]), adj_close, int(r["Volume"]), "yfinance"))
print(f"rows to insert: {len(rows)}")
if not rows:
    sys.exit(1)

with writer_lock(label="backfill-new-tickers"):
    con = duckdb.connect("data/sma.duckdb")
    run_id = con.execute(
        "SELECT COALESCE(MAX(run_id), 0) + 1 FROM ingest_log"
    ).fetchone()[0]
    con.executemany(
        "INSERT OR REPLACE INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [r + (run_id,) for r in rows],
    )
    n = con.execute(
        "SELECT COUNT(DISTINCT ticker) FROM prices WHERE date >= '2023-01-01'"
    ).fetchone()[0]
    con.close()
print(f"done; distinct tickers with 2023+ prices now: {n}")
