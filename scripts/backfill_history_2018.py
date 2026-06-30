"""Backfill 2018-01-01..2023-01-01 prices for the full universe (2026-06-15).

The IC analysis showed the model inverts in reversal regimes because it only
trained on 2023+ (one trending regime). This backfill adds the 2018 Q4
selloff, the 2020 COVID crash, and the 2022 bear — real momentum crashes —
so a retrain can learn regime-robust weights. INSERT OR REPLACE under
writer_lock; source='yfinance_hist'. Survivorship caveat: names that didn't
trade in 2018-2022 simply return no rows (logged), which is correct.
"""
import duckdb
import yfinance as yf

from sma.ingest.universe import load_universe
from sma.locks import writer_lock

universe = load_universe("src/sma/universe.yaml")
print(f"backfilling 2018-2022 for {len(universe)} tickers")
df = yf.download(
    [t.replace(".", "-") for t in universe], start="2018-01-01", end="2023-01-02",
    group_by="ticker", auto_adjust=False, progress=False, threads=True,
)
rows, missing = [], []
for t in universe:
    yt = t.replace(".", "-")
    try:
        sub = df[yt].dropna(subset=["Close"])
    except KeyError:
        missing.append(t); continue
    if sub.empty:
        missing.append(t); continue
    for dt, r in sub.iterrows():
        rows.append((t, dt.date(), float(r["Open"]), float(r["High"]), float(r["Low"]),
                     float(r["Close"]), float(r["Adj Close"]), int(r["Volume"]), "yfinance_hist"))
print(f"rows={len(rows)} | no pre-2023 data: {len(missing)} ({missing[:15]})")
if rows:
    with writer_lock(label="backfill-2018"):
        con = duckdb.connect("data/sma.duckdb")
        rid = con.execute("SELECT COALESCE(MAX(run_id),0)+1 FROM ingest_log").fetchone()[0]
        con.executemany("INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?,?,?,?)",
                        [r + (rid,) for r in rows])
        rng = con.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM prices").fetchone()
        con.close()
    print(f"done. new price range: {rng}")
