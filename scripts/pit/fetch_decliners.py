"""STEP 1 backfill: fetch free-yfinance OHLCV for the OBTAINABLE adverse
DECLINER set (post-crash S&P names omitted from the bot's survivor universe)
into the DURABLE de-biased copy /Users/youruser/.sma-pit/sma-pit.duckdb.

Candidate set = the ~42 genuinely-adverse omissions named in
pit-membership-findings.md §B/C (market-cap-decline + distress-event names).
We fetch ALL 42; yfinance availability filters to the obtainable subset
(rows logged per ticker; <500 rows or 0 rows => dropped + logged).
STI (reused SunTrust ticker, now a different listing) is EXCLUDED by design.

auto_adjust=False so close & adj_close differ (matches prod schema). Rows are
written with source='yfinance' and a sentinel run_id so they are identifiable
and survive the make_memcon dedup (decliners exist under no other source).
Prod DB is NEVER touched — we write only to the .sma-pit copy.
"""

import duckdb
import yfinance as yf

DB = "/Users/youruser/.sma-pit/sma-pit.duckdb"
SENTINEL_RUN_ID = 9_000_000_000_000_001
START = "2018-01-01"
END = "2026-06-25"  # yf end is exclusive -> through 2026-06-24

# 42 adverse candidates (market-cap-decline + distress) from findings B/C.
# STI deliberately excluded.
DECLINERS = [
    # retail collapses
    "KSS", "M", "GPS", "JWN", "BBWI", "HBI", "FL", "SIG", "HOG", "CPRI",
    "VFC", "UAA", "NWL", "COTY",
    # 2022 unprofitable-tech / clean-energy deratings
    "ETSY", "SEDG", "NKTR", "LUMN", "DXC", "PENN", "QRVO", "ILMN", "PRGO",
    # energy distress
    "CHK", "RRC", "NOV", "HP", "HFC",
    # airlines / financials stress
    "AAL", "ZION", "CMA", "LNC", "UNM",
    # legacy decliners
    "WU", "XRX", "DISH", "SRCL", "ADS",
    # explicit distress events
    "SIVB", "SBNY", "FRC", "PCG",
]

con = duckdb.connect(DB)
# safety: ensure we are NOT pointed at prod
assert DB.startswith("/Users/youruser/.sma-pit/"), DB

# clear any prior sentinel rows (idempotent re-run)
con.execute("DELETE FROM prices WHERE run_id = ?", [SENTINEL_RUN_ID])

obtained, dropped = [], []
for t in DECLINERS:
    try:
        df = yf.download(t, start=START, end=END, auto_adjust=False,
                         progress=False, threads=False)
    except Exception as e:  # noqa: BLE001
        dropped.append((t, f"error:{e}"))
        print(f"{t:6} ERROR {e}")
        continue
    if df is None or df.empty:
        dropped.append((t, "empty"))
        print(f"{t:6}    0  DROP empty")
        continue
    # flatten possible MultiIndex columns (yf returns (field,ticker))
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    rows = []
    for _, r in df.iterrows():
        d = r["Date"]
        d = d.date() if hasattr(d, "date") else d
        o, h, lo = r.get("Open"), r.get("High"), r.get("Low")
        c, ac, v = r.get("Close"), r.get("Adj Close"), r.get("Volume")
        if c is None or ac is None:
            continue
        try:
            rows.append((t, d, float(o), float(h), float(lo), float(c),
                         float(ac), int(v) if v == v else 0,
                         "yfinance", SENTINEL_RUN_ID))
        except (TypeError, ValueError):
            continue
    if len(rows) < 500:
        dropped.append((t, f"only {len(rows)} rows"))
        print(f"{t:6} {len(rows):5d}  DROP (<500)")
        continue
    con.executemany(
        "INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    obtained.append((t, len(rows)))
    print(f"{t:6} {len(rows):5d}  OK")

con.commit()
print("\n=== SUMMARY ===")
print(f"obtained: {len(obtained)} / {len(DECLINERS)}")
print("obtained tickers:", [t for t, _ in obtained])
print("dropped:", dropped)
tot = con.execute("SELECT COUNT(*), COUNT(DISTINCT ticker) FROM prices WHERE run_id=?",
                  [SENTINEL_RUN_ID]).fetchone()
print(f"sentinel rows inserted: {tot[0]} across {tot[1]} tickers")
print("date range:", con.execute(
    "SELECT MIN(date), MAX(date) FROM prices WHERE run_id=?", [SENTINEL_RUN_ID]).fetchone())
con.close()
