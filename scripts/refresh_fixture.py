#!/usr/bin/env python
"""Refresh the integration test fixture.

Hits REAL APIs (yfinance is enough for prices; news omitted to keep fixture
small) for a small ticker set over a fixed date range, writes the result to
tests/fixtures/sma-fixture.duckdb.

Run manually after schema changes or when the fixture goes stale. Do NOT
run on every test invocation. The fixture must be deterministic and committed.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

from sma.ingest.store import Store

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sma-fixture.duckdb"
TICKERS = ["AAPL", "MSFT", "GOOGL"]
START = "2025-10-01"
END = "2026-04-01"


def main() -> None:
    if FIXTURE_PATH.exists():
        FIXTURE_PATH.unlink()
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    store = Store(path=str(FIXTURE_PATH)).connect()
    rid = store.allocate_run_id()
    store.log_run_start(rid, source="yfinance")

    df = yf.download(TICKERS, start=START, end=END, progress=False,
                     group_by="ticker", auto_adjust=False)
    rows = []
    for t in TICKERS:
        sub = df[t].dropna(how="all")
        for ts, r in sub.iterrows():
            # 2026-09-18 incident audit: never write a row whose close is
            # NaN/None into the committed fixture.
            if pd.isna(r["Close"]):
                continue
            rows.append((
                t, ts.date(),
                float(r["Open"]), float(r["High"]), float(r["Low"]),
                float(r["Close"]),
                None if pd.isna(r["Adj Close"]) else float(r["Adj Close"]),
                int(r["Volume"]), "yfinance", rid,
            ))
    store.conn.executemany(
        "INSERT OR REPLACE INTO prices "
        "(ticker, date, open, high, low, close, adj_close, volume, source, run_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    store.log_run_end(rid, source="yfinance",
                      rows_inserted=len(rows), status="ok", error=None)
    store.close()
    print(f"Wrote {len(rows)} rows to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
