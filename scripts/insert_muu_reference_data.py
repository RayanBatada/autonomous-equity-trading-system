"""One-off loader: MUU (Direxion Daily MU Bull 2X Shares) reference prices.

Rayan asked for MUU daily prices as REFERENCE data only. MUU must NEVER be
added to src/sma/universe.yaml (not tradeable) — this script does not touch
universe.yaml and never will. Rows are written with the reserved reference
run_id 9000000000000002 so they're identifiable as a manual one-off load
distinct from a normal scheduled ingest run.

Fetches full available daily history via yfinance and inserts into the
`prices` table using the project's own Store/writer_lock pattern (mirrors
sma.ingest.sources.yfinance_prices._insert_single_ticker's row shape). The
writer_lock is held only for the brief insert, not for the network fetch,
so this never blocks the evening writer-lock jobs (ingest/decide/reconcile)
for longer than a local DB write.

Usage:
    ./.venv/bin/python scripts/insert_muu_reference_data.py

Idempotent: INSERT OR REPLACE keyed on (ticker, date, source), safe to re-run.

History: fetched once (per Rayan's request weeks ago) but the insert was
dropped before landing. This script exists so the load can be re-run/finished
without re-deriving the insert idiom from scratch.
"""

from __future__ import annotations

from pathlib import Path

import yfinance as yf
from loguru import logger

from sma.ingest.store import Store
from sma.locks import writer_lock

TICKER = "MUU"
SOURCE = "yfinance"
REFERENCE_RUN_ID = 9000000000000002  # reserved: manual reference-data loads, never a real ingest run
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "sma.duckdb"


def fetch_muu_history():
    """Full available daily history for MUU, unadjusted-columns intact (same
    shape as YFinancePricesSource so downstream readers see identical fields)."""
    return yf.download(
        TICKER,
        period="max",
        progress=False,
        auto_adjust=False,
        multi_level_index=False,
    )


def _rows_from_df(df) -> list[tuple]:
    """Flatten a yfinance OHLCV frame into `prices` insert tuples. Mirrors
    YFinancePricesSource._insert_single_ticker's row construction."""
    cols = {c.lower().replace(" ", "_"): c for c in df.columns}
    rows = []
    for ts, r in df.iterrows():
        d = ts.date() if hasattr(ts, "date") else ts
        rows.append((
            TICKER,
            d,
            float(r[cols["open"]]) if "open" in cols and r[cols["open"]] == r[cols["open"]] else None,
            float(r[cols["high"]]) if "high" in cols and r[cols["high"]] == r[cols["high"]] else None,
            float(r[cols["low"]]) if "low" in cols and r[cols["low"]] == r[cols["low"]] else None,
            float(r[cols["close"]]) if "close" in cols and r[cols["close"]] == r[cols["close"]] else None,
            (
                float(r[cols["adj_close"]])
                if "adj_close" in cols and r[cols["adj_close"]] == r[cols["adj_close"]]
                else None
            ),
            int(r[cols["volume"]]) if "volume" in cols and r[cols["volume"]] == r[cols["volume"]] else None,
            SOURCE,
            REFERENCE_RUN_ID,
        ))
    return rows


def main() -> None:
    df = fetch_muu_history()
    if df is None or df.empty:
        logger.error("yfinance returned no data for {}", TICKER)
        raise SystemExit(1)
    rows = _rows_from_df(df)
    if not rows:
        logger.error("no rows parsed from yfinance response for {}", TICKER)
        raise SystemExit(1)
    logger.info(
        "fetched {} rows for {} ({} .. {})", len(rows), TICKER, rows[0][1], rows[-1][1]
    )

    # writer_lock held only for the insert itself — fetch above happens
    # outside the lock so a slow/rate-limited network call never holds up
    # other writers (ingest/decide/reconcile all take the same lock).
    with writer_lock(label="muu-reference-insert"):
        store = Store(path=DB_PATH).connect()
        try:
            store.conn.executemany(
                "INSERT OR REPLACE INTO prices "
                "(ticker, date, open, high, low, close, adj_close, volume, source, run_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        finally:
            store.conn.close()
    logger.info("inserted {} MUU reference rows (run_id={})", len(rows), REFERENCE_RUN_ID)


if __name__ == "__main__":
    main()
