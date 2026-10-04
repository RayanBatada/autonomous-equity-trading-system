"""One-off: backfill the 2023-01-03..2023-04-24 prices gap (77 trading
sessions) between the end of the yfinance_hist historical load
(2022-12-30) and the start of alpaca coverage (2023-04-25).

Found by the label-horizon study: ~110 of 266 universe tickers have ZERO
prices rows in this window. backfill_history_2018.py (2026-06-15) only
had rolling per-ticker yfinance coverage for ~154 names through the gap;
the rest went dark until Alpaca ingestion picked them up on 2023-04-25.
This degrades the 2023-H1 walk-forward training window and contaminated
the 2023-h1-junk-rally regime window: src/sma/eval/regime.py hardcodes
`WHERE source='yfinance'` (no ELSE fallback) for its trend/reversal
factor calc, so the gap silently zeroed out ~110 names' contribution to
that regime's cross-section.

The missing set is computed live from the DB each run (COUNT(DISTINCT
date) in the window per universe ticker == 0), so this script is
idempotent — a second run finds nothing left to do.

Adjacency / adj-drift check (2026-08-14, Rayan's ask): yfinance today
returns TODAY's split/dividend-adjusted closes. The stored yfinance_hist
row for 2022-12-30 was fetched back on 2026-06-15. If a split/large div
happened on a ticker since then, today's adjusted series is rescaled
relative to the stored series, and naively splicing gap-fill rows onto
it would leave a level discontinuity at the 2022-12-30/2023-01-03 seam.
For every candidate ticker we compare today's fetched adj_close on the
overlap day (2022-12-30, the last yfinance_hist session) against the
stored yfinance_hist row for that date. >0.5% relative drift means DON'T
stitch -- refetch that ticker's full history (2016-01-01..2023-04-24)
instead, exactly like yfinance_prices.py's own adj-drift self-heal
("adj-drift detected ... full-history refetch re-synced"). The refetch
rows land with source='yfinance' (this script's source for every row,
see below), which outranks the old 'yfinance_hist' rows in every
downstream priority CASE (model/__main__.py, model/predictor.py:
yfinance=0, alpaca=1, else=2), so the corrected series wins without
deleting the stale rows.

Source label: 'yfinance' for every row this script writes (both the
gap-only stitch and any full-history refetch), because the entire gap
(2023-01-03..2023-04-24) is post-2023 -- the existing pipeline's own
convention for that period (scripts/backfill_new_tickers.py; documented
again in scripts/pit/shared_pit.py: "prod DB stores pre-2023 as
source='yfinance_hist' and 2023+ as 'yfinance'"). Using 'yfinance' here
(rather than a novel source string) is deliberate: src/sma/eval/regime.py
hardcodes `WHERE source='yfinance'` with no ELSE fallback, so a new
source label would be silently invisible to it and this backfill would
not actually de-contaminate the 2023-h1-junk-rally regime window it
exists to fix.

Auditability/reversibility follows scripts/insert_muu_reference_data.py's
precedent: every row from this script carries the reserved run_id
9000000000000003 (next unused id after MUU's 9000000000000002 -- checked
`SELECT DISTINCT run_id FROM prices WHERE run_id >= 9e15` before picking
it). `DELETE FROM prices WHERE run_id = 9000000000000003` cleanly
reverses this script's writes without touching any organic row.

Usage:
    ./.venv/bin/python scripts/backfill_2023h1_gap.py             # do it
    ./.venv/bin/python scripts/backfill_2023h1_gap.py --dry-run   # fetch + report only, no DB write
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

from sma.db_connect import read_only_connect
from sma.ingest.store import Store
from sma.ingest.universe import load_universe
from sma.locks import writer_lock

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "sma.duckdb"
UNIVERSE_PATH = Path(__file__).resolve().parents[1] / "src" / "sma" / "universe.yaml"

SOURCE = "yfinance"
REFERENCE_RUN_ID = 9000000000000003  # reserved: 2023-H1 gap backfill, never a real ingest run

GAP_START = "2023-01-03"  # first trading session after yfinance_hist ends (2022-12-30)
GAP_END = "2023-04-24"  # last trading session before alpaca coverage starts (2023-04-25)
OVERLAP_DATE = "2022-12-30"  # last yfinance_hist session; used for the adj-drift check
FETCH_END_EXCLUSIVE = "2023-04-25"  # yfinance `end` is exclusive; covers OVERLAP_DATE..GAP_END
DRIFT_THRESHOLD = 0.005  # 0.5% relative, per task spec (tighter than yfinance_prices.py's 1%)
FULL_HISTORY_START = "2016-01-01"  # mirrors yfinance_prices.py's resync fetch


def _find_missing(con) -> list[str]:
    universe = load_universe(UNIVERSE_PATH)
    rows = con.execute(
        """
        SELECT ticker, COUNT(DISTINCT date) AS n
        FROM prices
        WHERE ticker = ANY(?) AND date BETWEEN ? AND ?
        GROUP BY ticker
        """,
        [universe, GAP_START, GAP_END],
    ).fetchall()
    counts = {t: n for t, n in rows}
    return sorted(t for t in universe if counts.get(t, 0) == 0)


def _stored_overlap_adj_close(con, ticker: str) -> float | None:
    row = con.execute(
        "SELECT adj_close FROM prices WHERE ticker=? AND date=? AND source='yfinance_hist'",
        [ticker, OVERLAP_DATE],
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        known = {"open", "high", "low", "close", "adj close", "volume"}
        level0_hits = sum(1 for v in df.columns.get_level_values(0) if str(v).lower() in known)
        level1_hits = sum(1 for v in df.columns.get_level_values(1) if str(v).lower() in known)
        field_level = 0 if level0_hits >= level1_hits else 1
        df.columns = df.columns.get_level_values(field_level)
    return df


def _rows_from_df(ticker: str, df: pd.DataFrame, start: str | None, end: str | None) -> list[tuple]:
    """Mirror YFinancePricesSource._insert_single_ticker's row shape exactly."""
    if df is None or df.empty:
        return []
    df = _flatten(df)
    cols = {c.lower().replace(" ", "_"): c for c in df.columns}
    if "close" not in cols:
        return []
    rows = []
    for ts, r in df.iterrows():
        d = ts.date() if hasattr(ts, "date") else ts
        if start and d < pd.Timestamp(start).date():
            continue
        if end and d > pd.Timestamp(end).date():
            continue
        if pd.isna(r[cols["close"]]):
            continue
        rows.append((
            ticker,
            d,
            float(r[cols["open"]]) if "open" in cols and pd.notna(r[cols["open"]]) else None,
            float(r[cols["high"]]) if "high" in cols and pd.notna(r[cols["high"]]) else None,
            float(r[cols["low"]]) if "low" in cols and pd.notna(r[cols["low"]]) else None,
            float(r[cols["close"]]),
            (
                float(r[cols["adj_close"]])
                if "adj_close" in cols and pd.notna(r[cols["adj_close"]])
                else None
            ),
            int(r[cols["volume"]]) if "volume" in cols and pd.notna(r[cols["volume"]]) else None,
            SOURCE,
            REFERENCE_RUN_ID,
        ))
    return rows


def _yf_download(tickers: list[str], start: str, end: str, *, retries: int = 3):
    """Bulk download with capped-backoff retry, mirroring yfinance_prices.py's
    rate-limit posture (bulk first, small backoff on failure)."""
    backoff = 3.0
    last_exc = None
    for attempt in range(retries):
        try:
            return yf.download(
                tickers,
                start=start,
                end=end,
                progress=False,
                group_by="ticker",
                threads=True,
                auto_adjust=False,
                multi_level_index=False,
            )
        except Exception as e:  # noqa: BLE001
            last_exc = e
            logger.warning("bulk yfinance download failed (attempt {}): {}", attempt + 1, e)
            time.sleep(backoff)
            backoff *= 2
    raise last_exc


def _yf_download_single(ticker: str, start: str, end: str, *, retries: int = 3):
    backoff = 2.0
    last_exc = None
    for attempt in range(retries):
        try:
            return yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                multi_level_index=False,
            )
        except Exception as e:  # noqa: BLE001
            last_exc = e
            logger.warning("per-ticker yfinance download failed for {} (attempt {}): {}", ticker, attempt + 1, e)
            time.sleep(backoff)
            backoff *= 2
    raise last_exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="fetch + report only, no DB write")
    args = parser.parse_args()

    con = read_only_connect(DB_PATH)
    try:
        missing = _find_missing(con)
        logger.info("candidate tickers with ZERO rows in [{}, {}]: {}", GAP_START, GAP_END, len(missing))

        stored_overlap = {t: _stored_overlap_adj_close(con, t) for t in missing}
    finally:
        con.close()

    if not missing:
        logger.info("nothing to backfill.")
        return

    # Bulk fetch: overlap day through gap end, one request for every candidate.
    logger.info("bulk-fetching {} tickers from {} to {} (exclusive)...", len(missing), OVERLAP_DATE, FETCH_END_EXCLUSIVE)
    bulk_df = _yf_download(
        [t.replace(".", "-") for t in missing], OVERLAP_DATE, FETCH_END_EXCLUSIVE
    )

    expected_absent: dict[str, str] = {}
    drift_tickers: list[tuple[str, float, float, float]] = []  # ticker, stored, fetched, pct
    no_overlap_tickers: list[str] = []
    stitch_rows: dict[str, list[tuple]] = {}

    for t in missing:
        yt = t.replace(".", "-")
        try:
            sub = bulk_df[yt].dropna(how="all") if len(missing) > 1 else bulk_df.dropna(how="all")
        except KeyError:
            sub = pd.DataFrame()

        if sub.empty:
            # Retry solo before concluding genuinely no data — rules out a
            # bulk-call glitch/rate-limit masking as "no data" (backoff on
            # failure, per task spec).
            try:
                sub = _yf_download_single(yt, OVERLAP_DATE, FETCH_END_EXCLUSIVE)
            except Exception as e:  # noqa: BLE001
                logger.warning("{}: solo retry also failed: {}", t, e)
                sub = pd.DataFrame()
            if sub is None or sub.empty:
                expected_absent[t] = "yfinance returned no data anywhere in the fetch window (no trading history yet)"
                continue

        sub = _flatten(sub)
        cols = {c.lower().replace(" ", "_"): c for c in sub.columns}
        fetched_overlap = None
        if "adj_close" in cols:
            for ts, r in sub.iterrows():
                d = ts.date() if hasattr(ts, "date") else ts
                if d.isoformat() == OVERLAP_DATE and pd.notna(r[cols["adj_close"]]):
                    fetched_overlap = float(r[cols["adj_close"]])
                    break

        stored = stored_overlap.get(t)
        if stored is not None and fetched_overlap is not None:
            pct = abs(fetched_overlap / stored - 1.0)
            if pct > DRIFT_THRESHOLD:
                drift_tickers.append((t, stored, fetched_overlap, pct))
                continue  # handled by full-history refetch pass below
        elif fetched_overlap is None:
            no_overlap_tickers.append(t)
            # No overlap day to compare (e.g. ticker's history doesn't reach
            # back that far, or yfinance had a gap there too) -> nothing to
            # be inconsistent WITH, so a plain stitch is safe.

        rows = _rows_from_df(t, sub, GAP_START, GAP_END)
        if rows:
            stitch_rows[t] = rows
        else:
            # Had SOME data in the fetch window but none inside the actual
            # gap dates (e.g. IPO lands mid-gap) — check separately below.
            pass

    # Full-history refetch pass for drifted tickers.
    refetch_rows: dict[str, list[tuple]] = {}
    refetch_failed: list[str] = []
    for t, stored, fetched, pct in drift_tickers:
        yt = t.replace(".", "-")
        logger.warning(
            "{}: adj-drift {:.2%} (stored yfinance_hist={:.4f}, fetched today={:.4f}) "
            "-> full-history refetch {}..{}",
            t, pct, stored, fetched, FULL_HISTORY_START, GAP_END,
        )
        try:
            full = _yf_download_single(yt, FULL_HISTORY_START, FETCH_END_EXCLUSIVE)
            rows = _rows_from_df(t, full, None, GAP_END)
            if not rows:
                raise RuntimeError("full-history refetch returned no rows")
            refetch_rows[t] = rows
        except Exception as e:  # noqa: BLE001
            refetch_failed.append(t)
            logger.error("{}: full-history refetch FAILED: {}", t, e)

    # Tickers that had a mid-gap IPO/listing (data in fetch window, but zero
    # rows landed inside [GAP_START, GAP_END] after the date filter) — check
    # for real, don't silently drop.
    partial_ipo = {t: len(rows) for t, rows in stitch_rows.items()
                   if t in no_overlap_tickers}

    all_rows: list[tuple] = []
    for rows in stitch_rows.values():
        all_rows.extend(rows)
    for rows in refetch_rows.values():
        all_rows.extend(rows)

    stitch_count = len(stitch_rows) - len(partial_ipo)  # pure gap-fill, had a clean overlap
    logger.info("=" * 70)
    logger.info("PLAN: {} tickers stitched (gap-only insert, clean overlap), {} tickers "
                "full-history refetched, {} drift-refetch failures, {} expected-absent",
                stitch_count, len(refetch_rows), len(refetch_failed), len(expected_absent))
    logger.info("total rows to insert: {}", len(all_rows))
    for t in expected_absent:
        logger.info("  expected-absent: {} -- {}", t, expected_absent[t])
    for t, _stored, _fetched, pct in drift_tickers:
        status = "refetched" if t in refetch_rows else "REFETCH FAILED"
        logger.info("  drift: {} ({:.2%}) -> {}", t, pct, status)
    if partial_ipo:
        logger.info("  mid-gap listings (partial coverage expected): {}", partial_ipo)

    if args.dry_run:
        logger.info("--dry-run: no DB write performed.")
        return

    if not all_rows:
        logger.warning("no rows to insert; nothing written.")
        return

    with writer_lock(label="backfill-2023h1-gap"):
        store = Store(path=DB_PATH).connect()
        try:
            store.conn.executemany(
                "INSERT OR REPLACE INTO prices "
                "(ticker, date, open, high, low, close, adj_close, volume, source, run_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                all_rows,
            )
        finally:
            store.conn.close()
    logger.info("inserted {} rows (run_id={})", len(all_rows), REFERENCE_RUN_ID)

    # Post-write verification (read-only, outside the lock).
    con = read_only_connect(DB_PATH)
    try:
        still_missing = _find_missing(con)
        still_missing = [t for t in still_missing if t not in expected_absent]
        logger.info("post-verify: universe tickers still with ZERO rows in gap "
                    "(excluding expected-absent): {}", still_missing)
        for t in ["AAPL", "AMZN", "AVGO", "COST"]:
            rows = con.execute(
                "SELECT date, close, adj_close, source, run_id FROM prices "
                "WHERE ticker=? AND date BETWEEN ? AND ? ORDER BY date",
                [t, GAP_START, GAP_END],
            ).fetchall()
            logger.info("{}: {} rows in gap; first={} last={}", t, len(rows),
                        rows[0] if rows else None, rows[-1] if rows else None)
    finally:
        con.close()


if __name__ == "__main__":
    main()
