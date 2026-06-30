"""Backfill historical quarterly earnings from yfinance into DuckDB.

Fetches report dates, EPS estimate/actual, and revenue values when Yahoo
exposes them. Writes are idempotent on the earnings table primary key
``(ticker, report_date)``.
"""

from __future__ import annotations

import argparse
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
from loguru import logger

import sma.locks as _locks
from sma.ingest.store import Store
from sma.ingest.universe import load_universe
from sma.locks import writer_lock

SOURCE = "yfinance_hist"
DEFAULT_DB_PATH = Path("data/sma.duckdb")
DEFAULT_UNIVERSE_PATH = Path("src/sma/universe.yaml")
DEFAULT_MIN_REPORT_DATE = date(2022, 1, 1)

_EPS_ESTIMATE_COLUMNS = ("EPS Estimate", "Estimated EPS")
_EPS_ACTUAL_COLUMNS = ("Reported EPS", "Actual EPS", "EPS Actual")
_REVENUE_ESTIMATE_COLUMNS = (
    "Revenue Estimate",
    "Estimated Revenue",
    "Revenue Estimate Avg",
)
_REVENUE_ACTUAL_COLUMNS = (
    "Reported Revenue",
    "Actual Revenue",
    "Revenue Actual",
)


def _clean_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _first_present(row: pd.Series, columns: tuple[str, ...]) -> float | None:
    for column in columns:
        if column in row:
            return _clean_float(row[column])
    return None


def _report_date_from_index(value: Any) -> date | None:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return pd.Timestamp(value).date()
    except Exception:
        return None


def _fetch_earnings_dates(ticker: str, limit: int) -> pd.DataFrame | None:
    ticker_obj = yf.Ticker(ticker)
    getter = getattr(ticker_obj, "get_earnings_dates", None)
    if callable(getter):
        return getter(limit=limit)

    earnings_dates = getattr(ticker_obj, "earnings_dates", None)
    if callable(earnings_dates):
        return earnings_dates(limit=limit)
    return earnings_dates


def _rows_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    *,
    run_id: int,
    min_report_date: date,
) -> list[tuple]:
    rows: list[tuple] = []
    for index_value, row in frame.iterrows():
        report_date = _report_date_from_index(index_value)
        if report_date is None:
            logger.warning("skipping {} earnings row with invalid date {}", ticker, index_value)
            continue
        if report_date < min_report_date:
            continue

        rows.append(
            (
                ticker,
                report_date,
                _first_present(row, _EPS_ESTIMATE_COLUMNS),
                _first_present(row, _EPS_ACTUAL_COLUMNS),
                _first_present(row, _REVENUE_ESTIMATE_COLUMNS),
                _first_present(row, _REVENUE_ACTUAL_COLUMNS),
                SOURCE,
                run_id,
            )
        )
    return rows


def _allocate_run_id(store: Store) -> int:
    row = store.conn.execute(
        "SELECT COALESCE(MAX(run_id), 0) + 1 FROM ingest_log"
    ).fetchone()
    return int(row[0])


def backfill_earnings_history(
    *,
    tickers: list[str],
    db_path: str | Path = DEFAULT_DB_PATH,
    min_report_date: date = DEFAULT_MIN_REPORT_DATE,
    limit: int = 40,
    sleep_every: int = 10,
    sleep_seconds: float = 1.0,
    lock_path: str | Path | None = None,
) -> int:
    """Backfill yfinance historical earnings for ``tickers``.

    Returns the number of rows attempted via INSERT OR REPLACE. Per-ticker
    fetch/parse failures are logged and skipped.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    lock = Path(lock_path) if lock_path is not None else _locks.DEFAULT_LOCK_PATH

    with writer_lock(lock_path=lock, label="backfill-earnings-history"):
        store = Store(path=db_path).connect(read_only=False)
        try:
            run_id = _allocate_run_id(store)
            store.log_run_start(run_id, source=SOURCE)
            rows_inserted = 0
            error: str | None = None

            try:
                for index, ticker in enumerate(tickers, start=1):
                    try:
                        frame = _fetch_earnings_dates(ticker, limit=limit)
                    except Exception as exc:
                        logger.warning(
                            "yfinance historical earnings failed for {}: {}", ticker, exc
                        )
                        continue

                    if frame is None or len(frame) == 0:
                        continue
                    if not isinstance(frame, pd.DataFrame):
                        logger.warning(
                            "yfinance returned non-DataFrame earnings for {}: {}",
                            ticker,
                            type(frame).__name__,
                        )
                        continue

                    rows = _rows_from_frame(
                        ticker,
                        frame,
                        run_id=run_id,
                        min_report_date=min_report_date,
                    )
                    if rows:
                        store.conn.executemany(
                            "INSERT OR REPLACE INTO earnings "
                            "(ticker, report_date, eps_estimate, eps_actual, "
                            " revenue_estimate, revenue_actual, source, run_id) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            rows,
                        )
                        rows_inserted += len(rows)

                    if sleep_every > 0 and index % sleep_every == 0:
                        time.sleep(sleep_seconds)
            except Exception as exc:
                error = str(exc)
                raise
            finally:
                store.log_run_end(
                    run_id,
                    source=SOURCE,
                    rows_inserted=rows_inserted,
                    status="error" if error else "ok",
                    error=error,
                )
        finally:
            store.close()

    return rows_inserted


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill historical quarterly earnings from yfinance."
    )
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE_PATH, type=Path)
    parser.add_argument("--db", default=DEFAULT_DB_PATH, type=Path)
    parser.add_argument("--min-report-date", default=DEFAULT_MIN_REPORT_DATE.isoformat())
    parser.add_argument("--limit", default=40, type=int)
    parser.add_argument("--sleep-every", default=10, type=int)
    parser.add_argument("--sleep-seconds", default=1.0, type=float)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tickers = load_universe(args.universe)
    min_report_date = date.fromisoformat(args.min_report_date)
    rows = backfill_earnings_history(
        tickers=tickers,
        db_path=args.db,
        min_report_date=min_report_date,
        limit=args.limit,
        sleep_every=args.sleep_every,
        sleep_seconds=args.sleep_seconds,
    )
    print(f"inserted/replaced {rows} {SOURCE} earnings rows")


if __name__ == "__main__":
    main()
