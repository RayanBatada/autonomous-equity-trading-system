from contextlib import contextmanager
from datetime import date

import pandas as pd

from scripts import backfill_earnings_history as hist
from sma.ingest.store import Store


class FakeTicker:
    calls: list[tuple[str, int]] = []

    def __init__(self, ticker: str):
        self.ticker = ticker

    def get_earnings_dates(self, limit: int = 40):
        self.calls.append((self.ticker, limit))
        if self.ticker == "BAD":
            raise RuntimeError("boom")
        return pd.DataFrame(
            {
                "EPS Estimate": [1.25, 1.10, 0.95],
                "Reported EPS": [1.30, 1.05, 0.90],
                "Revenue Estimate": [100_000_000.0, None, 90_000_000.0],
                "Reported Revenue": [110_000_000.0, None, 88_000_000.0],
            },
            index=pd.to_datetime(["2026-04-25", "2025-01-25", "2021-12-20"]),
        )


def _init_db(path):
    Store(path=path).connect().close()


@contextmanager
def _already_locked(**_kwargs):
    yield


def test_backfill_writes_yfinance_history_and_skips_errors(tmp_path, monkeypatch):
    db_path = tmp_path / "earnings.duckdb"
    _init_db(db_path)
    FakeTicker.calls = []
    monkeypatch.setattr(hist.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(hist, "writer_lock", _already_locked)

    rows = hist.backfill_earnings_history(
        tickers=["AAPL", "BAD"],
        db_path=db_path,
        min_report_date=date(2022, 1, 1),
        sleep_every=1,
        sleep_seconds=0.01,
    )

    assert rows == 2
    assert FakeTicker.calls == [("AAPL", 40), ("BAD", 40)]

    store = Store(path=db_path).connect()
    try:
        earnings = store.conn.execute(
            """
            SELECT ticker, report_date, eps_estimate, eps_actual,
                   revenue_estimate, revenue_actual, source, run_id
            FROM earnings
            ORDER BY report_date DESC
            """
        ).fetchall()
        assert earnings == [
            ("AAPL", date(2026, 4, 25), 1.25, 1.30, 100_000_000.0, 110_000_000.0,
             "yfinance_hist", 1),
            ("AAPL", date(2025, 1, 25), 1.10, 1.05, None, None, "yfinance_hist", 1),
        ]
        log = store.conn.execute(
            "SELECT run_id, source, rows_inserted, status, error FROM ingest_log"
        ).fetchall()
        assert log == [(1, "yfinance_hist", 2, "ok", None)]
    finally:
        store.close()


def test_backfill_rerun_is_idempotent(tmp_path, monkeypatch):
    db_path = tmp_path / "earnings.duckdb"
    _init_db(db_path)
    FakeTicker.calls = []
    monkeypatch.setattr(hist.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(hist, "writer_lock", _already_locked)

    first = hist.backfill_earnings_history(
        tickers=["AAPL"],
        db_path=db_path,
        min_report_date=date(2022, 1, 1),
    )
    second = hist.backfill_earnings_history(
        tickers=["AAPL"],
        db_path=db_path,
        min_report_date=date(2022, 1, 1),
    )

    assert first == 2
    assert second == 2

    store = Store(path=db_path).connect()
    try:
        count = store.conn.execute("SELECT COUNT(*) FROM earnings").fetchone()[0]
        run_ids = store.conn.execute(
            "SELECT DISTINCT run_id FROM earnings ORDER BY run_id"
        ).fetchall()
        log = store.conn.execute(
            "SELECT run_id, source, rows_inserted, status FROM ingest_log ORDER BY run_id"
        ).fetchall()
        assert count == 2
        assert run_ids == [(2,)]
        assert log == [(1, "yfinance_hist", 2, "ok"), (2, "yfinance_hist", 2, "ok")]
    finally:
        store.close()
