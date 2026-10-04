"""Equity History staleness banner (dashboard/tabs/paper.py), added
2026-08-20 after the 2026-08-19 outage: the Mac was dark through that day's
16:30 reconcile, so account_snapshots had no row for it, and the dashboard
kept showing the 2026-08-18 snapshot as if it were current while live equity
had moved +19% overnight (MRNA earnings). `_snapshot_staleness_sessions` +
`snapshot_staleness_message` (sma.live.reconcile) give the Paper tab and
`sma.live status` the identical staleness number and caveat text.
"""

from __future__ import annotations

from datetime import date

import pytest

from dashboard.tabs import paper
from sma.ingest.store import Store


@pytest.fixture(autouse=True)
def _clear_caches():
    paper._snapshot_staleness_sessions.clear()
    yield
    paper._snapshot_staleness_sessions.clear()


def _make_db(tmp_path):
    db_path = tmp_path / "t.duckdb"
    Store(path=str(db_path)).connect().close()
    return db_path


def _seed_spy(db_path, *dates):
    store = Store(path=str(db_path)).connect()
    for i, d in enumerate(dates):
        store.conn.execute(
            "INSERT INTO prices VALUES ('SPY', ?, 450.0, 450.0, 450.0, 450.0, "
            "450.0, 1000000, 'yfinance', ?)",
            [d, i],
        )
    store.close()


def test_staleness_sessions_reads_from_prices_via_dashboard_wrapper(monkeypatch, tmp_path):
    db_path = _make_db(tmp_path)
    monkeypatch.setattr(paper, "DB_PATH", db_path)
    _seed_spy(db_path, date(2026, 8, 19), date(2026, 8, 20))

    sessions = paper._snapshot_staleness_sessions(date(2026, 8, 18))

    assert sessions == 2


def test_staleness_sessions_zero_when_snapshot_is_current(monkeypatch, tmp_path):
    db_path = _make_db(tmp_path)
    monkeypatch.setattr(paper, "DB_PATH", db_path)
    _seed_spy(db_path, date(2026, 8, 20))

    sessions = paper._snapshot_staleness_sessions(date(2026, 8, 20))

    assert sessions == 0


def test_banner_text_is_none_when_fresh():
    assert paper.snapshot_staleness_message(sessions=1, snapshot_date=date(2026, 8, 19)) is None


def test_banner_text_present_for_the_outage_case():
    msg = paper.snapshot_staleness_message(sessions=2, snapshot_date=date(2026, 8, 18))

    assert msg is not None
    assert "2 sessions old" in msg
    assert "2026-08-18" in msg
    assert "live account may differ" in msg
