"""Staleness surfacing (2026-08-20): the 2026-08-19 outage left the dashboard
and `sma.live status` quoting the 2026-08-18 snapshot as current while the
live account had moved +19% overnight (MRNA earnings). These two pure/DB
helpers give both surfaces the exact same "how stale is this" number and the
exact same caveat sentence, computed from `prices` (a trading-calendar proxy
already populated every evening by ingest) rather than an Alpaca API call --
both the dashboard and `sma.live status` are documented as no-broker-call
reads.
"""

from datetime import date

import duckdb
import pytest

from sma.live.reconcile import (
    STALE_SNAPSHOT_SESSIONS_THRESHOLD,
    snapshot_staleness_message,
    snapshot_staleness_sessions,
)


@pytest.fixture
def conn():
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE prices (ticker VARCHAR, date DATE, adj_close DOUBLE, source VARCHAR)"
    )
    return con


def _seed_spy(conn, *dates):
    for d in dates:
        conn.execute(
            "INSERT INTO prices VALUES ('SPY', ?, 450.0, 'yfinance')", [d]
        )


# ---- snapshot_staleness_sessions -------------------------------------------


def test_snapshot_is_the_latest_known_session_is_zero_sessions_old(conn):
    _seed_spy(conn, date(2026, 8, 18))

    sessions = snapshot_staleness_sessions(conn=conn, snapshot_date=date(2026, 8, 18))

    assert sessions == 0


def test_one_newer_session_is_one_session_old(conn):
    _seed_spy(conn, date(2026, 8, 18), date(2026, 8, 19))

    sessions = snapshot_staleness_sessions(conn=conn, snapshot_date=date(2026, 8, 18))

    assert sessions == 1


def test_the_2026_08_19_outage_case_is_two_sessions_old(conn):
    """Snapshot last written 8/18; SPY prices exist for 8/19 AND 8/20 (today) --
    a full session was skipped entirely."""
    _seed_spy(conn, date(2026, 8, 18), date(2026, 8, 19), date(2026, 8, 20))

    sessions = snapshot_staleness_sessions(conn=conn, snapshot_date=date(2026, 8, 18))

    assert sessions == 2


def test_no_newer_prices_at_all_is_zero(conn):
    sessions = snapshot_staleness_sessions(conn=conn, snapshot_date=date(2026, 8, 18))
    assert sessions == 0


def test_counts_distinct_dates_not_rows(conn):
    """Multiple sources for the same date (yfinance + alpaca) must not double-count."""
    conn.execute("INSERT INTO prices VALUES ('SPY', '2026-08-19', 450.0, 'yfinance')")
    conn.execute("INSERT INTO prices VALUES ('SPY', '2026-08-19', 450.1, 'alpaca')")

    sessions = snapshot_staleness_sessions(conn=conn, snapshot_date=date(2026, 8, 18))

    assert sessions == 1


def test_other_tickers_are_ignored(conn):
    conn.execute("INSERT INTO prices VALUES ('AAPL', '2026-08-19', 220.0, 'yfinance')")

    sessions = snapshot_staleness_sessions(conn=conn, snapshot_date=date(2026, 8, 18))

    assert sessions == 0


# ---- snapshot_staleness_message --------------------------------------------


def test_threshold_is_one_session(conn):
    assert STALE_SNAPSHOT_SESSIONS_THRESHOLD == 1


def test_zero_sessions_old_needs_no_message():
    assert snapshot_staleness_message(sessions=0, snapshot_date=date(2026, 8, 18)) is None


def test_exactly_one_session_old_needs_no_message():
    """Boundary: >1, not >=1 -- the routine one-day-behind gap before a
    session's reconcile has fired is not itself a problem."""
    assert snapshot_staleness_message(sessions=1, snapshot_date=date(2026, 8, 18)) is None


def test_two_sessions_old_produces_the_caveat():
    msg = snapshot_staleness_message(sessions=2, snapshot_date=date(2026, 8, 18))

    assert msg is not None
    assert "2 sessions old" in msg
    assert "2026-08-18" in msg
    assert "live account may differ" in msg


def test_message_text_matches_both_consumers_verbatim():
    """Locks the exact wording both the dashboard banner and `sma.live
    status` display -- a change here changes both call sites identically."""
    msg = snapshot_staleness_message(sessions=3, snapshot_date=date(2026, 8, 17))

    assert msg == (
        "Latest snapshot is 3 sessions old (asof 2026-08-17) "
        "— live account may differ."
    )
