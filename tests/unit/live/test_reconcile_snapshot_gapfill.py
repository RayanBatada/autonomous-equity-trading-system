"""Self-healing gap-fill: INSERT a missing account_snapshots row once
Alpaca's OFFICIAL daily close is available for that session.

Motivating incident (2026-08-20): the Mac was dark through the entire
2026-08-19 session, so reconcile never ran that evening and NO
account_snapshots row exists for 2026-08-19 at all -- unlike
backfill_official_closes (which only UPGRADES an existing row's equity
source), there is no row here to anchor a correction. This gap-fill looks at
the last GAPFILL_LOOKBACK_SESSIONS trading sessions (from the Alpaca
calendar, since a missing day has no row of its own to look back from) and
inserts a row for any with none, sourced from the OFFICIAL portfolio-history
daily bar only -- never the 1Min proxy, since an approximate equity for a day
whose cash/positions can never be reconstructed is not worth the false
precision. Migration 8 makes cash/buying_power/long_market_value/
position_count nullable for exactly this row shape.
"""

from datetime import date
from unittest.mock import MagicMock

import pytest

from sma.ingest.store import Store
from sma.live.reconcile import (
    GAPFILL_LOOKBACK_SESSIONS,
    backfill_missing_snapshots,
)


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(path=tmp_path / "t.duckdb").connect()


def _insert_snapshot(store, *, asof_date: date, equity: float = 100_000.0, run_id: int = 1):
    store.conn.execute(
        """
        INSERT INTO account_snapshots
        (asof_date, equity, cash, buying_power, long_market_value,
         position_count, total_unrealized_pnl, run_id, equity_source)
        VALUES (?, ?, ?, ?, ?, ?, NULL, ?, 'portfolio_history_daily_close')
        """,
        [asof_date, equity, equity * 0.1, equity, equity * 0.9, 10, run_id],
    )


def _alpaca(sessions, close_by_day=None, *, raises_close_for=None, calendar_raises=False):
    """MagicMock AlpacaClient. `sessions` drives sessions_between(); `close_by_day`
    is a {date: (equity, source) | None} map driving session_close_equity()."""
    close_by_day = close_by_day or {}
    raises_close_for = raises_close_for or set()

    m = MagicMock()
    if calendar_raises:
        m.sessions_between.side_effect = RuntimeError("calendar 503")
    else:
        m.sessions_between.return_value = list(sessions)

    def _close_side_effect(*, day):
        if day in raises_close_for:
            raise RuntimeError("portfolio-history 503")
        return close_by_day.get(day)

    m.session_close_equity.side_effect = _close_side_effect
    return m


# ---- core behavior ---------------------------------------------------------


def test_no_gaps_is_a_noop(store):
    """Every session in the window already has a row -- nothing to insert,
    no equity lookups spent."""
    sessions = [date(2026, 8, d) for d in (13, 14, 17, 18, 19)]
    for d in sessions:
        _insert_snapshot(store, asof_date=d)
    alpaca = _alpaca(sessions)

    inserted = backfill_missing_snapshots(store=store, alpaca=alpaca, today=date(2026, 8, 19))

    assert inserted == 0
    alpaca.session_close_equity.assert_not_called()


def test_inserts_a_missing_session_once_the_official_close_publishes(store, caplog):
    """The 2026-08-19 case: no row at all, and Alpaca now has the official bar."""
    sessions = [date(2026, 8, d) for d in (17, 18, 19, 20)]
    _insert_snapshot(store, asof_date=date(2026, 8, 17))
    _insert_snapshot(store, asof_date=date(2026, 8, 18))
    # 2026-08-19 missing entirely.
    _insert_snapshot(store, asof_date=date(2026, 8, 20))
    alpaca = _alpaca(
        sessions, {date(2026, 8, 19): (121_600.00, "portfolio_history_daily")}
    )

    with caplog.at_level("INFO", logger="sma.live.reconcile"):
        inserted = backfill_missing_snapshots(store=store, alpaca=alpaca, today=date(2026, 8, 20))

    assert inserted == 1
    row = store.conn.execute(
        "SELECT equity, cash, buying_power, long_market_value, position_count, "
        "total_unrealized_pnl, equity_source "
        "FROM account_snapshots WHERE asof_date = ?",
        [date(2026, 8, 19)],
    ).fetchone()
    equity, cash, bp, lmv, pos_count, pnl, source = row
    assert equity == 121_600.00
    assert source == "portfolio_history_daily_close"
    assert (cash, bp, lmv, pos_count, pnl) == (None, None, None, None, None), (
        "cash/positions for a missed day are unknowable retroactively"
    )
    assert sum("gap-fill: inserted" in r.message for r in caplog.records) == 1


def test_missing_session_without_a_published_close_yet_stays_missing(store):
    """Nothing published yet (normal, most evenings) -- no row inserted, and
    it will be retried on the next reconcile run."""
    sessions = [date(2026, 8, d) for d in (18, 19, 20)]
    _insert_snapshot(store, asof_date=date(2026, 8, 18))
    _insert_snapshot(store, asof_date=date(2026, 8, 20))
    alpaca = _alpaca(sessions, {date(2026, 8, 19): None})

    inserted = backfill_missing_snapshots(store=store, alpaca=alpaca, today=date(2026, 8, 20))

    assert inserted == 0
    row = store.conn.execute(
        "SELECT COUNT(*) FROM account_snapshots WHERE asof_date = ?",
        [date(2026, 8, 19)],
    ).fetchone()
    assert row[0] == 0


def test_only_the_1min_proxy_available_is_not_good_enough_to_insert(store):
    """session_close_equity falls back to the 1Min-proxy tier -- the official
    bar hasn't landed. An approximate equity for a day with no cash/position
    data is not worth inserting; wait for the real bar."""
    sessions = [date(2026, 8, d) for d in (18, 19, 20)]
    _insert_snapshot(store, asof_date=date(2026, 8, 18))
    _insert_snapshot(store, asof_date=date(2026, 8, 20))
    alpaca = _alpaca(
        sessions, {date(2026, 8, 19): (121_555.10, "portfolio_history_1min_close")}
    )

    inserted = backfill_missing_snapshots(store=store, alpaca=alpaca, today=date(2026, 8, 20))

    assert inserted == 0


def test_a_bad_lookup_does_not_stop_the_rest_of_the_batch(store):
    """Two missing sessions; one's close lookup blows up, the other succeeds."""
    sessions = [date(2026, 8, d) for d in (17, 18, 19, 20)]
    _insert_snapshot(store, asof_date=date(2026, 8, 20))
    # 2026-08-17, 08-18, 08-19 all missing.
    alpaca = _alpaca(
        sessions,
        {date(2026, 8, 19): (121_600.00, "portfolio_history_daily")},
        raises_close_for={date(2026, 8, 17), date(2026, 8, 18)},
    )

    inserted = backfill_missing_snapshots(store=store, alpaca=alpaca, today=date(2026, 8, 20))

    assert inserted == 1
    dates = {
        r[0] for r in store.conn.execute("SELECT asof_date FROM account_snapshots").fetchall()
    }
    assert dates == {date(2026, 8, 19), date(2026, 8, 20)}


def test_idempotent_a_second_run_does_not_reinsert_or_recheck(store):
    sessions = [date(2026, 8, d) for d in (18, 19, 20)]
    _insert_snapshot(store, asof_date=date(2026, 8, 18))
    _insert_snapshot(store, asof_date=date(2026, 8, 20))
    alpaca = _alpaca(
        sessions, {date(2026, 8, 19): (121_600.00, "portfolio_history_daily")}
    )

    first = backfill_missing_snapshots(store=store, alpaca=alpaca, today=date(2026, 8, 20))
    alpaca.session_close_equity.reset_mock()
    second = backfill_missing_snapshots(store=store, alpaca=alpaca, today=date(2026, 8, 20))

    assert first == 1
    assert second == 0
    alpaca.session_close_equity.assert_not_called()


def test_lookback_window_excludes_older_missing_sessions(store):
    """A session older than GAPFILL_LOOKBACK_SESSIONS trading sessions back is
    out of the bounded window and must not be checked at all."""
    assert GAPFILL_LOOKBACK_SESSIONS == 5
    old_missing_day = date(2026, 7, 1)
    sessions = [old_missing_day] + [
        date(2026, 8, d) for d in (13, 14, 17, 18, 19)
    ]
    for d in sessions[1:]:  # every recent session has a row; old_missing_day is out of window
        _insert_snapshot(store, asof_date=d)
    alpaca = _alpaca(sessions, {old_missing_day: (90_000.0, "portfolio_history_daily")})

    inserted = backfill_missing_snapshots(store=store, alpaca=alpaca, today=date(2026, 8, 19))

    assert inserted == 0
    alpaca.session_close_equity.assert_not_called()
    row = store.conn.execute(
        "SELECT COUNT(*) FROM account_snapshots WHERE asof_date = ?", [old_missing_day]
    ).fetchone()
    assert row[0] == 0


def test_multiple_missing_sessions_in_one_pass(store):
    sessions = [date(2026, 8, d) for d in (17, 18, 19, 20)]
    _insert_snapshot(store, asof_date=date(2026, 8, 17))
    # 08-18 and 08-19 both missing.
    _insert_snapshot(store, asof_date=date(2026, 8, 20))
    alpaca = _alpaca(
        sessions,
        {
            date(2026, 8, 18): (119_200.00, "portfolio_history_daily"),
            date(2026, 8, 19): (121_600.00, "portfolio_history_daily"),
        },
    )

    inserted = backfill_missing_snapshots(store=store, alpaca=alpaca, today=date(2026, 8, 20))

    assert inserted == 2
    rows = dict(
        store.conn.execute("SELECT asof_date, equity FROM account_snapshots").fetchall()
    )
    assert rows[date(2026, 8, 18)] == 119_200.00
    assert rows[date(2026, 8, 19)] == 121_600.00


def test_calendar_lookup_failure_is_a_safe_noop(store):
    sessions = []  # unused when calendar_raises
    alpaca = _alpaca(sessions, calendar_raises=True)

    inserted = backfill_missing_snapshots(store=store, alpaca=alpaca, today=date(2026, 8, 20))

    assert inserted == 0
    alpaca.session_close_equity.assert_not_called()


def test_empty_calendar_window_is_a_noop(store):
    alpaca = _alpaca([])

    inserted = backfill_missing_snapshots(store=store, alpaca=alpaca, today=date(2026, 8, 20))

    assert inserted == 0
    alpaca.session_close_equity.assert_not_called()


def test_defaults_today_to_now_when_not_given(store):
    """No `today` kwarg -- falls back to the real wall clock rather than
    crashing (matches reconcile()'s `now` default pattern)."""
    alpaca = _alpaca([])

    inserted = backfill_missing_snapshots(store=store, alpaca=alpaca)

    assert inserted == 0
