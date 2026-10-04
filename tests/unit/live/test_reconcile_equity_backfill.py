"""Self-healing backfill: replace a 1Min-proxy snapshot equity with Alpaca's
OFFICIAL daily close once it publishes.

7081d81 (2026-08-12) made evening snapshots source equity from Alpaca's
portfolio-history when get_account() would otherwise return an after-hours
mark, preferring the published 1D bar and falling back to the 16:00 ET point
of the 1Min series (validated ~0.01% error) when the 1D bar isn't posted yet
-- which is nearly always true for an evening reconcile (measured ABSENT at
21:20 ET the same session). That correction was a one-time manual fix for the
8/12 row. This backfill makes it automatic: at the start of every reconcile
run, any snapshot still on the 1Min proxy from the last few trading days gets
re-checked, and is upgraded to the official close once Alpaca finally
publishes it.
"""

from datetime import date
from unittest.mock import MagicMock

import pytest

from sma.ingest.store import Store
from sma.live.reconcile import (
    BACKFILL_LOOKBACK_TRADING_DAYS,
    backfill_official_closes,
)


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(path=tmp_path / "t.duckdb").connect()


def _insert_snapshot(
    store, *, asof_date: date, equity: float, cash: float, lmv: float,
    source: str | None, run_id: int = 1,
):
    store.conn.execute(
        """
        INSERT INTO account_snapshots
        (asof_date, equity, cash, buying_power, long_market_value,
         position_count, total_unrealized_pnl, run_id, equity_source)
        VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)
        """,
        [asof_date, equity, cash, cash + lmv, lmv, 5, run_id, source],
    )


def _alpaca(close_by_day: dict | None = None, *, raises_for: set | None = None):
    """MagicMock AlpacaClient whose session_close_equity(day=...) is driven by
    a {date: (equity, source)} map. `raises_for` maps specific days to a
    lookup failure instead."""
    close_by_day = close_by_day or {}
    raises_for = raises_for or set()

    def _side_effect(*, day):
        if day in raises_for:
            raise RuntimeError("portfolio-history 503")
        return close_by_day.get(day)

    m = MagicMock()
    m.session_close_equity.side_effect = _side_effect
    return m


def test_no_candidate_rows_is_a_noop(store):
    """No snapshot carries the proxy flag -- nothing to check, no API calls."""
    _insert_snapshot(
        store, asof_date=date(2026, 8, 12), equity=100_000, cash=10_000,
        lmv=90_000, source="portfolio_history_daily",
    )
    alpaca = _alpaca()

    corrected = backfill_official_closes(store=store, alpaca=alpaca)

    assert corrected == 0
    alpaca.session_close_equity.assert_not_called()


def test_corrects_a_proxy_row_once_the_official_close_publishes(store, caplog):
    """The 2026-08-12 case: a 1Min-proxy row, and the official 1D bar has
    since been published with a (slightly) different value."""
    _insert_snapshot(
        store, asof_date=date(2026, 8, 12), equity=102_516.39, cash=7_720.23,
        lmv=94_796.16, source="portfolio_history_1min_close",
    )
    alpaca = _alpaca({date(2026, 8, 12): (102_343.20, "portfolio_history_daily")})

    with caplog.at_level("INFO", logger="sma.live.reconcile"):
        corrected = backfill_official_closes(store=store, alpaca=alpaca)

    assert corrected == 1
    row = store.conn.execute(
        "SELECT equity, long_market_value, cash, equity_source "
        "FROM account_snapshots WHERE asof_date = ?",
        [date(2026, 8, 12)],
    ).fetchone()
    equity, lmv, cash, source = row
    assert equity == 102_343.20
    assert source == "portfolio_history_daily_close"
    assert cash == 7_720.23, "cash is mark-independent -- must be untouched"
    assert equity == pytest.approx(cash + lmv, abs=0.01), "equity == cash + lmv must hold"
    assert sum("backfill: corrected" in r.message for r in caplog.records) == 1


def test_still_only_the_1min_proxy_leaves_the_row_untouched(store):
    """session_close_equity falls back to the SAME 1Min-proxy tier again --
    the official bar has not landed. Must not be mistaken for a correction."""
    _insert_snapshot(
        store, asof_date=date(2026, 8, 12), equity=102_343.20, cash=7_720.23,
        lmv=94_622.97, source="portfolio_history_1min_close",
    )
    alpaca = _alpaca({
        date(2026, 8, 12): (102_343.20, "portfolio_history_1min_close"),
    })

    corrected = backfill_official_closes(store=store, alpaca=alpaca)

    assert corrected == 0
    source = store.conn.execute(
        "SELECT equity_source FROM account_snapshots WHERE asof_date = ?",
        [date(2026, 8, 12)],
    ).fetchone()[0]
    assert source == "portfolio_history_1min_close"


def test_unpublished_session_leaves_the_row_untouched(store):
    """session_close_equity returns None (nothing published yet at either
    tier) -- normal, retried on the next reconcile run."""
    _insert_snapshot(
        store, asof_date=date(2026, 8, 13), equity=105_101.25, cash=7_720.10,
        lmv=97_381.15, source="portfolio_history_1min_close",
    )
    alpaca = _alpaca({date(2026, 8, 13): None})

    corrected = backfill_official_closes(store=store, alpaca=alpaca)

    assert corrected == 0
    source = store.conn.execute(
        "SELECT equity_source FROM account_snapshots WHERE asof_date = ?",
        [date(2026, 8, 13)],
    ).fetchone()[0]
    assert source == "portfolio_history_1min_close"


def test_exact_agreement_needs_no_write(store):
    """The official close happens to equal the proxy exactly -- no material
    difference, so leave the row alone (still correct either way)."""
    _insert_snapshot(
        store, asof_date=date(2026, 8, 12), equity=102_343.20, cash=7_720.23,
        lmv=94_622.97, source="portfolio_history_1min_close",
    )
    alpaca = _alpaca({date(2026, 8, 12): (102_343.20, "portfolio_history_daily")})

    corrected = backfill_official_closes(store=store, alpaca=alpaca)

    assert corrected == 0
    source = store.conn.execute(
        "SELECT equity_source FROM account_snapshots WHERE asof_date = ?",
        [date(2026, 8, 12)],
    ).fetchone()[0]
    assert source == "portfolio_history_1min_close"


def test_a_bad_lookup_does_not_stop_the_rest_of_the_batch(store):
    """One row's history lookup blows up; the other candidate row in the
    window must still be checked and corrected."""
    _insert_snapshot(
        store, asof_date=date(2026, 8, 11), equity=101_957.79, cash=9_893.08,
        lmv=92_064.71, source="portfolio_history_1min_close",
    )
    _insert_snapshot(
        store, asof_date=date(2026, 8, 12), equity=102_516.39, cash=7_720.23,
        lmv=94_796.16, source="portfolio_history_1min_close",
    )
    alpaca = _alpaca(
        close_by_day={date(2026, 8, 12): (102_343.20, "portfolio_history_daily")},
        raises_for={date(2026, 8, 11)},
    )

    corrected = backfill_official_closes(store=store, alpaca=alpaca)

    assert corrected == 1
    rows = dict(store.conn.execute(
        "SELECT asof_date, equity_source FROM account_snapshots"
    ).fetchall())
    assert rows[date(2026, 8, 11)] == "portfolio_history_1min_close", (
        "the row whose lookup failed must be left exactly as it was"
    )
    assert rows[date(2026, 8, 12)] == "portfolio_history_daily_close"


def test_idempotent_a_second_run_does_not_recheck_a_corrected_row(store):
    """Once corrected, a row's source is no longer the proxy flag, so a
    second backfill pass must not touch it (or call Alpaca for it) again."""
    _insert_snapshot(
        store, asof_date=date(2026, 8, 12), equity=102_516.39, cash=7_720.23,
        lmv=94_796.16, source="portfolio_history_1min_close",
    )
    alpaca = _alpaca({date(2026, 8, 12): (102_343.20, "portfolio_history_daily")})

    first = backfill_official_closes(store=store, alpaca=alpaca)
    alpaca.session_close_equity.reset_mock()
    second = backfill_official_closes(store=store, alpaca=alpaca)

    assert first == 1
    assert second == 0
    alpaca.session_close_equity.assert_not_called()


def test_after_hours_and_null_rows_are_healed_too(store):
    """2026-08-16: the old filter healed ONLY 'portfolio_history_1min_close',
    which excluded exactly the wrong rows. 'get_account_after_hours' is the
    label _snapshot_equity writes when portfolio-history failed or had not
    published -- the read that was off by THOUSANDS on 2026-07-29 -- and NULL
    is legacy provenance. Both are less trustworthy than the 1Min proxy, and
    both must be upgraded once the official bar lands."""
    _insert_snapshot(
        store, asof_date=date(2026, 8, 10), equity=101_551.57, cash=9_912.83,
        lmv=91_638.74, source=None,
    )
    _insert_snapshot(
        store, asof_date=date(2026, 8, 11), equity=101_999.99, cash=9_893.08,
        lmv=92_106.91, source="get_account_after_hours",
    )
    _insert_snapshot(
        store, asof_date=date(2026, 8, 12), equity=102_500.00, cash=7_720.23,
        lmv=94_779.77, source="get_account",
    )
    alpaca = _alpaca({
        date(2026, 8, 10): (101_500.00, "portfolio_history_daily"),
        date(2026, 8, 11): (101_957.79, "portfolio_history_daily"),
        date(2026, 8, 12): (102_343.20, "portfolio_history_daily"),
    })

    corrected = backfill_official_closes(store=store, alpaca=alpaca)

    assert corrected == 3
    rows = {
        r[0]: r[1:]
        for r in store.conn.execute(
            "SELECT asof_date, equity, cash, long_market_value, equity_source "
            "FROM account_snapshots"
        ).fetchall()
    }
    assert rows[date(2026, 8, 10)][0] == 101_500.00, "NULL-provenance row healed"
    assert rows[date(2026, 8, 11)][0] == 101_957.79, "after-hours row healed"
    assert rows[date(2026, 8, 12)][0] == 102_343.20, "intraday get_account healed"
    for equity, cash, lmv, source in rows.values():
        assert source == "portfolio_history_daily_close"
        assert equity == pytest.approx(cash + lmv, abs=0.01), (
            "equity == cash + long_market_value must survive the heal"
        )


def test_rows_already_on_the_official_bar_are_never_candidates(store):
    """A row whose equity ALREADY came from the official daily bar (either the
    snapshot got it directly, or a previous backfill wrote it) needs no heal
    and must cost no API call."""
    _insert_snapshot(
        store, asof_date=date(2026, 8, 11), equity=101_957.79, cash=9_893.08,
        lmv=92_064.71, source="portfolio_history_daily",
    )
    _insert_snapshot(
        store, asof_date=date(2026, 8, 12), equity=102_343.20, cash=7_720.23,
        lmv=94_622.97, source="portfolio_history_daily_close",
    )
    alpaca = _alpaca()

    corrected = backfill_official_closes(store=store, alpaca=alpaca)

    assert corrected == 0
    alpaca.session_close_equity.assert_not_called()


def test_lookback_window_excludes_older_proxy_rows(store):
    """A proxy row older than BACKFILL_LOOKBACK_TRADING_DAYS trading days is
    out of the bounded window and must not be touched, even though nothing
    else in the table is newer than it."""
    assert BACKFILL_LOOKBACK_TRADING_DAYS == 5
    old_day = date(2026, 7, 1)
    _insert_snapshot(
        store, asof_date=old_day, equity=90_000, cash=10_000, lmv=80_000,
        source="portfolio_history_1min_close",
    )
    # Five newer rows push `old_day` outside the last-5-trading-days window.
    for i, d in enumerate(
        [date(2026, 8, 6), date(2026, 8, 7), date(2026, 8, 10),
         date(2026, 8, 11), date(2026, 8, 12)]
    ):
        _insert_snapshot(
            store, asof_date=d, equity=100_000 + i, cash=10_000, lmv=90_000 + i,
            source="portfolio_history_daily_close",
        )
    alpaca = _alpaca({old_day: (95_000, "portfolio_history_daily")})

    corrected = backfill_official_closes(store=store, alpaca=alpaca)

    assert corrected == 0
    alpaca.session_close_equity.assert_not_called()
    source = store.conn.execute(
        "SELECT equity_source FROM account_snapshots WHERE asof_date = ?",
        [old_day],
    ).fetchone()[0]
    assert source == "portfolio_history_1min_close"


def test_corrects_multiple_candidates_within_the_window(store):
    """Two proxy rows in the window, both with published official closes --
    both get corrected in one pass."""
    _insert_snapshot(
        store, asof_date=date(2026, 8, 12), equity=102_516.39, cash=7_720.23,
        lmv=94_796.16, source="portfolio_history_1min_close",
    )
    _insert_snapshot(
        store, asof_date=date(2026, 8, 13), equity=105_150.00, cash=7_720.10,
        lmv=97_429.90, source="portfolio_history_1min_close",
    )
    alpaca = _alpaca({
        date(2026, 8, 12): (102_343.20, "portfolio_history_daily"),
        date(2026, 8, 13): (105_101.25, "portfolio_history_daily"),
    })

    corrected = backfill_official_closes(store=store, alpaca=alpaca)

    assert corrected == 2
    rows = {
        r[0]: (r[1], r[2])
        for r in store.conn.execute(
            "SELECT asof_date, equity, equity_source FROM account_snapshots"
        ).fetchall()
    }
    assert rows[date(2026, 8, 12)] == (102_343.20, "portfolio_history_daily_close")
    assert rows[date(2026, 8, 13)] == (105_101.25, "portfolio_history_daily_close")
