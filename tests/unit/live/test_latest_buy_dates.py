"""Tests for `_latest_buy_dates` in decide.py.

Two concerns:
1. Backend query shape — returns `dict[ticker -> date]` for tickers with
   any BUY fill; absent for tickers with none.
2. Timestamp convention (revised 2026-09-27; see decide.py docstrings and
   sma.live.reconcile._record_fills): `paper_fills.filled_at` is a naive
   TIMESTAMP already in ET wall-clock (duckdb's Python client converts
   alpaca-py's tz-aware UTC datetime to this host's local [ET] timezone
   before storing it into a naive column) -- NOT naive UTC as originally
   assumed (codex LOW finding, 2026-05-13, which added a UTC->ET conversion
   that has since been removed as double-converting an already-ET value).
   A plain `CAST(filled_at AS DATE)` is therefore already the ET trading
   day for any fill in the real 09:30-20:00 ET session window.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from sma.ingest.store import Store
from sma.live.decide import _current_holding_entry_dates, _latest_buy_dates


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(path=tmp_path / "t.duckdb").connect()
    yield s
    s.close()


def _insert_fill(
    store: Store, ticker: str, side: str, filled_at: datetime,
    shares: int = 10, price: float = 100.0, order_id: str | None = None,
) -> None:
    """Insert one row into paper_fills with the minimum required columns."""
    store.conn.execute(
        """
        INSERT INTO paper_fills (
            alpaca_order_id, ticker, side, filled_shares,
            fill_price, submitted_at, filled_at, asof_date, status, run_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'filled', 1)
        """,
        [
            order_id or f"{ticker}-{filled_at.isoformat()}",
            ticker, side, shares, price, filled_at, filled_at,
            filled_at.date(),
        ],
    )


def test_returns_empty_for_empty_tickers(store):
    assert _latest_buy_dates(store=store, tickers=set()) == {}


def test_returns_max_buy_date_per_ticker(store):
    _insert_fill(store, "AAPL", "BUY", datetime(2026, 5, 4, 14, 30))
    _insert_fill(store, "AAPL", "BUY", datetime(2026, 5, 6, 14, 30))  # later
    _insert_fill(store, "MSFT", "BUY", datetime(2026, 5, 5, 14, 30))
    out = _latest_buy_dates(store=store, tickers={"AAPL", "MSFT"})
    assert out == {"AAPL": date(2026, 5, 6), "MSFT": date(2026, 5, 5)}


def test_ignores_sells(store):
    """SELL fills must not count as entry dates."""
    _insert_fill(store, "AAPL", "BUY", datetime(2026, 5, 4, 14, 30))
    _insert_fill(store, "AAPL", "SELL", datetime(2026, 5, 7, 14, 30))
    out = _latest_buy_dates(store=store, tickers={"AAPL"})
    assert out == {"AAPL": date(2026, 5, 4)}


def test_ignores_zero_share_fills(store):
    """Zero-share rows are filtered out."""
    _insert_fill(store, "AAPL", "BUY", datetime(2026, 5, 4, 14, 30), shares=0)
    _insert_fill(store, "AAPL", "BUY", datetime(2026, 5, 6, 14, 30), shares=5)
    out = _latest_buy_dates(store=store, tickers={"AAPL"})
    assert out == {"AAPL": date(2026, 5, 6)}


def test_tickers_without_fills_absent(store):
    _insert_fill(store, "AAPL", "BUY", datetime(2026, 5, 4, 14, 30))
    out = _latest_buy_dates(store=store, tickers={"AAPL", "NEVER"})
    assert out == {"AAPL": date(2026, 5, 4)}
    assert "NEVER" not in out


def test_naive_et_timestamp_is_not_reinterpreted_as_utc(store):
    """Regression for the 2026-09-27 fix (supersedes the 2026-05-13 codex LOW
    finding, which had it backwards): `filled_at` is naive ET wall-clock, not
    naive UTC, so it must NOT be run through a UTC->America/New_York
    conversion. A fill naively timestamped 2026-05-04 02:30 -- early ET
    hours, picked because the OLD (buggy) conversion would reinterpret it as
    UTC and rotate it to 2026-05-03 22:30 ET, i.e. the WRONG calendar day --
    must attribute to 5/4, its own naive date."""
    _insert_fill(store, "AAPL", "BUY", datetime(2026, 5, 4, 2, 30))
    out = _latest_buy_dates(store=store, tickers={"AAPL"})
    assert out == {"AAPL": date(2026, 5, 4)}, (
        "A naive-ET filled_at must attribute to its own calendar date, not be "
        f"re-interpreted as UTC and rotated to the wrong day (got {out})"
    )


def test_am_and_pm_et_fills_attribute_to_their_own_trading_date(store):
    """The two real fill-time shapes seen in production (see reconcile.py's
    timestamp-convention docstring): a 09:32 ET fill near the open and a
    15:58 ET fill near the close. Both must land on their own naive-ET
    calendar date -- no conversion, no shift."""
    _insert_fill(store, "AAPL", "BUY", datetime(2026, 5, 4, 9, 32))
    _insert_fill(store, "MSFT", "BUY", datetime(2026, 5, 4, 15, 58))
    out = _latest_buy_dates(store=store, tickers={"AAPL", "MSFT"})
    assert out == {"AAPL": date(2026, 5, 4), "MSFT": date(2026, 5, 4)}


# --- _current_holding_entry_dates (trailing-peak window; Bug 2, 2026-07-01) ---
def test_current_holding_entry_returns_earliest_buy_of_streak(store):
    """A top-up must NOT move the entry date: it stays at the FIRST buy of the
    continuous holding (this is exactly where it differs from _latest_buy_dates,
    which returns the top-up date and caused the live-vs-sim trailing drift)."""
    _insert_fill(store, "AAPL", "BUY", datetime(2026, 5, 1, 14, 30), shares=25)
    _insert_fill(store, "AAPL", "BUY", datetime(2026, 5, 6, 14, 30), shares=10)  # top-up
    assert _current_holding_entry_dates(store=store, tickers={"AAPL"}) == {
        "AAPL": date(2026, 5, 1)
    }
    # Contrast: _latest_buy_dates returns the most-recent buy (the drift source).
    assert _latest_buy_dates(store=store, tickers={"AAPL"}) == {"AAPL": date(2026, 5, 6)}


def test_current_holding_entry_resets_after_full_close_and_reopen(store):
    """A full close (shares → 0) then a re-buy starts a NEW holding: the entry is
    the reopen date, not the original open."""
    _insert_fill(store, "AAPL", "BUY", datetime(2026, 5, 1, 14, 30), shares=25)
    _insert_fill(store, "AAPL", "SELL", datetime(2026, 5, 3, 14, 30), shares=25)  # flat
    _insert_fill(store, "AAPL", "BUY", datetime(2026, 5, 6, 14, 30), shares=10)   # reopen
    assert _current_holding_entry_dates(store=store, tickers={"AAPL"}) == {
        "AAPL": date(2026, 5, 6)
    }


def test_current_holding_entry_absent_when_net_flat(store):
    """A fully-closed position (net 0 shares) has no current holding → absent
    (caller falls back to a cost_basis-only peak, trailing off)."""
    _insert_fill(store, "AAPL", "BUY", datetime(2026, 5, 1, 14, 30), shares=25)
    _insert_fill(store, "AAPL", "SELL", datetime(2026, 5, 3, 14, 30), shares=25)
    assert _current_holding_entry_dates(store=store, tickers={"AAPL"}) == {}


def test_current_holding_entry_empty_for_empty_tickers(store):
    assert _current_holding_entry_dates(store=store, tickers=set()) == {}


def test_current_holding_entry_partial_sell_keeps_original_entry(store):
    """A partial trim (still net-long) does NOT reset the holding: the entry
    stays at the original open, since the position never went flat."""
    _insert_fill(store, "AAPL", "BUY", datetime(2026, 5, 1, 14, 30), shares=25)
    _insert_fill(store, "AAPL", "SELL", datetime(2026, 5, 3, 14, 30), shares=10)  # trim
    assert _current_holding_entry_dates(store=store, tickers={"AAPL"}) == {
        "AAPL": date(2026, 5, 1)
    }


# --- asof cutoff (replay point-in-time; live-attribution Bug 3, 2026-10-01) ---
def test_latest_buy_dates_asof_excludes_fills_after_asof(store):
    """Replaying a past night must not see buys that happened later. A fill ON
    the asof session (the morning OPG fill) is visible to that evening's
    decide; one the next day is not."""
    _insert_fill(store, "COIN", "BUY", datetime(2026, 8, 19, 9, 31))
    _insert_fill(store, "COIN", "BUY", datetime(2026, 8, 24, 9, 31))  # asof session
    _insert_fill(store, "COIN", "BUY", datetime(2026, 8, 26, 9, 31))  # future
    assert _latest_buy_dates(store=store, tickers={"COIN"}, asof=date(2026, 8, 24)) == {
        "COIN": date(2026, 8, 24)
    }
    assert _latest_buy_dates(store=store, tickers={"COIN"}, asof=date(2026, 8, 23)) == {
        "COIN": date(2026, 8, 19)
    }
    # No asof → unchanged legacy behaviour (all fills).
    assert _latest_buy_dates(store=store, tickers={"COIN"}) == {"COIN": date(2026, 8, 26)}


def test_current_holding_entry_asof_ignores_a_later_close_and_reopen(store):
    """A full close + reopen AFTER asof must not move the asof-night entry date
    (without the cutoff the streak start is the future reopen, held_days goes
    negative, and min_hold reads as expired)."""
    _insert_fill(store, "F", "BUY", datetime(2026, 9, 1, 9, 31), shares=50)
    _insert_fill(store, "F", "SELL", datetime(2026, 9, 8, 9, 31), shares=50)
    _insert_fill(store, "F", "BUY", datetime(2026, 9, 10, 9, 31), shares=40)
    assert _current_holding_entry_dates(
        store=store, tickers={"F"}, asof=date(2026, 9, 4)
    ) == {"F": date(2026, 9, 1)}
    assert _current_holding_entry_dates(store=store, tickers={"F"}) == {
        "F": date(2026, 9, 10)
    }

