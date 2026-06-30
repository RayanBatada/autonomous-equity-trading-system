"""Tests for `_latest_buy_dates` in decide.py.

Two concerns:
1. Backend query shape — returns `dict[ticker -> date]` for tickers with
   any BUY fill; absent for tickers with none.
2. UTC → ET conversion (codex LOW finding, 2026-05-13). A late-day ET
   fill (e.g. 23:30 ET = 03:30 UTC next day) must attribute to the ET
   trading session, not the UTC date.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from sma.ingest.store import Store
from sma.live.decide import _latest_buy_dates


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


def test_late_day_et_fill_attributes_to_et_session_not_utc(store):
    """Codex LOW finding (2026-05-13): a fill at 2026-05-04T23:30 ET equals
    2026-05-05T03:30 UTC. The naïve `CAST AS DATE` would yield 5/5 — but
    the trader thinks of that fill as belonging to the 5/4 session. With
    the UTC→America/New_York conversion in place, `_latest_buy_dates`
    returns 5/4, not 5/5."""
    # Store the timestamp as UTC (after 7pm ET = next UTC day).
    _insert_fill(
        store, "AAPL", "BUY",
        datetime(2026, 5, 5, 3, 30),   # 5/5 03:30 UTC == 5/4 23:30 ET
    )
    out = _latest_buy_dates(store=store, tickers={"AAPL"})
    assert out == {"AAPL": date(2026, 5, 4)}, (
        "Late-day ET fill must attribute to the ET trading session, "
        f"not the UTC date (got {out})"
    )


def test_midday_et_fill_unaffected_by_conversion(store):
    """Sanity check: a midday ET fill (well inside both ET and UTC days)
    attributes to the same date with or without the conversion."""
    _insert_fill(
        store, "AAPL", "BUY",
        datetime(2026, 5, 4, 17, 30),  # 5/4 17:30 UTC == 5/4 13:30 ET
    )
    out = _latest_buy_dates(store=store, tickers={"AAPL"})
    assert out == {"AAPL": date(2026, 5, 4)}
