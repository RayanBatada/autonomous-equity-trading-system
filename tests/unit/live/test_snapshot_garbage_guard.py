"""Defensive filter for garbage account_snapshots rows (2026-07-30).

The 2026-07-07 Alpaca broker wipe wrote a garbage $6,935 equity row (real
incident, self-healed the next day) — proving the broker CAN report garbage.
Both readers of account_snapshots must not trust an isolated outlier row:
  - the catastrophic-loss abort's "prior day" reference (a spuriously HIGH
    prior row would make today's real equity look like a huge loss and
    wrongly abort trading)
  - the drawdown rail's running peak, MAX(equity) (a spuriously HIGH row
    would inflate the peak FOREVER, permanently overstating drawdown and
    wedging the derisk rail into blocking all buys)

`_clean_snapshot_equities` drops rows more than 50% away from the trailing
median of the surrounding snapshots before either read uses them.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from sma.ingest.store import Store
from sma.live.decide import _clean_snapshot_equities


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(path=tmp_path / "t.duckdb").connect()
    yield s
    s.close()


def _seed(store: Store, asof: date, equity: float) -> None:
    store.conn.execute(
        "INSERT INTO account_snapshots "
        "(asof_date, equity, cash, buying_power, long_market_value, "
        " position_count, total_unrealized_pnl, run_id) "
        "VALUES (?, ?, 0, 0, ?, 0, 0, 1)",
        [asof, equity, equity],
    )


def test_drops_garbage_high_row(store):
    """A $500k row among ~$100k rows is >50% off the trailing median -> dropped."""
    for i, eq in enumerate([98_000.0, 101_000.0, 99_500.0, 500_000.0, 102_000.0]):
        _seed(store, date(2026, 7, i + 1), eq)
    out = _clean_snapshot_equities(store, asof=date(2026, 7, 10))
    equities = {e for _, e in out}
    assert 500_000.0 not in equities
    assert equities == {98_000.0, 101_000.0, 99_500.0, 102_000.0}


def test_drops_garbage_low_row(store):
    """The real 2026-07-07 incident: a $6,935 row among ~$100k rows -> dropped."""
    for i, eq in enumerate([99_000.0, 100_500.0, 6_935.0, 101_200.0, 98_800.0]):
        _seed(store, date(2026, 7, i + 1), eq)
    out = _clean_snapshot_equities(store, asof=date(2026, 7, 10))
    equities = {e for _, e in out}
    assert 6_935.0 not in equities
    assert equities == {99_000.0, 100_500.0, 101_200.0, 98_800.0}


def test_clean_data_returns_all_rows_unfiltered(store):
    """No filtering must occur when every row is close to the median — the
    guard must be a no-op on normal data."""
    rows_in = [98_000.0, 101_000.0, 99_500.0, 103_000.0, 97_000.0]
    for i, eq in enumerate(rows_in):
        _seed(store, date(2026, 7, i + 1), eq)
    out = _clean_snapshot_equities(store, asof=date(2026, 7, 10))
    assert sorted(e for _, e in out) == sorted(rows_in)


def test_respects_asof_cutoff(store):
    """Rows AFTER asof must never be considered (matches the original <= asof
    query semantics)."""
    _seed(store, date(2026, 7, 1), 100_000.0)
    _seed(store, date(2026, 7, 2), 101_000.0)
    _seed(store, date(2026, 7, 15), 999_999.0)  # future relative to asof
    out = _clean_snapshot_equities(store, asof=date(2026, 7, 10))
    dates = {d for d, _ in out}
    assert date(2026, 7, 15) not in dates


def test_too_few_rows_skips_filtering(store):
    """With <3 rows there isn't enough history for a meaningful median — trust
    the data rather than guess (matches legacy behavior when history is thin)."""
    _seed(store, date(2026, 7, 1), 100_000.0)
    _seed(store, date(2026, 7, 2), 6_935.0)  # would look like garbage with more history
    out = _clean_snapshot_equities(store, asof=date(2026, 7, 10))
    assert sorted(e for _, e in out) == [6_935.0, 100_000.0]
