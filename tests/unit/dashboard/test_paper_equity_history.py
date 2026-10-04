"""Equity History section (dashboard/tabs/paper.py): day-by-day account
equity view, added 2026-08-05 per Rayan's ask to "view what we were at on
different days".

Covers the two known account_snapshots artifacts:
  - 2026-07-07 Alpaca broker-wipe garbage row (~$6.9k) must be dropped, via
    the same >50%-off-trailing-median guard as decide.py's
    _clean_snapshot_equities (imported and reused, not reimplemented).
  - A missing day (e.g. 2026-08-04) must render as a gap, never a
    zero-filled row.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from dashboard.tabs import paper


@pytest.fixture(autouse=True)
def _clear_equity_history_cache():
    # st.cache_data caches by (function identity, args); _equity_history_df
    # takes no args, so a stale result from a prior test's monkeypatched
    # DB_PATH would otherwise leak across tests.
    paper._equity_history_df.clear()
    yield
    paper._equity_history_df.clear()


def _make_db(tmp_path) -> Path:
    db_path = tmp_path / "t.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE account_snapshots (
            asof_date DATE PRIMARY KEY, equity DOUBLE, cash DOUBLE,
            buying_power DOUBLE, long_market_value DOUBLE,
            position_count INTEGER, total_unrealized_pnl DOUBLE,
            run_id BIGINT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    con.close()
    return db_path


def _seed(
    con,
    asof: str,
    equity: float,
    cash: float = 10_000.0,
    long_market_value: float | None = None,
    position_count: int = 5,
) -> None:
    lmv = long_market_value if long_market_value is not None else equity - cash
    con.execute(
        "INSERT INTO account_snapshots "
        "(asof_date, equity, cash, buying_power, long_market_value, "
        " position_count, total_unrealized_pnl, run_id) "
        "VALUES (?, ?, ?, ?, ?, ?, 0, 1)",
        [asof, equity, cash, cash, lmv, position_count],
    )


# ── _equity_history_df (DB-backed) ────────────────────────────────────────


def test_equity_history_drops_the_707_garbage_row(monkeypatch, tmp_path):
    db_path = _make_db(tmp_path)
    monkeypatch.setattr(paper, "DB_PATH", db_path)
    con = duckdb.connect(str(db_path))
    for d, eq in [
        ("2026-07-03", 108_943.25),
        ("2026-07-06", 111_202.45),
        ("2026-07-07", 6_935.70),
        ("2026-07-08", 106_139.81),
        ("2026-07-09", 108_965.90),
    ]:
        _seed(con, d, eq)
    con.close()

    out = paper._equity_history_df()

    assert 6_935.70 not in set(out["equity"])
    assert date(2026, 7, 7) not in set(out["asof_date"].dt.date)
    assert len(out) == 4


def test_equity_history_gap_is_absent_not_zero(monkeypatch, tmp_path):
    db_path = _make_db(tmp_path)
    monkeypatch.setattr(paper, "DB_PATH", db_path)
    con = duckdb.connect(str(db_path))
    for d, eq in [
        ("2026-08-01", 97_000.0),
        ("2026-08-03", 97_932.76),
        # 2026-08-04 deliberately absent -- the real gap in the live DB.
    ]:
        _seed(con, d, eq)
    con.close()

    out = paper._equity_history_df()

    dates = set(out["asof_date"].dt.date)
    assert date(2026, 8, 4) not in dates
    assert 0.0 not in set(out["equity"])
    assert len(out) == 2


def test_equity_history_daily_change_skips_across_a_gap(monkeypatch, tmp_path):
    """The change on the row after a gap is the multi-day delta against the
    last EXISTING row, not a delta from a fabricated zero-filled row."""
    db_path = _make_db(tmp_path)
    monkeypatch.setattr(paper, "DB_PATH", db_path)
    con = duckdb.connect(str(db_path))
    _seed(con, "2026-08-01", 100_000.0)
    _seed(con, "2026-08-03", 102_000.0)  # 2026-08-02 absent
    con.close()

    out = paper._equity_history_df().sort_values("asof_date").reset_index(drop=True)

    assert pd.isna(out["daily_change_dollars"].iloc[0])
    assert out["daily_change_dollars"].iloc[1] == pytest.approx(2_000.0)
    assert out["daily_change_pct"].iloc[1] == pytest.approx(2.0)


def test_equity_history_too_few_rows_trusts_the_data(monkeypatch, tmp_path):
    """With <3 rows the garbage guard is a documented no-op (matches
    decide.py's _clean_snapshot_equities) -- an early-history 2-row DB must
    not be wiped out just because one value looks extreme."""
    db_path = _make_db(tmp_path)
    monkeypatch.setattr(paper, "DB_PATH", db_path)
    con = duckdb.connect(str(db_path))
    _seed(con, "2026-04-30", 100_000.0)
    _seed(con, "2026-05-01", 6_935.70)
    con.close()

    out = paper._equity_history_df()

    assert len(out) == 2


def test_equity_history_empty_db_returns_empty_frame(monkeypatch, tmp_path):
    db_path = _make_db(tmp_path)
    monkeypatch.setattr(paper, "DB_PATH", db_path)

    out = paper._equity_history_df()

    assert out.empty


def test_equity_history_tolerates_a_gap_filled_row_with_null_cash(monkeypatch, tmp_path):
    """A gap-filled row (sma.live.reconcile.backfill_missing_snapshots, added
    2026-08-20) stores NULL cash/buying_power/long_market_value/
    position_count for a day reconcile never ran at all -- those columns are
    genuinely unknowable in retrospect. The dashboard must render the equity
    series (which IS known) without crashing on that row."""
    db_path = _make_db(tmp_path)
    monkeypatch.setattr(paper, "DB_PATH", db_path)
    con = duckdb.connect(str(db_path))
    _seed(con, "2026-08-18", 102_700.0)
    con.execute(
        "INSERT INTO account_snapshots "
        "(asof_date, equity, cash, buying_power, long_market_value, "
        " position_count, total_unrealized_pnl, run_id) "
        "VALUES ('2026-08-19', 121600.0, NULL, NULL, NULL, NULL, NULL, 2)"
    )
    con.close()

    out = paper._equity_history_df()

    assert len(out) == 2
    row = out[out["asof_date"] == pd.Timestamp("2026-08-19")].iloc[0]
    assert row["equity"] == pytest.approx(121_600.0)
    assert pd.isna(row["cash"])
    assert pd.isna(row["long_market_value"])
    assert pd.isna(row["position_count"])


def test_equity_history_includes_cash_and_position_count(monkeypatch, tmp_path):
    db_path = _make_db(tmp_path)
    monkeypatch.setattr(paper, "DB_PATH", db_path)
    con = duckdb.connect(str(db_path))
    _seed(con, "2026-04-30", 100_000.0, cash=8_123.27, position_count=19)
    _seed(con, "2026-05-01", 99_919.28, cash=90_415.44, position_count=2)
    con.close()

    out = paper._equity_history_df().sort_values("asof_date").reset_index(drop=True)

    assert out["cash"].iloc[0] == pytest.approx(8_123.27)
    assert int(out["position_count"].iloc[1]) == 2


# ── pure helpers: no DB/Streamlit needed ─────────────────────────────────


def test_equity_summary_stats_current_start_peak_drawdown():
    df = pd.DataFrame(
        {
            "asof_date": pd.to_datetime(
                ["2026-04-30", "2026-05-01", "2026-05-04", "2026-05-05"]
            ),
            "equity": [100_000.0, 110_000.0, 99_000.0, 105_000.0],
        }
    )

    stats = paper._equity_summary_stats(df)

    assert stats["current"] == pytest.approx(105_000.0)
    assert stats["start"] == pytest.approx(100_000.0)
    assert stats["peak"] == pytest.approx(110_000.0)
    # Max drawdown from the 110k peak down to 99k = -10%.
    assert stats["max_drawdown_pct"] == pytest.approx(-10.0)


def test_equity_summary_stats_empty_input():
    stats = paper._equity_summary_stats(pd.DataFrame(columns=["asof_date", "equity"]))

    assert stats == {
        "current": None,
        "start": None,
        "peak": None,
        "max_drawdown_pct": None,
    }


def test_align_spy_to_equity_window_drops_spy_rows_past_last_equity_date():
    """SPY prices update daily regardless of account_snapshots gaps -- the
    real 2026-08-04 case: SPY has a price that day but the reconcile
    snapshot is missing, so the equity history's last date is 2026-08-03.
    The SPY comparison must not run ahead of that."""
    spy = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-01", "2026-08-03", "2026-08-04"]),
            "spy_return_pct": [5.0, 5.43, 7.33],
        }
    )

    out = paper._align_spy_to_equity_window(spy, pd.Timestamp("2026-08-03"))

    assert list(out["date"]) == list(pd.to_datetime(["2026-08-01", "2026-08-03"]))
    assert out["spy_return_pct"].iloc[-1] == pytest.approx(5.43)


def test_align_spy_to_equity_window_empty_input_is_a_noop():
    out = paper._align_spy_to_equity_window(pd.DataFrame(), pd.Timestamp("2026-08-03"))
    assert out.empty


def test_with_daily_changes_first_row_is_nan_rest_computed():
    df = pd.DataFrame(
        {
            "asof_date": pd.to_datetime(["2026-04-30", "2026-05-01"]),
            "equity": [100_000.0, 101_000.0],
        }
    )

    out = paper._with_daily_changes(df)

    assert pd.isna(out["daily_change_dollars"].iloc[0])
    assert pd.isna(out["daily_change_pct"].iloc[0])
    assert out["daily_change_dollars"].iloc[1] == pytest.approx(1_000.0)
    assert out["daily_change_pct"].iloc[1] == pytest.approx(1.0)
