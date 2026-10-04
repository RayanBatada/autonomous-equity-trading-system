"""Unit tests for sma.live.replay's building blocks.

See tests/integration/live/test_replay.py for the end-to-end guarantees
(never submits, never mutates the DB, regression against a real recorded
decide night).
"""

from datetime import date, datetime

import duckdb
import pytest

from sma.ingest.store import Store
from sma.live.replay import (
    StoredPredictionsPredictor,
    _NoSubmitAlpaca,
    build_book,
    reconstruct_book_asof,
    replay_connection,
)


def _seeded_store(tmp_path):
    db = tmp_path / "test.duckdb"
    store = Store(path=str(db)).connect()
    return store, db


def test_stored_predictions_predictor_picks_latest_computed_at(tmp_path):
    store, _db = _seeded_store(tmp_path)
    store.conn.execute(
        "INSERT INTO predictions "
        "(asof_date, ticker, target, predicted_value, model_id, computed_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [date(2026, 8, 28), "AAPL", "ret_30d_forward", 0.01, "model-a",
         datetime(2026, 8, 28, 19, 30)],
    )
    store.conn.execute(
        "INSERT INTO predictions "
        "(asof_date, ticker, target, predicted_value, model_id, computed_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [date(2026, 8, 28), "AAPL", "ret_30d_forward", 0.02, "model-b",
         datetime(2026, 8, 28, 19, 45)],
    )
    # A different ticker + a different target: must be excluded.
    store.conn.execute(
        "INSERT INTO predictions "
        "(asof_date, ticker, target, predicted_value, model_id, computed_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [date(2026, 8, 28), "AAPL", "ret_5d_forward", 0.99, "model-a",
         datetime(2026, 8, 28, 19, 45)],
    )

    predictor = StoredPredictionsPredictor(store.conn)
    scores = predictor.predict_for(date(2026, 8, 28), ["AAPL", "MSFT"])

    assert scores == {"AAPL": 0.02}  # the later computed_at row wins


def test_replay_connection_pit_filters_theses_passes_through_prices(tmp_path):
    store, db = _seeded_store(tmp_path)
    store.conn.execute(
        "INSERT INTO theses (ticker, asof_date, run_id, conviction, created_at) "
        "VALUES (?, ?, 1, 'bullish', ?)",
        ["AAPL", date(2026, 8, 28), datetime(2026, 8, 28, 19, 0)],
    )
    store.conn.execute(
        "INSERT INTO theses (ticker, asof_date, run_id, conviction, created_at) "
        "VALUES (?, ?, 1, 'bearish', ?)",
        ["MSFT", date(2026, 8, 28), datetime(2026, 8, 28, 20, 30)],
    )
    store.conn.execute(
        "INSERT INTO prices (ticker, date, open, high, low, close, adj_close, "
        "volume, source, run_id) VALUES ('AAPL', ?, 1, 1, 1, 1, 1, 100, 'yfinance', 1)",
        [date(2026, 8, 28)],
    )
    store.close()

    conn = replay_connection(db, asof=date(2026, 8, 28))
    try:
        theses_tickers = {
            r[0] for r in conn.execute("SELECT ticker FROM theses").fetchall()
        }
        assert theses_tickers == {"AAPL"}  # MSFT's thesis was created AFTER 20:00
        assert conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0] == 1
    finally:
        conn.close()


def test_reconstruct_book_asof_respects_cutoff_and_sums_fills(tmp_path):
    store, db = _seeded_store(tmp_path)

    def _fill(ticker, side, shares, price, filled_at):
        store.conn.execute(
            "INSERT INTO paper_fills (alpaca_order_id, asof_date, ticker, side, "
            "filled_shares, fill_price, status, submitted_at, filled_at, run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, 'filled', ?, ?, 1)",
            [f"ord-{ticker}-{filled_at}", filled_at.date(), ticker, side, shares,
             price, filled_at, filled_at],
        )

    _fill("AAPL", "BUY", 100, 200.0, datetime(2026, 8, 27, 13, 30))   # before cutoff
    _fill("AAPL", "SELL", 30, 210.0, datetime(2026, 8, 28, 13, 30))   # before cutoff
    _fill("MSFT", "BUY", 50, 400.0, datetime(2026, 8, 29, 13, 30))    # AFTER cutoff
    store.close()

    conn = duckdb.connect(str(db), read_only=True)
    try:
        positions = reconstruct_book_asof(conn, asof=date(2026, 8, 28))
    finally:
        conn.close()

    assert set(positions) == {"AAPL"}
    assert positions["AAPL"]["shares"] == 70


def test_build_book_current_requires_alpaca(tmp_path):
    store, db = _seeded_store(tmp_path)
    store.close()
    conn = duckdb.connect(str(db), read_only=True)
    try:
        with pytest.raises(ValueError, match="requires a live AlpacaClient"):
            build_book(mode="current", conn=conn, asof=date(2026, 8, 28))
    finally:
        conn.close()


def test_build_book_empty_has_no_positions(tmp_path):
    store, db = _seeded_store(tmp_path)
    store.conn.execute(
        "INSERT INTO account_snapshots (asof_date, equity, cash, buying_power, "
        "long_market_value, position_count, run_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [date(2026, 8, 27), 100_000.0, 5_000.0, 5_000.0, 95_000.0, 3, 1],
    )
    store.close()
    conn = duckdb.connect(str(db), read_only=True)
    try:
        book = build_book(mode="empty", conn=conn, asof=date(2026, 8, 28))
    finally:
        conn.close()

    assert book.get_positions() == {}
    assert book.get_account()["equity"] == 100_000.0
    assert book.get_account()["cash"] == 100_000.0  # all cash, by construction


def test_no_submit_alpaca_blocks_submits_but_passes_reads():
    class _Inner:
        def get_account(self):
            return {"equity": 1.0}

        def get_positions(self):
            return {"AAPL": {"shares": 1, "cost_basis": 1.0}}

        def submit_day_market_buy(self, *a, **kw):
            raise AssertionError("should never be called through the guard")

    guarded = _NoSubmitAlpaca(_Inner())
    assert guarded.get_account() == {"equity": 1.0}
    assert guarded.get_positions() == {"AAPL": {"shares": 1, "cost_basis": 1.0}}

    from sma.live.replay import SubmitBlockedError

    with pytest.raises(SubmitBlockedError):
        guarded.submit_day_market_buy("AAPL", 1)
    with pytest.raises(SubmitBlockedError):
        guarded.submit_day_sell("AAPL", 1)
    with pytest.raises(SubmitBlockedError):
        guarded.submit_day_opg_buy("AAPL", 1)
    with pytest.raises(SubmitBlockedError):
        guarded.submit_market_sell("AAPL", 1)
    with pytest.raises(SubmitBlockedError):
        guarded.get_order_by_client_order_id("coid")
    with pytest.raises(AttributeError):
        guarded.some_unrelated_method()


def test_reconstruct_book_asof_cutoff_is_naive_et_not_utc(tmp_path):
    """`filled_at` is naive ET wall-clock (decide.py docstring, 6ebf07d). The
    cutoff is asof 20:00 ET compared directly against it. The old code
    converted 20:00 ET to naive UTC (00:00 the next day), which let a fill
    stamped 21:00 ET on asof into the book."""
    store, db = _seeded_store(tmp_path)

    def _fill(ticker, filled_at):
        store.conn.execute(
            "INSERT INTO paper_fills (alpaca_order_id, asof_date, ticker, side, "
            "filled_shares, fill_price, status, submitted_at, filled_at, run_id) "
            "VALUES (?, ?, ?, 'BUY', 10, 100.0, 'filled', ?, ?, 1)",
            [f"ord-{ticker}", filled_at.date(), ticker, filled_at, filled_at],
        )

    _fill("AAPL", datetime(2026, 8, 28, 19, 59))   # before 20:00 ET decide
    _fill("MSFT", datetime(2026, 8, 28, 21, 0))    # after decide, same ET date
    store.close()

    conn = duckdb.connect(str(db), read_only=True)
    try:
        positions = reconstruct_book_asof(conn, asof=date(2026, 8, 28))
    finally:
        conn.close()
    assert set(positions) == {"AAPL"}
