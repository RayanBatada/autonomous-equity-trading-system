"""Integration tests for live.decide.decide_once."""

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from alpaca.trading.client import TradingClient

from sma.backtest.strategies.base import StrategyDecision
from sma.ingest.quality import EXPECTED_SOURCES
from sma.ingest.store import Store
from sma.live.alpaca_client import AlpacaClient
from sma.live.decide import (
    _last_prices_per_ticker,
    _sector_exposure,
    _to_dollars,
    decide_once,
)
from sma.live.exceptions import EmptyDecisionsWithHeldPositionsError
from sma.risk.rails import RiskRails


@pytest.fixture
def db_with_prices(tmp_path):
    """Tmp DuckDB seeded with prices + an ingest_log success row for today."""
    db = tmp_path / "test.duckdb"
    store = Store(path=str(db)).connect()
    rid = store.allocate_run_id()

    # ingest_log: all EXPECTED_SOURCES at status='ok' for today (matches what
    # the strict preflight ingest check requires).
    started = datetime(2026, 5, 1, 18, 30)
    finished = datetime(2026, 5, 1, 18, 31)
    for src in EXPECTED_SOURCES:
        store.conn.execute(
            "INSERT INTO ingest_log (run_id, source, started_at, finished_at, "
            " rows_inserted, status) VALUES (?, ?, ?, ?, 100, 'ok')",
            [rid, src, started, finished],
        )
    # Prices for AAPL and MSFT, last-known on 2026-05-01.
    for ticker, base_price in [("AAPL", 200.0), ("MSFT", 400.0)]:
        for offset in range(5):
            d = date(2026, 4, 27) + timedelta(days=offset)
            store.conn.execute(
                "INSERT INTO prices (ticker, date, open, high, low, close, "
                " adj_close, volume, source, run_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 1000000, 'yfinance', ?)",
                [ticker, d, base_price, base_price, base_price,
                 base_price, base_price, rid],
            )

    return store, db


def _alpaca_mock(*, equity=100_000.0, positions=()):
    tc = MagicMock(spec=TradingClient)

    acct = MagicMock()
    acct.equity = str(equity)
    acct.cash = str(equity * 0.5)
    acct.buying_power = str(equity)
    acct.long_market_value = str(equity * 0.5)
    acct.trading_blocked = False
    acct.account_blocked = False
    tc.get_account.return_value = acct

    pos_objs = []
    for ticker, shares, cost in positions:
        p = MagicMock()
        p.symbol = ticker
        p.qty = str(shares)
        p.avg_entry_price = str(cost)
        pos_objs.append(p)
    tc.get_all_positions.return_value = pos_objs

    cal_entry = MagicMock()
    cal_entry.date = date(2026, 5, 4)   # Mon
    tc.get_calendar.return_value = [cal_entry]

    submitted_ids = []

    def submit_order(req):
        oid = f"ord-{len(submitted_ids)}"
        submitted_ids.append(oid)
        ret = MagicMock()
        ret.id = oid
        return ret

    tc.submit_order.side_effect = submit_order
    # Default: the broker has no pre-existing order for any client_order_id, so
    # the recovery path (crash-after-accept idempotency) resubmits normally.
    tc.get_order_by_client_id.return_value = None
    return AlpacaClient(trading_client=tc), tc


class FakeStrategy:
    def __init__(self, decisions):
        self._decisions = decisions

    def decide(self, *, asof_date, prices):
        return self._decisions


def test_decide_writes_intended_orders_and_submits(db_with_prices):
    store, db = db_with_prices
    alpaca, tc = _alpaca_mock(equity=100_000.0)
    decisions = [
        StrategyDecision(asof_date=date(2026, 5, 1), ticker="AAPL", target_weight=0.05),
        StrategyDecision(asof_date=date(2026, 5, 1), ticker="MSFT", target_weight=0.05),
    ]
    strategy = FakeStrategy(decisions)

    result = decide_once(
        asof=date(2026, 5, 1),
        store=store, alpaca=alpaca,
        universe=["AAPL", "MSFT"],
        strategy=strategy,
        sector_for=lambda t: "Tech",
        rails=RiskRails(stop_loss_pct=0.0),
    )

    assert result.dry_run is False
    assert result.submitted == 2
    assert result.failed == 0
    # 2 intended_orders rows
    rows = store.conn.execute(
        "SELECT ticker, side, target_shares, last_price, status, alpaca_order_id "
        "FROM intended_orders ORDER BY ticker"
    ).fetchall()
    assert len(rows) == 2
    aapl_row = [r for r in rows if r[0] == "AAPL"][0]
    assert aapl_row[1] == "BUY"
    assert aapl_row[2] == 25   # 100000 * 0.05 / 200
    assert aapl_row[3] == 200.0
    assert aapl_row[4] == "submitted"
    assert aapl_row[5] is not None   # alpaca_order_id populated


def test_decide_dry_run_writes_nothing(db_with_prices, capsys):
    store, db = db_with_prices
    alpaca, tc = _alpaca_mock()
    decisions = [
        StrategyDecision(asof_date=date(2026, 5, 1), ticker="AAPL", target_weight=0.05),
    ]
    strategy = FakeStrategy(decisions)

    result = decide_once(
        asof=date(2026, 5, 1),
        store=store, alpaca=alpaca,
        universe=["AAPL", "MSFT"],
        strategy=strategy,
        sector_for=lambda t: "Tech",
        rails=RiskRails(stop_loss_pct=0.0),
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.submitted == 0
    rows = store.conn.execute("SELECT COUNT(*) FROM intended_orders").fetchone()
    assert rows[0] == 0
    # Alpaca submit_order never called
    tc.submit_order.assert_not_called()
    captured = capsys.readouterr()
    assert "[dry-run]" in captured.out


def test_decide_canary_filters_to_one_ticker(db_with_prices):
    store, db = db_with_prices
    alpaca, tc = _alpaca_mock()
    decisions = [
        StrategyDecision(asof_date=date(2026, 5, 1), ticker="AAPL", target_weight=0.05),
        StrategyDecision(asof_date=date(2026, 5, 1), ticker="MSFT", target_weight=0.05),
    ]
    strategy = FakeStrategy(decisions)

    result = decide_once(
        asof=date(2026, 5, 1),
        store=store, alpaca=alpaca,
        universe=["AAPL", "MSFT"],
        strategy=strategy,
        sector_for=lambda t: "Tech",
        rails=RiskRails(stop_loss_pct=0.0),
        canary="AAPL",
    )

    assert result.submitted == 1
    rows = store.conn.execute("SELECT ticker FROM intended_orders").fetchall()
    assert [r[0] for r in rows] == ["AAPL"]


def test_decide_canary_force_includes_ticker_not_in_strategy_output(db_with_prices):
    """Regression: --canary should force-include the named ticker even when
    the strategy doesn't surface it. Previously --canary filtered down to
    nothing if the ticker wasn't in top-K, producing 0 orders.
    """
    store, db = db_with_prices
    alpaca, tc = _alpaca_mock(equity=100_000.0)
    # Strategy surfaces only MSFT; canary is AAPL (NOT in output).
    decisions = [
        StrategyDecision(asof_date=date(2026, 5, 1), ticker="MSFT", target_weight=0.05),
    ]
    strategy = FakeStrategy(decisions)

    result = decide_once(
        asof=date(2026, 5, 1),
        store=store, alpaca=alpaca,
        universe=["AAPL", "MSFT"],
        strategy=strategy,
        sector_for=lambda t: "Tech",
        rails=RiskRails(stop_loss_pct=0.0, max_position_pct=0.05),
        canary="AAPL",
    )

    # Canary AAPL synthesized at max_position_pct → 1 order submitted for AAPL.
    assert result.submitted == 1
    rows = store.conn.execute(
        "SELECT ticker, target_weight FROM intended_orders ORDER BY ticker"
    ).fetchall()
    assert [r[0] for r in rows] == ["AAPL"]
    # Synthesized weight should equal max_position_pct
    assert rows[0][1] == 0.05


def test_decide_raises_paranoia_rail_on_empty_decisions_with_held_positions(db_with_prices):
    store, db = db_with_prices
    alpaca, tc = _alpaca_mock(positions=[("AAPL", 10, 200.0)])
    strategy = FakeStrategy([])  # empty decisions

    with pytest.raises(EmptyDecisionsWithHeldPositionsError):
        decide_once(
            asof=date(2026, 5, 1),
            store=store, alpaca=alpaca,
            universe=["AAPL", "MSFT"],
            strategy=strategy,
            sector_for=lambda t: "Tech",
            rails=RiskRails(stop_loss_pct=0.0),
        )


def test_decide_retry_after_failed_submit_replaces_failed_row(db_with_prices):
    """Regression: re-running decide on the same date for the same ticker after
    a previous failed attempt must NOT raise the unique-key constraint error.
    The prior failed row should be replaced; a fresh submit attempt should
    proceed.
    """
    store, db = db_with_prices
    alpaca, tc = _alpaca_mock()
    decisions = [
        StrategyDecision(asof_date=date(2026, 5, 1), ticker="AAPL", target_weight=0.05),
    ]
    strategy = FakeStrategy(decisions)

    # First run: Alpaca rejects.
    from alpaca.common.exceptions import APIError
    tc.submit_order.side_effect = APIError("first attempt rejected")
    decide_once(
        asof=date(2026, 5, 1), store=store, alpaca=alpaca,
        universe=["AAPL"], strategy=strategy, sector_for=lambda t: "Tech",
        rails=RiskRails(stop_loss_pct=0.0),
    )
    failed_row = store.conn.execute(
        "SELECT status FROM intended_orders WHERE ticker = 'AAPL'",
    ).fetchone()
    assert failed_row[0] == "submission_failed"

    # Second run: Alpaca succeeds. Must NOT raise constraint error.
    returned = MagicMock()
    returned.id = "ord-retry-1"
    tc.submit_order.side_effect = lambda req: returned
    result = decide_once(
        asof=date(2026, 5, 1), store=store, alpaca=alpaca,
        universe=["AAPL"], strategy=strategy, sector_for=lambda t: "Tech",
        rails=RiskRails(stop_loss_pct=0.0),
    )
    assert result.submitted == 1
    assert result.failed == 0
    rows = store.conn.execute(
        "SELECT status, alpaca_order_id FROM intended_orders WHERE ticker = 'AAPL'",
    ).fetchall()
    # Exactly one row remains (the prior failed row was deleted, retry inserted).
    assert len(rows) == 1
    assert rows[0][0] == "submitted"
    assert rows[0][1] == "ord-retry-1"


def test_decide_skips_re_submit_when_already_submitted_with_alpaca_id(db_with_prices):
    """If a prior run submitted successfully (alpaca_order_id populated),
    a re-run for the same date+ticker must SKIP the submit (no double-submit)
    and not crash on the unique key.
    """
    store, db = db_with_prices
    alpaca, tc = _alpaca_mock()
    decisions = [
        StrategyDecision(asof_date=date(2026, 5, 1), ticker="AAPL", target_weight=0.05),
    ]
    strategy = FakeStrategy(decisions)

    # First run: succeeds.
    returned = MagicMock()
    returned.id = "ord-first-1"
    tc.submit_order.side_effect = lambda req: returned
    decide_once(
        asof=date(2026, 5, 1), store=store, alpaca=alpaca,
        universe=["AAPL"], strategy=strategy, sector_for=lambda t: "Tech",
        rails=RiskRails(stop_loss_pct=0.0),
    )
    first_call_count = tc.submit_order.call_count

    # Second run: must NOT call submit_order again.
    decide_once(
        asof=date(2026, 5, 1), store=store, alpaca=alpaca,
        universe=["AAPL"], strategy=strategy, sector_for=lambda t: "Tech",
        rails=RiskRails(stop_loss_pct=0.0),
    )
    assert tc.submit_order.call_count == first_call_count, (
        "second run double-submitted; idempotency broken"
    )
    rows = store.conn.execute(
        "SELECT alpaca_order_id FROM intended_orders WHERE ticker = 'AAPL'",
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "ord-first-1"


def test_decide_records_submission_failed_on_alpaca_error(db_with_prices):
    """Per-order isolation: one failed submit doesn't kill the batch."""
    store, db = db_with_prices
    alpaca, tc = _alpaca_mock()
    # Make the first submit fail; subsequent calls succeed.
    from alpaca.common.exceptions import APIError
    call_count = [0]

    def maybe_fail(req):
        call_count[0] += 1
        if call_count[0] == 1:
            raise APIError("test failure")
        ret = MagicMock()
        ret.id = f"ord-{call_count[0]}"
        return ret

    tc.submit_order.side_effect = maybe_fail

    decisions = [
        StrategyDecision(asof_date=date(2026, 5, 1), ticker="AAPL", target_weight=0.05),
        StrategyDecision(asof_date=date(2026, 5, 1), ticker="MSFT", target_weight=0.05),
    ]
    strategy = FakeStrategy(decisions)

    result = decide_once(
        asof=date(2026, 5, 1),
        store=store, alpaca=alpaca,
        universe=["AAPL", "MSFT"],
        strategy=strategy,
        sector_for=lambda t: "Tech",
        rails=RiskRails(stop_loss_pct=0.0),
    )

    assert result.submitted == 1
    assert result.failed == 1
    statuses = store.conn.execute(
        "SELECT ticker, status, error FROM intended_orders ORDER BY ticker"
    ).fetchall()
    assert len(statuses) == 2
    by_ticker = {row[0]: row for row in statuses}
    assert by_ticker["AAPL"][1] == "submission_failed"
    assert "test failure" in by_ticker["AAPL"][2]
    assert by_ticker["MSFT"][1] == "submitted"


# Pure-function helpers
def test_last_prices_per_ticker_picks_latest_date():
    import pandas as pd
    df = pd.DataFrame([
        {"ticker": "AAPL", "date": date(2026, 4, 28), "adj_close": 195.0},
        {"ticker": "AAPL", "date": date(2026, 5, 1), "adj_close": 200.0},
        {"ticker": "MSFT", "date": date(2026, 5, 1), "adj_close": 400.0},
    ])
    out = _last_prices_per_ticker(df, ["AAPL", "MSFT", "GOOGL"])
    assert out["AAPL"] == (200.0, date(2026, 5, 1))
    assert out["MSFT"] == (400.0, date(2026, 5, 1))
    assert "GOOGL" not in out


def test_to_dollars_uses_last_price():
    out = _to_dollars(
        positions={"AAPL": {"shares": 25, "cost_basis": 180.0}},
        last_prices={"AAPL": (200.0, date(2026, 5, 1))},
    )
    assert out == {"AAPL": 5_000.0}   # 25 * 200 (current) not 25 * 180 (cost)


def test_sector_exposure_aggregates_by_sector():
    out = _sector_exposure(
        positions={
            "AAPL": {"shares": 25, "cost_basis": 200.0},
            "MSFT": {"shares": 10, "cost_basis": 400.0},
            "JPM": {"shares": 20, "cost_basis": 150.0},
        },
        last_prices={
            "AAPL": (200.0, date(2026, 5, 1)),
            "MSFT": (400.0, date(2026, 5, 1)),
            "JPM": (150.0, date(2026, 5, 1)),
        },
        equity=100_000.0,
        sector_for=lambda t: "Tech" if t in ("AAPL", "MSFT") else "Financials",
    )
    # AAPL: 25*200=5000, MSFT: 10*400=4000 → Tech = 9000/100000 = 0.09
    # JPM:  20*150=3000 → Financials = 0.03
    assert out["Tech"] == 0.09
    assert out["Financials"] == 0.03
