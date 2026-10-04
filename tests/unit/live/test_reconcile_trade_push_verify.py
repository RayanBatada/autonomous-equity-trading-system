"""_detect_trade_push_drift (2026-09-04): verifies a persisted trade-push
sentinel against the paper_fills that actually booked for the same asof --
see sma.live.reconcile's module docstring and sma.live.trade_push's
record_trade_push. Unit-level: a bare Store, no Alpaca client, exercising the
detector function directly (reconcile()'s own wiring of this into the
order_drift_open block is covered by
tests/integration/live/test_reconcile.py).
"""

import logging
import uuid
from datetime import date, datetime, time

from sma.ingest.store import Store
from sma.live.reconcile import _detect_trade_push_drift
from sma.live.trade_push import record_trade_push


def _make_store(tmp_path):
    return Store(path=str(tmp_path / "test.duckdb")).connect()


def _seed_intended_and_fill(
    store, *, asof, ticker, side, alpaca_order_id,
    filled_shares=None, fill_price=None, target_shares=10,
    last_price=None, status="filled", source="decide",
):
    """Write an intended_orders row (as decide would after submit), and --
    only when `filled_shares` is truthy -- a matching paper_fills row sharing
    `alpaca_order_id` (as reconcile._record_fills would after the order
    filled). A falsy filled_shares (None or 0) leaves NO paper_fills row,
    simulating a missed fill: the LEFT JOIN in _detect_trade_push_drift's
    query then reads back realized_shares=0 via COALESCE."""
    rid = store.allocate_run_id()
    iid = str(uuid.uuid4())
    store.conn.execute(
        """
        INSERT INTO intended_orders
        (intended_order_id, asof_date, ticker, side, target_shares,
         target_weight, last_price, source, alpaca_order_id, status, run_id)
        VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)
        """,
        [iid, asof, ticker, side, target_shares, last_price, source,
         alpaca_order_id, status, rid],
    )
    if filled_shares:
        store.conn.execute(
            """
            INSERT INTO paper_fills
            (alpaca_order_id, intended_order_id, asof_date, ticker, side,
             filled_shares, fill_price, commission, fees, status,
             submitted_at, filled_at, run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 'filled', ?, ?, ?)
            """,
            [alpaca_order_id, iid, asof, ticker, side, filled_shares,
             fill_price, datetime.combine(asof, time(18, 35)),
             datetime.combine(asof, time(9, 30)), rid],
        )
    return iid


def _push_order(*, ticker, side, shares, decide_price, equity, full_exit=False):
    """Build a pushed-order dict with the SAME math
    sma.live.trade_push._push_order_payload uses, for fixtures where the
    push was accurate. Tests exercising a wrong/buggy push (e.g. the LLY
    historical regression) construct the dict inline instead, with a
    deliberately mismatched order_pct_of_equity."""
    notional = shares * decide_price if decide_price else None
    pct = notional / equity if (notional is not None and equity > 0) else None
    return {
        "ticker": ticker, "side": side, "shares": shares,
        "decide_price": decide_price, "order_notional": notional,
        "order_pct_of_equity": pct, "full_exit": full_exit,
    }


def test_missing_push_file_skips_silently(tmp_path):
    store = _make_store(tmp_path)
    asof = date(2026, 9, 2)
    # No record_trade_push call at all for this asof.
    assert _detect_trade_push_drift(asof=asof, store=store) == []


def test_exact_match_is_silent_pass(tmp_path):
    store = _make_store(tmp_path)
    asof = date(2026, 9, 3)
    equity = 100_000.0

    _seed_intended_and_fill(
        store, asof=asof, ticker="AAPL", side="BUY", alpaca_order_id="ord-1",
        filled_shares=10, fill_price=200.0, last_price=200.0,
    )
    record_trade_push(
        asof=asof, title="t", message="m",
        orders=[_push_order(
            ticker="AAPL", side="BUY", shares=10, decide_price=200.0, equity=equity,
        )],
        equity=equity, delivered=True,
    )

    assert _detect_trade_push_drift(asof=asof, store=store) == []


def test_lly_historical_regression_fires(tmp_path):
    """The real 2026-09-02 incident: a SELL of 1 LLY share (real fill notional
    ~1.0% of equity) was pushed claiming 5.7% of equity -- LLY's post-trade
    POSITION weight, not the order's own size (see sma.live.trade_push's
    module docstring). Shares match (1 pushed, 1 filled), so this exercises
    the PCT check specifically: the ~82% relative diff must fire, nowhere
    near the 20% default tolerance."""
    store = _make_store(tmp_path)
    asof = date(2026, 9, 2)
    equity = 116_174.92

    _seed_intended_and_fill(
        store, asof=asof, ticker="LLY", side="SELL", alpaca_order_id="ord-lly",
        filled_shares=1, fill_price=1160.08, last_price=1160.08,
    )
    record_trade_push(
        asof=asof, title="t", message="m",
        orders=[{
            "ticker": "LLY", "side": "SELL", "shares": 1,
            "decide_price": 1160.08, "order_notional": 1160.08,
            # The actual bug: this claimed the POST-TRADE position weight
            # (5.7%) instead of the order's own ~1.0% notional.
            "order_pct_of_equity": 0.057142857142857155,
            "full_exit": False,
        }],
        equity=equity, delivered=True,
    )

    alerts = _detect_trade_push_drift(asof=asof, store=store)
    assert len(alerts) == 1
    assert alerts[0].kind == "trade_push_mismatch"
    assert "LLY" in alerts[0].detail
    assert "5.7%" in alerts[0].detail
    assert "1.0%" in alerts[0].detail


def test_shares_mismatch_fires(tmp_path):
    store = _make_store(tmp_path)
    asof = date(2026, 9, 3)
    equity = 100_000.0

    _seed_intended_and_fill(
        store, asof=asof, ticker="NVDA", side="SELL", alpaca_order_id="ord-2",
        filled_shares=5, fill_price=200.0, last_price=200.0,
    )
    record_trade_push(
        asof=asof, title="t", message="m",
        orders=[_push_order(
            ticker="NVDA", side="SELL", shares=10, decide_price=200.0, equity=equity,
        )],
        equity=equity, delivered=True,
    )

    alerts = _detect_trade_push_drift(asof=asof, store=store)
    assert len(alerts) == 1
    assert alerts[0].kind == "trade_push_mismatch"
    assert "NVDA" in alerts[0].detail
    assert "10" in alerts[0].detail
    assert "5" in alerts[0].detail


def test_missing_fill_reads_back_as_shares_mismatch(tmp_path):
    """The order was pushed but never filled at all -- no paper_fills row
    exists for it. The LEFT JOIN + COALESCE reads that back as 0 realized
    shares, which is a shares mismatch against any nonzero pushed count."""
    store = _make_store(tmp_path)
    asof = date(2026, 9, 3)
    equity = 100_000.0

    _seed_intended_and_fill(
        store, asof=asof, ticker="MSFT", side="BUY", alpaca_order_id="ord-3",
        filled_shares=None, last_price=300.0,
    )
    record_trade_push(
        asof=asof, title="t", message="m",
        orders=[_push_order(
            ticker="MSFT", side="BUY", shares=3, decide_price=300.0, equity=equity,
        )],
        equity=equity, delivered=True,
    )

    alerts = _detect_trade_push_drift(asof=asof, store=store)
    assert len(alerts) == 1
    assert alerts[0].kind == "trade_push_mismatch"
    assert "MSFT" in alerts[0].detail


def test_tolerance_boundary_just_under_passes_just_over_fires(tmp_path):
    """Deliberately NOT bit-exact-at-0.20 (avoids float boundary flakiness):
    pushed_pct=5.0%, realized 5.95% is a 19% relative diff (passes, just
    under the 20% default), realized 6.05% is a 21% relative diff (fires,
    just over). Shares match in both cases -- this isolates the pct check."""
    store = _make_store(tmp_path)
    equity = 100_000.0

    asof_pass = date(2026, 9, 3)
    _seed_intended_and_fill(
        store, asof=asof_pass, ticker="CAT", side="BUY", alpaca_order_id="ord-under",
        filled_shares=10, fill_price=595.0, last_price=500.0,
    )
    record_trade_push(
        asof=asof_pass, title="t", message="m",
        orders=[_push_order(
            ticker="CAT", side="BUY", shares=10, decide_price=500.0, equity=equity,
        )],
        equity=equity, delivered=True,
    )
    assert _detect_trade_push_drift(asof=asof_pass, store=store) == []

    asof_fail = date(2026, 9, 4)
    _seed_intended_and_fill(
        store, asof=asof_fail, ticker="CAT", side="BUY", alpaca_order_id="ord-over",
        filled_shares=10, fill_price=605.0, last_price=500.0,
    )
    record_trade_push(
        asof=asof_fail, title="t", message="m",
        orders=[_push_order(
            ticker="CAT", side="BUY", shares=10, decide_price=500.0, equity=equity,
        )],
        equity=equity, delivered=True,
    )
    alerts = _detect_trade_push_drift(asof=asof_fail, store=store)
    assert len(alerts) == 1
    assert alerts[0].kind == "trade_push_mismatch"
    assert "CAT" in alerts[0].detail


def test_no_prior_session_push_still_verifies_this_asofs_push(tmp_path, caplog):
    """Regression lock (2026-09-11 investigation): a bug report claimed
    _detect_trade_push_drift(asof=asof) should instead look up the push for
    the PREVIOUS trading session before `asof`, on the theory that `asof`
    reads as "today" rather than "the decide day being reconciled". That
    theory is wrong -- record_trade_push(asof=asof, ...) and this asof's
    intended_orders/paper_fills rows are written by the SAME decide_once
    call, so they always describe the identical batch. Proof: seed ONLY
    push(asof) (deliberately no push for the prior session) and confirm the
    detector actually verifies it -- a clean pass with the "matched" INFO
    log firing -- rather than "wrongly verifying against nothing" (silently
    finding no push at all, which is what asof-1 lookup would do here, since
    no push for the prior session exists)."""
    caplog.set_level(logging.INFO, logger="sma.live.reconcile")
    store = _make_store(tmp_path)
    asof = date(2026, 9, 10)
    equity = 100_000.0

    _seed_intended_and_fill(
        store, asof=asof, ticker="AFRM", side="BUY", alpaca_order_id="ord-afrm",
        filled_shares=18, fill_price=68.94, last_price=67.99,
    )
    # No push recorded for the prior session (2026-09-09) at all -- proves
    # the detector isn't (even accidentally) falling back to it.
    record_trade_push(
        asof=asof, title="t", message="m",
        orders=[_push_order(
            ticker="AFRM", side="BUY", shares=18, decide_price=67.99, equity=equity,
        )],
        equity=equity, delivered=True,
    )

    alerts = _detect_trade_push_drift(asof=asof, store=store)
    assert alerts == []
    assert any(
        "matched booked fills for 2026-09-10" in r.message for r in caplog.records
    )


def test_catch_up_two_sessions_each_verify_their_own_push(tmp_path):
    """Two decide batches reconciled in the same catch-up run (mirrors
    sma.live.__main__'s per-asof loop over `reconciled_asofs`) must each be
    checked against THEIR OWN push -- session 1's mismatch must not leak
    into session 2's clean result or vice versa."""
    store = _make_store(tmp_path)
    equity = 100_000.0

    asof1 = date(2026, 9, 8)
    _seed_intended_and_fill(
        store, asof=asof1, ticker="INTC", side="SELL", alpaca_order_id="ord-intc",
        filled_shares=20, fill_price=102.0, last_price=100.32,
    )
    record_trade_push(
        asof=asof1, title="t", message="m",
        orders=[{
            "ticker": "INTC", "side": "SELL", "shares": 20,
            "decide_price": 100.32, "order_notional": 2006.4,
            # Deliberately wrong pct (mirrors the LLY bug shape) so this
            # batch's alert can be distinguished from session 2's.
            "order_pct_of_equity": 0.50,
            "full_exit": False,
        }],
        equity=equity, delivered=True,
    )

    asof2 = date(2026, 9, 9)
    _seed_intended_and_fill(
        store, asof=asof2, ticker="PWR", side="BUY", alpaca_order_id="ord-pwr",
        filled_shares=2, fill_price=627.24, last_price=618.73,
    )
    record_trade_push(
        asof=asof2, title="t", message="m",
        orders=[_push_order(
            ticker="PWR", side="BUY", shares=2, decide_price=618.73, equity=equity,
        )],
        equity=equity, delivered=True,
    )

    alerts1 = _detect_trade_push_drift(asof=asof1, store=store)
    alerts2 = _detect_trade_push_drift(asof=asof2, store=store)

    assert len(alerts1) == 1
    assert "INTC" in alerts1[0].detail
    assert alerts2 == []
