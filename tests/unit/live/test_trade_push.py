"""build_trade_push: renders tonight's SUBMITTED orders as percent-of-equity
lines, never share counts, so the message scales to any account size Rayan
mirrors it into by hand. See __main__.py's decide flow (wired at the end,
after orders are submitted) and docs/REAL_MONEY_CHECKLIST.md "Mirroring
signals manually".

Two different percentages, by design (2026-09-03 bug fix): a full-exit SELL
reports the PRIOR POSITION's weight ("was X% of book") -- useful context for
a full liquidation, since the whole position is being sold regardless of its
size. Every other line (a partial SELL trim or a BUY, fresh or a top-up)
reports THIS ORDER's own notional as % of equity -- a mirrorer trades the
order, not the position it leaves behind. Conflating the two (using the
decision's post-trade TARGET weight for a partial order) was exactly the bug:
a 2026-09-02 SELL of 1-of-6 LLY shares (~1.0% of equity) was pushed as "5.7%
of equity", LLY's post-trade POSITION weight -- a manual mirrorer would have
sold 5.7x too much.
"""

from datetime import date

from sma.backtest.strategies.base import StrategyDecision
from sma.live.orders import Order
from sma.live.trade_push import (
    MAX_MESSAGE_CHARS,
    build_push_orders_payload,
    build_trade_push,
)


def test_full_exit_sell_shows_book_weight_and_buy_shows_order_size():
    """The exact two example lines from the spec: a full-exit SELL shows
    'all' + the PRIOR weight it held, a BUY shows THIS ORDER's size. ENPH is
    a fresh full-size entry (no prior shares), so its order notional and its
    decision target_weight are numerically the same 6.3% here -- see
    test_partial_buy_topup_shows_order_pct_not_position_pct for a case where
    they diverge and only the order size is correct."""
    orders = [
        Order(ticker="MRNA", side="SELL", shares=100, type="DAY",
              last_price=142.0, full_exit=True),
        Order(ticker="ENPH", side="BUY", shares=50, type="DAY", last_price=126.0),
    ]
    decisions = [
        StrategyDecision(asof_date=date(2026, 8, 28), ticker="ENPH", target_weight=0.063),
    ]

    title, message = build_trade_push(
        asof=date(2026, 8, 28),
        submitted_orders=orders,
        decisions=decisions,
        equity=100_000.0,
        position_count=9,
    )

    assert title == "SMA trades - Fri 8/28"
    lines = message.split("\n")
    assert lines[0] == "2026-08-28"
    assert lines[1] == "SELL MRNA — all (was 14.2% of book)"
    assert lines[2] == "BUY ENPH — 6.3% of equity"
    assert lines[-1] == "Equity $100,000 | 9 positions"


def test_partial_sell_trim_shows_order_pct_not_position_pct():
    """NVDA's decision target_weight (9.0%, the POST-TRADE position size) is
    deliberately far from the order's own notional (1.0%) -- if the fix
    regresses to reading decisions instead of the order, this asserts the
    wrong (9.0%) number and fails."""
    orders = [
        Order(ticker="NVDA", side="SELL", shares=5, type="DAY",
              last_price=200.0, full_exit=False),
    ]
    decisions = [
        StrategyDecision(asof_date=date(2026, 8, 28), ticker="NVDA", target_weight=0.09),
    ]

    _, message = build_trade_push(
        asof=date(2026, 8, 28),
        submitted_orders=orders,
        decisions=decisions,
        equity=100_000.0,
        position_count=5,
    )

    # 5 shares * $200 / $100,000 equity = 1.0% -- the ORDER's notional.
    assert "SELL NVDA — 1.0% of equity" in message.split("\n")
    assert "9.0%" not in message


def test_partial_sell_regression_lly_1_of_6_shares_shows_order_pct():
    """Regression for the exact 2026-09-02 bug: a 1-share trim of a 6-share
    LLY position (order notional ~1.0% of equity) was pushed as "5.7% of
    equity" -- LLY's POST-TRADE position weight, not the order's size. A
    manual mirrorer reading "5.7%" would have sold 5.7x too much. Numbers
    are the real fill: equity $116,174.92, LLY last_price $1160.08, 1 share
    sold, decisions target_weight 0.05714285714... (the new, smaller,
    5.7%-of-book position after the trim)."""
    orders = [
        Order(ticker="LLY", side="SELL", shares=1, type="DAY",
              last_price=1160.08, full_exit=False),
    ]
    decisions = [
        StrategyDecision(
            asof_date=date(2026, 9, 2), ticker="LLY",
            target_weight=0.057142857142857155,
        ),
    ]

    _, message = build_trade_push(
        asof=date(2026, 9, 2),
        submitted_orders=orders,
        decisions=decisions,
        equity=116_174.92,
        position_count=11,
    )

    assert "SELL LLY — 1.0% of equity" in message.split("\n")
    assert "5.7%" not in message


def test_partial_buy_topup_shows_order_pct_not_position_pct():
    """A BUY that TOPS UP an existing position (not a fresh entry) must also
    report the order's own size, not the resulting position's weight -- the
    identical bug class as the LLY sell, just on the buy side. Numbers are a
    real fill: CAT topped up by 2 shares at $779.16, equity $116,174.92,
    decisions target_weight 0.069388... (the post-trade ~6.9%-of-book
    position, after adding these 2 shares to ~8 already held)."""
    orders = [
        Order(ticker="CAT", side="BUY", shares=2, type="DAY", last_price=779.16),
    ]
    decisions = [
        StrategyDecision(
            asof_date=date(2026, 9, 1), ticker="CAT",
            target_weight=0.06938775510204084,
        ),
    ]

    _, message = build_trade_push(
        asof=date(2026, 9, 1),
        submitted_orders=orders,
        decisions=decisions,
        equity=116_174.92,
        position_count=11,
    )

    assert "BUY CAT — 1.3% of equity" in message.split("\n")
    assert "6.9%" not in message


def test_full_exit_without_a_price_omits_the_was_parenthetical():
    """A force-sell of a delisted/priceless ticker still has last_price=None
    (orders.py's force-sell branch never checks freshness). The line must
    degrade gracefully, not crash on a None * float."""
    orders = [
        Order(ticker="ZOMB", side="SELL", shares=10, type="DAY",
              last_price=None, full_exit=True),
    ]
    _, message = build_trade_push(
        asof=date(2026, 8, 28),
        submitted_orders=orders,
        decisions=[],
        equity=100_000.0,
        position_count=4,
    )
    assert "SELL ZOMB — all" in message.split("\n")
    assert "was" not in message


def test_zero_orders_message():
    _, message = build_trade_push(
        asof=date(2026, 8, 31),
        submitted_orders=[],
        decisions=[],
        equity=50_000.0,
        position_count=7,
    )
    assert message == (
        "2026-08-31\n"
        "No trades tonight — book unchanged (7 holds)\n"
        "Equity $50,000 | 7 positions"
    )


def test_singular_position_and_hold_nouns():
    _, message = build_trade_push(
        asof=date(2026, 8, 31),
        submitted_orders=[],
        decisions=[],
        equity=50_000.0,
        position_count=1,
    )
    assert "1 hold)" in message
    assert "1 position" in message
    assert "1 positions" not in message
    assert "1 holds" not in message


def test_title_has_no_leading_zeros_in_month_or_day():
    title, _ = build_trade_push(
        asof=date(2026, 8, 5),
        submitted_orders=[],
        decisions=[],
        equity=1_000.0,
        position_count=0,
    )
    assert title == "SMA trades - Wed 8/5"


def test_message_never_exceeds_max_chars_and_preserves_footer():
    orders = [
        Order(ticker=f"TICK{i:03d}", side="BUY", shares=1, type="DAY", last_price=100.0)
        for i in range(60)
    ]
    # decisions target_weight is deliberately different from the order's own
    # 0.1% notional (1 share * $100 / $100,000) -- these lines must render
    # from the ORDER, not the decision.
    decisions = [
        StrategyDecision(asof_date=date(2026, 8, 28), ticker=o.ticker, target_weight=0.05)
        for o in orders
    ]
    raw_len = len("\n".join(
        [date(2026, 8, 28).isoformat()]
        + [f"BUY {o.ticker} — 0.1% of equity" for o in orders]
        + ["Equity $100,000 | 60 positions"]
    ))
    assert raw_len > MAX_MESSAGE_CHARS, "test setup must actually exceed the bound"

    _, message = build_trade_push(
        asof=date(2026, 8, 28),
        submitted_orders=orders,
        decisions=decisions,
        equity=100_000.0,
        position_count=60,
    )

    assert len(message) <= MAX_MESSAGE_CHARS
    lines = message.split("\n")
    assert lines[0] == "2026-08-28"
    assert lines[-1] == "Equity $100,000 | 60 positions"
    assert any("more" in line for line in lines)


def test_message_under_bound_is_untruncated():
    orders = [
        Order(ticker="ENPH", side="BUY", shares=50, type="DAY", last_price=126.0),
    ]
    decisions = [
        StrategyDecision(asof_date=date(2026, 8, 28), ticker="ENPH", target_weight=0.063),
    ]
    _, message = build_trade_push(
        asof=date(2026, 8, 28),
        submitted_orders=orders,
        decisions=decisions,
        equity=100_000.0,
        position_count=9,
    )
    assert "more" not in message


def test_deterministic_order_is_preserved_not_resorted():
    """build_trade_push must not re-sort submitted_orders -- decide_once
    already emits sells-then-buys in a fixed order; callers rely on that."""
    orders = [
        Order(ticker="ZZZZ", side="BUY", shares=1, type="DAY", last_price=10.0),
        Order(ticker="AAAA", side="BUY", shares=1, type="DAY", last_price=10.0),
    ]
    decisions = [
        StrategyDecision(asof_date=date(2026, 8, 28), ticker="ZZZZ", target_weight=0.01),
        StrategyDecision(asof_date=date(2026, 8, 28), ticker="AAAA", target_weight=0.02),
    ]
    _, message = build_trade_push(
        asof=date(2026, 8, 28),
        submitted_orders=orders,
        decisions=decisions,
        equity=10_000.0,
        position_count=2,
    )
    lines = message.split("\n")
    assert lines[1].startswith("BUY ZZZZ")
    assert lines[2].startswith("BUY AAAA")


def test_weight_falls_back_to_order_notional_when_ticker_missing_from_decisions():
    """An ADV-capped force-sell loses full_exit (orders.py clears it when
    trimmed) but the ticker was never in decisions (it's a force-sell) --
    the line must still render a sane percentage from the order's own
    notional rather than "unavailable"."""
    orders = [
        Order(ticker="NANC", side="SELL", shares=20, type="DAY",
              last_price=50.0, full_exit=False, capped_by="adv_participation"),
    ]
    _, message = build_trade_push(
        asof=date(2026, 8, 28),
        submitted_orders=orders,
        decisions=[],
        equity=100_000.0,
        position_count=6,
    )
    # 20 * 50 / 100_000 = 1.0%
    assert "SELL NANC — 1.0% of equity" in message.split("\n")


# ---------------------------------------------------------------------------
# build_push_orders_payload (2026-09-04): the structured record persisted by
# record_trade_push and later checked by
# sma.live.reconcile._detect_trade_push_drift. Must share the exact math
# _order_line renders so the persisted numbers can never diverge from the
# pushed text -- these fixtures reuse the same real numbers as the
# build_trade_push regression tests above.
# ---------------------------------------------------------------------------


def test_payload_matches_rendered_pct_lly_regression():
    """Same LLY fixture as
    test_partial_sell_regression_lly_1_of_6_shares_shows_order_pct: the
    payload's order_pct_of_equity must be the CORRECT 1.0%, never the buggy
    5.7% position weight."""
    orders = [
        Order(ticker="LLY", side="SELL", shares=1, type="DAY",
              last_price=1160.08, full_exit=False),
    ]
    payload = build_push_orders_payload(submitted_orders=orders, equity=116_174.92)

    assert payload == [{
        "ticker": "LLY",
        "side": "SELL",
        "shares": 1,
        "decide_price": 1160.08,
        "order_notional": 1160.08,
        "order_pct_of_equity": 1160.08 / 116_174.92,
        "full_exit": False,
    }]
    assert round(payload[0]["order_pct_of_equity"], 3) == 0.010


def test_payload_full_exit_uses_prior_position_value():
    """A full-exit SELL's payload pct is the PRIOR position's weight (shares
    sold * decide price / equity) -- the same number _order_line renders as
    "was X% of book". MRNA: 100 shares @ $142, equity $100k -> 14.2%."""
    orders = [
        Order(ticker="MRNA", side="SELL", shares=100, type="DAY",
              last_price=142.0, full_exit=True),
    ]
    payload = build_push_orders_payload(submitted_orders=orders, equity=100_000.0)

    assert payload[0]["order_notional"] == 14_200.0
    assert round(payload[0]["order_pct_of_equity"], 3) == 0.142
    assert payload[0]["full_exit"] is True


def test_payload_none_last_price_gives_none_fields():
    """A force-sell of a delisted/priceless ticker has last_price=None --
    decide_price, order_notional, and order_pct_of_equity must all degrade to
    None rather than crash on None * float (same case build_trade_push's
    'omits the was parenthetical' test covers for the rendered text)."""
    orders = [
        Order(ticker="ZOMB", side="SELL", shares=10, type="DAY",
              last_price=None, full_exit=True),
    ]
    payload = build_push_orders_payload(submitted_orders=orders, equity=100_000.0)

    assert payload[0]["decide_price"] is None
    assert payload[0]["order_notional"] is None
    assert payload[0]["order_pct_of_equity"] is None
    assert payload[0]["shares"] == 10


def test_payload_zero_equity_gives_none_pct_but_keeps_notional():
    """Zero/negative equity can't produce a percentage, but the order's own
    notional (shares * decide price) doesn't depend on equity at all and
    should still resolve."""
    orders = [
        Order(ticker="AAPL", side="BUY", shares=5, type="DAY", last_price=200.0),
    ]
    payload = build_push_orders_payload(submitted_orders=orders, equity=0.0)

    assert payload[0]["order_notional"] == 1_000.0
    assert payload[0]["order_pct_of_equity"] is None


def test_payload_empty_orders_is_empty_list():
    assert build_push_orders_payload(submitted_orders=[], equity=100_000.0) == []
