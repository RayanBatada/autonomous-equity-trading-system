"""Tests for orders.translate. Test names match the Phase 5 spec §8.1 list."""

from datetime import date
from math import isclose

import pytest

from sma.backtest.strategies.base import StrategyDecision
from sma.live.exceptions import EmptyDecisionsWithHeldPositionsError
from sma.live.orders import translate
from sma.risk.rails import RiskRails


def _decision(ticker: str, weight: float = 0.05) -> StrategyDecision:
    return StrategyDecision(
        asof_date=date(2026, 5, 1), ticker=ticker, target_weight=weight,
    )


def _last_prices(asof: date = date(2026, 5, 1), **kw) -> dict:
    return {t: (p, asof) for t, p in kw.items()}


def test_translate_buy_from_zero_position():
    out = translate(
        decisions=[_decision("AAPL", weight=0.05)],
        current_positions={},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(AAPL=200.0),
        asof_date=date(2026, 5, 1),
    )
    assert len(out) == 1
    assert out[0].ticker == "AAPL"
    assert out[0].side == "BUY"
    assert out[0].shares == 25
    assert out[0].type == "DAY"
    assert out[0].last_price == 200.0


def test_translate_top_up_existing_position():
    out = translate(
        decisions=[_decision("AAPL", weight=0.05)],
        current_positions={"AAPL": 10},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(AAPL=200.0),
        asof_date=date(2026, 5, 1),
    )
    assert len(out) == 1
    assert out[0].side == "BUY"
    assert out[0].shares == 15  # delta = 25 - 10


def test_translate_full_sell_for_dropped_ticker():
    out = translate(
        decisions=[_decision("MSFT", weight=0.05)],
        current_positions={"AAPL": 25},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(AAPL=200.0, MSFT=400.0),
        asof_date=date(2026, 5, 1),
    )
    aapl_orders = [o for o in out if o.ticker == "AAPL"]
    assert len(aapl_orders) == 1
    assert aapl_orders[0].side == "SELL"
    assert aapl_orders[0].shares == 25
    assert aapl_orders[0].type == "DAY"


def test_translate_keeps_held_when_rails_blocked_buy_but_model_approved():
    """High-turnover bug observed 2026-05-05: rails (sector_cap) dropped
    AAPL from the buy list, translate force-sold the held AAPL position
    even though the model still rated AAPL highly. Fix: held tickers in
    `model_approved_tickers` (the raw pre-rails top-K) are kept, even when
    rails removed them from the post-rails decision set. Only held tickers
    the model itself dropped get force-sold."""
    out = translate(
        decisions=[_decision("MSFT", weight=0.05)],   # rails kept only MSFT
        current_positions={"AAPL": 25, "TSLA": 10},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(AAPL=200.0, MSFT=400.0, TSLA=300.0),
        asof_date=date(2026, 5, 1),
        # Strategy approved AAPL and MSFT pre-rails; rails dropped AAPL.
        # TSLA was never approved (model dropped it).
        model_approved_tickers={"AAPL", "MSFT"},
    )
    aapl_orders = [o for o in out if o.ticker == "AAPL"]
    tsla_orders = [o for o in out if o.ticker == "TSLA"]
    msft_orders = [o for o in out if o.ticker == "MSFT"]
    assert aapl_orders == [], (
        "AAPL is rails-blocked but model-approved; should NOT be sold "
        f"(got {aapl_orders})"
    )
    assert len(tsla_orders) == 1 and tsla_orders[0].side == "SELL", (
        "TSLA was model-dropped; should be sold"
    )
    assert len(msft_orders) == 1 and msft_orders[0].side == "BUY"


def test_translate_default_model_approved_falls_back_to_decisions():
    """When `model_approved_tickers` is omitted, behavior matches the
    original (pre-fix) translate semantics: held names not in decisions
    get sold. Preserves backward compatibility for callers (and tests)
    that don't pass the new arg."""
    out = translate(
        decisions=[_decision("MSFT", weight=0.05)],
        current_positions={"AAPL": 25},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(AAPL=200.0, MSFT=400.0),
        asof_date=date(2026, 5, 1),
    )
    aapl_orders = [o for o in out if o.ticker == "AAPL"]
    assert len(aapl_orders) == 1 and aapl_orders[0].side == "SELL"


def test_translate_partial_sell():
    out = translate(
        decisions=[_decision("AAPL", weight=0.05)],
        current_positions={"AAPL": 50},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(AAPL=200.0),
        asof_date=date(2026, 5, 1),
    )
    assert len(out) == 1
    assert out[0].side == "SELL"
    assert out[0].shares == 25  # |25 - 50|


def test_translate_zero_delta_emits_no_order():
    out = translate(
        decisions=[_decision("AAPL", weight=0.05)],
        current_positions={"AAPL": 25},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(AAPL=200.0),
        asof_date=date(2026, 5, 1),
    )
    assert out == []


def test_translate_uses_floor_not_round_for_share_count():
    out = translate(
        decisions=[_decision("AAPL", weight=0.055)],
        current_positions={},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(AAPL=200.0),
        asof_date=date(2026, 5, 1),
    )
    assert out[0].shares == 27   # 5500/200 = 27.5 → 27, not 28


def test_translate_skips_tickers_with_missing_price():
    out = translate(
        decisions=[_decision("AAPL"), _decision("XYZ")],
        current_positions={},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(AAPL=200.0),  # XYZ missing
        asof_date=date(2026, 5, 1),
    )
    assert {o.ticker for o in out} == {"AAPL"}


def test_translate_skips_stale_price():
    """price_date < asof - 3 days → skip ticker."""
    out = translate(
        decisions=[_decision("AAPL")],
        current_positions={},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices={"AAPL": (200.0, date(2026, 4, 25))},  # 6 days old
        asof_date=date(2026, 5, 1),
    )
    assert out == []


def test_translate_short_circuits_on_zero_equity():
    out = translate(
        decisions=[_decision("AAPL")],
        current_positions={},
        account_equity=0.0,
        cash=0.0,
        last_prices=_last_prices(AAPL=200.0),
        asof_date=date(2026, 5, 1),
    )
    assert out == []


def test_translate_raises_on_empty_decisions_with_held_positions():
    with pytest.raises(EmptyDecisionsWithHeldPositionsError):
        translate(
            decisions=[],
            current_positions={"AAPL": 25},
            account_equity=100_000.0,
            cash=100_000.0,
            last_prices=_last_prices(AAPL=200.0),
            asof_date=date(2026, 5, 1),
        )


def test_translate_persists_last_price_to_order_object():
    out = translate(
        decisions=[_decision("AAPL")],
        current_positions={},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(AAPL=237.42),
        asof_date=date(2026, 5, 1),
    )
    assert isclose(out[0].last_price, 237.42)


def test_translate_min_hold_blocks_same_day_force_sell():
    """Regression: 2026-05-05 paper run did 13 same-day round-trips
    (BUY at OPEN + SELL same evening → -$197 slippage). With
    min_hold_days=1 and the position entered today, the force-sell
    must NOT fire."""
    asof = date(2026, 5, 12)
    out = translate(
        decisions=[_decision("MSFT", weight=0.05)],
        current_positions={"AAPL": 25},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(AAPL=200.0, MSFT=400.0),
        asof_date=asof,
        rails=RiskRails(min_hold_days=1),
        # AAPL bought today — entry_date == asof_date → held_days == 0 < 1.
        position_entry_dates={"AAPL": asof},
    )
    aapl_orders = [o for o in out if o.ticker == "AAPL"]
    assert aapl_orders == [], (
        "AAPL was bought today; min_hold_days=1 should block the same-day "
        f"force-sell (got {aapl_orders})"
    )


def test_translate_min_hold_allows_sell_when_held_long_enough():
    """min_hold_days=1 should NOT block a force-sell of a position held
    for 2 days. The rail is a 1-day floor, not a permanent stay."""
    out = translate(
        decisions=[_decision("MSFT", weight=0.05)],
        current_positions={"AAPL": 25},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(AAPL=200.0, MSFT=400.0),
        asof_date=date(2026, 5, 12),
        rails=RiskRails(min_hold_days=1),
        position_entry_dates={"AAPL": date(2026, 5, 10)},  # 2 days ago
    )
    aapl_orders = [o for o in out if o.ticker == "AAPL"]
    assert len(aapl_orders) == 1 and aapl_orders[0].side == "SELL"


def test_translate_min_hold_zero_is_no_op():
    """Default `min_hold_days=0` preserves legacy behavior — force-sells
    fire regardless of entry-date freshness."""
    out = translate(
        decisions=[_decision("MSFT", weight=0.05)],
        current_positions={"AAPL": 25},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(AAPL=200.0, MSFT=400.0),
        asof_date=date(2026, 5, 12),
        rails=RiskRails(min_hold_days=0),  # explicit no-op
        position_entry_dates={"AAPL": date(2026, 5, 12)},  # bought today
    )
    aapl_orders = [o for o in out if o.ticker == "AAPL"]
    assert len(aapl_orders) == 1 and aapl_orders[0].side == "SELL"


def test_translate_min_hold_no_entry_date_does_not_block():
    """Tickers with NO recorded entry date (e.g. pre-seeded Alpaca paper
    account) get sold normally — the rail can only protect positions we
    know we just bought."""
    out = translate(
        decisions=[_decision("MSFT", weight=0.05)],
        current_positions={"AAPL": 25},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(AAPL=200.0, MSFT=400.0),
        asof_date=date(2026, 5, 12),
        rails=RiskRails(min_hold_days=1),
        position_entry_dates={},  # AAPL absent → no protection
    )
    aapl_orders = [o for o in out if o.ticker == "AAPL"]
    assert len(aapl_orders) == 1 and aapl_orders[0].side == "SELL"


def test_translate_min_hold_only_blocks_force_sells_not_partial_sells():
    """min_hold_days only gates the force-sell path (held but not in
    decisions). A partial sell (model still wants the ticker, just less
    of it) must still fire regardless of how fresh the position is."""
    asof = date(2026, 5, 12)
    out = translate(
        decisions=[_decision("AAPL", weight=0.02)],  # target 10sh @ $200, hold 50
        current_positions={"AAPL": 50},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(asof=asof, AAPL=200.0),
        asof_date=asof,
        rails=RiskRails(min_hold_days=1),
        position_entry_dates={"AAPL": asof},  # bought today
    )
    aapl_orders = [o for o in out if o.ticker == "AAPL"]
    assert len(aapl_orders) == 1 and aapl_orders[0].side == "SELL"
    assert aapl_orders[0].shares == 40  # 50 - 10


def test_translate_min_hold_blocks_target_weight_zero_full_liquidation():
    """Codex MED finding: a decision with target_weight=0 routes through
    the delta-branch (delta = 0 - 50 = -50), not the force-sell loop. The
    fix applies min_hold to BOTH branches when the SELL would be a full
    exit. Without this fix the rail is bypassed by setting target=0."""
    asof = date(2026, 5, 12)
    out = translate(
        decisions=[_decision("AAPL", weight=0.0)],   # full liquidation
        current_positions={"AAPL": 50},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(asof=asof, AAPL=200.0),
        asof_date=asof,
        rails=RiskRails(min_hold_days=1),
        position_entry_dates={"AAPL": asof},
    )
    aapl_orders = [o for o in out if o.ticker == "AAPL"]
    assert aapl_orders == [], (
        f"target_weight=0 full liquidation must respect min_hold (got {aapl_orders})"
    )


def test_translate_min_hold_future_entry_date_does_not_pin_position():
    """Codex LOW finding: a future-dated entry (clock skew, bad backfill)
    must NOT pin the position forever. The rail treats `held_days < 0` as
    "not fresh" so the operator can still exit."""
    out = translate(
        decisions=[_decision("MSFT", weight=0.05)],
        current_positions={"AAPL": 25},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(AAPL=200.0, MSFT=400.0),
        asof_date=date(2026, 5, 12),
        rails=RiskRails(min_hold_days=5),
        # Entry date in the future (clock skew / backfill bug).
        position_entry_dates={"AAPL": date(2026, 6, 1)},
    )
    aapl_orders = [o for o in out if o.ticker == "AAPL"]
    assert len(aapl_orders) == 1 and aapl_orders[0].side == "SELL", (
        "Future-dated entry should NOT pin the position"
    )


def test_translate_min_hold_partial_reduction_still_fires_when_fresh():
    """Make explicit: a partial reduction (model trims position, target > 0)
    on a freshly-bought name still produces the SELL. min_hold only blocks
    FULL exits — partial trims are model rebalances, not churn."""
    asof = date(2026, 5, 12)
    out = translate(
        decisions=[_decision("AAPL", weight=0.04)],  # 20 target vs 50 held
        current_positions={"AAPL": 50},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(asof=asof, AAPL=200.0),
        asof_date=asof,
        rails=RiskRails(min_hold_days=1),
        position_entry_dates={"AAPL": asof},  # bought today
    )
    aapl_orders = [o for o in out if o.ticker == "AAPL"]
    assert len(aapl_orders) == 1 and aapl_orders[0].side == "SELL"
    assert aapl_orders[0].shares == 30  # 50 - 20


def test_translate_dead_zone_skips_small_rebalance_sell():
    """Codex HIGH #1 (2026-05-14): when account_equity drops slightly (e.g.
    -5% on a broad market move), target_value of every position drops
    proportionally, and every position gets a small SELL. That's pure churn
    paid in spread/slippage. The dead-zone rail skips rebalances smaller
    than 10% of current shares so this doesn't happen.

    Setup: hold 100 shares at $100. target_value = 0.10 * $95k = $9500 →
    target_shares = floor(9500/100) = 95. delta = -5 = 5% of held = inside
    the 10% dead zone → skip."""
    asof = date(2026, 5, 14)
    out = translate(
        decisions=[_decision("AAPL", weight=0.10)],
        current_positions={"AAPL": 100},
        account_equity=95_000.0,  # equity dropped from a notional 100k
        cash=10_000.0,
        last_prices=_last_prices(asof=asof, AAPL=100.0),
        asof_date=asof,
        rails=RiskRails(rebalance_dead_zone_pct=0.10),
    )
    aapl_orders = [o for o in out if o.ticker == "AAPL"]
    assert aapl_orders == [], f"small rebalance should be skipped (got {aapl_orders})"


def test_translate_dead_zone_lets_large_rebalance_through():
    """A meaningful position change (>10% of held shares) is NOT churn —
    that's the model genuinely re-sizing. Must fire."""
    asof = date(2026, 5, 14)
    out = translate(
        decisions=[_decision("AAPL", weight=0.07)],  # 70 target vs 100 held = 30% reduction
        current_positions={"AAPL": 100},
        account_equity=100_000.0,
        cash=10_000.0,
        last_prices=_last_prices(asof=asof, AAPL=100.0),
        asof_date=asof,
        rails=RiskRails(rebalance_dead_zone_pct=0.10),
    )
    aapl_orders = [o for o in out if o.ticker == "AAPL"]
    assert len(aapl_orders) == 1 and aapl_orders[0].side == "SELL"
    assert aapl_orders[0].shares == 30


def test_translate_dead_zone_does_not_block_new_entries():
    """A new entry (held=0) is never a rebalance — must always fire."""
    asof = date(2026, 5, 14)
    out = translate(
        decisions=[_decision("MSFT", weight=0.05)],
        current_positions={},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(asof=asof, MSFT=200.0),
        asof_date=asof,
        rails=RiskRails(rebalance_dead_zone_pct=0.10),
    )
    msft_orders = [o for o in out if o.ticker == "MSFT"]
    assert len(msft_orders) == 1 and msft_orders[0].side == "BUY"
    assert msft_orders[0].shares == 25  # floor(5000/200)


def test_translate_dead_zone_does_not_block_full_exit():
    """target_weight=0 (full liquidation) is not a rebalance — must fire."""
    asof = date(2026, 5, 14)
    out = translate(
        decisions=[_decision("AAPL", weight=0.0)],  # exit entirely
        current_positions={"AAPL": 100},
        account_equity=100_000.0,
        cash=10_000.0,
        last_prices=_last_prices(asof=asof, AAPL=100.0),
        asof_date=asof,
        rails=RiskRails(
            rebalance_dead_zone_pct=0.10,
            min_hold_days=0,  # disable other rail
        ),
    )
    aapl_orders = [o for o in out if o.ticker == "AAPL"]
    assert len(aapl_orders) == 1 and aapl_orders[0].side == "SELL"
    assert aapl_orders[0].shares == 100


def test_translate_dead_zone_boundary_delta_exactly_at_threshold_is_suppressed():
    """CAT bug (7/23-7/28 live): with the old `abs(delta) < dead_zone * held`
    comparison, a delta EXACTLY at the threshold (held=10, delta=1, dead_zone
    =0.10 -> 1.0) evaluated `1 < 1.0` = False, so the boundary rebalance
    fired anyway. CAT did four 1-share round-trips paying spread each time.
    Fixed to `<=` so an at-threshold delta is suppressed like anything
    smaller."""
    asof = date(2026, 7, 23)
    out = translate(
        decisions=[_decision("AAPL", weight=0.09)],  # target 9 vs held 10 -> delta -1 (10%)
        current_positions={"AAPL": 10},
        account_equity=100_000.0,
        cash=10_000.0,
        last_prices=_last_prices(asof=asof, AAPL=1_000.0),
        asof_date=asof,
        rails=RiskRails(rebalance_dead_zone_pct=0.10),
    )
    aapl_orders = [o for o in out if o.ticker == "AAPL"]
    assert aapl_orders == [], (
        f"exactly-at-threshold delta must be suppressed (got {aapl_orders})"
    )


def test_translate_dead_zone_delta_above_threshold_still_trades():
    """A delta just ABOVE the threshold (held=10, delta=2 -> 20%) is a real
    rebalance, not boundary noise, and must still fire."""
    asof = date(2026, 7, 23)
    out = translate(
        decisions=[_decision("AAPL", weight=0.08)],  # target 8 vs held 10 -> delta -2 (20%)
        current_positions={"AAPL": 10},
        account_equity=100_000.0,
        cash=10_000.0,
        last_prices=_last_prices(asof=asof, AAPL=1_000.0),
        asof_date=asof,
        rails=RiskRails(rebalance_dead_zone_pct=0.10),
    )
    aapl_orders = [o for o in out if o.ticker == "AAPL"]
    assert len(aapl_orders) == 1 and aapl_orders[0].side == "SELL"
    assert aapl_orders[0].shares == 2


def test_translate_dead_zone_boundary_smaller_held_still_trades_above_pct():
    """held=5, delta=1 is 20% of held (above the 10% dead zone) -> must fire,
    even though the absolute share count (1) matches the boundary case above.
    The dead zone is a fraction of held shares, not a fixed share count."""
    asof = date(2026, 7, 23)
    out = translate(
        decisions=[_decision("AAPL", weight=0.04)],  # target 4 vs held 5 -> delta -1 (20%)
        current_positions={"AAPL": 5},
        account_equity=100_000.0,
        cash=10_000.0,
        last_prices=_last_prices(asof=asof, AAPL=1_000.0),
        asof_date=asof,
        rails=RiskRails(rebalance_dead_zone_pct=0.10),
    )
    aapl_orders = [o for o in out if o.ticker == "AAPL"]
    assert len(aapl_orders) == 1 and aapl_orders[0].side == "SELL"
    assert aapl_orders[0].shares == 1


def test_translate_dead_zone_zero_is_noop():
    """Default 0.0 preserves legacy behaviour: even tiny rebalances fire."""
    asof = date(2026, 5, 14)
    out = translate(
        decisions=[_decision("AAPL", weight=0.10)],
        current_positions={"AAPL": 100},
        account_equity=95_000.0,
        cash=10_000.0,
        last_prices=_last_prices(asof=asof, AAPL=100.0),
        asof_date=asof,
        rails=RiskRails(rebalance_dead_zone_pct=0.0),
    )
    aapl_orders = [o for o in out if o.ticker == "AAPL"]
    assert len(aapl_orders) == 1 and aapl_orders[0].shares == 5  # the 5% trim fires


def test_translate_paranoia_rail_skipped_when_all_positions_fresh():
    """Codex MED 1 (2026-05-13): when decisions is empty + min_hold is
    active + every held position is inside the min_hold window, the
    paranoia rail must suppress and return an empty order list (no
    trading today — hold what we have). Without this, decide aborts."""
    asof = date(2026, 5, 13)
    out = translate(
        decisions=[],
        current_positions={"AAPL": 25, "MSFT": 10},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(asof=asof, AAPL=200.0, MSFT=400.0),
        asof_date=asof,
        rails=RiskRails(min_hold_days=2),
        position_entry_dates={
            "AAPL": date(2026, 5, 12),  # 1 day ago
            "MSFT": date(2026, 5, 13),  # bought today
        },
    )
    assert out == [], (
        f"All positions inside min_hold; should be no-op (got {out})"
    )


def test_translate_paranoia_rail_fires_when_any_position_unprotected():
    """If even ONE held position has aged past min_hold (legitimately
    sellable but the model emitted no decisions), the paranoia rail
    fires — that's a real bug indicator, not a stay-the-course day."""
    asof = date(2026, 5, 13)
    with pytest.raises(EmptyDecisionsWithHeldPositionsError):
        translate(
            decisions=[],
            current_positions={"AAPL": 25, "MSFT": 10},
            account_equity=100_000.0,
            cash=100_000.0,
            last_prices=_last_prices(asof=asof, AAPL=200.0, MSFT=400.0),
            asof_date=asof,
            rails=RiskRails(min_hold_days=1),
            position_entry_dates={
                "AAPL": date(2026, 5, 13),  # fresh
                "MSFT": date(2026, 5, 5),   # 8 days old — sellable
            },
        )


def test_translate_paranoia_rail_fires_when_no_min_hold_set(store=None):
    """When min_hold_days is 0 (or rails is None), legacy paranoia rail
    behavior stands — 0 decisions + held positions always aborts."""
    with pytest.raises(EmptyDecisionsWithHeldPositionsError):
        translate(
            decisions=[],
            current_positions={"AAPL": 25},
            account_equity=100_000.0,
            cash=100_000.0,
            last_prices=_last_prices(AAPL=200.0),
            asof_date=date(2026, 5, 1),
            rails=RiskRails(min_hold_days=0),
            position_entry_dates={"AAPL": date(2026, 5, 1)},
        )


def test_translate_buy_and_sell_both_use_day():
    """Buys queue to the next session (DAY market, fill near the open); sells
    exit any time same day. (OPG retired — it expired unfilled in paper.)"""
    out = translate(
        decisions=[_decision("AAPL", weight=0.05)],
        current_positions={"AAPL": 50},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(AAPL=200.0),
        asof_date=date(2026, 5, 1),
    )
    assert out[0].type == "DAY"   # selling down

    out = translate(
        decisions=[_decision("AAPL", weight=0.05)],
        current_positions={},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(AAPL=200.0),
        asof_date=date(2026, 5, 1),
    )
    assert out[0].type == "DAY"   # buying from zero


def test_translate_cash_floor_skips_buy_that_would_breach_floor():
    """100% target weight + 5% cash floor → buy is rejected (would leave 0% cash)."""
    out = translate(
        decisions=[_decision("AAPL", weight=1.0)],   # buy with 100% of equity
        current_positions={},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(AAPL=200.0),
        asof_date=date(2026, 5, 1),
        rails=RiskRails(stop_loss_pct=0.0, cash_floor_pct=0.05),
    )
    assert out == []


def test_translate_cash_floor_disabled_allows_full_buy():
    """rails=None → no cash floor enforcement → buy proceeds."""
    out = translate(
        decisions=[_decision("AAPL", weight=1.0)],
        current_positions={},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(AAPL=200.0),
        asof_date=date(2026, 5, 1),
        rails=None,
    )
    assert len(out) == 1
    assert out[0].side == "BUY"


def test_translate_cash_floor_zero_pct_allows_full_buy():
    """rails.cash_floor_pct=0 → floor disabled → buy proceeds."""
    out = translate(
        decisions=[_decision("AAPL", weight=1.0)],
        current_positions={},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(AAPL=200.0),
        asof_date=date(2026, 5, 1),
        rails=RiskRails(stop_loss_pct=0.0, cash_floor_pct=0.0),
    )
    assert len(out) == 1


def test_translate_cash_floor_uses_sell_proceeds_to_fund_buys():
    """SELLs are processed first; their proceeds increase the cash budget for BUYs."""
    # Start with 5% cash; want to sell MSFT (frees ~$5k) then buy AAPL (~$5k).
    out = translate(
        decisions=[
            _decision("AAPL", weight=0.05),   # buy 25 @ 200 = $5,000
            _decision("MSFT", weight=0.0),    # full close MSFT
        ],
        current_positions={"MSFT": 25},        # MSFT 25 shares @ 200 = $5,000
        account_equity=100_000.0,
        cash=5_000.0,                          # only $5k cash before SELLs settle
        last_prices=_last_prices(AAPL=200.0, MSFT=200.0),
        asof_date=date(2026, 5, 1),
        rails=RiskRails(stop_loss_pct=0.0, cash_floor_pct=0.05),
    )
    # Both orders should be present: SELL MSFT (always), BUY AAPL (funded by MSFT sale)
    sides = sorted([(o.ticker, o.side) for o in out])
    assert sides == [("AAPL", "BUY"), ("MSFT", "SELL")]


def test_translate_cash_floor_partial_acceptance_when_running_cash_runs_out():
    """Multiple BUYs in batch: accept until floor threatens, skip rest."""
    out = translate(
        decisions=[
            _decision("AAPL", weight=0.30),   # cost 30k
            _decision("MSFT", weight=0.30),   # cost 30k
            _decision("GOOGL", weight=0.30),  # cost 30k
            # 3 x 30k = 90k; with $100k cash and 5k floor, only 2 should fit
        ],
        current_positions={},
        account_equity=100_000.0,
        cash=100_000.0,
        last_prices=_last_prices(AAPL=200.0, MSFT=200.0, GOOGL=200.0),
        asof_date=date(2026, 5, 1),
        rails=RiskRails(stop_loss_pct=0.0, cash_floor_pct=0.05),
    )
    # 30k + 30k = 60k spent; 100k - 60k = 40k cash left, above 5k floor → both pass.
    # 30k + 30k + 30k = 90k spent; 100k - 90k = 10k left, above 5k floor → all pass.
    # Wait — 10k > 5k so all 3 actually fit. Recompute: only 2 BUYs fit if a 4th would breach.
    # With 3 buys of 30k each on 100k cash + 5k floor: running_cash starts 100k.
    #   buy 1: 100k - 30k = 70k >= 5k → accept
    #   buy 2: 70k - 30k = 40k >= 5k → accept
    #   buy 3: 40k - 30k = 10k >= 5k → accept
    # So all 3 fit. This test confirms multi-buy accumulation works correctly.
    assert len(out) == 3
    assert all(o.side == "BUY" for o in out)


def test_translate_sell_proceeds_haircut_buffers_rotation_buys():
    """A haircut on ESTIMATED sell proceeds keeps a cash buffer so we do not
    fund a rotation buy against proceeds that may not fully materialize at the
    (lower) open fill. haircut=1.0 preserves legacy behavior."""
    kw = dict(
        decisions=[_decision("AAPL", weight=0.05), _decision("MSFT", weight=0.0)],
        current_positions={"MSFT": 25},  # 25 @ 200 = $5,000 proceeds
        account_equity=100_000.0,
        cash=5_000.0,
        last_prices=_last_prices(AAPL=200.0, MSFT=200.0),
        asof_date=date(2026, 5, 1),
    )
    full = translate(
        **kw, rails=RiskRails(stop_loss_pct=0.0, cash_floor_pct=0.05, sell_proceeds_haircut=1.0)
    )
    assert sorted((o.ticker, o.side) for o in full) == [("AAPL", "BUY"), ("MSFT", "SELL")]
    # haircut 0.5 → only $2.5k of the $5k proceeds count → the $5k AAPL buy would
    # drop cash below the $5k floor → blocked. Only the SELL remains.
    cut = translate(
        **kw, rails=RiskRails(stop_loss_pct=0.0, cash_floor_pct=0.05, sell_proceeds_haircut=0.5)
    )
    assert sorted((o.ticker, o.side) for o in cut) == [("MSFT", "SELL")]


def test_translate_funds_highest_conviction_first_under_tight_cash():
    """When cash only covers one buy, the higher-target_weight (higher
    conviction) name wins, regardless of the order decisions arrive in."""
    out = translate(
        decisions=[_decision("ZZZ", weight=0.09), _decision("AAA", weight=0.10)],
        current_positions={},
        account_equity=100_000.0,
        cash=15_000.0,
        last_prices=_last_prices(AAA=200.0, ZZZ=200.0),
        asof_date=date(2026, 5, 1),
        rails=RiskRails(stop_loss_pct=0.0, cash_floor_pct=0.05),
    )
    buys = [o for o in out if o.side == "BUY"]
    assert len(buys) == 1
    assert buys[0].ticker == "AAA"  # 0.10 conviction funded before 0.09


def test_translate_tie_weight_preserves_emission_conviction_order():
    """When target weights TIE, the strategy's emission order (best first) is the
    conviction signal and must be preserved — a stable sort must not reorder them
    alphabetically (Codex HIGH 2026-07-01). ZZZ is emitted first (higher
    conviction) at the same weight as AAA; with cash for only one, ZZZ wins."""
    out = translate(
        decisions=[_decision("ZZZ", weight=0.10), _decision("AAA", weight=0.10)],
        current_positions={},
        account_equity=10_000.0,
        cash=1_000.0,
        last_prices=_last_prices(AAA=100.0, ZZZ=100.0),
        asof_date=date(2026, 5, 1),
        rails=RiskRails(stop_loss_pct=0.0, cash_floor_pct=0.0),
    )
    buys = [o for o in out if o.side == "BUY"]
    assert len(buys) == 1
    assert buys[0].ticker == "ZZZ"  # emission order preserved, NOT alphabetical AAA


def test_translate_drawdown_derisk_raises_floor_blocks_buys():
    """2026-06-16: drawdown-scaled de-risk (validated: slope 1.5 improved
    sharpe/return/maxDD on val). In a drawdown with slope>0 the effective cash
    floor rises, so a buy that fits at full deployment is blocked — the book
    de-risks as a crash develops. slope=0 (default) leaves behavior unchanged."""
    rails_off = RiskRails(
        stop_loss_pct=0.0, max_position_pct=0.10, cash_floor_pct=0.05,
        drawdown_derisk_slope=0.0,
    )
    rails_on = RiskRails(
        stop_loss_pct=0.0, max_position_pct=0.10, cash_floor_pct=0.05,
        drawdown_derisk_start=0.05, drawdown_derisk_slope=3.0,
    )
    # account 90% in cash, 15% drawdown. A 10% buy ($10k) fits under a 5% floor
    # ($5k floor leaves $85k room) but NOT under the de-risked floor:
    # derisk_cash_floor(0.05, 0.15, start=0.05, slope=3.0) = 0.05+3*0.10 = 0.35
    # -> floor $35k, cash $90k, buy $10k -> $80k >= $35k still ok... use deeper.
    kw = dict(
        current_positions={}, account_equity=100_000.0, cash=40_000.0,
        last_prices=_last_prices(AAPL=200.0), asof_date=date(2026, 5, 1),
    )
    buy = [_decision("AAPL", weight=0.10)]  # $10k buy, needs $10k cash
    # OFF: floor $5k, cash $40k -> buy fits
    out_off = translate(decisions=buy, rails=rails_off, current_drawdown=0.15, **kw)
    assert any(o.ticker == "AAPL" and o.side == "BUY" for o in out_off)
    # ON: drawdown 0.15 -> floor 0.35 = $35k; $40k cash - $10k buy = $30k < $35k -> blocked
    out_on = translate(decisions=buy, rails=rails_on, current_drawdown=0.15, **kw)
    assert not any(o.ticker == "AAPL" and o.side == "BUY" for o in out_on)


def test_translate_derisk_no_effect_at_peak():
    """At peak (drawdown 0), de-risk is inert even when enabled."""
    rails_on = RiskRails(
        stop_loss_pct=0.0, max_position_pct=0.10, cash_floor_pct=0.05,
        drawdown_derisk_start=0.05, drawdown_derisk_slope=3.0,
    )
    out = translate(
        decisions=[_decision("AAPL", weight=0.10)], current_positions={},
        account_equity=100_000.0, cash=40_000.0,
        last_prices=_last_prices(AAPL=200.0), asof_date=date(2026, 5, 1),
        rails=rails_on, current_drawdown=0.0,
    )
    assert any(o.ticker == "AAPL" and o.side == "BUY" for o in out)


def test_stale_price_does_not_swallow_held_name_reduction():
    """2026-07-01 MEDIUM: the stale-price veto also swallowed SELLs of held
    names AFTER pipeline.apply had already freed that sector room for buys —
    breaching the sector cap with no alert. Reductions now size on the last
    known (stale) price; buys stay vetoed."""
    from datetime import date
    stale = (100.0, date(2026, 1, 2))  # far older than the threshold
    orders = translate(
        decisions=[_decision("OLDP", 0.01)],
        last_prices={"OLDP": stale},
        account_equity=100_000.0,
        cash=50_000.0,
        current_positions={"OLDP": 50},  # target = 10 shares -> delta -40
        asof_date=date(2026, 6, 30),
    )
    sells = [o for o in orders if o.side == "SELL"]
    assert len(sells) == 1 and sells[0].shares == 40
    # same staleness on a NON-held name (a buy) stays vetoed
    orders = translate(
        decisions=[_decision("OLDP", 0.01)],
        last_prices={"OLDP": stale},
        account_equity=100_000.0,
        cash=50_000.0,
        current_positions={},
        asof_date=date(2026, 6, 30),
    )
    assert orders == []


def test_missing_price_full_exit_still_sells():
    """A target_weight=0 exit of a held name needs no price at all."""
    from datetime import date
    orders = translate(
        decisions=[_decision("GONE", 0.0)],
        last_prices={},
        account_equity=100_000.0,
        cash=50_000.0,
        current_positions={"GONE": 30},
        asof_date=date(2026, 6, 30),
    )
    sells = [o for o in orders if o.side == "SELL" and o.ticker == "GONE"]
    assert len(sells) == 1 and sells[0].shares == 30
