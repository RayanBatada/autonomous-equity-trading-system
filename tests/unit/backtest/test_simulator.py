"""Simulator tests including a hand-computed 3-trade P&L example.

The hand-computed example is the most important test in this whole codebase.
If the simulator gets it wrong, every Phase 2+ measurement is corrupted.
"""

from datetime import date

import pandas as pd
import pytest

from sma.backtest.simulator import simulate
from sma.backtest.slippage import SlippageModel
from sma.backtest.strategies.base import StrategyDecision


def _build_price_df(rows: list[tuple]) -> pd.DataFrame:
    """rows is a list of (ticker, date, open, close, adj_close, volume) tuples."""
    df = pd.DataFrame(rows, columns=["ticker", "date", "open", "close", "adj_close", "volume"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


class FixedDecisionStrategy:
    """Strategy that returns canned decisions per date."""
    name = "fixed_decision"

    def __init__(self, plan: dict):  # date -> list of (ticker, weight)
        self.plan = plan

    def decide(self, asof_date, prices, fundamentals=None):
        if asof_date not in self.plan:
            return []
        return [
            StrategyDecision(asof_date=asof_date, ticker=t, target_weight=w)
            for t, w in self.plan[asof_date]
        ]


class _InternalTypeErrorStrategy:
    """decide() has the standard signature but raises a TypeError from DEEP
    inside (mimics a crashing tilt() under tilt_strict). The simulator must NOT
    mistake this for a decide()-signature mismatch and silently retry — that
    masked broken autoresearch proposals as untilted-baseline scores."""

    name = "internal_typeerror"

    def __init__(self):
        self.calls = 0

    def decide(self, asof_date, prices, fundamentals=None):
        self.calls += 1
        raise TypeError(
            "StrategyDecision.__init__() got an unexpected keyword argument "
            "'signal_metadata'"
        )


def test_simulator_does_not_swallow_internal_typeerror_from_decide():
    prices = _build_price_df([
        ("AAPL", "2025-07-01", 100.0, 101.0, 101.0, 50_000_000),
        ("AAPL", "2025-07-02", 101.0, 102.0, 102.0, 50_000_000),
    ])
    strat = _InternalTypeErrorStrategy()
    with pytest.raises(TypeError, match="signal_metadata"):
        simulate(
            strategy=strat,
            universe=["AAPL"],
            prices=prices,
            sector_map={"AAPL": "Technology"},
            window_name="val",
            start_date=date(2025, 7, 1),
            end_date=date(2025, 7, 2),
            initial_cash=100_000.0,
        )
    # Must fail fast on the first call — NOT retry decide() (which would mask
    # the real error behind the signature-compat fallback).
    assert strat.calls == 1


def test_stop_loss_pct_zero_disables_the_stop():
    """stop_loss_pct=0.0 must DISABLE the stop (Phase 5 ships disabled). The
    condition `pct_loss >= stop_loss_pct` fired on EVERY flat/down position when
    stop_loss_pct=0 (0 >= 0), silently liquidating break-even holdings and
    confounding every no-stop baseline backtest."""
    from sma.risk.rails import RiskRails
    prices = _build_price_df([
        ("AAA", "2025-07-01", 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", "2025-07-02", 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", "2025-07-03", 100.0, 100.0, 100.0, 50_000_000),
    ])
    # Keep AAA every day (no drop → isolates the stop logic from force-sell).
    strat = FixedDecisionStrategy(plan={
        date(2025, 7, 1): [("AAA", 0.5)],
        date(2025, 7, 2): [("AAA", 0.5)],
        date(2025, 7, 3): [("AAA", 0.5)],
    })
    rails = RiskRails(
        max_position_pct=0.6, max_sector_pct=0.9, stop_loss_pct=0.0,
        cash_floor_pct=0.0, max_drawdown_pct=1.0,
    )
    result = simulate(
        strategy=strat, universe=["AAA"], prices=prices,
        sector_map={"AAA": "Technology"}, window_name="train",
        start_date=date(2025, 7, 1), end_date=date(2025, 7, 3),
        initial_cash=100_000.0,
        slippage_model=SlippageModel(base_bps=0.0, illiquidity_penalty_bps=0.0),
        rails=rails,
    )
    actions = [(t["ticker"], t["action"]) for t in result.trades]
    assert ("AAA", "stop_loss") not in actions, (
        f"stop_loss fired with stop_loss_pct=0.0 (should be disabled); trades={actions}"
    )


def test_simulator_force_sells_names_dropped_from_decisions():
    """A top-K strategy holds ONLY its current picks. When a name drops out of
    the decision set, the simulator must force-sell it (live `translate()` does,
    via keep_set). Previously the simulator held dropped names forever — which
    made the stop-loss the de-facto sole exit and inflated its backtested value.
    """
    prices = _build_price_df([
        ("AAA", "2025-07-01", 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", "2025-07-02", 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", "2025-07-03", 100.0, 100.0, 100.0, 50_000_000),
        ("BBB", "2025-07-01", 50.0, 50.0, 50.0, 50_000_000),
        ("BBB", "2025-07-02", 50.0, 50.0, 50.0, 50_000_000),
        ("BBB", "2025-07-03", 50.0, 50.0, 50.0, 50_000_000),
    ])
    # 7/1 pick AAA; 7/2+ pick BBB only (AAA dropped from top-K).
    strat = FixedDecisionStrategy(plan={
        date(2025, 7, 1): [("AAA", 0.5)],
        date(2025, 7, 2): [("BBB", 0.5)],
        date(2025, 7, 3): [("BBB", 0.5)],
    })
    from sma.risk.rails import RiskRails
    permissive = RiskRails(
        max_position_pct=0.6, max_sector_pct=0.9, stop_loss_pct=0.0,
        cash_floor_pct=0.0, max_drawdown_pct=1.0,
    )
    result = simulate(
        strategy=strat,
        universe=["AAA", "BBB"],
        prices=prices,
        sector_map={"AAA": "Technology", "BBB": "Technology"},
        window_name="train",
        start_date=date(2025, 7, 1),
        end_date=date(2025, 7, 3),
        initial_cash=100_000.0,
        slippage_model=SlippageModel(base_bps=0.0, illiquidity_penalty_bps=0.0),
        rails=permissive,
    )
    actions = [(t["ticker"], t["action"]) for t in result.trades]
    assert ("AAA", "buy") in actions, f"AAA never bought; trades={actions}"
    assert ("AAA", "sell") in actions, (
        f"AAA dropped from decisions but was never force-sold; trades={actions}"
    )


def test_rebalance_dead_zone_skips_small_rebalances():
    """rails.rebalance_dead_zone_pct must suppress tiny rebalances of existing
    positions (live orders.py parity). The simulator ignored it, so the backtest
    churned on sub-threshold weight drift that live would never trade."""
    from sma.risk.rails import RiskRails
    prices = _build_price_df([
        ("AAA", "2025-07-01", 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", "2025-07-02", 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", "2025-07-03", 100.0, 100.0, 100.0, 50_000_000),
    ])
    # Day1 enter AAA at 10%; day2 nudge to 10.5% (a 5% position change — inside
    # a 10% dead-zone, so no rebalance trade should fire on day 3).
    strat = FixedDecisionStrategy(plan={
        date(2025, 7, 1): [("AAA", 0.10)],
        date(2025, 7, 2): [("AAA", 0.105)],
        date(2025, 7, 3): [("AAA", 0.105)],
    })
    rails = RiskRails(
        max_position_pct=0.5, max_sector_pct=0.9, stop_loss_pct=0.0,
        cash_floor_pct=0.0, max_drawdown_pct=1.0, rebalance_dead_zone_pct=0.10,
    )
    result = simulate(
        strategy=strat, universe=["AAA"], prices=prices,
        sector_map={"AAA": "Technology"}, window_name="train",
        start_date=date(2025, 7, 1), end_date=date(2025, 7, 3),
        initial_cash=100_000.0,
        slippage_model=SlippageModel(base_bps=0.0, illiquidity_penalty_bps=0.0),
        rails=rails,
    )
    aaa_buys = [t for t in result.trades if t["ticker"] == "AAA" and t["action"] == "buy"]
    assert len(aaa_buys) == 1, (
        f"expected only the entry buy (dead-zone skips the tiny rebalance); "
        f"got {len(aaa_buys)} buys: {result.trades}"
    )


def test_simulator_returns_backtest_result_for_no_op_strategy():
    """Strategy that always returns empty decisions => 100% cash, zero return."""
    prices = _build_price_df([
        ("AAPL", "2025-07-01", 100.0, 101.0, 101.0, 50_000_000),
        ("AAPL", "2025-07-02", 101.0, 102.0, 102.0, 50_000_000),
        ("AAPL", "2025-07-03", 102.0, 103.0, 103.0, 50_000_000),
    ])
    strat = FixedDecisionStrategy(plan={})

    result = simulate(
        strategy=strat,
        universe=["AAPL"],
        prices=prices,
        sector_map={"AAPL": "Technology"},
        window_name="train",
        start_date=date(2025, 7, 1),
        end_date=date(2025, 7, 3),
        initial_cash=100_000.0,
    )

    assert result.strategy_name == "fixed_decision"
    assert result.window == "train"
    assert result.total_return == 0.0
    assert result.num_trades == 0
    assert isinstance(result.daily_returns, list)


def test_simulator_lookahead_protection():
    """Strategy that tries to use future data should fail or get future-blind data only."""
    prices = _build_price_df([
        ("AAPL", "2025-07-01", 100.0, 101.0, 101.0, 50_000_000),
        ("AAPL", "2025-07-02", 101.0, 102.0, 102.0, 50_000_000),
    ])

    received_dates = []
    class PeekingStrategy:
        name = "peeker"
        def decide(self, asof_date, prices, fundamentals=None):
            received_dates.append((asof_date, prices["date"].max() if not prices.empty else None))
            return []

    simulate(
        strategy=PeekingStrategy(),
        universe=["AAPL"],
        prices=prices,
        sector_map={"AAPL": "Technology"},
        window_name="train",
        start_date=date(2025, 7, 1),
        end_date=date(2025, 7, 2),
        initial_cash=100_000.0,
    )

    # On asof 2025-07-01, the strategy should NOT see 2025-07-02's row.
    for asof, max_seen in received_dates:
        assert max_seen is None or max_seen <= asof, \
            f"Strategy on {asof} saw data up to {max_seen} (lookahead leak)"


def test_simulator_executes_buy_at_next_day_open():
    """Decision on day 1 with target_weight=0.04 (within 5% cap) should buy at day 2's open."""
    prices = _build_price_df([
        ("AAPL", "2025-07-01", 100.0, 101.0, 101.0, 50_000_000),
        ("AAPL", "2025-07-02", 105.0, 106.0, 106.0, 50_000_000),  # day 2 open is 105
        ("AAPL", "2025-07-03", 106.0, 107.0, 107.0, 50_000_000),
    ])
    strat = FixedDecisionStrategy(plan={
        date(2025, 7, 1): [("AAPL", 0.04)],   # 4% target
    })

    result = simulate(
        strategy=strat,
        universe=["AAPL"],
        prices=prices,
        sector_map={"AAPL": "Technology"},
        window_name="train",
        start_date=date(2025, 7, 1),
        end_date=date(2025, 7, 3),
        initial_cash=100_000.0,
        slippage_model=SlippageModel(base_bps=0.0, illiquidity_penalty_bps=0.0),
    )

    # 4% of 100k = $4000. At $105 buy price = 38 shares rounded down = $3990 invested.
    assert result.num_trades >= 1


def test_simulator_rejects_decisions_above_position_cap_but_continues():
    """Decision tries 50% (above 5% cap). Order is rejected; rest of run completes cleanly."""
    prices = _build_price_df([
        ("AAPL", "2025-07-01", 100.0, 101.0, 101.0, 50_000_000),
        ("AAPL", "2025-07-02", 100.0, 101.0, 101.0, 50_000_000),
    ])
    strat = FixedDecisionStrategy(plan={
        date(2025, 7, 1): [("AAPL", 0.50)],   # 50% target violates 5% cap
    })

    result = simulate(
        strategy=strat,
        universe=["AAPL"],
        prices=prices,
        sector_map={"AAPL": "Technology"},
        window_name="train",
        start_date=date(2025, 7, 1),
        end_date=date(2025, 7, 2),
        initial_cash=100_000.0,
    )

    # Rejected order means no trade; total_return ~ 0
    assert result.num_trades == 0


def test_avg_holding_days_includes_open_positions():
    """Open positions at end-of-backtest must contribute to avg_holding_days.

    Without the fix the simulator only counts closed positions; a buy-and-hold
    that never sells would report avg_holding_days=0. After the fix it counts
    the duration of positions still open when the backtest ends.
    """
    # 10 trading days, buy AAA on day 0 and never sell.
    start = date(2025, 7, 1)
    dates = [start + __import__("datetime").timedelta(days=i) for i in range(10)]
    rows = [
        ("AAA", d, 100.0, 100.0, 100.0, 1_000_000)
        for d in dates
    ]
    prices = _build_price_df(rows)

    # Strategy buys 4% on day 0; never issues a sell.
    strat = FixedDecisionStrategy(plan={
        dates[0]: [("AAA", 0.04)],
    })

    result = simulate(
        strategy=strat,
        universe=["AAA"],
        prices=prices,
        sector_map={"AAA": "Technology"},
        window_name="train",
        start_date=dates[0],
        end_date=dates[-1],
        initial_cash=100_000.0,
        slippage_model=SlippageModel(base_bps=0.0, illiquidity_penalty_bps=0.0),
    )

    # Position opens on day 1 (fill at day-1 open), end_date is day 9.
    # Entry date recorded as day 1; holding = end_date - entry_date >= 8 days.
    assert result.avg_holding_days >= 5, (
        f"Expected avg_holding_days >= 5 for open position held ~8 days, "
        f"got {result.avg_holding_days}"
    )


def test_simulator_hand_computed_three_trade_pnl():
    """The most important test in the codebase. A 3-trade scenario whose P&L we compute by hand.

    Scenario:
    - Day 1: open=100. Strategy says buy 4% AAPL. Executes day 2 open at $100 (no slippage in test).
      Buys 40 shares (4% of 100k = $4000, $4000/$100 = 40 shares). Cash left: $96000.
    - Day 2: position is 40 * day-2-close ($110) = $4400. Total equity = $96000 + $4400 = $100400.
      Daily return = +0.4%
    - Day 3: strategy says SELL (target_weight=0). Sells at day 3 open of $115.
      Realized: 40 * $115 = $4600. Cash now $96000 + $4600 = $100600. Total return = +0.6%.
    """
    prices = _build_price_df([
        ("AAPL", "2025-07-01", 100.0, 100.0, 100.0, 50_000_000),
        ("AAPL", "2025-07-02", 100.0, 110.0, 110.0, 50_000_000),
        ("AAPL", "2025-07-03", 115.0, 115.0, 115.0, 50_000_000),
    ])
    strat = FixedDecisionStrategy(plan={
        date(2025, 7, 1): [("AAPL", 0.04)],   # buy 4%
        date(2025, 7, 2): [("AAPL", 0.0)],     # close out
    })

    result = simulate(
        strategy=strat,
        universe=["AAPL"],
        prices=prices,
        sector_map={"AAPL": "Technology"},
        window_name="train",
        start_date=date(2025, 7, 1),
        end_date=date(2025, 7, 3),
        initial_cash=100_000.0,
        slippage_model=SlippageModel(base_bps=0.0, illiquidity_penalty_bps=0.0),
    )

    # Final equity should be close to $100600. Total return ~0.6%.
    assert result.total_return == pytest.approx(0.006, abs=1e-3)
    assert result.num_trades == 2  # one buy, one sell


def test_simulate_membership_drops_not_yet_added_ticker():
    """Point-in-time: with membership, a buy of a name added AFTER the decision
    day is dropped (no hindsight selection); an already-present name still
    trades. Features still score over the full universe (training parity) — only
    the tradable SELECTION is restricted."""
    prices = _build_price_df([
        ("AAPL", "2026-05-01", 100.0, 100.0, 100.0, 50_000_000),
        ("AAPL", "2026-05-02", 100.0, 100.0, 100.0, 50_000_000),
        ("NEW", "2026-05-01", 50.0, 50.0, 50.0, 50_000_000),
        ("NEW", "2026-05-02", 50.0, 50.0, 50.0, 50_000_000),
    ])
    strat = FixedDecisionStrategy({date(2026, 5, 1): [("AAPL", 0.04), ("NEW", 0.04)]})
    membership = {"AAPL": date(2026, 1, 1), "NEW": date(2026, 6, 10)}  # NEW added later
    result = simulate(
        strategy=strat, universe=["AAPL", "NEW"], prices=prices,
        sector_map={"AAPL": "Tech", "NEW": "Tech"},
        window_name="val", start_date=date(2026, 5, 1), end_date=date(2026, 5, 2),
        membership=membership,
    )
    bought = {t["ticker"] for t in result.trades if t.get("action") == "buy"}
    assert "NEW" not in bought, "must not buy a name added after the decision day"
    assert "AAPL" in bought


def test_compute_adv_dollars_excludes_fill_day_no_lookahead():
    """ADV for a fill at day d's OPEN must use data STRICTLY BEFORE d — day d's
    full-day volume is not known at the open. Including it (date <= d) leaks
    fill-day volume (look-ahead), inflating ADV and understating slippage on
    high-volume days."""
    from sma.backtest.simulator import _compute_adv_dollars

    df = _build_price_df([
        ("AAPL", date(2026, 1, 2), 100, 100, 100, 1_000),
        ("AAPL", date(2026, 1, 5), 100, 100, 100, 1_000),
        ("AAPL", date(2026, 1, 6), 100, 100, 100, 1_000),
        ("AAPL", date(2026, 1, 7), 100, 100, 100, 1_000),
        ("AAPL", date(2026, 1, 8), 100, 100, 100, 1_000),
        # Fill day: a huge volume spike that must NOT enter the ADV.
        ("AAPL", date(2026, 1, 9), 100, 100, 100, 1_000_000),
    ])
    adv = _compute_adv_dollars(df, "AAPL", date(2026, 1, 9))
    assert adv == 100 * 1_000  # mean over the 5 PRIOR days only, not the spike


def test_stop_loss_exit_applies_slippage():
    """A stop-loss exit is a market sell — it must incur slippage like any other
    fill, not execute frictionlessly at the open (which overstated the stop's
    backtested value)."""
    from sma.risk.rails import RiskRails
    prices = _build_price_df([
        ("AAA", "2025-07-01", 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", "2025-07-02", 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", "2025-07-03", 80.0, 80.0, 80.0, 50_000_000),  # -20% open triggers stop
    ])
    strat = FixedDecisionStrategy(plan={
        date(2025, 7, 1): [("AAA", 0.5)],
        date(2025, 7, 2): [("AAA", 0.5)],
    })
    rails = RiskRails(
        max_position_pct=0.6, max_sector_pct=0.9, stop_loss_pct=0.10,
        cash_floor_pct=0.0, max_drawdown_pct=1.0,
    )
    result = simulate(
        strategy=strat, universe=["AAA"], prices=prices,
        sector_map={"AAA": "Technology"}, window_name="train",
        start_date=date(2025, 7, 1), end_date=date(2025, 7, 3),
        initial_cash=100_000.0,
        slippage_model=SlippageModel(base_bps=100.0, illiquidity_penalty_bps=0.0),
        rails=rails,
    )
    stop_trades = [t for t in result.trades if t["action"] == "stop_loss"]
    assert len(stop_trades) == 1
    assert stop_trades[0]["price"] < 80.0  # sell slippage pushes exit below the open


def test_fills_are_split_consistent_with_adj_close_marks():
    """A 2:1 split between buy and mark must NOT fake a -50% loss (audit
    simulator.py:176): the sim marks at adj_close but used to FILL at raw
    open, so shares bought pre-split never doubled and the position halved.
    Fills must execute in ADJUSTED space: open * adj_close/close.

    AAA trades flat at raw $100, splits 2:1 before the last day (raw $50).
    Backward-adjusted, adj_close = 50 throughout. Buying on day 1 and holding
    must produce ~zero return, not -25% of equity.
    """
    prices = _build_price_df([
        ("AAA", date(2025, 1, 6), 100.0, 100.0, 50.0, 1_000_000),
        ("AAA", date(2025, 1, 7), 100.0, 100.0, 50.0, 1_000_000),  # buy fills here
        ("AAA", date(2025, 1, 8), 50.0, 50.0, 50.0, 2_000_000),    # post-split
    ])
    strat = FixedDecisionStrategy({date(2025, 1, 6): [("AAA", 0.04)]})
    result = simulate(
        strategy=strat,
        universe=["AAA"],
        prices=prices,
        sector_map={"AAA": "tech"},
        window_name="train",
        start_date=date(2025, 1, 6),
        end_date=date(2025, 1, 8),
        initial_cash=100_000.0,
    )
    assert result.num_trades >= 1, "buy must actually fill for this test to mean anything"
    assert result.total_return == pytest.approx(0.0, abs=1e-4), (
        "split between fill and mark created phantom P&L"
    )
    assert result.max_drawdown == pytest.approx(0.0, abs=1e-4)


def test_sector_rotation_buy_allowed_when_dropped_holding_frees_room():
    """Codex module review (2026-06-11 HIGH, sim-vs-live parity): live
    risk.apply() frees sector room for held names the model DROPPED (their
    force-sell lands the same batch) — the simulator counted them, rejected
    the rotation buy that live accepts, force-sold anyway, and ended in cash.
    Same-sector rotation must work: drop A (28% tech), buy B (5% tech) under
    a 30% sector cap."""
    from sma.risk.rails import RiskRails

    days = [date(2025, 1, 6), date(2025, 1, 7), date(2025, 1, 8), date(2025, 1, 9)]
    rows = []
    for t in ("AAA", "BBB"):
        for dd in days:
            rows.append((t, dd, 100.0, 100.0, 100.0, 5_000_000))
    prices = _build_price_df(rows)
    strat = FixedDecisionStrategy({
        days[0]: [("AAA", 0.28)],
        days[1]: [("BBB", 0.05)],   # AAA dropped → rotation into BBB
        # no further decisions: the rotation must complete from THIS batch —
        # pre-fix it only completed a day late via a re-decision (lag, drag)
    })
    rails = RiskRails(
        stop_loss_pct=0.0, max_position_pct=0.30, max_sector_pct=0.30,
        min_hold_days=0, rebalance_dead_zone_pct=0.10,
    )
    result = simulate(
        strategy=strat, universe=["AAA", "BBB"], prices=prices,
        sector_map={"AAA": "tech", "BBB": "tech"},
        window_name="train", start_date=days[0], end_date=days[-1],
        initial_cash=100_000.0, rails=rails,
    )
    # AAA buy + AAA force-sell + BBB buy. Pre-fix the BBB buy was rejected
    # (AAA still counted against the tech cap) → num_trades == 2.
    assert result.num_trades == 3, "rotation buy must fill once AAA's room is freed"


def test_sector_room_not_freed_in_sim_when_min_hold_protects_exit():
    """Mirror of the live-side reservation: a min-hold-protected dropped name
    cannot actually exit this batch, so its sector room must stay counted."""
    from sma.risk.rails import RiskRails

    days = [date(2025, 1, 6), date(2025, 1, 7), date(2025, 1, 8)]
    rows = []
    for t in ("AAA", "BBB"):
        for dd in days:
            rows.append((t, dd, 100.0, 100.0, 100.0, 5_000_000))
    prices = _build_price_df(rows)
    strat = FixedDecisionStrategy({
        days[0]: [("AAA", 0.28)],
        days[1]: [("BBB", 0.05)],  # AAA dropped next day — but min_hold=3 protects it
    })
    rails = RiskRails(
        stop_loss_pct=0.0, max_position_pct=0.30, max_sector_pct=0.30,
        min_hold_days=3, rebalance_dead_zone_pct=0.10,
    )
    result = simulate(
        strategy=strat, universe=["AAA", "BBB"], prices=prices,
        sector_map={"AAA": "tech", "BBB": "tech"},
        window_name="train", start_date=days[0], end_date=days[-1],
        initial_cash=100_000.0, rails=rails,
    )
    # AAA buy only: its exit is min-hold-vetoed, so room is NOT freed and the
    # BBB buy must stay rejected (28% + 5% > 30% cap).
    assert result.num_trades == 1


def test_rotation_buy_funded_by_force_sell_under_cash_floor():
    """Codex parity finding (2026-06-16): live translate computes ALL sells
    first (running_cash = cash + sell_proceeds) THEN gates buys on the cash
    floor; the simulator gated buys BEFORE force-sell proceeds, so a rotation
    that live funds was blocked. Hold A at ~full deployment, drop A + buy B:
    B must fill (funded by A's force-sale), respecting the 5% cash floor."""
    from sma.risk.rails import RiskRails

    days = [date(2025, 1, 6), date(2025, 1, 7), date(2025, 1, 8)]
    rows = []
    for t in ("AAA", "BBB"):
        for dd in days:
            rows.append((t, dd, 100.0, 100.0, 100.0, 5_000_000))
    prices = _build_price_df(rows)
    # day 0: buy AAA at 95% (near-fully deployed). day 1: drop AAA, buy BBB 95%.
    strat = FixedDecisionStrategy({days[0]: [("AAA", 0.95)], days[1]: [("BBB", 0.95)]})
    rails = RiskRails(
        stop_loss_pct=0.0, max_position_pct=1.0, max_sector_pct=1.0,
        cash_floor_pct=0.05, min_hold_days=0, rebalance_dead_zone_pct=0.0,
    )
    result = simulate(
        strategy=strat, universe=["AAA", "BBB"], prices=prices,
        sector_map={"AAA": "tech", "BBB": "tech"},
        window_name="train", start_date=days[0], end_date=days[-1],
        initial_cash=100_000.0, rails=rails,
    )
    actions = [(t["date"], t["ticker"], t["action"]) for t in result.trades]
    # the rotation BUY of BBB on day 1 must occur (funded by AAA's force-sale)
    assert any(tk == "BBB" and a in ("buy",) for (_, tk, a) in actions), (
        f"BBB rotation buy was blocked by the cash floor (sim/live parity); trades={actions}"
    )


# ---------------------------------------------------------------------------
# Price-based exits (trailing stop + take-profit) — 2026-07-01.
# All three exits default OFF; these assert they fire correctly when enabled
# and stay inert at their neutral defaults.
# ---------------------------------------------------------------------------
_NO_SLIP = SlippageModel(base_bps=0.0, illiquidity_penalty_bps=0.0)


def _keep_plan(ticker: str, days: list[date], weight: float = 0.5) -> "FixedDecisionStrategy":
    """Strategy that keeps `ticker` every day (isolates exits from force-sell)."""
    return FixedDecisionStrategy(plan={d: [(ticker, weight)] for d in days})


def test_trailing_stop_fires_on_peak_then_drop():
    from sma.risk.rails import RiskRails
    days = [date(2025, 7, d) for d in (1, 2, 3, 4)]
    # buy fills at 7/2 open (100); peak rises to 120 on 7/3; 7/4 opens at 100,
    # which is >10% below the 120 peak → trailing stop fires.
    prices = _build_price_df([
        ("AAA", days[0], 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", days[1], 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", days[2], 120.0, 120.0, 120.0, 50_000_000),
        ("AAA", days[3], 100.0, 100.0, 100.0, 50_000_000),
    ])
    rails = RiskRails(
        max_position_pct=0.6, max_sector_pct=0.9, stop_loss_pct=0.0,
        take_profit_pct=0.0, trailing_stop_pct=0.10, cash_floor_pct=0.0,
        max_drawdown_pct=1.0, min_hold_days=0,
    )
    result = simulate(
        strategy=_keep_plan("AAA", days), universe=["AAA"], prices=prices,
        sector_map={"AAA": "tech"}, window_name="train",
        start_date=days[0], end_date=days[3], initial_cash=100_000.0,
        slippage_model=_NO_SLIP, rails=rails,
    )
    actions = [(t["ticker"], t["action"]) for t in result.trades]
    assert ("AAA", "trailing_stop") in actions, f"trailing stop never fired; {actions}"


def test_trailing_stop_does_not_fire_on_steady_climb():
    from sma.risk.rails import RiskRails
    days = [date(2025, 7, d) for d in (1, 2, 3, 4, 5)]
    prices = _build_price_df([
        ("AAA", days[0], 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", days[1], 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", days[2], 110.0, 110.0, 110.0, 50_000_000),
        ("AAA", days[3], 120.0, 120.0, 120.0, 50_000_000),
        ("AAA", days[4], 130.0, 130.0, 130.0, 50_000_000),
    ])
    rails = RiskRails(
        max_position_pct=0.6, max_sector_pct=0.9, stop_loss_pct=0.0,
        take_profit_pct=0.0, trailing_stop_pct=0.10, cash_floor_pct=0.0,
        max_drawdown_pct=1.0, min_hold_days=0,
    )
    result = simulate(
        strategy=_keep_plan("AAA", days), universe=["AAA"], prices=prices,
        sector_map={"AAA": "tech"}, window_name="train",
        start_date=days[0], end_date=days[4], initial_cash=100_000.0,
        slippage_model=_NO_SLIP, rails=rails,
    )
    actions = [t["action"] for t in result.trades]
    assert "trailing_stop" not in actions, f"trailing stop fired on a monotone climb; {actions}"


def test_take_profit_fires_at_threshold():
    from sma.risk.rails import RiskRails
    days = [date(2025, 7, d) for d in (1, 2, 3)]
    # entry 100 on 7/2; 7/3 opens at 125 (+25%) >= +20% take-profit → fire.
    prices = _build_price_df([
        ("AAA", days[0], 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", days[1], 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", days[2], 125.0, 125.0, 125.0, 50_000_000),
    ])
    rails = RiskRails(
        max_position_pct=0.6, max_sector_pct=0.9, stop_loss_pct=0.0,
        take_profit_pct=0.20, trailing_stop_pct=0.0, cash_floor_pct=0.0,
        max_drawdown_pct=1.0, min_hold_days=0,
    )
    result = simulate(
        strategy=_keep_plan("AAA", days), universe=["AAA"], prices=prices,
        sector_map={"AAA": "tech"}, window_name="train",
        start_date=days[0], end_date=days[2], initial_cash=100_000.0,
        slippage_model=_NO_SLIP, rails=rails,
    )
    actions = [(t["ticker"], t["action"]) for t in result.trades]
    assert ("AAA", "take_profit") in actions, f"take-profit never fired; {actions}"


def test_take_profit_does_not_fire_below_threshold():
    from sma.risk.rails import RiskRails
    days = [date(2025, 7, d) for d in (1, 2, 3)]
    prices = _build_price_df([
        ("AAA", days[0], 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", days[1], 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", days[2], 115.0, 115.0, 115.0, 50_000_000),  # +15% < +20%
    ])
    rails = RiskRails(
        max_position_pct=0.6, max_sector_pct=0.9, stop_loss_pct=0.0,
        take_profit_pct=0.20, trailing_stop_pct=0.0, cash_floor_pct=0.0,
        max_drawdown_pct=1.0, min_hold_days=0,
    )
    result = simulate(
        strategy=_keep_plan("AAA", days), universe=["AAA"], prices=prices,
        sector_map={"AAA": "tech"}, window_name="train",
        start_date=days[0], end_date=days[2], initial_cash=100_000.0,
        slippage_model=_NO_SLIP, rails=rails,
    )
    actions = [t["action"] for t in result.trades]
    assert "take_profit" not in actions, f"take-profit fired below threshold; {actions}"


def test_new_price_exits_are_noops_at_default():
    """Peak-then-crash path: the NEW knobs (trailing_stop_pct, take_profit_pct)
    default to 0 and must stay inert. The legacy fixed stop_loss_pct is set to 0
    here too so this isolates the new capability (its own default is exercised
    elsewhere)."""
    from sma.risk.rails import RiskRails
    days = [date(2025, 7, d) for d in (1, 2, 3, 4)]
    prices = _build_price_df([
        ("AAA", days[0], 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", days[1], 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", days[2], 150.0, 150.0, 150.0, 50_000_000),
        ("AAA", days[3], 60.0, 60.0, 60.0, 50_000_000),   # -60% from peak
    ])
    rails = RiskRails(  # new exits at their default 0; legacy fixed stop off too
        max_position_pct=0.6, max_sector_pct=0.9, stop_loss_pct=0.0,
        cash_floor_pct=0.0, max_drawdown_pct=1.0, min_hold_days=0,
    )
    assert rails.trailing_stop_pct == 0.0 and rails.take_profit_pct == 0.0
    result = simulate(
        strategy=_keep_plan("AAA", days), universe=["AAA"], prices=prices,
        sector_map={"AAA": "tech"}, window_name="train",
        start_date=days[0], end_date=days[3], initial_cash=100_000.0,
        slippage_model=_NO_SLIP, rails=rails,
    )
    actions = {t["action"] for t in result.trades}
    assert actions.isdisjoint({"trailing_stop", "take_profit"}), (
        f"a NEW price exit fired at its default 0; {actions}"
    )


def test_trailing_stop_exit_applies_slippage():
    """A trailing-stop exit is a market sell — it incurs sell slippage."""
    from sma.risk.rails import RiskRails
    days = [date(2025, 7, d) for d in (1, 2, 3, 4)]
    prices = _build_price_df([
        ("AAA", days[0], 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", days[1], 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", days[2], 120.0, 120.0, 120.0, 50_000_000),
        ("AAA", days[3], 100.0, 100.0, 100.0, 50_000_000),
    ])
    rails = RiskRails(
        max_position_pct=0.6, max_sector_pct=0.9, stop_loss_pct=0.0,
        trailing_stop_pct=0.10, cash_floor_pct=0.0, max_drawdown_pct=1.0,
        min_hold_days=0,
    )
    result = simulate(
        strategy=_keep_plan("AAA", days), universe=["AAA"], prices=prices,
        sector_map={"AAA": "tech"}, window_name="train",
        start_date=days[0], end_date=days[3], initial_cash=100_000.0,
        slippage_model=SlippageModel(base_bps=100.0, illiquidity_penalty_bps=0.0),
        rails=rails,
    )
    ts = [t for t in result.trades if t["action"] == "trailing_stop"]
    assert len(ts) == 1
    assert ts[0]["price"] < 100.0  # sell slippage pushes the exit below the open


def test_price_exit_not_re_bought_same_day_but_allowed_later():
    """Bug 1: a Step-0 price exit at the open must NOT be undone by yesterday's
    still-standing target re-buying the SAME ticker at the SAME open (a no-op
    round-trip that made take-profit/trailing backtests meaningless). The name
    stays flat that day and may be re-bought on a LATER day if still wanted."""
    from sma.risk.rails import RiskRails
    days = [date(2025, 7, d) for d in (1, 2, 3, 4)]
    prices = _build_price_df([
        ("AAA", days[0], 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", days[1], 100.0, 100.0, 100.0, 50_000_000),  # buy fills @100
        ("AAA", days[2], 130.0, 130.0, 130.0, 50_000_000),  # +30% → take-profit
        ("AAA", days[3], 100.0, 100.0, 100.0, 50_000_000),  # re-entry allowed
    ])
    rails = RiskRails(
        max_position_pct=0.6, max_sector_pct=0.9, stop_loss_pct=0.0,
        take_profit_pct=0.20, trailing_stop_pct=0.0, cash_floor_pct=0.0,
        max_drawdown_pct=1.0, min_hold_days=0, rebalance_dead_zone_pct=0.0,
    )
    result = simulate(
        strategy=_keep_plan("AAA", days), universe=["AAA"], prices=prices,
        sector_map={"AAA": "tech"}, window_name="train",
        start_date=days[0], end_date=days[3], initial_cash=100_000.0,
        slippage_model=_NO_SLIP, rails=rails,
    )
    by_day: dict = {}
    for t in result.trades:
        by_day.setdefault(t["date"], []).append((t["action"], t["ticker"]))
    # 7/3: exactly the take-profit sell — and NO same-day re-buy (position flat).
    assert ("take_profit", "AAA") in by_day.get(days[2], [])
    assert ("buy", "AAA") not in by_day.get(days[2], []), (
        f"AAA was re-bought the same day it took profit; {by_day.get(days[2])}"
    )
    # 7/4: re-entry on a LATER day is allowed (model still wants it).
    assert ("buy", "AAA") in by_day.get(days[3], []), (
        f"AAA was never re-entered on a later day; {by_day.get(days[3])}"
    )


def test_price_exit_reentry_guard_is_inert_at_default():
    """At the default take_profit/trailing/stop knobs (0.0) NO ticker is exited in
    Step 0, so the same-day re-entry guard never engages: AAA enters once and is
    held — byte-identical to the pre-fix baseline (nothing is ever skipped)."""
    from sma.risk.rails import RiskRails
    days = [date(2025, 7, d) for d in (1, 2, 3, 4)]
    prices = _build_price_df([
        ("AAA", days[0], 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", days[1], 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", days[2], 130.0, 130.0, 130.0, 50_000_000),
        ("AAA", days[3], 100.0, 100.0, 100.0, 50_000_000),
    ])
    rails = RiskRails(  # all price exits at their default 0
        max_position_pct=0.6, max_sector_pct=0.9, stop_loss_pct=0.0,
        cash_floor_pct=0.0, max_drawdown_pct=1.0, min_hold_days=0,
    )
    assert rails.take_profit_pct == 0.0 and rails.trailing_stop_pct == 0.0
    result = simulate(
        strategy=_keep_plan("AAA", days), universe=["AAA"], prices=prices,
        sector_map={"AAA": "tech"}, window_name="train",
        start_date=days[0], end_date=days[3], initial_cash=100_000.0,
        slippage_model=_NO_SLIP, rails=rails,
    )
    actions = {t["action"] for t in result.trades}
    assert actions.isdisjoint({"take_profit", "trailing_stop", "stop_loss"})
    buys = [t for t in result.trades if t["action"] == "buy" and t["ticker"] == "AAA"]
    assert buys and buys[0]["date"] == days[1], (
        f"AAA should enter once on 7/2 with no exit-driven churn; {result.trades}"
    )


def test_trailing_stop_peak_survives_top_up():
    """Bug 2 (sim reference): adding to a winner KEEPS the older post-entry peak,
    so a later rollover still trips the trailing stop. This is the peak semantic
    live must match — the earliest buy of the continuous holding, not the top-up
    date. Entry @100 (7/2), peak 200 (7/2 close), top-up @190 (7/3), then 175
    (7/4) which is >10% below the 200 peak → trailing fires."""
    from sma.risk.rails import RiskRails
    days = [date(2025, 7, d) for d in (1, 2, 3, 4)]
    prices = _build_price_df([
        ("AAA", days[0], 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", days[1], 100.0, 200.0, 200.0, 50_000_000),  # entry @100; peak→200
        ("AAA", days[2], 190.0, 190.0, 190.0, 50_000_000),  # top-up @190; peak kept 200
        ("AAA", days[3], 175.0, 175.0, 175.0, 50_000_000),  # 175 <= 200*0.9 → fire
    ])
    strat = FixedDecisionStrategy(plan={
        days[0]: [("AAA", 0.4)],
        days[1]: [("AAA", 0.6)],   # raise the target → averages up on 7/3
        days[2]: [("AAA", 0.6)],
    })
    rails = RiskRails(
        max_position_pct=0.9, max_sector_pct=0.95, stop_loss_pct=0.0,
        take_profit_pct=0.0, trailing_stop_pct=0.10, cash_floor_pct=0.0,
        max_drawdown_pct=1.0, min_hold_days=0, rebalance_dead_zone_pct=0.0,
    )
    result = simulate(
        strategy=strat, universe=["AAA"], prices=prices,
        sector_map={"AAA": "tech"}, window_name="train",
        start_date=days[0], end_date=days[3], initial_cash=100_000.0,
        slippage_model=_NO_SLIP, rails=rails,
    )
    topups = [t for t in result.trades if t["action"] == "buy" and t["date"] == days[2]]
    assert topups, f"expected a top-up buy on 7/3; trades={result.trades}"
    actions = [(t["ticker"], t["action"]) for t in result.trades]
    assert ("AAA", "trailing_stop") in actions, (
        f"trailing stop did not fire after top-up (peak wrongly reset?); {result.trades}"
    )


def test_extra_cash_floor_by_date_blocks_buys_on_flagged_dates():
    """The dispersion de-risk seam (research 2026-07-20): an optional per-date
    extra cash floor. A date with floor=1.0 forces all-cash — the buy must be
    rejected that day and go through on an unflagged day. None = inert."""
    from sma.risk.rails import RiskRails
    prices = _build_price_df([
        ("AAA", "2025-07-01", 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", "2025-07-02", 100.0, 100.0, 100.0, 50_000_000),
        ("AAA", "2025-07-03", 100.0, 100.0, 100.0, 50_000_000),
    ])
    strat = FixedDecisionStrategy(plan={
        date(2025, 7, 1): [("AAA", 0.5)],
        date(2025, 7, 2): [("AAA", 0.5)],
    })
    rails = RiskRails(
        max_position_pct=0.6, max_sector_pct=0.9, stop_loss_pct=0.0,
        cash_floor_pct=0.0, max_drawdown_pct=1.0,
    )
    common = dict(
        strategy=strat, universe=["AAA"], prices=prices,
        sector_map={"AAA": "Technology"}, window_name="train",
        start_date=date(2025, 7, 1), end_date=date(2025, 7, 3),
        initial_cash=100_000.0,
        slippage_model=SlippageModel(base_bps=0.0, illiquidity_penalty_bps=0.0),
        rails=rails,
    )
    # Flagged on the day the 7/01 decision executes (fills 7/02 open): no buy.
    r_on = simulate(**common, extra_cash_floor_by_date={date(2025, 7, 2): 1.0})
    buys_on = [t for t in r_on.trades if t["action"] == "buy"
               and t["date"] == date(2025, 7, 2)]
    assert buys_on == [], f"buy executed through a 100% floor: {r_on.trades}"

    # Same sim without the floor: the buy happens (proves the flag caused it).
    r_off = simulate(**common)
    buys_off = [t for t in r_off.trades if t["action"] == "buy"
                and t["date"] == date(2025, 7, 2)]
    assert buys_off, "control: buy should execute without the extra floor"
