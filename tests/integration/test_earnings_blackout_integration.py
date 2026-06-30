"""Integration tests: earnings blackout wired into the simulator."""

from datetime import date, timedelta

import pandas as pd

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


def test_simulator_skips_buy_during_blackout():
    """Strategy targets AAA on day 0 (fills day 1). Earnings is day 1 + 2 days.
    Day 1 is within the 3-day blackout window, so the buy is rejected.
    """
    day0 = date(2025, 7, 1)
    day1 = date(2025, 7, 2)
    day2 = date(2025, 7, 3)
    day3 = date(2025, 7, 4)
    day4 = date(2025, 7, 5)

    prices = _build_price_df([
        ("AAA", day0.isoformat(), 100.0, 101.0, 101.0, 5_000_000),
        ("AAA", day1.isoformat(), 101.0, 102.0, 102.0, 5_000_000),
        ("AAA", day2.isoformat(), 102.0, 103.0, 103.0, 5_000_000),
        ("AAA", day3.isoformat(), 103.0, 104.0, 104.0, 5_000_000),
        ("AAA", day4.isoformat(), 104.0, 105.0, 105.0, 5_000_000),
    ])

    # Earnings is 2 days after the fill date (day1). Fill date + 2 = day3.
    # day1 is within [day1, day1 + 3] window around earnings on day3.
    earnings_date = day1 + timedelta(days=2)
    earnings_blackouts = {"AAA": [earnings_date]}

    strat = FixedDecisionStrategy(plan={day0: [("AAA", 0.04)]})

    result = simulate(
        strategy=strat,
        universe=["AAA"],
        prices=prices,
        sector_map={"AAA": "Information Technology"},
        earnings_blackouts=earnings_blackouts,
        window_name="train",
        start_date=day0,
        end_date=day4,
        initial_cash=100_000.0,
        slippage_model=SlippageModel(base_bps=0.0, illiquidity_penalty_bps=0.0),
    )

    assert result.num_trades == 0, (
        f"Expected buy to be blocked by earnings blackout, got {result.num_trades} trades"
    )


def test_simulator_allows_sell_during_blackout():
    """Buy on day 0 (fills day 1, no blackout then). Earnings day 5+2.
    Sell on day 7 during blackout. Sell should still fill.
    """
    base = date(2025, 7, 1)
    days = [base + timedelta(days=i) for i in range(10)]

    rows = [
        ("AAA", d.isoformat(), 100.0 + i, 101.0 + i, 101.0 + i, 5_000_000)
        for i, d in enumerate(days)
    ]
    prices = _build_price_df(rows)

    # Earnings on day 7 (days[7]). Blackout covers days[4] through days[7].
    # Buy decision on day 0 fills on day 1 -- no blackout on day 1.
    # Sell decision on day 6 fills on day 7 -- inside blackout, but sells are allowed.
    earnings_date = days[7]
    earnings_blackouts = {"AAA": [earnings_date]}

    plan = {
        days[0]: [("AAA", 0.04)],   # buy
        days[6]: [("AAA", 0.0)],    # sell (target 0 = exit)
    }
    strat = FixedDecisionStrategy(plan=plan)

    result = simulate(
        strategy=strat,
        universe=["AAA"],
        prices=prices,
        sector_map={"AAA": "Information Technology"},
        earnings_blackouts=earnings_blackouts,
        window_name="train",
        start_date=days[0],
        end_date=days[-1],
        initial_cash=100_000.0,
        slippage_model=SlippageModel(base_bps=0.0, illiquidity_penalty_bps=0.0),
    )

    # At least the buy fills; the sell during blackout should also fill.
    assert result.num_trades >= 2, (
        f"Expected buy + sell (sell allowed during blackout), got {result.num_trades} trades"
    )
    trade_actions = [t["action"] for t in result.trades]
    assert "buy" in trade_actions, "Buy should have filled on day 1 (outside blackout)"
    assert "sell" in trade_actions, "Sell should have filled on day 7 (sells bypass blackout)"


def test_simulator_no_blackout_when_earnings_blackouts_is_none():
    """Passing earnings_blackouts=None (default) should not affect buy behavior."""
    day0 = date(2025, 7, 1)
    day1 = date(2025, 7, 2)
    day2 = date(2025, 7, 3)

    prices = _build_price_df([
        ("AAA", day0.isoformat(), 100.0, 101.0, 101.0, 5_000_000),
        ("AAA", day1.isoformat(), 101.0, 102.0, 102.0, 5_000_000),
        ("AAA", day2.isoformat(), 102.0, 103.0, 103.0, 5_000_000),
    ])

    strat = FixedDecisionStrategy(plan={day0: [("AAA", 0.04)]})

    result = simulate(
        strategy=strat,
        universe=["AAA"],
        prices=prices,
        sector_map={"AAA": "Information Technology"},
        earnings_blackouts=None,
        window_name="train",
        start_date=day0,
        end_date=day2,
        initial_cash=100_000.0,
        slippage_model=SlippageModel(base_bps=0.0, illiquidity_penalty_bps=0.0),
    )

    assert result.num_trades >= 1, "Buy should fill when earnings_blackouts=None"
