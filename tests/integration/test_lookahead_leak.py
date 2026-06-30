"""Adversarial tests proving the backtest simulator cannot be tricked into using
future data.

Three threat vectors:
1. asof_date mismatch: strategy returns a decision claiming a future asof_date.
2. visible_prices filter: strategy.decide only ever receives rows with date <= asof_date.
3. Trade execution timing: a buy on day D fills at day D+1 open, not day D close.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from sma.backtest.simulator import LookaheadLeakError, simulate
from sma.backtest.slippage import SlippageModel
from sma.backtest.strategies.base import StrategyDecision


def _prices(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ticker", "date", "open", "close", "adj_close", "volume"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# ---------------------------------------------------------------------------
# Test 1: future-date StrategyDecision is rejected
# ---------------------------------------------------------------------------

class _FutureDateStrategy:
    name = "future_date"

    def decide(self, asof_date, prices, fundamentals=None):
        return [
            StrategyDecision(
                asof_date=asof_date + timedelta(days=10),
                ticker="AAA",
                target_weight=0.5,
            )
        ]


def test_simulator_rejects_decision_with_future_asof_date():
    """Strategy returning a future asof_date decision must raise LookaheadLeakError."""
    prices = _prices([
        ("AAA", "2025-01-01", 10.0, 11.0, 11.0, 1_000_000),
        ("AAA", "2025-01-02", 11.0, 12.0, 12.0, 1_000_000),
        ("AAA", "2025-01-03", 12.0, 13.0, 13.0, 1_000_000),
        ("AAA", "2025-01-04", 13.0, 14.0, 14.0, 1_000_000),
        ("AAA", "2025-01-05", 14.0, 15.0, 15.0, 1_000_000),
    ])

    with pytest.raises(LookaheadLeakError):
        simulate(
            strategy=_FutureDateStrategy(),
            universe=["AAA"],
            prices=prices,
            sector_map={"AAA": "Technology"},
            window_name="train",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 5),
            initial_cash=10_000.0,
        )


# ---------------------------------------------------------------------------
# Test 2: visible_prices passed to strategy.decide never contains future rows
# ---------------------------------------------------------------------------

class _PriceAuditStrategy:
    """Records (asof_date, max_date_in_prices) for every call to decide."""
    name = "price_audit"

    def __init__(self):
        self.audit: list[tuple[date, date | None]] = []

    def decide(self, asof_date, prices, fundamentals=None):
        max_date = prices["date"].max() if not prices.empty else None
        self.audit.append((asof_date, max_date))
        return []


def test_strategy_only_sees_prices_up_to_asof_date():
    """The prices DataFrame passed to .decide must contain no row with date > asof_date."""
    prices = _prices([
        ("AAA", f"2025-01-{d:02d}", float(d), float(d), float(d), 1_000_000)
        for d in range(1, 11)
    ])

    strat = _PriceAuditStrategy()
    simulate(
        strategy=strat,
        universe=["AAA"],
        prices=prices,
        sector_map={"AAA": "Technology"},
        window_name="train",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 10),
        initial_cash=10_000.0,
    )

    assert strat.audit, "strategy was never called"
    for asof, max_seen in strat.audit:
        assert max_seen is None or max_seen <= asof, (
            f"On {asof}, strategy saw prices up to {max_seen} (lookahead leak)"
        )


# ---------------------------------------------------------------------------
# Test 3: trades execute at next-day open, not same-day close
# ---------------------------------------------------------------------------

class _BuyOnDayTwoStrategy:
    """On day 2, signals a 4% buy (under the default 5% position cap)."""
    name = "buy_on_day_two"

    def decide(self, asof_date, prices, fundamentals=None):
        if asof_date == date(2025, 1, 2):
            return [StrategyDecision(asof_date=asof_date, ticker="AAA", target_weight=0.04)]
        return []


def test_decision_made_on_d_executes_at_d_plus_1_open():
    """A buy decision made on day D fills at day D+1 open, not day D close.

    day 1: open=100, close=100
    day 2: open=100, close=100  -- decision made here; day-2 close is $100
    day 3: open=200, close=200  -- trade MUST fill here at $200

    The strategy targets 4% of $10,000 = $400. At the correct $200 fill:
      shares bought = 400 // 200 = 2 shares, cost = $400.
    At the incorrect $100 fill (day-2 close):
      shares bought = 400 // 100 = 4 shares, cost = $400.

    day-3 MTM: position = shares * $200 close.
      correct path: 2 * 200 = $400 => position unchanged, day-3 return ~0%.
      wrong path:   4 * 200 = $800 => position doubled, day-3 return ~+4%.

    So day-3 return ~0% proves the fill was at $200 (day-3 open).
    """
    prices = _prices([
        ("AAA", "2025-01-01", 100.0, 100.0, 100.0, 10_000_000),
        ("AAA", "2025-01-02", 100.0, 100.0, 100.0, 10_000_000),
        ("AAA", "2025-01-03", 200.0, 200.0, 200.0, 10_000_000),
    ])

    strat = _BuyOnDayTwoStrategy()
    result = simulate(
        strategy=strat,
        universe=["AAA"],
        prices=prices,
        sector_map={"AAA": "Technology"},
        window_name="train",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 3),
        initial_cash=10_000.0,
        slippage_model=SlippageModel(base_bps=0.0, illiquidity_penalty_bps=0.0),
    )

    assert result.num_trades >= 1, "expected at least one buy trade"
    day3_return = result.daily_returns[2]
    # 2 shares bought at $200, MTM at $200 close: no P&L on day 3 => ~0%.
    assert day3_return == pytest.approx(0.0, abs=0.005), (
        f"day-3 return was {day3_return:.4f}; expected ~0% "
        "(fill at $200 open == $200 close). "
        "A positive return means the fill used day-2 close ($100), "
        "giving 4 shares at $200 MTM."
    )
