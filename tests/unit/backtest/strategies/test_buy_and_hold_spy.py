from datetime import date

import pandas as pd

from sma.backtest.strategies.buy_and_hold_spy import BuyAndHoldSPYStrategy


def test_emits_one_decision_on_first_day():
    strat = BuyAndHoldSPYStrategy()
    decisions = strat.decide(
        asof_date=date(2025, 7, 1),
        prices=pd.DataFrame({
            "ticker": ["SPY"], "date": [date(2025, 7, 1)],
            "open": [500.0], "close": [500.0], "adj_close": [500.0], "volume": [1_000_000_000],
        }),
    )
    assert len(decisions) == 1
    assert decisions[0].ticker == "SPY"
    assert decisions[0].target_weight == 0.05  # 5% to match position cap
    assert decisions[0].asof_date == date(2025, 7, 1)


def test_emits_no_decisions_after_first_day():
    strat = BuyAndHoldSPYStrategy()
    # First call: emits buy
    strat.decide(asof_date=date(2025, 7, 1), prices=pd.DataFrame())
    # Second call: should emit nothing (already bought)
    decisions = strat.decide(asof_date=date(2025, 7, 2), prices=pd.DataFrame())
    assert decisions == []


def test_strategy_name():
    assert BuyAndHoldSPYStrategy().name == "buy_and_hold_spy"


def test_strategy_state_resets_per_instance():
    strat1 = BuyAndHoldSPYStrategy()
    strat2 = BuyAndHoldSPYStrategy()
    strat1.decide(asof_date=date(2025, 7, 1), prices=pd.DataFrame())
    # New instance should still emit on its first day
    decisions = strat2.decide(asof_date=date(2025, 7, 1), prices=pd.DataFrame())
    assert len(decisions) == 1


def test_buy_and_hold_spy_accepts_custom_target_weight():
    strat = BuyAndHoldSPYStrategy(target_weight=1.0)
    decisions = strat.decide(
        asof_date=date(2025, 7, 1),
        prices=pd.DataFrame({
            "ticker": ["SPY"], "date": [date(2025, 7, 1)],
            "open": [500.0], "close": [500.0], "adj_close": [500.0], "volume": [1_000_000_000],
        }),
    )
    assert len(decisions) == 1
    assert decisions[0].target_weight == 1.0
