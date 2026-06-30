from datetime import date

import pandas as pd

from sma.backtest.strategies.equal_weight import EqualWeightStrategy


def test_strategy_name():
    assert EqualWeightStrategy(universe=["AAPL", "MSFT"]).name == "equal_weight"


def test_emits_one_decision_per_ticker_on_first_day():
    strat = EqualWeightStrategy(universe=["AAPL", "MSFT", "GOOGL", "AMZN"])
    decisions = strat.decide(asof_date=date(2025, 7, 1), prices=pd.DataFrame())
    assert len(decisions) == 4
    tickers = sorted(d.ticker for d in decisions)
    assert tickers == ["AAPL", "AMZN", "GOOGL", "MSFT"]


def test_target_weights_sum_to_full_allocation():
    """Each weight = 1/N. Sum across N tickers = 1.0 (or close due to float)."""
    strat = EqualWeightStrategy(universe=["AAPL", "MSFT", "GOOGL", "AMZN"])
    decisions = strat.decide(asof_date=date(2025, 7, 1), prices=pd.DataFrame())
    total = sum(d.target_weight for d in decisions)
    assert abs(total - 1.0) < 1e-9


def test_individual_weight_is_one_over_n():
    strat = EqualWeightStrategy(universe=["AAPL", "MSFT", "GOOGL", "AMZN"])
    decisions = strat.decide(asof_date=date(2025, 7, 1), prices=pd.DataFrame())
    for d in decisions:
        assert d.target_weight == 0.25  # 1/4


def test_emits_no_decisions_after_first_day():
    strat = EqualWeightStrategy(universe=["AAPL", "MSFT"])
    strat.decide(asof_date=date(2025, 7, 1), prices=pd.DataFrame())
    decisions = strat.decide(asof_date=date(2025, 7, 2), prices=pd.DataFrame())
    assert decisions == []


def test_empty_universe_raises():
    import pytest
    with pytest.raises(ValueError):
        EqualWeightStrategy(universe=[])


def test_decisions_use_asof_date():
    strat = EqualWeightStrategy(universe=["AAPL"])
    decisions = strat.decide(asof_date=date(2025, 7, 5), prices=pd.DataFrame())
    assert decisions[0].asof_date == date(2025, 7, 5)
