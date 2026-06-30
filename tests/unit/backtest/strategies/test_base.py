from datetime import date

from sma.backtest.strategies.base import Strategy, StrategyDecision


def test_strategy_decision_constructs_and_is_frozen():
    d = StrategyDecision(
        asof_date=date(2025, 7, 1),
        ticker="AAPL",
        target_weight=0.05,
    )
    assert d.ticker == "AAPL"
    assert d.target_weight == 0.05
    import pytest
    with pytest.raises(Exception):
        d.target_weight = 0.99  # type: ignore


def test_strategy_protocol_accepts_a_conforming_class():
    """A class with the right shape satisfies the Strategy Protocol structurally."""
    class FakeStrat:
        name = "fake"
        def decide(self, asof_date, prices, fundamentals):
            return []

    s: Strategy = FakeStrat()  # mypy/type-checker would assert; runtime is structural
    assert s.name == "fake"
    assert s.decide(date(2025, 7, 1), None, None) == []
