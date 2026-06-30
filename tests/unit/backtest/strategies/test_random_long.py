from datetime import date

import pandas as pd

from sma.backtest.strategies.random_long import RandomLongStrategy


def test_strategy_name():
    s = RandomLongStrategy(universe=["AAPL", "MSFT"], seed=42)
    assert s.name == "random_long"


def test_emits_decisions_each_day():
    s = RandomLongStrategy(universe=["AAPL", "MSFT", "GOOGL"], seed=42, num_picks=2)
    d1 = s.decide(asof_date=date(2025, 7, 1), prices=pd.DataFrame())
    d2 = s.decide(asof_date=date(2025, 7, 2), prices=pd.DataFrame())
    assert len(d1) == 2
    assert len(d2) == 2


def test_picks_are_within_universe():
    universe = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    s = RandomLongStrategy(universe=universe, seed=42, num_picks=2)
    decisions = s.decide(asof_date=date(2025, 7, 1), prices=pd.DataFrame())
    for d in decisions:
        assert d.ticker in universe


def test_picks_are_distinct_within_a_day():
    """K=3 picks from 5-ticker universe should be 3 distinct tickers, no duplicates."""
    s = RandomLongStrategy(
        universe=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
        seed=42,
        num_picks=3,
    )
    decisions = s.decide(asof_date=date(2025, 7, 1), prices=pd.DataFrame())
    tickers = [d.ticker for d in decisions]
    assert len(set(tickers)) == len(tickers)


def test_target_weights_are_within_position_cap():
    s = RandomLongStrategy(
        universe=["AAPL", "MSFT", "GOOGL"],
        seed=42,
        num_picks=2,
        max_weight_per_pick=0.05,
    )
    decisions = s.decide(asof_date=date(2025, 7, 1), prices=pd.DataFrame())
    for d in decisions:
        assert 0.0 <= d.target_weight <= 0.05


def test_same_seed_produces_identical_decisions():
    """Critical for autoresearch reproducibility."""
    s1 = RandomLongStrategy(universe=["AAPL", "MSFT", "GOOGL"], seed=123, num_picks=2)
    s2 = RandomLongStrategy(universe=["AAPL", "MSFT", "GOOGL"], seed=123, num_picks=2)
    d1 = s1.decide(asof_date=date(2025, 7, 1), prices=pd.DataFrame())
    d2 = s2.decide(asof_date=date(2025, 7, 1), prices=pd.DataFrame())
    assert [(d.ticker, d.target_weight) for d in d1] == [(d.ticker, d.target_weight) for d in d2]


def test_different_seeds_produce_different_decisions():
    s1 = RandomLongStrategy(universe=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                            seed=1, num_picks=2)
    s2 = RandomLongStrategy(universe=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                            seed=2, num_picks=2)
    d1 = s1.decide(asof_date=date(2025, 7, 1), prices=pd.DataFrame())
    d2 = s2.decide(asof_date=date(2025, 7, 1), prices=pd.DataFrame())
    # Should differ in at least one of (ticker set, weights)
    assert [(d.ticker, d.target_weight) for d in d1] != [(d.ticker, d.target_weight) for d in d2]


def test_num_picks_above_universe_size_caps_to_universe():
    s = RandomLongStrategy(universe=["AAPL", "MSFT"], seed=42, num_picks=10)
    decisions = s.decide(asof_date=date(2025, 7, 1), prices=pd.DataFrame())
    assert len(decisions) == 2  # capped to universe size
