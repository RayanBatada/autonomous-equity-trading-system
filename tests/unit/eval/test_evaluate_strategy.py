from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from sma.eval import evaluate_strategy


class _NoOpStrategy:
    name = "noop"
    def decide(self, asof_date, prices, fundamentals=None):
        return []


def _build_min_prices_df():
    return pd.DataFrame({
        "ticker": ["SPY", "SPY", "SPY"],
        "date": [date(2025, 7, 1), date(2025, 7, 2), date(2025, 7, 3)],
        "open": [500.0, 501.0, 502.0],
        "close": [501.0, 502.0, 503.0],
        "adj_close": [501.0, 502.0, 503.0],
        "volume": [1_000_000_000] * 3,
    })


def test_evaluate_strategy_returns_backtest_result(tmp_path):
    strat = _NoOpStrategy()
    with patch("sma.eval.evaluate_strategy._load_prices_for_window",
               return_value=_build_min_prices_df()):
        result = evaluate_strategy(
            strategy=strat,
            window="train",
            universe=["SPY"],
            membership="current",
        )
    assert result.strategy_name == "noop"
    assert result.window == "train"


def test_test_window_requires_promotion_flag():
    strat = _NoOpStrategy()
    loader = "sma.eval.evaluate_strategy._load_prices_for_window"
    with (
        patch(loader, return_value=_build_min_prices_df()),
        pytest.raises(PermissionError, match="test window"),
    ):
        evaluate_strategy(strategy=strat, window="test", universe=["SPY"], membership="current")


def test_test_window_works_with_promotion_flag():
    strat = _NoOpStrategy()
    with patch("sma.eval.evaluate_strategy._load_prices_for_window",
               return_value=_build_min_prices_df()):
        result = evaluate_strategy(
            strategy=strat, window="test", universe=["SPY"], membership="current",
            i_promise_this_is_a_promotion_decision=True,
        )
    assert result.window == "test"


def test_invalid_window_raises():
    with pytest.raises(ValueError):
        evaluate_strategy(strategy=_NoOpStrategy(), window="banana", universe=["SPY"],
            membership="current")  # type: ignore


def test_seed_propagates_to_result():
    strat = _NoOpStrategy()
    with patch("sma.eval.evaluate_strategy._load_prices_for_window",
               return_value=_build_min_prices_df()):
        result = evaluate_strategy(strategy=strat, window="train", membership="current",
                                   universe=["SPY"], seed=12345)
    assert result.seed == 12345


def test_evaluate_strategy_is_importable_from_package():
    """`from sma.eval import evaluate_strategy` must work for autoresearch."""
    from sma.eval import evaluate_strategy as func
    assert callable(func)


# --- membership is an explicit choice (audit evaluate_strategy.py:109) --------
# The old default (None = trade the full CURRENT universe over history) baked
# survivorship bias silently into every eval. The universe.yaml added-dates are
# vault-addition dates (all 2026-04+), so silently auto-loading them would
# instead zero out historical windows. Callers must now CHOOSE.


def test_membership_default_none_raises_with_guidance():
    strat = _NoOpStrategy()
    with (
        patch("sma.eval.evaluate_strategy._load_prices_for_window",
              return_value=_build_min_prices_df()),
        pytest.raises(ValueError, match="membership"),
    ):
        evaluate_strategy(strategy=strat, window="train", universe=["SPY"])


def test_membership_current_opts_into_survivorship_bias():
    strat = _NoOpStrategy()
    with patch("sma.eval.evaluate_strategy._load_prices_for_window",
               return_value=_build_min_prices_df()):
        result = evaluate_strategy(
            strategy=strat, window="train", universe=["SPY"],
            membership="current",
        )
    assert result.strategy_name == "noop"


def test_membership_map_is_passed_through_to_simulate():
    strat = _NoOpStrategy()
    member_map = {"SPY": date(2020, 1, 1)}
    with (
        patch("sma.eval.evaluate_strategy._load_prices_for_window",
              return_value=_build_min_prices_df()),
        patch("sma.eval.evaluate_strategy.simulate") as sim,
    ):
        evaluate_strategy(
            strategy=strat, window="train", universe=["SPY"],
            membership=member_map,
        )
    assert sim.call_args.kwargs["membership"] == member_map
