"""Integration tests for sma.backtest CLI entry point."""

import importlib
import sys
from datetime import date

import pandas as pd
import pytest
from click.testing import CliRunner


def _get_ev_mod():
    importlib.import_module("sma.eval.evaluate_strategy")
    return sys.modules["sma.eval.evaluate_strategy"]


def _make_prices(tickers: list[str], n_days: int = 100) -> pd.DataFrame:
    """Minimal OHLCV DataFrame for the simulator."""
    import datetime as dt

    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    base = date(2023, 1, 3)
    for i in range(n_days):
        d = base + dt.timedelta(days=i)
        for t in tickers:
            close = 100.0 * (1 + rng.normal(0, 0.01))
            rows.append({
                "ticker": t,
                "date": d,
                "open": close * 0.999,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "adj_close": close,
                "volume": 1_000_000,
            })
    return pd.DataFrame(rows)


def _make_backtest_result(
    strategy_name: str,
    window: str,
    sharpe: float,
    daily_returns: list[float] | None = None,
    num_trades: int = 50,
) -> object:
    from sma.backtest.result import BacktestResult

    dr = daily_returns if daily_returns is not None else [0.001] * 252
    return BacktestResult(
        strategy_name=strategy_name,
        window=window,  # type: ignore[arg-type]
        start_date=date(2023, 1, 1),
        end_date=date(2025, 6, 30),
        sharpe=sharpe,
        sortino=sharpe * 1.2,
        calmar=sharpe * 0.5,
        max_drawdown=-0.10,
        total_return=0.20,
        annualized_return=0.08,
        hit_rate=0.55,
        num_trades=num_trades,
        avg_holding_days=5.0,
        daily_returns=dr,
        monthly_returns=[0.01] * 12,
        code_commit="abc12345",
        data_run_id=1,
        seed=None,
    )


# ---------------------------------------------------------------------------
# Test 1
# ---------------------------------------------------------------------------


def test_evaluate_runs_buy_and_hold_spy_on_train(monkeypatch: pytest.MonkeyPatch):
    from sma.backtest.__main__ import cli

    universe = ["SPY", "AAPL", "MSFT"]
    prices = _make_prices(universe, n_days=100)
    monkeypatch.setattr(
        _get_ev_mod(), "_load_prices_for_window", lambda db_path, u, start, end: prices
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate", "--strategy", "buy_and_hold_spy", "--window", "train"])
    assert result.exit_code == 0, result.output
    assert "buy_and_hold_spy" in result.output
    assert "sharpe" in result.output.lower()


# ---------------------------------------------------------------------------
# Test 2
# ---------------------------------------------------------------------------


def test_evaluate_test_window_requires_promotion_flag():
    from sma.backtest.__main__ import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate", "--strategy", "buy_and_hold_spy", "--window", "test"])
    assert result.exit_code != 0
    assert "promot" in result.output.lower() or "permission" in result.output.lower()


# ---------------------------------------------------------------------------
# Test 3
# ---------------------------------------------------------------------------


def test_evaluate_test_window_with_flag_works(monkeypatch: pytest.MonkeyPatch):
    from sma.backtest.__main__ import cli

    universe = ["SPY", "AAPL", "MSFT"]
    prices = _make_prices(universe, n_days=30)
    monkeypatch.setattr(
        _get_ev_mod(), "_load_prices_for_window", lambda db_path, u, start, end: prices
    )

    runner = CliRunner()
    result = runner.invoke(cli, [
        "evaluate",
        "--strategy", "buy_and_hold_spy",
        "--window", "test",
        "--i-promise-this-is-a-promotion-decision",
    ])
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# Test 4
# ---------------------------------------------------------------------------


def test_evaluate_unknown_strategy_errors():
    from sma.backtest.__main__ import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate", "--strategy", "does_not_exist", "--window", "train"])
    assert result.exit_code != 0
    combined = result.output + str(result.exception or "")
    assert "Unknown strategy" in combined


def test_evaluate_help_shows_use_theses_flag():
    from sma.backtest.__main__ import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "--use-theses" in result.output


def test_build_strategy_xgb_top_k_use_theses_passes_store(monkeypatch: pytest.MonkeyPatch):
    """When --use-theses is set, _build_strategy should construct XGBoostTopKStrategy
    with use_theses=True and a non-None store. Otherwise both are off."""
    from sma.backtest.__main__ import _build_strategy

    captured = {}

    class _FakePredictor:
        def __init__(self, *a, **kw):
            pass

    class _FakeXGB:
        def __init__(self, *, predictor, universe, use_theses=False, store=None,
                       k=20, target_weight_per_position=0.05, sector_neutralize=0.0):
            captured["use_theses"] = use_theses
            captured["store_provided"] = store is not None
            captured["sector_neutralize"] = sector_neutralize

    monkeypatch.setattr("sma.model.predictor.Predictor", _FakePredictor)
    monkeypatch.setattr(
        "sma.backtest.strategies.xgb_top_k.XGBoostTopKStrategy", _FakeXGB
    )

    _build_strategy("xgb_top_k", seed=None, universe=["AAPL"], use_theses=True)
    assert captured["use_theses"] is True
    assert captured["store_provided"] is True
    assert captured["sector_neutralize"] == 0.0  # default off unless explicitly set

    captured.clear()
    _build_strategy("xgb_top_k", seed=None, universe=["AAPL"], use_theses=False)
    assert captured["use_theses"] is False
    assert captured["store_provided"] is False


# ---------------------------------------------------------------------------
# Test 5
# ---------------------------------------------------------------------------


def test_smoke_runs_all_three_baselines(monkeypatch: pytest.MonkeyPatch):
    from sma.backtest.__main__ import cli

    universe = ["SPY", "AAPL", "MSFT"]
    prices = _make_prices(universe, n_days=100)
    monkeypatch.setattr(
        _get_ev_mod(), "_load_prices_for_window", lambda db_path, u, start, end: prices
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["smoke"])
    assert result.exit_code == 0, result.output
    assert "buy_and_hold_spy" in result.output
    assert "equal_weight" in result.output
    assert "random_long" in result.output


# ---------------------------------------------------------------------------
# Test 6
# ---------------------------------------------------------------------------


def test_detect_overfit_passes_on_clean_strategy(monkeypatch: pytest.MonkeyPatch):
    from sma.backtest.__main__ import cli

    train_result = _make_backtest_result(
        "buy_and_hold_spy", "train", sharpe=0.8, daily_returns=[0.001] * 252, num_trades=50
    )
    val_result = _make_backtest_result(
        "buy_and_hold_spy", "val", sharpe=0.75, daily_returns=[0.001] * 125, num_trades=50
    )

    call_count = {"n": 0}

    def fake_evaluate_strategy(**kwargs):
        n = call_count["n"]
        call_count["n"] += 1
        return train_result if n == 0 else val_result

    monkeypatch.setattr("sma.backtest.__main__.evaluate_strategy", fake_evaluate_strategy)

    runner = CliRunner()
    result = runner.invoke(cli, ["detect-overfit", "--strategy", "buy_and_hold_spy"])
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# Test 7
# ---------------------------------------------------------------------------


def test_verify_spy_exits_zero_when_returns_match(monkeypatch: pytest.MonkeyPatch):
    """verify-spy exits 0 when simulated return is within 1% of ground truth."""
    from sma.backtest import __main__ as bt_main

    truth = (0.18, date(2025, 1, 2), date(2025, 12, 31))
    monkeypatch.setattr(bt_main, "_spy_ground_truth_return", lambda db_path: truth)
    monkeypatch.setattr(bt_main, "_spy_simulated_return", lambda db_path: 0.179)

    from sma.backtest.__main__ import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["verify-spy"])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_verify_spy_exits_one_when_returns_diverge(monkeypatch: pytest.MonkeyPatch):
    """verify-spy exits 1 when simulated return differs by more than 1% from ground truth."""
    from sma.backtest import __main__ as bt_main

    truth = (0.18, date(2025, 1, 2), date(2025, 12, 31))
    monkeypatch.setattr(bt_main, "_spy_ground_truth_return", lambda db_path: truth)
    monkeypatch.setattr(bt_main, "_spy_simulated_return", lambda db_path: 0.05)

    from sma.backtest.__main__ import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["verify-spy"])
    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output


def test_detect_overfit_flags_overfit_strategy(monkeypatch: pytest.MonkeyPatch):
    from sma.backtest.__main__ import cli

    # Train Sharpe 3.0, val Sharpe -0.5: gap=3.5 > 1.0 triggers train_val_gap.
    # Val returns: 3 of 4 quarters are negative, triggering walk_forward_consistency.
    n_val = 252
    val_returns = [-0.005] * n_val
    val_returns[n_val - n_val // 4:] = [0.0001] * (n_val // 4)

    train_result = _make_backtest_result(
        "buy_and_hold_spy", "train", sharpe=3.0, daily_returns=[0.005] * 252, num_trades=50
    )
    val_result = _make_backtest_result(
        "buy_and_hold_spy", "val", sharpe=-0.5, daily_returns=val_returns, num_trades=50
    )

    call_count = {"n": 0}

    def fake_evaluate_strategy(**kwargs):
        n = call_count["n"]
        call_count["n"] += 1
        return train_result if n == 0 else val_result

    monkeypatch.setattr("sma.backtest.__main__.evaluate_strategy", fake_evaluate_strategy)

    runner = CliRunner()
    result = runner.invoke(cli, ["detect-overfit", "--strategy", "buy_and_hold_spy"])
    assert result.exit_code == 1, result.output
