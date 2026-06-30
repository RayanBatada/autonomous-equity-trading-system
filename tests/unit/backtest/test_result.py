from datetime import date

from sma.backtest.result import BacktestResult


def test_backtest_result_constructs_with_required_fields():
    r = BacktestResult(
        strategy_name="buy_and_hold_spy",
        window="val",
        start_date=date(2025, 7, 1),
        end_date=date(2025, 12, 31),
        sharpe=1.42,
        sortino=1.8,
        calmar=0.9,
        max_drawdown=-0.12,
        total_return=0.08,
        annualized_return=0.16,
        hit_rate=0.55,
        num_trades=42,
        avg_holding_days=14.3,
        daily_returns=[0.001, -0.002, 0.003],
        monthly_returns=[0.02, -0.01, 0.015],
        code_commit="abc1234",
        data_run_id=1234567890,
        seed=None,
    )
    assert r.sharpe == 1.42
    assert r.window == "val"
    assert r.num_trades == 42


def test_backtest_result_is_frozen():
    import pytest
    r = BacktestResult(
        strategy_name="x", window="train",
        start_date=date(2024, 1, 1), end_date=date(2024, 6, 30),
        sharpe=1.0, sortino=1.0, calmar=0.5, max_drawdown=-0.1,
        total_return=0.05, annualized_return=0.1, hit_rate=0.5,
        num_trades=10, avg_holding_days=5.0,
        daily_returns=[], monthly_returns=[],
        code_commit="abc", data_run_id=1, seed=42,
    )
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        r.sharpe = 99  # type: ignore
