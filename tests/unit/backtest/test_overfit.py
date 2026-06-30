from datetime import date

from sma.backtest.overfit import (
    detect_overfit,
)
from sma.backtest.result import BacktestResult


def _result(
    *,
    window="val",
    sharpe=1.0,
    num_trades=100,
    daily_returns=None,
    monthly_returns=None,
    start=date(2025, 7, 1),
    end=date(2025, 12, 31),
) -> BacktestResult:
    if daily_returns is None:
        daily_returns = [0.001] * 100
    if monthly_returns is None:
        monthly_returns = [0.02] * 6
    return BacktestResult(
        strategy_name="x",
        window=window,
        start_date=start,
        end_date=end,
        sharpe=sharpe,
        sortino=sharpe * 1.1,
        calmar=0.5,
        max_drawdown=-0.1,
        total_return=0.05,
        annualized_return=0.10,
        hit_rate=0.5,
        num_trades=num_trades,
        avg_holding_days=10.0,
        daily_returns=daily_returns,
        monthly_returns=monthly_returns,
        code_commit="abc",
        data_run_id=1,
        seed=42,
    )


def test_clean_strategy_passes_all_checks():
    train = _result(window="train", sharpe=1.5, num_trades=200)
    val = _result(window="val", sharpe=1.3, num_trades=120,
                   daily_returns=[0.001 if i % 3 else -0.0005 for i in range(120)])
    report = detect_overfit(train_result=train, val_result=val)
    assert report.passed, report.summary()


def test_train_vs_val_gap_flag():
    train = _result(window="train", sharpe=3.0, num_trades=200)
    val = _result(window="val", sharpe=1.0, num_trades=100)  # 2.0 gap > 1.0 threshold
    report = detect_overfit(train_result=train, val_result=val)
    assert not report.passed
    assert any(c.name == "train_val_gap" and not c.passed for c in report.checks)


def test_low_trade_count_flag():
    train = _result(window="train", sharpe=1.0, num_trades=200)
    val = _result(window="val", sharpe=1.0, num_trades=10)  # < 30
    report = detect_overfit(train_result=train, val_result=val)
    assert not report.passed
    assert any(c.name == "trade_count_adequacy" and not c.passed for c in report.checks)


def test_recency_bias_flag():
    """Last 60 days are spectacular, earlier portion is mediocre."""
    daily = [0.0001] * 60 + [0.005] * 60  # second half much higher
    val = _result(
        window="val",
        sharpe=1.0,
        num_trades=120,
        daily_returns=daily,
    )
    train = _result(window="train", sharpe=1.0, num_trades=200)
    report = detect_overfit(train_result=train, val_result=val)
    assert not report.passed
    assert any(c.name == "recency_bias" and not c.passed for c in report.checks)


def test_walk_forward_inconsistency_flag():
    """4 quarters of val, 3 negative, aggregate positive due to one big winner."""
    # ~120 trading days; 4 quarters of ~30 days each
    quarter1 = [-0.001] * 30   # negative
    quarter2 = [-0.001] * 30   # negative
    quarter3 = [0.020] * 30     # huge positive (60% over the quarter)
    quarter4 = [-0.001] * 30   # negative
    daily = quarter1 + quarter2 + quarter3 + quarter4
    val = _result(window="val", sharpe=0.5, num_trades=120, daily_returns=daily)
    train = _result(window="train", sharpe=0.5, num_trades=200)
    report = detect_overfit(train_result=train, val_result=val)
    assert not report.passed
    assert any(c.name == "walk_forward_consistency" and not c.passed for c in report.checks)


def test_feature_importance_stub_passes_with_note():
    train = _result(window="train", sharpe=1.0, num_trades=200)
    val = _result(window="val", sharpe=1.0, num_trades=100)
    report = detect_overfit(train_result=train, val_result=val)
    fi_check = next(c for c in report.checks if c.name == "feature_importance_shift")
    assert fi_check.passed
    assert "stub" in fi_check.detail.lower()


def test_synthetic_overfit_trigger():
    """The most important test: deliberately overfit predictions trigger detector."""
    # Train: amazing Sharpe; val: terrible Sharpe
    train = _result(window="train", sharpe=5.0, num_trades=500)
    val = _result(window="val", sharpe=-1.0, num_trades=50,
                   daily_returns=[-0.001] * 50)
    report = detect_overfit(train_result=train, val_result=val)
    assert not report.passed
    # At least 2 checks should fail
    failed = [c for c in report.checks if not c.passed]
    assert len(failed) >= 2


def test_summary_format():
    train = _result(window="train", sharpe=1.0, num_trades=200)
    val = _result(window="val", sharpe=1.0, num_trades=100)
    report = detect_overfit(train_result=train, val_result=val)
    s = report.summary()
    assert "Overfit report" in s
    assert "OVERALL" in s
