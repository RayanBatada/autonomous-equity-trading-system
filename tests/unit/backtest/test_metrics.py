
import pytest

from sma.backtest.metrics import (
    annualized_return,
    calmar,
    hit_rate,
    max_drawdown,
    sharpe,
    sortino,
    total_return,
)

# 252 trading days/year is the convention.
TRADING_DAYS = 252


def test_sharpe_zero_for_zero_returns():
    assert sharpe([0.0] * 100, risk_free_annual=0.0) == 0.0


def test_sharpe_known_value_for_constant_positive_returns():
    """Constant 0.001 daily return = 25.2% annualized, with zero std.
    Sharpe = (mean - rf) / std. Std is 0 here so we should return inf or a
    sentinel; pick whatever the impl does. For this test, use slight noise."""
    # Add tiny noise so std > 0.
    rets = [0.001, 0.0011, 0.0009, 0.001, 0.001] * 50  # daily mean ~0.001
    s = sharpe(rets, risk_free_annual=0.0)
    # Annualized mean ~ 0.001 * 252 = 0.252; std small => high Sharpe
    assert s > 5.0


def test_sharpe_negative_for_losing_strategy():
    rets = [-0.001, -0.002, -0.0015, -0.001, -0.001] * 50
    s = sharpe(rets, risk_free_annual=0.0)
    assert s < 0


def test_sharpe_subtracts_risk_free_rate():
    """A strategy returning the risk-free rate exactly should have Sharpe ~0."""
    daily_rf = (1 + 0.05) ** (1 / TRADING_DAYS) - 1  # 5% annual -> daily
    # Add tiny noise to avoid div by zero
    rets = [daily_rf + 0.0001 * (i % 3 - 1) for i in range(252)]
    s = sharpe(rets, risk_free_annual=0.05)
    assert abs(s) < 0.5  # close to zero


def test_sortino_only_penalizes_downside():
    """Sortino should be HIGHER than Sharpe when downside is small relative to total volatility."""
    # Mostly small positive returns with rare large positives (skewed up)
    rets = [0.001] * 100 + [0.05] * 5 + [-0.0005] * 50
    sh = sharpe(rets, risk_free_annual=0.0)
    so = sortino(rets, risk_free_annual=0.0)
    assert so > sh


def test_max_drawdown_returns_negative_number():
    """Path goes 1.0 -> 1.20 -> 0.96 -> 1.30. Max DD is (0.96-1.20)/1.20 = -0.20."""
    # Convert to daily returns from prices
    prices = [1.0, 1.20, 0.96, 1.30]
    rets = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
    dd = max_drawdown(rets)
    assert dd == pytest.approx(-0.20, abs=1e-6)


def test_max_drawdown_zero_for_monotonic_growth():
    rets = [0.01, 0.01, 0.01, 0.01]  # never drops
    assert max_drawdown(rets) == 0.0


def test_calmar_is_annualized_return_over_abs_max_drawdown():
    """Construct a sequence where we know both numbers."""
    rets = [0.001] * 100 + [-0.10] + [0.001] * 151  # one bad day, drawdown ~10%
    c = calmar(rets)
    expected_ann_ret = annualized_return(rets)
    expected_dd = abs(max_drawdown(rets))
    assert c == pytest.approx(expected_ann_ret / expected_dd, rel=1e-6)


def test_calmar_zero_when_no_drawdown():
    rets = [0.001] * 252
    assert calmar(rets) == 0.0  # convention: 0/0 is 0


def test_hit_rate_fraction_positive():
    rets = [0.01, -0.01, 0.02, -0.005, 0.0]
    # Positive: 2 of 5 = 0.4. Zero excluded from numerator.
    assert hit_rate(rets) == pytest.approx(0.4)


def test_hit_rate_empty_returns_zero():
    assert hit_rate([]) == 0.0


def test_total_return_compounds():
    """+10% then -10% is 99% of original = -1% total."""
    rets = [0.10, -0.10]
    assert total_return(rets) == pytest.approx(-0.01, abs=1e-9)


def test_annualized_return_for_one_year_of_data():
    """One year of constant 0.1% daily returns."""
    rets = [0.001] * TRADING_DAYS
    ar = annualized_return(rets)
    # (1.001)^252 - 1 = ~0.288
    expected = (1.001 ** TRADING_DAYS) - 1
    assert ar == pytest.approx(expected, rel=1e-6)
