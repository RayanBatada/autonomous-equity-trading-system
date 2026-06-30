"""Backtest metrics.

Standalone functions for Sharpe, Sortino, Calmar, max drawdown, hit rate,
total return, and annualized return. Each takes a list of daily returns
(decimals like 0.012 = +1.2%) and returns a scalar.

Sharpe is the primary metric the auto-research agent in Phase 6 will
optimize. The others are diagnostics for humans and the overfit detector.

Convention: 252 trading days per year. Risk-free rate is annualized;
internally converted to daily for Sharpe / Sortino.
"""

import math
import statistics

TRADING_DAYS = 252


def total_return(returns: list[float]) -> float:
    if not returns:
        return 0.0
    product = 1.0
    for r in returns:
        product *= (1.0 + r)
    return product - 1.0


def annualized_return(returns: list[float]) -> float:
    """Geometric annualization: (1 + total)^(252/n) - 1."""
    if not returns:
        return 0.0
    tot = total_return(returns)
    return (1.0 + tot) ** (TRADING_DAYS / len(returns)) - 1.0


def _daily_rf(annual_rf: float) -> float:
    return (1.0 + annual_rf) ** (1.0 / TRADING_DAYS) - 1.0


def sharpe(returns: list[float], risk_free_annual: float = 0.05) -> float:
    """Annualized Sharpe ratio.

    sqrt(252) * (mean_excess_daily) / std_daily
    where excess = daily_return - daily_risk_free.
    """
    if len(returns) < 2:
        return 0.0
    rf = _daily_rf(risk_free_annual)
    excess = [r - rf for r in returns]
    mu = statistics.mean(excess)
    sigma = statistics.stdev(excess)
    if sigma == 0:
        return 0.0
    return math.sqrt(TRADING_DAYS) * mu / sigma


def sortino(returns: list[float], risk_free_annual: float = 0.05) -> float:
    """Annualized Sortino ratio: like Sharpe but penalizes downside only.

    sqrt(252) * (mean_excess_daily) / downside_deviation
    where downside_deviation = sqrt(mean(min(0, excess)^2)).
    """
    if len(returns) < 2:
        return 0.0
    rf = _daily_rf(risk_free_annual)
    excess = [r - rf for r in returns]
    mu = statistics.mean(excess)
    downside_squares = [min(0.0, e) ** 2 for e in excess]
    downside_var = sum(downside_squares) / len(downside_squares)
    downside_dev = math.sqrt(downside_var)
    if downside_dev == 0:
        return 0.0
    return math.sqrt(TRADING_DAYS) * mu / downside_dev


def max_drawdown(returns: list[float]) -> float:
    """Largest peak-to-trough fractional loss along the equity curve.

    Returned as a NEGATIVE number, e.g. -0.18 for an 18% drawdown.
    Returns 0.0 if the equity curve never declines.
    """
    if not returns:
        return 0.0
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for r in returns:
        equity *= (1.0 + r)
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak
        if dd < worst:
            worst = dd
    return worst


def calmar(returns: list[float]) -> float:
    """Annualized return divided by abs(max drawdown).

    By convention, returns 0.0 when there is no drawdown (avoids div by zero).
    """
    dd = abs(max_drawdown(returns))
    if dd == 0:
        return 0.0
    return annualized_return(returns) / dd


def hit_rate(returns: list[float]) -> float:
    """Fraction of returns that are strictly positive."""
    if not returns:
        return 0.0
    return sum(1 for r in returns if r > 0) / len(returns)
