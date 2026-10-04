"""Honest performance measurement: alpha vs a buy-and-hold benchmark, and how
much to trust it given the live sample size.

The bot is fully long, so absolute return is dominated by market beta — a +6%
return in a +6% market is ZERO skill. The only evidence of edge is EXCESS return
over buy-and-hold SPY. And with a tiny live sample, even a positive excess is
noise. These pure helpers make both explicit so early P&L isn't over-read.
"""

from __future__ import annotations


def alpha_vs_benchmark(
    equity_curve: list[float], benchmark_curve: list[float]
) -> dict[str, float | None]:
    """Strategy vs buy-and-hold benchmark over the same period.

    Returns strategy_return, benchmark_return, and excess_return (the alpha
    proxy). All None when either curve has < 2 points or a non-positive start.
    """
    none = {"strategy_return": None, "benchmark_return": None, "excess_return": None}
    if len(equity_curve) < 2 or len(benchmark_curve) < 2:
        return none
    if equity_curve[0] <= 0 or benchmark_curve[0] <= 0:
        return none
    strat = equity_curve[-1] / equity_curve[0] - 1.0
    bench = benchmark_curve[-1] / benchmark_curve[0] - 1.0
    return {
        "strategy_return": strat,
        "benchmark_return": bench,
        "excess_return": strat - bench,
    }


def regression_alpha(
    strategy_returns: list[float],
    benchmark_returns: list[float],
    min_n: int = 20,
) -> dict[str, float | None]:
    """OLS regression alpha of daily strategy returns on benchmark returns.

    The naive excess (strat − bench) assumes beta=1; on a ~1.17-beta long book
    it overstates selection skill by the beta-tilt's share of the market return
    (~2.5pp at the 6/25 diagnosis). Regressing r_s = a + b·r_b separates the
    leverage-like beta tilt (b) from actual selection alpha (a), and the t-stat
    of `a` says whether that alpha is distinguishable from luck — require
    |t| > 2 before claiming ANY skill. Returns alpha_daily, alpha_annualized
    (×252), beta, t_stat, n; all-None when n < min_n or the benchmark series
    has no variance.
    """
    none: dict[str, float | None] = {
        "alpha_daily": None, "alpha_annualized": None,
        "beta": None, "t_stat": None, "n": None,
    }
    n = min(len(strategy_returns), len(benchmark_returns))
    if n < min_n:
        return none
    s = strategy_returns[:n]
    b = benchmark_returns[:n]
    # DB DOUBLEs can carry NaN/inf (e.g. a corrupted snapshot row); NaN slips
    # through every downstream guard (`NaN <= 0` is False) and would render a
    # confident-looking nan banner. Reject non-finite inputs outright.
    import math
    if not all(math.isfinite(v) for v in s) or not all(math.isfinite(v) for v in b):
        return none
    mean_b = sum(b) / n
    mean_s = sum(s) / n
    sxx = sum((x - mean_b) ** 2 for x in b)
    if sxx <= 0:
        return none
    beta = sum((x - mean_b) * (y - mean_s) for x, y in zip(b, s, strict=True)) / sxx
    alpha = mean_s - beta * mean_b
    resid = [y - (alpha + beta * x) for x, y in zip(b, s, strict=True)]
    dof = n - 2
    s2 = sum(e * e for e in resid) / dof
    # Degenerate perfect-fit guard: with ~zero residual variance both alpha and
    # its SE are float noise and the ratio is garbage — report t=0 (no evidence).
    se_alpha = 0.0 if s2 < 1e-20 else (s2 * (1.0 / n + mean_b * mean_b / sxx)) ** 0.5
    t_stat = (alpha / se_alpha) if se_alpha > 0 else 0.0
    return {
        "alpha_daily": alpha,
        "alpha_annualized": alpha * 252.0,
        "beta": beta,
        "t_stat": t_stat,
        "n": n,
    }


def trust_level(n_live_days: int) -> tuple[str, str]:
    """How much the live P&L can be trusted, by sample size. Returns
    (level, explanation). The thresholds are deliberately conservative: a few
    dozen daily returns cannot distinguish skill from luck."""
    if n_live_days < 60:
        return (
            "LOW",
            f"{n_live_days} live trading days — far too few to distinguish skill "
            "from luck. Treat all P&L as noise.",
        )
    if n_live_days < 250:
        return (
            "MEDIUM",
            f"{n_live_days} live days — suggestive but not conclusive; one good "
            "month can flatter the record.",
        )
    return (
        "HIGH",
        f"{n_live_days} live days — a meaningful track record (still judge on "
        "EXCESS return vs SPY, not absolute).",
    )
