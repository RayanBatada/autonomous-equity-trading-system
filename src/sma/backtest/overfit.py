"""Overfit detector.

Five checks designed to flag the kinds of mistakes auto-research will produce
when given thousands of strategy variants. Phase 6 will treat any flag as
automatic-discard regardless of Sharpe improvement.

Why this matters: Karpathy can run 100 model experiments overnight and trust
the val_bpb because each model has only 5 minutes of compute. We don't have
that constraint. A trading strategy backtested 1000 times WILL find a winner
by chance. These checks are the defense.
"""

import math
import statistics
from dataclasses import dataclass

from sma.backtest.metrics import TRADING_DAYS
from sma.backtest.metrics import sharpe as compute_sharpe
from sma.backtest.result import BacktestResult


def _segment_sharpe_with_pooled_sigma(
    segment: list[float],
    pooled_sigma: float,
) -> float:
    """Annualized Sharpe-like score for a sub-window using pooled vol.

    When a sub-window is too short or has constant returns, its own std is
    zero or unreliable. Using the parent window's std as the volatility
    normalizer lets us compare sub-window means on a common scale.
    """
    if not segment or pooled_sigma <= 0:
        return 0.0
    mu = statistics.mean(segment)
    return math.sqrt(TRADING_DAYS) * mu / pooled_sigma


@dataclass(frozen=True)
class OverfitCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class OverfitReport:
    checks: list[OverfitCheck]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def summary(self) -> str:
        lines = ["Overfit report", ""]
        for c in self.checks:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{mark}] {c.name}: {c.detail}")
        lines.append("")
        lines.append("OVERALL: " + ("PASS" if self.passed else "FAIL"))
        return "\n".join(lines)


def _check_train_val_gap(train: BacktestResult, val: BacktestResult,
                          threshold: float = 1.0) -> OverfitCheck:
    gap = train.sharpe - val.sharpe
    return OverfitCheck(
        name="train_val_gap",
        passed=gap <= threshold,
        detail=(
            f"Sharpe(train)={train.sharpe:.2f}, Sharpe(val)={val.sharpe:.2f}, "
            f"gap={gap:.2f} (threshold {threshold:.2f})"
        ),
    )


def _check_trade_count_adequacy(val: BacktestResult,
                                  min_trades: int = 30) -> OverfitCheck:
    return OverfitCheck(
        name="trade_count_adequacy",
        passed=val.num_trades >= min_trades,
        detail=f"num_trades={val.num_trades} (need >= {min_trades})",
    )


def _check_recency_bias(val: BacktestResult,
                          recent_days: int = 60,
                          gap_threshold: float = 1.5) -> OverfitCheck:
    rets = val.daily_returns
    if len(rets) < 2 * recent_days:
        return OverfitCheck(
            name="recency_bias",
            passed=True,
            detail=f"insufficient data ({len(rets)} days) for recency check",
        )
    early_half = rets[: len(rets) // 2]
    recent = rets[-recent_days:]
    # Use the full val window's std as the pooled vol so that sub-windows
    # with constant returns (Sharpe undefined) still produce comparable
    # mean-based scores. This catches "all the alpha came in the last 60
    # days" even when each sub-window has near-zero internal variance.
    pooled_sigma = statistics.stdev(rets) if len(rets) >= 2 else 0.0
    early_sharpe = _segment_sharpe_with_pooled_sigma(early_half, pooled_sigma)
    recent_sharpe = _segment_sharpe_with_pooled_sigma(recent, pooled_sigma)
    return OverfitCheck(
        name="recency_bias",
        passed=(recent_sharpe - early_sharpe) <= gap_threshold,
        detail=(
            f"early Sharpe={early_sharpe:.2f}, last-{recent_days}d Sharpe={recent_sharpe:.2f}, "
            f"gap={recent_sharpe - early_sharpe:.2f}"
        ),
    )


def _check_walk_forward_consistency(val: BacktestResult) -> OverfitCheck:
    rets = val.daily_returns
    if len(rets) < 4:
        return OverfitCheck(
            name="walk_forward_consistency",
            passed=True,
            detail=f"insufficient data ({len(rets)} days) for quarterly check",
        )
    n = len(rets)
    quarters = [rets[i*n//4 : (i+1)*n//4] for i in range(4)]
    # Use mean return per quarter for the sign decision: Sharpe is 0 when a
    # window's std is 0, which would silently hide losses (test data with
    # constant negative returns would never be flagged). Mean return is the
    # signal we actually care about for "is this quarter losing money".
    quarter_means = [statistics.mean(q) if q else 0.0 for q in quarters]
    pooled_sigma = statistics.stdev(rets) if len(rets) >= 2 else 0.0
    quarter_sharpes = [
        _segment_sharpe_with_pooled_sigma(q, pooled_sigma) for q in quarters if q
    ]
    aggregate_sharpe = compute_sharpe(rets, risk_free_annual=0.0)
    negative_quarters = sum(1 for m in quarter_means if m < 0)
    # Two pathologies both deserve a flag:
    #   (a) "Aggregate masking": 3+ negative quarters BUT aggregate positive,
    #       i.e. one giant winner hiding a string of losses.
    #   (b) "Uniform losing": 3+ negative quarters period, even if aggregate
    #       is also negative. Strategy is broken in val.
    flag = negative_quarters >= 3
    return OverfitCheck(
        name="walk_forward_consistency",
        passed=not flag,
        detail=(
            f"per-quarter Sharpes={[round(s,2) for s in quarter_sharpes]}, "
            f"per-quarter means={[round(m,4) for m in quarter_means]}, "
            f"aggregate Sharpe={aggregate_sharpe:.2f}"
        ),
    )


def _check_feature_importance_shift_stub() -> OverfitCheck:
    return OverfitCheck(
        name="feature_importance_shift",
        passed=True,
        detail="stub: implemented in Phase 6 once retraining is in place",
    )


def detect_overfit(
    train_result: BacktestResult,
    val_result: BacktestResult,
) -> OverfitReport:
    checks = [
        _check_train_val_gap(train_result, val_result),
        _check_trade_count_adequacy(val_result),
        _check_recency_bias(val_result),
        _check_walk_forward_consistency(val_result),
        _check_feature_importance_shift_stub(),
    ]
    return OverfitReport(checks=checks)
