"""Drawdown rail. Aborts ALL new buys when current_drawdown exceeds max_drawdown_pct."""

from sma.risk.rails import RiskRails


def check_drawdown(
    *,
    rails: RiskRails,
    current_drawdown: float,
) -> tuple[bool, str]:
    """Return (triggered, reason).

    triggered=True means ALL new buys should be REJECTED (account in deep drawdown).
    `current_drawdown` is a positive 0..1 fraction (e.g., 0.10 = 10% from peak).
    Disabled when rails.max_drawdown_pct >= 1.0.
    """
    if rails.max_drawdown_pct >= 1.0:
        return False, ""
    if current_drawdown > rails.max_drawdown_pct:
        return True, (
            f"drawdown {current_drawdown:.2%} exceeds limit {rails.max_drawdown_pct:.2%}"
        )
    return False, ""
