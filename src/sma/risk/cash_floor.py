"""Cash-floor rail. Rejects buys that would drop cash below cash_floor_pct of equity."""

from sma.risk.rails import RiskRails


def check_cash_floor(
    *,
    rails: RiskRails,
    account_value: float,
    cash_after_order: float,
) -> tuple[bool, str]:
    """Return (triggered, reason).

    triggered=True means the order should be REJECTED (cash would go below floor).
    Disabled when rails.cash_floor_pct <= 0.
    """
    if rails.cash_floor_pct <= 0:
        return False, ""
    if account_value <= 0:
        return False, ""
    floor = rails.cash_floor_pct * account_value
    if cash_after_order < floor:
        return True, (
            f"cash floor: post-order cash ${cash_after_order:.2f} would drop "
            f"below floor ${floor:.2f} ({rails.cash_floor_pct:.0%} of equity)"
        )
    return False, ""
