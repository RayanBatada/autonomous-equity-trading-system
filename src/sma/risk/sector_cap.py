"""Sector-cap rail. Rejects buys that push a single GICS sector above max_sector_pct."""

from sma.risk.rails import RiskRails


def check_sector_cap(
    *,
    rails: RiskRails,
    sector: str,
    sector_exposure_after: float,
) -> tuple[bool, str]:
    """Return (triggered, reason).

    triggered=True means the order should be REJECTED (would push sector over cap).
    Disabled when rails.max_sector_pct >= 1.0.
    """
    if rails.max_sector_pct >= 1.0:
        return False, ""
    if sector_exposure_after > rails.max_sector_pct:
        return True, (
            f"sector cap: {sector} would be {sector_exposure_after:.2%}, "
            f"exceeds cap {rails.max_sector_pct:.2%}"
        )
    return False, ""
