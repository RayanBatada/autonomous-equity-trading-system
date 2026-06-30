"""Stop-loss rail. Triggers a forced sell when current_price <= cost_basis * (1 - stop_loss_pct).

Phase 5 ships with stop_loss_pct=0 (rail disabled) per the 2026-04-28 rail
diagnostic: with the 8% threshold active, val Sharpe was -0.272; with it off,
val Sharpe is +0.441. The rail is preserved as code so future tuning can
re-enable it with a higher threshold (e.g., 15%).
"""

from sma.risk.rails import RiskRails


def check_stop_loss(
    *,
    rails: RiskRails,
    ticker: str,
    cost_basis: float,
    current_price: float,
) -> tuple[bool, str]:
    """Return (triggered, reason). Reason is empty when triggered=False.

    Disabled when rails.stop_loss_pct <= 0 OR cost_basis <= 0 (defensive).
    """
    if rails.stop_loss_pct <= 0:
        return False, ""
    if cost_basis <= 0:
        return False, ""
    pct_loss = (cost_basis - current_price) / cost_basis
    if pct_loss >= rails.stop_loss_pct:
        return True, (
            f"stop-loss for {ticker}: price {current_price:.2f} is "
            f"{pct_loss:.2%} below cost basis {cost_basis:.2f}"
        )
    return False, ""
