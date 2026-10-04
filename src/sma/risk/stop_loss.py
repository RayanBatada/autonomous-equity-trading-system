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

    `cost_basis` and `current_price` must be on the SAME (raw, unadjusted)
    basis — see the live sweep's price lookup (sma.live.stop_loss) for why.

    Known behavior: averaging down (buying more of a name at a lower price)
    lowers avg_entry_price, i.e. `cost_basis` — which lowers the trigger
    price too. Flagged, not changed, in the 2026-07-30 review (Finding 8).
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


def check_take_profit(
    *,
    rails: RiskRails,
    ticker: str,
    cost_basis: float,
    current_price: float,
) -> tuple[bool, str]:
    """Return (triggered, reason) for the take-profit exit.

    Fires when current_price >= cost_basis * (1 + take_profit_pct). Disabled
    when rails.take_profit_pct <= 0 OR cost_basis <= 0 (defensive). Mirrors the
    simulator's take-profit so live == sim.

    `cost_basis` and `current_price` must be on the SAME (raw, unadjusted)
    basis — see the live sweep's price lookup (sma.live.stop_loss) for why.

    Known behavior: averaging down lowers avg_entry_price (`cost_basis`),
    which also lowers the take-profit trigger price. Flagged, not changed,
    in the 2026-07-30 review (Finding 8).
    """
    if rails.take_profit_pct <= 0:
        return False, ""
    if cost_basis <= 0:
        return False, ""
    gain = (current_price - cost_basis) / cost_basis
    if gain >= rails.take_profit_pct:
        return True, (
            f"take-profit for {ticker}: price {current_price:.2f} is "
            f"{gain:.2%} above cost basis {cost_basis:.2f}"
        )
    return False, ""


def check_trailing_stop(
    *,
    rails: RiskRails,
    ticker: str,
    peak_price: float,
    current_price: float,
) -> tuple[bool, str]:
    """Return (triggered, reason) for the trailing-stop exit.

    Fires when current_price <= peak_price * (1 - trailing_stop_pct), where
    peak_price is the post-entry high. Disabled when rails.trailing_stop_pct <= 0
    OR peak_price <= 0 (defensive). Mirrors the simulator's trailing stop.

    `peak_price` and `current_price` must be on the SAME (raw, unadjusted)
    basis — see the live sweep's peak/price lookups (sma.live.stop_loss) for
    why (2026-07-30 review, Finding 8).

    Known behavior: `peak_price` floors at cost_basis (see
    _peak_prices_from_store), so averaging down — which lowers
    avg_entry_price — can also lower that floor and thus the trigger price,
    same as the other two price exits.
    """
    if rails.trailing_stop_pct <= 0:
        return False, ""
    if peak_price <= 0:
        return False, ""
    drop = (peak_price - current_price) / peak_price
    if drop >= rails.trailing_stop_pct:
        return True, (
            f"trailing-stop for {ticker}: price {current_price:.2f} is "
            f"{drop:.2%} below post-entry peak {peak_price:.2f}"
        )
    return False, ""
