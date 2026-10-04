"""Regulatory sell-side fees.

Paper trading charges NONE of these — Alpaca's paper docs list regulatory fees
alongside market impact and dividends as things paper does not model — so every
default here is 0.0 and a default `FeeSchedule()` leaves the backtest exactly
where it has always been, comparable to the paper track record.

Rates verified 2026-08-27 against the Alpaca Brokerage Fee Schedule (rev.
2026-07-20) and FINRA By-Laws Schedule A §1. Alpaca passes all three through at
cost and takes no spread on them.

    SEC Section 31   sells only   $20.60 per $1,000,000 of principal (eff. 2026-04-04)
    FINRA TAF        sells only   $0.000195/share, capped $9.79 per trade (2026)
    FINRA CAT        BOTH sides   $0.000003/share

Two traps worth stating, because both are widely repeated wrongly:

  * The Section 31 rate is NOT stable. It was $27.80/M in May 2024, ZERO for
    most of FY2025, and $20.60/M since April 2026, expiring 60 days after the
    FY2027 appropriation. Never hardcode it into a study's conclusions; pass the
    rate that applied on the trade date.
  * There is NO $0.01 TAF minimum. The opposite exists: a de-minimis waiver when
    the execution price is below the per-share rate. And Alpaca aggregates each
    fee type per account per DAY before rounding up to the cent, so charging a
    penny per order overstates the cost of a many-small-orders strategy — which
    is precisely the strategy a small fractional account runs.

The 2026 numbers are exposed as constants so a real-money study can opt in
without re-deriving them, but they are NOT the defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

#: SEC Section 31, dollars per $1M of sale principal, effective 2026-04-04.
SEC_FEE_PER_MILLION_2026 = 20.60
#: FINRA Trading Activity Fee, dollars per share sold, effective 2026-01-01.
FINRA_TAF_PER_SHARE_2026 = 0.000195
#: Per-trade cap on the TAF, effective 2026-01-01.
FINRA_TAF_MAX_PER_TRADE_2026 = 9.79
#: FINRA CAT fee, dollars per share, charged on BUYS AND SELLS.
FINRA_CAT_PER_SHARE_2026 = 0.000003


@dataclass(frozen=True)
class FeeSchedule:
    """All-zero by default = paper behaviour = today's backtest."""

    sec_fee_per_million: float = 0.0
    finra_taf_per_share: float = 0.0
    finra_taf_max_per_trade: float = 0.0
    finra_cat_per_share: float = 0.0

    @classmethod
    def real_money_2026(cls) -> FeeSchedule:
        """The published rates in force as of 2026-08-27. Check them before
        relying on a study run in a later year — see the module docstring."""
        return cls(
            sec_fee_per_million=SEC_FEE_PER_MILLION_2026,
            finra_taf_per_share=FINRA_TAF_PER_SHARE_2026,
            finra_taf_max_per_trade=FINRA_TAF_MAX_PER_TRADE_2026,
            finra_cat_per_share=FINRA_CAT_PER_SHARE_2026,
        )

    @property
    def is_zero(self) -> bool:
        return not (
            self.sec_fee_per_million
            or self.finra_taf_per_share
            or self.finra_cat_per_share
        )

    def sell_fees(self, *, shares: float, price: float) -> float:
        """Dollars deducted from the proceeds of one SELL."""
        if self.is_zero:
            return 0.0
        principal = shares * price
        sec = principal * self.sec_fee_per_million / 1_000_000.0
        taf = shares * self.finra_taf_per_share
        if self.finra_taf_max_per_trade > 0:
            taf = min(taf, self.finra_taf_max_per_trade)
        cat = shares * self.finra_cat_per_share
        return sec + taf + cat

    def buy_fees(self, *, shares: float, price: float) -> float:
        """Dollars added to the cost of one BUY. Only CAT is charged on buys —
        Section 31 and the TAF are sell-side only."""
        if self.is_zero:
            return 0.0
        return shares * self.finra_cat_per_share
