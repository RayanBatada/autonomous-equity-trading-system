"""Capital-scale sizing policy: whole vs fractional shares, order-size floors,
and a liquidity cap on how much of a name's daily volume one order may take.

Why this exists (2026-08-27 money-path scale audit). `translate()` sized every
order as `floor(target_weight * equity / price)`. That single `floor` is the
entire scale story:

  * BELOW ~$25k it silently drops names. Running the real 2026-08-27 decide
    output against a fresh book: at $10k, LLY ($1,176/share, 8.35% target
    weight = $835) rounds to ZERO shares; at $500 only 3 of 9 names are
    fillable; at $50 NOTHING is — the account holds pure cash and the model's
    signal reaches the market not at all.
  * ABOVE ~$1M the opposite failure waits. Nothing in the money path has ever
    looked at how big an order is relative to the name's traded volume. The
    universe is large-cap and the largest order this bot has ever placed was
    0.0197% of the name's 20-day dollar ADV — but NANC, in the universe since
    2026-05-09, trades ~$0.4M/day, and a 10% slot of a $10M book would be
    ~$1M: 250% of a full day's volume, market-on-open.

Every policy field defaults to the no-op that reproduces the pre-audit path
exactly. `SizingPolicy()` is today's live behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

from sma.live.quantity import truncate_qty


@dataclass(frozen=True)
class SizingPolicy:
    """Runtime mirror of `config.SizingConfig` (same relationship RiskRails has
    to config.LiveRails). All-default = today's exact live sizing."""

    fractional_shares: bool = False
    fractional_precision: int = 9
    min_order_notional: float = 0.0
    max_participation_of_adv: float = 0.0
    adv_lookback_days: int = 20

    def target_qty(self, weight: float, equity: float, price: float) -> float:
        """Shares to hold for `weight` of `equity` at `price`.

        Whole-share mode floors, exactly as before. Fractional mode truncates to
        `fractional_precision` decimals — also downward, so neither mode can
        ever size an order the cash-floor check did not budget for.
        """
        raw = weight * equity / price
        if self.fractional_shares:
            return truncate_qty(raw, self.fractional_precision)
        return floor(raw)

    def participation_cap_qty(self, *, adv_dollars: float, price: float) -> float | None:
        """Max shares one order may take of a name with this ADV, or None when
        the cap is off / the ADV is unknown (unknown ADV must not silently
        block trading — the rail fails OPEN and says so in the log)."""
        if self.max_participation_of_adv <= 0 or adv_dollars <= 0 or price <= 0:
            return None
        cap_dollars = self.max_participation_of_adv * adv_dollars
        if self.fractional_shares:
            return truncate_qty(cap_dollars / price, self.fractional_precision)
        return floor(cap_dollars / price)


def _flag(value, default: bool) -> bool:
    """A boolean knob is on ONLY for a real `True`. Not for the string "true",
    not for a non-empty object, not for a test double — anything that is not
    literally a bool falls back to the default. Bare truthiness here would let a
    stray value silently turn a money-path rail on."""
    return value if isinstance(value, bool) else default


def _num(value, default: float) -> float:
    """Same idea for numeric knobs: a real number or the default. Falling back
    is the safe direction — every default in SizingPolicy is the no-op."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def build_sizing_policy(settings) -> SizingPolicy:
    """Construct a SizingPolicy from `settings.live.sizing`, tolerating an old
    config with no `sizing:` block at all (→ today's behaviour).

    Every field is coerced rather than passed through. Pydantic already
    validates the real config, so this is about everything that is NOT the real
    config: a partially-built settings object, a test double, a hand-edited
    YAML. `getattr` on any of those hands back something that is not a number,
    and an un-coerced `sizing.max_participation_of_adv > 0` then raises inside
    translate — at 18:35, on the money path. Fall back to the no-op instead.
    """
    cfg = getattr(getattr(settings, "live", None), "sizing", None)
    if cfg is None:
        return SizingPolicy()
    return SizingPolicy(
        fractional_shares=_flag(getattr(cfg, "fractional_shares", False), False),
        fractional_precision=int(_num(getattr(cfg, "fractional_precision", 9), 9)),
        min_order_notional=_num(getattr(cfg, "min_order_notional", 0.0), 0.0),
        max_participation_of_adv=_num(
            getattr(cfg, "max_participation_of_adv", 0.0), 0.0
        ),
        adv_lookback_days=int(_num(getattr(cfg, "adv_lookback_days", 20), 20)),
    )


@dataclass(frozen=True)
class UnfillableName:
    ticker: str
    target_weight: float
    price: float
    target_dollars: float


def unfillable_names(
    *,
    decisions,
    account_equity: float,
    last_prices: dict[str, tuple[float, object]],
    sizing: SizingPolicy,
) -> list[UnfillableName]:
    """Names whose target slot is smaller than one share, so whole-share sizing
    buys none of them. Empty in fractional mode (a fraction of a share always
    fits) and empty whenever equity is large enough for the whole book.

    This is the minimum-viable-equity check. It lives here rather than in
    `live.preflight` on purpose: preflight is deliberately sentinel-only and
    opens neither the DB nor the broker (the 2026-04-30 writer-blocks-self
    deadlock), so it cannot see equity or prices. decide_once calls this once
    it has both, before translating.
    """
    if sizing.fractional_shares or account_equity <= 0:
        return []
    out: list[UnfillableName] = []
    for d in decisions:
        entry = last_prices.get(d.ticker)
        if not entry:
            continue
        price = entry[0]
        if not price or price <= 0:
            continue
        target_dollars = d.target_weight * account_equity
        if target_dollars < price:
            out.append(UnfillableName(
                ticker=d.ticker, target_weight=d.target_weight,
                price=price, target_dollars=target_dollars,
            ))
    return out


def min_viable_equity(
    *,
    decisions,
    last_prices: dict[str, tuple[float, object]],
) -> float | None:
    """Smallest equity at which EVERY decided name is fillable in whole shares,
    i.e. max over names of price / target_weight. None when nothing is priced."""
    worst = 0.0
    for d in decisions:
        entry = last_prices.get(d.ticker)
        if not entry or d.target_weight <= 0:
            continue
        price = entry[0]
        if not price or price <= 0:
            continue
        worst = max(worst, price / d.target_weight)
    return worst or None
