"""Drawdown-scaled cash floor: graduated de-risking in persistent crashes.

The 2026-06-13 IC analysis showed the model is a momentum factor that inverts
in reversal regimes — and momentum crashes are persistent (weeks-months). With
the bot ~91% deployed, a crash hits at full exposure behind only the blunt 30%
catastrophic-abort breaker. This raises the EFFECTIVE cash floor as the
portfolio's drawdown from peak deepens, smoothly cutting exposure as a crash
develops. Default-off (slope=0) until a walk-forward backtest shows it reduces
drawdown without unduly hurting return.
"""

from __future__ import annotations


def derisk_cash_floor(
    base_floor: float,
    drawdown: float,
    *,
    start: float,
    slope: float,
    cap: float = 0.60,
) -> float:
    """Effective cash floor given current drawdown from peak (a positive
    fraction). Below `start` (or slope<=0) returns base_floor unchanged; past
    it the floor rises by `slope` per unit of excess drawdown, clamped to
    `cap` (so the book is never forced fully to cash). Look-ahead-free —
    drawdown is known at decision time.
    """
    if slope <= 0.0 or drawdown <= start:
        return base_floor
    return min(cap, base_floor + slope * (drawdown - start))
