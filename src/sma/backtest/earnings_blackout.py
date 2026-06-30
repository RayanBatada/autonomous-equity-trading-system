"""Backwards-compatibility shim. earnings_blackout moved to sma.risk in Phase 5."""

from sma.risk.earnings_blackout import (
    BLACKOUT_DAYS,
    in_earnings_blackout,
    load_upcoming_earnings,
)

__all__ = ["BLACKOUT_DAYS", "in_earnings_blackout", "load_upcoming_earnings"]
