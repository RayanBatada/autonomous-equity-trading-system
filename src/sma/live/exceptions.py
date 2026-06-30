"""Live module exceptions. All inherit from PhaseFiveError so callers can
catch broadly when graceful degradation is desired.
"""


class PhaseFiveError(Exception):
    """Base for all live-module errors."""


class PreflightAbortError(PhaseFiveError):
    """Pre-flight check failed; no submissions made."""


class EmptyDecisionsWithHeldPositionsError(PhaseFiveError):
    """decide() returned 0 decisions while positions are held — paranoia rail aborted.

    Real cause is almost always a model-load failure or upstream crash, not
    the strategy legitimately wanting zero positions.
    """


class StaleDataError(PhaseFiveError):
    """Required data (price, ingest run, last decide fire) is too old to use safely."""


# CalendarHolidayError is the legacy name for the preflight calendar guard.
# The actual class lives in `sma.live.preflight` (NextSessionNotTomorrow); we
# re-export it here so `from sma.live.exceptions import CalendarHolidayError`
# and `from sma.live.preflight import CalendarHolidayError` resolve to the
# SAME class — `except CalendarHolidayError` works either way. Pre-fix these
# were two unrelated classes and a caller using the exceptions-module import
# would silently miss every actual raise.
from sma.live.preflight import NextSessionNotTomorrow as CalendarHolidayError  # noqa: E402, F401


class IngestNotCompleteError(PhaseFiveError):
    """Today's daily-ingest run hasn't reached status='success' yet."""
