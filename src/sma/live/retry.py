"""Shared transient-failure retry for read-only Alpaca calls on the live path.

Extracted from live.__main__ so preflight (which __main__ imports) can reuse it
without a circular import. 2026-06-09: a single DNS blip killed the 09:25
stop-loss sweep and the bot ran unprotected all day. 2026-06-24 audit: the
decide-preflight next_session_date call was the one remaining un-wrapped
read-only Alpaca call on the money path, and decide runs shortly after the
laptop wakes (warm-DNS today, but the same failure mode as the stop-loss kill).
A read-only call is always safe to retry.
"""

from __future__ import annotations

import time

# Patchable sleep for retry tests.
_SLEEP = time.sleep


def _retry_transient(fn, *, label: str, attempts: int = 3, delay_s: float = 60.0):
    """Call `fn()` retrying transient failures (network/DNS blips).

    A read-only call is always safe to retry; the last failure propagates.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts:
                raise
            from loguru import logger

            logger.warning(
                f"{label}: attempt {attempt}/{attempts} failed ({e!r}); "
                f"retrying in {delay_s}s"
            )
            _SLEEP(delay_s)
    raise AssertionError("unreachable")
