"""Shared retry-within-run helpers for Finnhub per-ticker fetch loops.

Two situations covered:

- **429 (rate limited).** Since the `per_minute_bucket()` fix (2026-07-29),
  429s within a run are rare -- but when one does happen the affected
  ticker used to be dropped silently for the night
  (`except -> logger.warning -> continue`). `fetch_with_429_retry` retries
  the SAME ticker in place instead: sleep ~15-30s (giving Finnhub's
  per-minute window room to clear) while still respecting the shared rate
  limiter, then retry up to `max_retries` more times before giving up with
  the pre-existing warning text.

- **Read timeout.** The 2026-08-03 earnings backfill hit repeated
  `company_earnings` read timeouts against a hardcoded 10s client timeout
  (transient Finnhub slowness, not a rate limit). Callers use the
  lower-level `fetch_with_retry` directly with a
  `requests.exceptions.ReadTimeout` predicate and a single retry.

Both build on the same generic `fetch_with_retry`: retry only when
`is_retryable(exc)` says so, sleep (respecting an optional shared rate
limiter) between attempts, and give up with a warning that matches the
"<source> failed for <ticker>: <exc>" text callers already logged before
this module existed. Non-retryable exceptions are not retried -- retrying
blind adds latency for no benefit and risks masking a real failure.

**Aggregate sleep budget (2026-08-16).** The per-ticker retry is bounded
(2 retries x 15-30s = ~60s), but nothing bounded the SUM across a run. An
ingest pass runs inside the global writer lock, so a 429-storm night over a
267-name universe could add 1-2h of lock hold and starve predict/decide --
re-arming the exact mechanism behind the 8/4 no-trade outage. Callers pass a
`RetrySleepBudget` (default 180s, one per `fetch()` call); once it is spent
the source stops retrying for the rest of that run and falls back to the
original skip-with-warning behavior.
"""

import random
import re
import time
from collections.abc import Callable
from typing import TypeVar

from finnhub.exceptions import FinnhubAPIException
from loguru import logger

from sma.ingest.ratelimit import TokenBucket

T = TypeVar("T")

DEFAULT_MAX_RETRIES = 2
DEFAULT_SLEEP_RANGE_S = (15.0, 30.0)
# Cumulative seconds a single source run may spend sleeping between retries.
# 180s = ~6-12 retry sleeps at the 15-30s range: enough for a handful of
# genuinely rate-limited tickers, nowhere near enough to hold the writer lock
# for hours.
DEFAULT_RETRY_SLEEP_BUDGET_S = 180.0


class RetrySleepBudget:
    """Cumulative cap on retry sleeping for ONE source run.

    Allocate one per `fetch()` call and thread it through every per-ticker
    retry in that run. Not thread-safe by design: the Finnhub sources fetch
    sequentially, and a shared mutable budget across threads would need
    locking for no benefit here.
    """

    __slots__ = ("_exhausted_logged", "spent_s", "total_s")

    def __init__(self, total_s: float = DEFAULT_RETRY_SLEEP_BUDGET_S):
        self.total_s = float(total_s)
        self.spent_s = 0.0
        self._exhausted_logged = False

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.total_s - self.spent_s)

    def exhausted(self) -> bool:
        return self.remaining_s <= 0.0

    def take(self, wait_s: float) -> float:
        """Reserve up to `wait_s` seconds, clamped to what is left. Returns the
        amount granted, so the TOTAL slept in a run never exceeds total_s."""
        granted = min(float(wait_s), self.remaining_s)
        self.spent_s += granted
        return granted

    def note_exhausted(self, source_name: str) -> None:
        """Log the give-up transition once per run, not once per ticker."""
        if self._exhausted_logged:
            return
        self._exhausted_logged = True
        logger.warning(
            "{}: retry-sleep budget of {:.0f}s exhausted; no further retries "
            "this run (remaining rate-limited tickers are skipped with a "
            "warning, as they were before retry-in-place existed)",
            source_name, self.total_s,
        )


# API keys ride in query strings (Finnhub's ?token=, others' apiKey/api_key),
# and requests' ConnectionError text quotes the whole URL. Anything that logs
# an exception from an HTTP client goes through this first (flaw hunt
# 2026-10-01 D1: 726 lines of ingest.err.log held the Finnhub key).
_SECRET_PARAM_RE = re.compile(r"((?:token|api_?key|apikey)=)[^&\s'\")]+", re.IGNORECASE)


def redact_secrets(text: object) -> str:
    return _SECRET_PARAM_RE.sub(r"\1REDACTED", str(text))


def fetch_with_retry(
    source_name: str,
    ticker: str,
    call: Callable[[], T],
    *,
    is_retryable: Callable[[Exception], bool],
    reason: str,
    rate_limiter: TokenBucket | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    max_retries: int = DEFAULT_MAX_RETRIES,
    sleep_range: tuple[float, float] = DEFAULT_SLEEP_RANGE_S,
    sleep_budget: "RetrySleepBudget | None" = None,
) -> T | None:
    """Call `call()`, retrying up to `max_retries` times when `is_retryable(e)`.

    Returns the call's result, or None if it never succeeded. A warning
    matching the pre-existing "<source> failed for <ticker>: <exc>" text is
    logged in the give-up case, so callers can just skip the ticker.

    `sleep_budget` caps the CUMULATIVE retry sleeping across every ticker in
    one source run (see the module docstring). Once it is spent, retries stop
    immediately for the rest of the run — the ticker is skipped with the same
    warning it got before retry-in-place existed. None = unbounded (tests and
    one-off scripts; every scheduled source passes one).
    """
    attempt = 0
    while True:
        try:
            return call()
        except Exception as e:
            if is_retryable(e) and attempt < max_retries:
                if sleep_budget is not None and sleep_budget.exhausted():
                    sleep_budget.note_exhausted(source_name)
                else:
                    attempt += 1
                    wait = random.uniform(*sleep_range)
                    if sleep_budget is not None:
                        wait = sleep_budget.take(wait)
                    logger.warning(
                        "{} {} for {} (retry {}/{}); sleeping {:.0f}s",
                        source_name, reason, ticker, attempt, max_retries, wait,
                    )
                    sleep_fn(wait)
                    if rate_limiter is not None:
                        rate_limiter.acquire()
                    continue
            logger.warning("{} failed for {}: {}", source_name, ticker, redact_secrets(e))
            return None


def _is_finnhub_429(e: Exception) -> bool:
    return isinstance(e, FinnhubAPIException) and getattr(e, "status_code", None) == 429


def fetch_with_429_retry(
    source_name: str,
    ticker: str,
    call: Callable[[], T],
    *,
    rate_limiter: TokenBucket | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    max_retries: int = DEFAULT_MAX_RETRIES,
    sleep_range: tuple[float, float] = DEFAULT_SLEEP_RANGE_S,
    sleep_budget: RetrySleepBudget | None = None,
) -> T | None:
    """Convenience wrapper: retry `call()` on a Finnhub 429 specifically."""
    return fetch_with_retry(
        source_name, ticker, call,
        is_retryable=_is_finnhub_429,
        reason="429",
        rate_limiter=rate_limiter,
        sleep_fn=sleep_fn,
        max_retries=max_retries,
        sleep_range=sleep_range,
        sleep_budget=sleep_budget,
    )
