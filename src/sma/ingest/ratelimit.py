"""Token-bucket rate limiter.

Each source gets one bucket whose SUSTAINED rate matches its provider's limit.
Build them with `per_minute_bucket()`, not the raw constructor: `capacity` is the
BURST allowance and the bucket starts full, so `capacity=requests_per_minute`
lets a whole minute's quota fire instantaneously (see per_minute_bucket docs).

`try_acquire()` is non-blocking; `acquire()` blocks (sleeping) until a token
is available. Time source uses `time.time()` (not monotonic) so freezegun
can drive it in tests. The trade-off is small: system clock changes are very
rare in normal operation, and it makes the limiter testable.

Thread safety (2026-09-02): finnhub_news and finnhub_fundamentals share ONE
bucket instance (`sma.ingest.__main__._build_source`'s `finnhub_limiter`).
Since the ingest runner now fetches non-price sources concurrently in threads
(see `sma.ingest.runner`), both can call try_acquire()/acquire() on the SAME
bucket at once. The refill-then-decrement sequence is a read-modify-write and
is NOT safe across threads without a lock -- verified via a reproducing test
(`tests/unit/test_ratelimit.py::test_try_acquire_is_thread_safe_under_concurrent_access`)
before this fix: concurrent callers could both pass the `_tokens >= n` check
before either decremented, over-issuing tokens beyond capacity and re-risking
the 2026-07-29 429-storm bug this bucket exists to prevent. `try_acquire()`
now holds `_lock` around its whole refill+decide+decrement sequence, which is
sufficient: `acquire()`'s retry loop only ever succeeds via a `try_acquire()`
call that itself atomically claimed the token, so no separate lock is needed
around the loop itself.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    capacity: int
    refill_rate_per_sec: float
    _tokens: float = field(init=False)
    _last_refill: float = field(init=False)
    _lock: threading.Lock = field(
        init=False, default_factory=threading.Lock, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self._tokens = float(self.capacity)
        self._last_refill = time.time()

    def _refill(self) -> None:
        # Caller must hold self._lock.
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(
            float(self.capacity),
            self._tokens + elapsed * self.refill_rate_per_sec,
        )
        self._last_refill = now

    def try_acquire(self, n: float = 1.0) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False

    def acquire(self, n: float = 1.0, sleep: Callable[[float], None] = time.sleep) -> None:
        while not self.try_acquire(n):
            with self._lock:
                self._refill()
                deficit = n - self._tokens
                wait = deficit / self.refill_rate_per_sec
            sleep(max(wait, 0.001))


# Default burst allowance. Small on purpose: enough to absorb normal jitter
# without letting a run front-load a whole minute's quota into one second.
DEFAULT_BURST = 5


def per_minute_bucket(requests_per_minute: int, *, burst: int = DEFAULT_BURST) -> TokenBucket:
    """Bucket whose SUSTAINED rate is `requests_per_minute`, with a small burst.

    Use this instead of `TokenBucket(capacity=rpm, refill_rate_per_sec=rpm/60)`.

    `capacity` is the burst allowance AND the bucket's starting balance
    (`__post_init__` sets `_tokens = capacity`), so sizing capacity to the
    per-minute quota lets the first `rpm` calls fire with **zero delay**. That is
    a full minute of traffic in one instant, which trips providers that enforce
    their limit over a short window even though the per-minute average is legal.

    Observed 2026-07-29: the Finnhub limiter was built that way (capacity=55).
    Every ingest burst-fired ~55 requests, Finnhub throttled, and it never
    recovered inside the run — 4,303 HTTP 429s in the log, the first ~65 tickers
    succeeding alphabetically and the rest of the universe getting no news, every
    night for weeks. Decoupling burst from rate is the fix; the sustained
    requests_per_minute is unchanged.

    `burst` is floored at 1 — a zero-capacity bucket can never satisfy
    `acquire(1)` and would spin forever.
    """
    return TokenBucket(
        capacity=max(1, int(burst)),
        refill_rate_per_sec=requests_per_minute / 60.0,
    )
