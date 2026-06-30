"""Token-bucket rate limiter.

Each source gets one bucket sized to its provider's rate limit (e.g. Finnhub
55 req/min, capacity 55, refill 55/60 = 0.917/s).

`try_acquire()` is non-blocking; `acquire()` blocks (sleeping) until a token
is available. Time source uses `time.time()` (not monotonic) so freezegun
can drive it in tests. The trade-off is small: system clock changes are very
rare in normal operation, and it makes the limiter testable.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    capacity: int
    refill_rate_per_sec: float
    _tokens: float = field(init=False)
    _last_refill: float = field(init=False)

    def __post_init__(self) -> None:
        self._tokens = float(self.capacity)
        self._last_refill = time.time()

    def _refill(self) -> None:
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(
            float(self.capacity),
            self._tokens + elapsed * self.refill_rate_per_sec,
        )
        self._last_refill = now

    def try_acquire(self, n: float = 1.0) -> bool:
        self._refill()
        if self._tokens >= n:
            self._tokens -= n
            return True
        return False

    def acquire(self, n: float = 1.0, sleep: Callable[[float], None] = time.sleep) -> None:
        while not self.try_acquire(n):
            self._refill()
            deficit = n - self._tokens
            wait = deficit / self.refill_rate_per_sec
            sleep(max(wait, 0.001))
