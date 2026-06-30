"""Exponential-backoff retry with jitter.

`retry(fn, cfg)` calls fn() up to cfg.max times, sleeping
`base_delay * 4**attempt * uniform(1 - jitter, 1 + jitter)` between attempts.

Pass `do_not_retry=(SomeError,)` for exceptions that should fail fast (e.g.,
401 Unauthorized: retrying won't help).
"""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryConfig:
    max: int
    base_delay: float
    jitter: float


def retry(
    fn: Callable[[], T],
    cfg: RetryConfig,
    *,
    sleep: Callable[[float], None] = time.sleep,
    do_not_retry: tuple[type[BaseException], ...] = (),
) -> T:
    last_exc: BaseException | None = None
    for attempt in range(cfg.max):
        try:
            return fn()
        except do_not_retry:
            raise
        except Exception as e:
            last_exc = e
            if attempt == cfg.max - 1:
                break
            base = cfg.base_delay * (4 ** attempt)
            jitter_factor = 1.0 + random.uniform(-cfg.jitter, cfg.jitter)
            sleep(base * jitter_factor)
    assert last_exc is not None
    raise last_exc
