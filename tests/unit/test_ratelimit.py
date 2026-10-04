import pytest
from freezegun import freeze_time

from sma.ingest.ratelimit import TokenBucket


def test_bucket_starts_full():
    b = TokenBucket(capacity=10, refill_rate_per_sec=1.0)
    for _ in range(10):
        assert b.try_acquire() is True
    assert b.try_acquire() is False


def test_bucket_refills_over_time():
    with freeze_time("2026-01-01 12:00:00") as frozen:
        b = TokenBucket(capacity=2, refill_rate_per_sec=1.0)
        assert b.try_acquire() is True
        assert b.try_acquire() is True
        assert b.try_acquire() is False
        frozen.tick(delta=2.0)
        assert b.try_acquire() is True
        assert b.try_acquire() is True
        assert b.try_acquire() is False


def test_bucket_capacity_caps_refill():
    with freeze_time("2026-01-01 12:00:00") as frozen:
        b = TokenBucket(capacity=2, refill_rate_per_sec=1.0)
        b.try_acquire(); b.try_acquire()
        frozen.tick(delta=10.0)
        assert b.try_acquire() is True
        assert b.try_acquire() is True
        assert b.try_acquire() is False


def test_acquire_waits_when_empty():
    with freeze_time("2026-01-01 12:00:00") as frozen:
        b = TokenBucket(capacity=1, refill_rate_per_sec=10.0)
        b.try_acquire()
        slept_for: list[float] = []

        def fake_sleep(s: float) -> None:
            slept_for.append(s)
            frozen.tick(delta=s)

        b.acquire(sleep=fake_sleep)
        assert 0.05 < slept_for[0] < 0.2


# ---------------------------------------------------------------------------
# per_minute_bucket: burst must be decoupled from the sustained rate (2026-07-29)
# ---------------------------------------------------------------------------


def test_per_minute_bucket_does_not_allow_a_full_minute_burst():
    """A 55/min limiter must NOT fire 55 requests instantly.

    Regression: the Finnhub limiter was built as
    `TokenBucket(capacity=rpm, refill_rate_per_sec=rpm/60)`, and TokenBucket
    starts FULL (`_tokens = capacity`). So the first ~55 calls returned with zero
    delay — a 55-request instantaneous burst against a provider that enforces
    60/min on a short window. Finnhub then throttled for the rest of the run:
    4,303 HTTP 429s in the ingest log, the first ~65 tickers succeeding
    alphabetically and everything after them failing, every single night.
    """
    from sma.ingest.ratelimit import per_minute_bucket

    bucket = per_minute_bucket(55)
    instant = 0
    while bucket.try_acquire() and instant < 100:
        instant += 1
    assert instant < 55, f"fired {instant} requests with no delay — still bursty"
    assert instant >= 1, "must allow at least one immediate request"


def test_per_minute_bucket_keeps_the_sustained_rate():
    """Sustained throughput still tracks requests_per_minute/60 per second."""
    from sma.ingest.ratelimit import per_minute_bucket

    bucket = per_minute_bucket(60)
    assert bucket.refill_rate_per_sec == pytest.approx(1.0)


def test_per_minute_bucket_burst_is_configurable_and_at_least_one():
    from sma.ingest.ratelimit import per_minute_bucket

    assert per_minute_bucket(55, burst=3).capacity == 3
    # A zero/negative burst would deadlock acquire(); floor it at 1.
    assert per_minute_bucket(55, burst=0).capacity == 1


# ---------------------------------------------------------------------------
# Thread safety (2026-09-02, ingest wall-clock task): finnhub_news and
# finnhub_fundamentals share ONE TokenBucket instance in production (see
# sma.ingest.__main__._build_source's shared `finnhub_limiter`). Parallelizing
# source fetches means both can now call try_acquire()/acquire() from separate
# threads on the SAME bucket concurrently. The original _refill-then-decrement
# sequence was a classic read-modify-write race with no lock -- concurrent
# callers could both pass the `_tokens >= n` check before either decremented,
# over-issuing tokens beyond capacity and re-risking the 2026-07-29 429-storm
# bug this bucket exists to prevent (see per_minute_bucket's docstring).
# ---------------------------------------------------------------------------


def test_try_acquire_is_thread_safe_under_concurrent_access():
    import sys
    import threading

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # force frequent GIL switches to expose the race
    try:
        bucket = TokenBucket(capacity=1, refill_rate_per_sec=0.0)
        n_threads = 200
        barrier = threading.Barrier(n_threads)
        results: list[bool | None] = [None] * n_threads

        def worker(i: int) -> None:
            barrier.wait()  # line every thread up to hit try_acquire() together
            results[i] = bucket.try_acquire()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        granted = sum(1 for r in results if r)
        assert granted == 1, (
            f"bucket capacity=1 granted {granted} tokens to {n_threads} "
            "concurrent callers -- try_acquire() is not thread-safe"
        )
    finally:
        sys.setswitchinterval(old_interval)


def test_try_acquire_never_over_issues_under_sustained_concurrent_load():
    """Broader stress test at a realistic capacity/refill rate: total tokens
    granted across many concurrent callers must never exceed what capacity +
    elapsed-time refill can physically justify."""
    import threading
    import time as _time

    bucket = TokenBucket(capacity=5, refill_rate_per_sec=1000.0)
    n_threads = 64
    calls_per_thread = 50
    granted_lock = threading.Lock()
    total_granted = [0]

    def worker() -> None:
        local = 0
        for _ in range(calls_per_thread):
            if bucket.try_acquire():
                local += 1
        with granted_lock:
            total_granted[0] += local

    start = _time.time()
    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = _time.time() - start

    max_possible = bucket.capacity + elapsed * bucket.refill_rate_per_sec + 1  # +1 slack
    assert total_granted[0] <= max_possible, (
        f"granted {total_granted[0]} tokens in {elapsed:.4f}s, but capacity="
        f"{bucket.capacity} + refill only justifies <= {max_possible:.1f} -- "
        "tokens were over-issued under concurrent access"
    )
