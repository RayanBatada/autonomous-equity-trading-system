
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
