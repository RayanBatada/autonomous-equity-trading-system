import pytest

from sma.ingest.retry import RetryConfig, retry


def test_retry_returns_value_on_first_success():
    calls = {"n": 0}
    def f():
        calls["n"] += 1
        return "ok"
    assert retry(f, RetryConfig(max=3, base_delay=0.0, jitter=0.0)) == "ok"
    assert calls["n"] == 1


def test_retry_retries_until_success():
    calls = {"n": 0}
    def f():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"
    assert retry(f, RetryConfig(max=3, base_delay=0.0, jitter=0.0)) == "ok"
    assert calls["n"] == 3


def test_retry_raises_after_max_attempts():
    calls = {"n": 0}
    def f():
        calls["n"] += 1
        raise RuntimeError("always fails")
    with pytest.raises(RuntimeError):
        retry(f, RetryConfig(max=3, base_delay=0.0, jitter=0.0))
    assert calls["n"] == 3


def test_retry_uses_exponential_backoff():
    sleeps: list[float] = []
    def fake_sleep(s: float) -> None:
        sleeps.append(s)
    def f():
        raise RuntimeError("nope")
    with pytest.raises(RuntimeError):
        retry(f, RetryConfig(max=3, base_delay=1.0, jitter=0.0), sleep=fake_sleep)
    assert sleeps == [1.0, 4.0]


def test_retry_applies_jitter():
    sleeps: list[float] = []
    def fake_sleep(s: float) -> None:
        sleeps.append(s)
    def f():
        raise RuntimeError("nope")
    with pytest.raises(RuntimeError):
        retry(f, RetryConfig(max=3, base_delay=1.0, jitter=0.5), sleep=fake_sleep)
    assert 0.5 <= sleeps[0] <= 1.5
    assert 2.0 <= sleeps[1] <= 6.0


def test_retry_skips_sleep_when_excluded_exception_raised():
    calls = {"n": 0}
    class FatalError(Exception):
        pass
    def f():
        calls["n"] += 1
        raise FatalError()
    with pytest.raises(FatalError):
        retry(f, RetryConfig(max=3, base_delay=0.0, jitter=0.0), do_not_retry=(FatalError,))
    assert calls["n"] == 1
