from unittest.mock import MagicMock

from finnhub.exceptions import FinnhubAPIException

from sma.ingest.ratelimit import TokenBucket
from sma.ingest.sources._finnhub_retry import (
    DEFAULT_RETRY_SLEEP_BUDGET_S,
    RetrySleepBudget,
    fetch_with_429_retry,
    fetch_with_retry,
)


def _fake_429():
    return FinnhubAPIException(MagicMock(status_code=429, text="rate"))


def _fake_500():
    return FinnhubAPIException(MagicMock(status_code=500, text="server error"))


# ---------- fetch_with_retry (generic) ----------


def test_fetch_with_retry_returns_value_on_first_success():
    sleeps = []
    result = fetch_with_retry(
        "src", "AAPL", lambda: "ok",
        is_retryable=lambda e: True,
        reason="whatever",
        sleep_fn=sleeps.append,
    )
    assert result == "ok"
    assert sleeps == []


def test_fetch_with_retry_retries_then_succeeds():
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        if calls["n"] < 2:
            raise TimeoutError("transient")
        return "ok"

    sleeps = []
    result = fetch_with_retry(
        "src", "AAPL", call,
        is_retryable=lambda e: isinstance(e, TimeoutError),
        reason="timeout",
        sleep_fn=sleeps.append,
        max_retries=2,
    )
    assert result == "ok"
    assert calls["n"] == 2
    assert len(sleeps) == 1


def test_fetch_with_retry_gives_up_after_max_retries():
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        raise TimeoutError("always")

    sleeps = []
    result = fetch_with_retry(
        "src", "AAPL", call,
        is_retryable=lambda e: isinstance(e, TimeoutError),
        reason="timeout",
        sleep_fn=sleeps.append,
        max_retries=2,
    )
    assert result is None
    assert calls["n"] == 3  # initial + 2 retries
    assert len(sleeps) == 2


def test_fetch_with_retry_does_not_retry_non_retryable_exception():
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        raise RuntimeError("fatal")

    sleeps = []
    result = fetch_with_retry(
        "src", "AAPL", call,
        is_retryable=lambda e: isinstance(e, TimeoutError),
        reason="timeout",
        sleep_fn=sleeps.append,
        max_retries=2,
    )
    assert result is None
    assert calls["n"] == 1
    assert sleeps == []


def test_fetch_with_retry_reacquires_rate_limiter_between_retries():
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        if calls["n"] < 2:
            raise TimeoutError("transient")
        return "ok"

    rl = MagicMock(spec=TokenBucket)
    fetch_with_retry(
        "src", "AAPL", call,
        is_retryable=lambda e: isinstance(e, TimeoutError),
        reason="timeout",
        rate_limiter=rl,
        sleep_fn=lambda s: None,
        max_retries=2,
    )
    assert rl.acquire.call_count == 1


# ---------- fetch_with_429_retry (Finnhub 429 convenience wrapper) ----------


def test_fetch_with_429_retry_retries_on_429_then_succeeds():
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        if calls["n"] < 2:
            raise _fake_429()
        return "ok"

    sleeps = []
    result = fetch_with_429_retry(
        "finnhub_news", "AAPL", call, sleep_fn=sleeps.append,
    )
    assert result == "ok"
    assert calls["n"] == 2
    assert len(sleeps) == 1
    assert 15.0 <= sleeps[0] <= 30.0


def test_fetch_with_429_retry_gives_up_after_three_429s():
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        raise _fake_429()

    result = fetch_with_429_retry(
        "finnhub_news", "AAPL", call, sleep_fn=lambda s: None,
    )
    assert result is None
    assert calls["n"] == 3  # initial + 2 retries


def test_fetch_with_429_retry_does_not_retry_non_429_finnhub_exception():
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        raise _fake_500()

    result = fetch_with_429_retry(
        "finnhub_news", "AAPL", call, sleep_fn=lambda s: None,
    )
    assert result is None
    assert calls["n"] == 1


def test_fetch_with_429_retry_does_not_retry_other_exceptions():
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        raise RuntimeError("boom")

    result = fetch_with_429_retry(
        "finnhub_news", "AAPL", call, sleep_fn=lambda s: None,
    )
    assert result is None
    assert calls["n"] == 1


# ---------- RetrySleepBudget (aggregate cap, 2026-08-16) ----------
#
# Per-ticker retries are bounded (2 x 15-30s) but the SUM across a run was not.
# Ingest runs inside the global writer lock, so a 429-storm night over 267
# names could add 1-2h of lock hold and starve predict/decide -- re-arming the
# 8/4 no-trade mechanism. These pin the cap.


def _always_429():
    def call():
        raise _fake_429()
    return call


def test_budget_default_is_180s():
    assert DEFAULT_RETRY_SLEEP_BUDGET_S == 180.0
    assert RetrySleepBudget().total_s == 180.0


def test_total_sleep_across_tickers_never_exceeds_the_budget():
    """The invariant that matters: a whole run of nothing but 429s sleeps at
    most `total_s`, no matter how many tickers are in the universe."""
    budget = RetrySleepBudget(50.0)
    sleeps = []
    for ticker in [f"T{i}" for i in range(50)]:
        fetch_with_429_retry(
            "finnhub_news", ticker, _always_429(),
            sleep_fn=sleeps.append, sleep_budget=budget,
        )

    assert sum(sleeps) <= 50.0
    assert budget.exhausted()


def test_once_exhausted_further_429s_skip_immediately():
    """After the budget is gone, a rate-limited ticker is skipped with a
    warning and ZERO sleep -- the pre-retry-in-place behavior."""
    budget = RetrySleepBudget(50.0)
    sleeps = []
    for ticker in [f"T{i}" for i in range(10)]:
        fetch_with_429_retry(
            "finnhub_news", ticker, _always_429(),
            sleep_fn=sleeps.append, sleep_budget=budget,
        )
    assert budget.exhausted()

    calls = {"n": 0}

    def call():
        calls["n"] += 1
        raise _fake_429()

    later_sleeps = []
    result = fetch_with_429_retry(
        "finnhub_news", "ZZZZ", call,
        sleep_fn=later_sleeps.append, sleep_budget=budget,
    )

    assert result is None
    assert later_sleeps == [], "an exhausted budget must not sleep at all"
    assert calls["n"] == 1, "exactly one attempt: no retries once exhausted"


def test_exhausted_budget_does_not_reacquire_the_rate_limiter():
    """No retry means no post-sleep re-acquire — the skip path must be free."""
    budget = RetrySleepBudget(0.0)
    rl = MagicMock(spec=TokenBucket)

    fetch_with_429_retry(
        "finnhub_news", "AAPL", _always_429(),
        rate_limiter=rl, sleep_fn=lambda s: None, sleep_budget=budget,
    )

    rl.acquire.assert_not_called()


def test_budget_still_allows_normal_retries_while_it_has_room():
    """The cap must not break the thing it bounds: with budget left, a 429
    still retries in place and can succeed."""
    budget = RetrySleepBudget(100.0)
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        if calls["n"] < 2:
            raise _fake_429()
        return "ok"

    sleeps = []
    result = fetch_with_429_retry(
        "finnhub_news", "AAPL", call, sleep_fn=sleeps.append, sleep_budget=budget,
    )

    assert result == "ok"
    assert len(sleeps) == 1
    assert budget.spent_s == sleeps[0]
    assert not budget.exhausted()


def test_no_budget_means_unbounded():
    """Callers that pass no budget (tests, one-off scripts) keep the old
    behavior rather than silently getting a cap they never asked for."""
    sleeps = []
    fetch_with_429_retry(
        "finnhub_news", "AAPL", _always_429(), sleep_fn=sleeps.append,
    )
    assert len(sleeps) == 2
