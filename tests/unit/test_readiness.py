from sma.readiness import ReadinessState, live_readiness


def _ingest_sentinel(
    quality_passed: bool, blocking: list[str] | None = None, waived: list[str] | None = None
):
    return {
        "label": "com.sma.ingest.daily",
        "asof": "2026-04-30",
        "quality": {
            "passed": quality_passed,
            "blocking_failures": blocking or [],
            "waived": waived or [],
        },
    }


def _predict_sentinel():
    """Predict's real sentinel shape: a success-only sentinel with NO `quality`
    key. Predict has no quality gate, and it writes this sentinel only after
    predictions persist (inside the writer lock), so the file's existence means
    the job completed successfully. Carries `completed_at` (real shape)."""
    return {
        "label": "com.sma.model.predict.daily",
        "asof": "2026-06-04",
        "completed_at": "2026-06-04T23:30:08.104246Z",
        "model_id": "xgb_ret_30d_forward_2026-06-01_4c54748c",
        "rows_written": 188,
        "tickers": 188,
    }


def test_ready_when_quality_passed():
    sentinel = _ingest_sentinel(quality_passed=True)
    r = live_readiness(label="com.sma.ingest.daily", sentinel=sentinel, waivers=frozenset())
    assert r.state == ReadinessState.READY
    assert r.is_ready()


def test_ready_when_sentinel_has_no_quality_block():
    """A non-gated job (predict) writes a success-only sentinel with no `quality`
    key. Its presence means the job completed (predict writes it only after
    predictions persist), so it must be READY — NOT swept up by the passed=False
    fail-closed branch, which is meant only for a present-but-malformed quality
    block. Regression guard: the fail-closed branch added 2026-06-04 turned every
    quality-less predict sentinel into FAIL and froze all live trading."""
    sentinel = _predict_sentinel()
    r = live_readiness(
        label="com.sma.model.predict.daily", sentinel=sentinel, waivers=frozenset()
    )
    assert r.state == ReadinessState.READY
    assert r.is_ready()


def test_fail_when_gated_job_sentinel_has_no_quality_block():
    """A GATED job (e.g. ingest, which ALWAYS writes a quality block) appearing
    without one is malformed/crashed — it must FAIL, not be mistaken for a
    non-gated success. Guards the quality-less branch from green-lighting a
    broken hard dependency and trading on bad data."""
    sentinel = {
        "label": "com.sma.ingest.daily",
        "asof": "2026-06-04",
        "completed_at": "2026-06-04T22:35:59Z",
    }
    r = live_readiness(label="com.sma.ingest.daily", sentinel=sentinel, waivers=frozenset())
    assert r.state == ReadinessState.FAIL


def test_fail_when_non_gated_sentinel_lacks_completion_evidence():
    """Even a recognized non-gated label must carry positive completion evidence
    (completed_at). An empty/truncated payload at the predict path must FAIL, not
    READY."""
    r = live_readiness(
        label="com.sma.model.predict.daily",
        sentinel={"label": "com.sma.model.predict.daily", "rows_written": 0},
        waivers=frozenset(),
    )
    assert r.state == ReadinessState.FAIL


def test_fail_when_quality_block_is_not_a_dict():
    """A present-but-malformed quality value (null/list/scalar) must FAIL closed,
    not raise an AttributeError that crashes preflight."""
    for bad in (None, [], "passed", 1):
        sentinel = {"label": "com.sma.ingest.daily", "quality": bad}
        r = live_readiness(label="com.sma.ingest.daily", sentinel=sentinel, waivers=frozenset())
        assert r.state == ReadinessState.FAIL, f"quality={bad!r} should FAIL"


def test_wait_when_sentinel_missing():
    r = live_readiness(label="com.sma.ingest.daily", sentinel=None, waivers=frozenset())
    assert r.state == ReadinessState.WAIT
    assert not r.is_ready()


def test_fail_when_blocking_failure_not_waived():
    sentinel = _ingest_sentinel(quality_passed=False, blocking=["all_tickers_have_price"])
    r = live_readiness(label="com.sma.ingest.daily", sentinel=sentinel, waivers=frozenset())
    assert r.state == ReadinessState.FAIL
    assert "all_tickers_have_price" in r.explanation


def test_fail_when_passed_false_and_no_blocking():
    """passed=False with an EMPTY blocking_failures list must FAIL, not READY —
    fail closed so a malformed/failed sentinel can't green-light the downstream
    job (the all-waived branch previously fired on an empty blocking list)."""
    sentinel = _ingest_sentinel(quality_passed=False, blocking=[])
    r = live_readiness(label="com.sma.ingest.daily", sentinel=sentinel, waivers=frozenset())
    assert r.state == ReadinessState.FAIL


def test_ready_when_only_blocking_failure_is_waived():
    sentinel = _ingest_sentinel(quality_passed=False, blocking=["theses_freshness"])
    r = live_readiness(
        label="com.sma.ingest.daily",
        sentinel=sentinel,
        waivers=frozenset({"theses_freshness"}),
    )
    assert r.state == ReadinessState.READY
    assert "theses_freshness" in r.explanation


def test_fail_when_some_blocking_waived_some_not():
    sentinel = _ingest_sentinel(
        quality_passed=False,
        blocking=["theses_freshness", "all_tickers_have_price"],
    )
    r = live_readiness(
        label="com.sma.ingest.daily",
        sentinel=sentinel,
        waivers=frozenset({"theses_freshness"}),
    )
    assert r.state == ReadinessState.FAIL
    assert "all_tickers_have_price" in r.explanation
    assert "theses_freshness" not in r.explanation  # waived; not in unwaived blocking list


def test_state_string_values():
    """The enum values are strings for easy serialization."""
    assert ReadinessState.READY.value == "READY"
    assert ReadinessState.WAIT.value == "WAIT"
    assert ReadinessState.FAIL.value == "FAIL"


def test_backup_sentinel_with_failed_verification_fails_closed():
    """backup is gated (not in _NON_GATED_LABELS): a sentinel whose verification
    failed (quality.passed=False) must read FAIL, not be auto-passed as ready."""
    sentinel = {
        "label": "com.sma.backup.daily",
        "completed_at": "2026-06-05T02:03:32Z",
        "verified": False,
        "quality": {"passed": False, "blocking_failures": ["backup_verification_failed"]},
    }
    r = live_readiness(label="com.sma.backup.daily", sentinel=sentinel, waivers=frozenset())
    assert r.state == ReadinessState.FAIL
