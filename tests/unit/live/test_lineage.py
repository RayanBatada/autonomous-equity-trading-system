"""Run-lineage between predict and ingest sentinels.

2026-06-09: ingest failed at 18:54 (DNS), predict at 19:30 silently built
predictions off STALE 6/08 features and wrote a healthy-looking sentinel.
When ingest later healed (manual rerun, 21:26), the watchdog-kicked decide
would have traded on the stale predictions — both sentinels said READY.
The predict sentinel must carry the ingest run_id it consumed, and decide's
preflight must treat a mismatch as not-ready.
"""

from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sma import schedule as sched
from sma.live.preflight import UpstreamMissedDeadline, run_preflight
from sma.readiness import sentinel_lineage_stale
from sma.sentinels import write_sentinel

# --- pure contract -----------------------------------------------------------


def test_lineage_stale_when_run_ids_differ():
    assert sentinel_lineage_stale(
        consumer={"ingest_run_id": 1}, upstream={"run_id": 2}
    ) is True


def test_lineage_fresh_when_run_ids_match():
    assert sentinel_lineage_stale(
        consumer={"ingest_run_id": 7}, upstream={"run_id": 7}
    ) is False


def test_lineage_tolerant_when_fields_or_sentinels_missing():
    # Old-format predict sentinel (pre-lineage) must not brick the pipeline.
    assert sentinel_lineage_stale(consumer={}, upstream={"run_id": 2}) is False
    assert sentinel_lineage_stale(consumer={"ingest_run_id": 1}, upstream={}) is False
    assert sentinel_lineage_stale(consumer=None, upstream={"run_id": 2}) is False
    assert sentinel_lineage_stale(consumer={"ingest_run_id": 1}, upstream=None) is False


# --- preflight integration ---------------------------------------------------


def _mock_alpaca():
    alpaca = MagicMock()
    alpaca.next_session_date.return_value = date(2026, 5, 1)
    return alpaca


def _write_chain(asof, *, ingest_run_id, predict_consumed_run_id):
    write_sentinel(
        label="com.sma.ingest.daily",
        asof=asof,
        payload={
            "run_id": ingest_run_id,
            "quality": {"passed": True, "blocking_failures": []},
        },
    )
    payload = {"run_id": 1, "completed_at": "2026-04-30T23:30:00Z"}
    if predict_consumed_run_id is not None:
        payload["ingest_run_id"] = predict_consumed_run_id
    write_sentinel(label="com.sma.model.predict.daily", asof=asof, payload=payload)
    write_sentinel(
        label="com.sma.agents.daily",
        asof=asof,
        payload={"run_id": 1, "quality": {"passed": True, "blocking_failures": []}},
    )


def test_preflight_rejects_stale_predict_lineage(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    asof = date(2026, 4, 30)
    _write_chain(asof, ingest_run_id=200, predict_consumed_run_id=100)
    with pytest.raises(UpstreamMissedDeadline, match="lineage"):
        run_preflight(asof=asof, db_path=Path("x"), alpaca=_mock_alpaca(), max_wait_s=0)


def test_preflight_accepts_matching_predict_lineage(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    asof = date(2026, 4, 30)
    _write_chain(asof, ingest_run_id=200, predict_consumed_run_id=200)
    run_preflight(asof=asof, db_path=Path("x"), alpaca=_mock_alpaca(), max_wait_s=0)


def test_preflight_accepts_old_format_predict_sentinel(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    asof = date(2026, 4, 30)
    _write_chain(asof, ingest_run_id=200, predict_consumed_run_id=None)
    run_preflight(asof=asof, db_path=Path("x"), alpaca=_mock_alpaca(), max_wait_s=0)


def test_preflight_late_start_gets_grace_window_for_kicked_upstream(
    tmp_path, monkeypatch
):
    """A watchdog-kicked decide at 22:00 (deadline 21:00) must POLL briefly —
    the watchdog kicks predict in the same pass; dying instantly on deadline
    math would strand the night even though predict lands seconds later."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    asof = date(2026, 4, 30)
    write_sentinel(
        label="com.sma.ingest.daily",
        asof=asof,
        payload={
            "run_id": 200,
            "quality": {"passed": True, "blocking_failures": []},
        },
    )
    # predict sentinel intentionally MISSING at start
    write_sentinel(
        label="com.sma.agents.daily",
        asof=asof,
        payload={"run_id": 1, "quality": {"passed": True, "blocking_failures": []}},
    )

    late_now = datetime.combine(asof, datetime.min.time(), tzinfo=sched.NY_TZ).replace(
        hour=22, minute=0
    )

    def fake_sleep(_s):
        # the concurrently-kicked predict lands during the first poll
        write_sentinel(
            label="com.sma.model.predict.daily",
            asof=asof,
            payload={
                "run_id": 1,
                "completed_at": "2026-05-01T02:01:00Z",
                "ingest_run_id": 200,
            },
        )

    run_preflight(
        asof=asof,
        db_path=Path("x"),
        alpaca=_mock_alpaca(),
        max_wait_s=600,
        poll_interval_s=1,
        now_fn=lambda: late_now,
        sleep_fn_poll=fake_sleep,
    )
