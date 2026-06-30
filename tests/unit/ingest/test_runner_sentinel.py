"""After Task 5: sma.ingest.runner.run() writes a sentinel after success."""

from datetime import date

import pytest

from sma.locks import writer_lock
from sma.sentinels import read_sentinel


def test_ingest_run_writes_sentinel_with_quality_payload(tmp_path, monkeypatch):
    """Run a stub ingest with no sources; verify sentinel is written with quality verdict."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", tmp_path / ".sma-writer.lock")

    db_path = tmp_path / "test.duckdb"
    asof = date(2026, 4, 30)

    from sma.ingest.runner import run as ingest_run

    with writer_lock(label="test-ingest", lock_path=tmp_path / ".sma-writer.lock"):
        ingest_run(asof=asof, db_path=db_path, sources=[], universe=[])

    sentinel = read_sentinel(label="com.sma.ingest.daily", asof=asof)
    assert sentinel is not None
    assert sentinel["label"] == "com.sma.ingest.daily"
    assert sentinel["asof"] == asof.isoformat()
    assert "run_id" in sentinel
    assert "started_at" in sentinel
    assert "completed_at" in sentinel
    assert "sources" in sentinel
    assert "quality" in sentinel
    assert "passed" in sentinel["quality"]
    assert "blocking_failures" in sentinel["quality"]


def test_ingest_writes_failure_sentinel_when_quality_checks_raise(tmp_path, monkeypatch):
    """If quality checks crash, run() must STILL write a quality.passed=False
    sentinel. Otherwise no sentinel exists, decide's preflight has nothing to
    read, and the pipeline blocks/polls until timeout instead of failing clean."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", tmp_path / ".sma-writer.lock")

    def boom(*a, **k):
        raise RuntimeError("quality check exploded")

    monkeypatch.setattr("sma.ingest.quality.run_quality_checks", boom)

    db_path = tmp_path / "test.duckdb"
    asof = date(2026, 4, 30)
    from sma.ingest.runner import run as ingest_run

    with (
        writer_lock(label="test-ingest", lock_path=tmp_path / ".sma-writer.lock"),
        pytest.raises(RuntimeError, match="quality check exploded"),
    ):
        ingest_run(asof=asof, db_path=db_path, sources=[], universe=[])

    sentinel = read_sentinel(label="com.sma.ingest.daily", asof=asof)
    assert sentinel is not None, "no sentinel written → decide would block forever"
    assert sentinel["quality"]["passed"] is False


class _FakeSource:
    def __init__(self, name, status="ok", rows=3, crash=False):
        self.name = name
        self._status = status
        self._rows = rows
        self._crash = crash

    def fetch(self, universe, asof, store, run_id):
        from sma.ingest.sources.base import IngestResult

        if self._crash:
            raise RuntimeError("boom")
        return IngestResult(
            source=self.name, rows_inserted=self._rows, status=self._status, error=None
        )


def test_ingest_sentinel_records_per_source_results(tmp_path, monkeypatch):
    """The sentinel's `sources` block was hardcoded {} — it must carry each
    source's final status/rows so a failed night is diagnosable from the
    sentinel alone (2026-06-09: sources:{} hid which source died)."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", tmp_path / ".sma-writer.lock")

    db_path = tmp_path / "test.duckdb"
    asof = date(2026, 4, 30)
    from sma.ingest.runner import run as ingest_run

    sources = [_FakeSource("good", rows=3), _FakeSource("bad", crash=True)]
    with writer_lock(label="test-ingest", lock_path=tmp_path / ".sma-writer.lock"):
        ingest_run(asof=asof, db_path=db_path, sources=sources, universe=["AAPL"])

    s = read_sentinel(label="com.sma.ingest.daily", asof=asof)
    assert s["sources"]["good"] == {"status": "ok", "rows_inserted": 3}
    assert s["sources"]["bad"]["status"] == "error"
    assert s["sources"]["bad"]["rows_inserted"] == 0
