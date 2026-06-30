from datetime import date
from unittest.mock import MagicMock

import pytest

from sma.ingest.runner import IngestRunner
from sma.ingest.sources.base import IngestResult
from sma.ingest.store import Store


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    yield s
    s.close()


def _ok_source(name: str, rows: int = 5):
    src = MagicMock()
    src.name = name
    src.fetch.return_value = IngestResult(name, rows, "ok", None)
    return src


def _failing_source(name: str):
    src = MagicMock()
    src.name = name
    src.fetch.side_effect = RuntimeError("boom")
    return src


def test_runner_calls_each_source_and_logs_results(store):
    sources = [_ok_source("s1"), _ok_source("s2", rows=3)]
    runner = IngestRunner(store=store, sources=sources, universe=["AAPL"])
    rid = runner.run(asof_date=date(2026, 4, 23))

    log = store.conn.execute(
        "SELECT source, status, rows_inserted FROM ingest_log "
        "WHERE run_id = ? ORDER BY source", [rid]
    ).fetchall()
    assert log == [("s1", "ok", 5), ("s2", "ok", 3)]


def test_runner_isolates_failed_source(store):
    sources = [_ok_source("s1"), _failing_source("s2"), _ok_source("s3")]
    runner = IngestRunner(store=store, sources=sources, universe=["AAPL"])
    rid = runner.run(asof_date=date(2026, 4, 23))

    log = dict(store.conn.execute(
        "SELECT source, status FROM ingest_log WHERE run_id = ?", [rid]
    ).fetchall())
    assert log["s1"] == "ok"
    assert log["s2"] == "error"
    assert log["s3"] == "ok"
