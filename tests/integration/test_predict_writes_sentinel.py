"""Task 10: predict adopts writer_lock + writes sentinel + routes through Store."""

from contextlib import contextmanager
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from sma.sentinels import read_sentinel, write_sentinel


def _write_passing_ingest_sentinel(asof: date, run_id: int = 7) -> None:
    """The scheduled predict path now gates on the ingest sentinel
    (2026-06-09 lineage fix); give these tests a healthy one."""
    write_sentinel(
        label="com.sma.ingest.daily",
        asof=asof,
        payload={
            "run_id": run_id,
            "quality": {"passed": True, "blocking_failures": []},
        },
    )


def _make_writer_lock_factory(lock_path: Path):
    """Return a drop-in for writer_lock that uses a test-scoped lock_path.

    The real writer_lock captures DEFAULT_LOCK_PATH as a default-arg value at
    import time, so monkeypatching the module attribute does not affect the
    default. This factory explicitly passes the tmp lock_path so the pid file
    lands in tmp_path and Store.connect() (which reads sma.locks.DEFAULT_LOCK_PATH
    at call time, after monkeypatch) agrees.
    """
    from sma.locks import writer_lock as _real_writer_lock

    @contextmanager
    def _writer_lock_with_path(*, label: str, **kwargs):
        with _real_writer_lock(lock_path=lock_path, label=label, **kwargs):
            yield

    return _writer_lock_with_path


def test_predict_writes_sentinel(monkeypatch, tmp_path):
    """predict CLI invocation writes the predict sentinel inside writer_lock."""
    import sma.locks as _locks

    lock_path = tmp_path / ".sma-writer.lock"
    sentinel_dir = tmp_path / "sentinels"

    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", lock_path)

    from click.testing import CliRunner

    from sma.model.__main__ import cli

    asof = date(2026, 4, 30)
    _write_passing_ingest_sentinel(asof)
    runner = CliRunner()

    with (
        patch(
            "sma.model.predictor.Predictor.predict_for_with_model_id",
            return_value=({"AAPL": 0.05, "MSFT": 0.03}, "xgb_ret_30d_forward_2026-04-01_abcd1234"),
        ),
        patch("sma.model.__main__.write_predictions", return_value=2),
        patch("sma.model.__main__.writer_lock", _make_writer_lock_factory(lock_path)),
    ):
        result = runner.invoke(
            cli,
            [
                "predict",
                "--asof",
                asof.isoformat(),
                "--db-path",
                str(tmp_path / "sma.duckdb"),
                "--models-dir",
                str(tmp_path / "models"),
                "--universe-path",
                "src/sma/universe.yaml",
            ],
        )

    assert result.exit_code == 0, result.output

    sentinel = read_sentinel(label="com.sma.model.predict.daily", asof=asof)
    assert sentinel is not None, "Sentinel was not written"
    assert sentinel["label"] == "com.sma.model.predict.daily"
    assert sentinel["asof"] == asof.isoformat()
    assert "model_id" in sentinel
    assert "completed_at" in sentinel
    assert sentinel["rows_written"] == 2
    assert sentinel["tickers"] == 2


def test_predict_sentinel_payload_fields(monkeypatch, tmp_path):
    """Sentinel payload contains all required fields with correct types."""
    import sma.locks as _locks

    lock_path = tmp_path / ".sma-writer.lock"
    sentinel_dir = tmp_path / "sentinels"

    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", lock_path)

    from click.testing import CliRunner

    from sma.model.__main__ import cli

    model_id = "xgb_ret_30d_forward_2026-04-01_abcd1234"
    asof = date(2026, 4, 28)
    _write_passing_ingest_sentinel(asof, run_id=99)
    runner = CliRunner()

    with (
        patch(
            "sma.model.predictor.Predictor.predict_for_with_model_id",
            return_value=(
                {"AAPL": 0.05, "MSFT": 0.03, "GOOGL": 0.07},
                model_id,
            ),
        ),
        patch("sma.model.__main__.write_predictions", return_value=3),
        patch("sma.model.__main__.writer_lock", _make_writer_lock_factory(lock_path)),
    ):
        result = runner.invoke(
            cli,
            [
                "predict",
                "--asof",
                asof.isoformat(),
                "--db-path",
                str(tmp_path / "sma.duckdb"),
                "--models-dir",
                str(tmp_path / "models"),
                "--universe-path",
                "src/sma/universe.yaml",
            ],
        )

    assert result.exit_code == 0, result.output

    sentinel = read_sentinel(label="com.sma.model.predict.daily", asof=asof)
    assert sentinel is not None
    assert sentinel["model_id"] == model_id
    assert sentinel["rows_written"] == 3
    assert sentinel["tickers"] == 3
    # completed_at must end with Z (UTC ISO 8601)
    assert sentinel["completed_at"].endswith("Z")
    # lineage: the consumed ingest run (2026-06-09 stale-prediction fix)
    assert sentinel["ingest_run_id"] == 99


def test_predict_uses_store_not_direct_duckdb(monkeypatch, tmp_path):
    """write_predictions routes through Store, so WriterLockNotHeld fires without lock."""
    import sma.locks as _locks
    from sma.ingest.store import WriterLockNotHeld
    from sma.model.persistence import write_predictions

    # Point DEFAULT_LOCK_PATH at a new path with NO lock held (the autouse
    # fixture holds a lock on a different path; we override here with a path
    # that has no pid file, so the assertion fires)
    unlocked_path = tmp_path / "no-lock" / ".sma-writer.lock"
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", unlocked_path)

    db_path = tmp_path / "test.duckdb"
    with pytest.raises(WriterLockNotHeld):
        write_predictions(
            db_path=db_path,
            asof_date=date(2026, 4, 30),
            target="ret_30d_forward",
            model_id="test-model",
            predictions={"AAPL": 0.05},
        )
