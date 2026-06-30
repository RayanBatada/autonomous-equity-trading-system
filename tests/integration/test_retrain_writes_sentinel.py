"""Task 14: model retrain adopts writer_lock + writes sentinel."""

from contextlib import contextmanager
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from sma.sentinels import read_sentinel


def _make_writer_lock_factory(lock_path: Path):
    """Return a drop-in for writer_lock that uses a test-scoped lock_path.

    The real writer_lock captures DEFAULT_LOCK_PATH as a default-arg value at
    import time, so monkeypatching the module attribute does not affect the
    default. This factory explicitly passes the tmp lock_path so Store.connect()
    (which reads sma.locks.DEFAULT_LOCK_PATH at call time, after monkeypatch) agrees.
    """
    from sma.locks import writer_lock as _real_writer_lock

    @contextmanager
    def _writer_lock_with_path(*, label: str, **kwargs):
        with _real_writer_lock(lock_path=lock_path, label=label, **kwargs):
            yield

    return _writer_lock_with_path


def test_train_writes_sentinel(monkeypatch, tmp_path):
    """train CLI writes the retrain sentinel inside writer_lock after training completes."""
    import sma.locks as _locks

    lock_path = tmp_path / ".sma-writer.lock"
    sentinel_dir = tmp_path / "sentinels"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", lock_path)

    asof = date(2026, 5, 3)
    models_dir = tmp_path / "models"
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text("universe:\n  tickers: [AAPL, MSFT]\n")

    fake_model_id = f"xgb_ret_30d_forward_{asof.isoformat()}_deadbeef"

    from click.testing import CliRunner

    from sma.model.__main__ import cli

    runner = CliRunner()

    # Stub out heavy training steps so test is fast. The sentinel is written
    # by the CLI handler after save_model, so we must let save_model run (or
    # mock it) and capture the model_id from its return.
    fake_X = pd.DataFrame({"f1": [1.0, 2.0]})  # noqa: N806
    fake_y = pd.Series([0.1, 0.2])
    fake_dates = pd.Series([asof, asof])
    fake_model = MagicMock()

    with (
        patch("sma.model.__main__._load_prices_for_range", return_value=pd.DataFrame()),
        patch(
            "sma.model.__main__.build_training_set",
            return_value=(fake_X, fake_y, fake_dates),
        ),
        patch("sma.model.__main__.train_xgb", return_value=fake_model),
        patch(
            "sma.model.__main__.save_model",
            return_value=(
                models_dir / f"{fake_model_id}.pkl",
                models_dir / f"{fake_model_id}.json",
            ),
        ),
        patch("sma.model.__main__.writer_lock", _make_writer_lock_factory(lock_path)),
    ):
        result = runner.invoke(
            cli,
            [
                "train",
                "--asof",
                asof.isoformat(),
                "--no-cv",
                "--db-path",
                str(tmp_path / "sma.duckdb"),
                "--models-dir",
                str(models_dir),
                "--universe-path",
                str(universe_path),
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, f"train CLI failed:\n{result.output}"

    sentinel = read_sentinel(label="com.sma.model.retrain.weekly", asof=asof)
    assert sentinel is not None, "sentinel was not written"
    assert sentinel["label"] == "com.sma.model.retrain.weekly"
    assert sentinel["asof"] == asof.isoformat()
    assert "completed_at" in sentinel
    assert sentinel["completed_at"].endswith("Z"), "completed_at must be UTC with Z suffix"
    assert "model_id" in sentinel
    assert "train_end_date" in sentinel
    assert sentinel["train_end_date"] == asof.isoformat()
    assert "cv_rmse" in sentinel
    assert "training_rows" in sentinel
    assert sentinel["training_rows"] == len(fake_X)
