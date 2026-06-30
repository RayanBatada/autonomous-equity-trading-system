"""Task 13: stop-loss-sweep adopts writer_lock + writes sentinel."""

from contextlib import contextmanager
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def _make_alpaca_mock() -> MagicMock:
    alpaca = MagicMock()
    # No positions: stop-loss sweep is a NOOP (stop_loss_pct=0 in default config).
    alpaca.get_positions.return_value = {}
    alpaca.get_account.return_value = {
        "equity": 100_000.0,
        "cash": 50_000.0,
        "buying_power": 100_000.0,
        "long_market_value": 50_000.0,
    }
    return alpaca


def test_stop_loss_sweep_writes_sentinel(monkeypatch, tmp_path):
    """stop-loss-sweep CLI writes the sentinel inside writer_lock after work completes."""
    import sma.locks as _locks

    lock_path = tmp_path / ".sma-writer.lock"
    sentinel_dir = tmp_path / "sentinels"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", lock_path)

    asof = date(2026, 5, 1)
    db_path = tmp_path / "test.duckdb"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")

    alpaca_mock = _make_alpaca_mock()

    from click.testing import CliRunner

    from sma.live.__main__ import stop_loss_sweep_cmd

    runner = CliRunner()

    from sma.risk.rails import RiskRails

    with (
        patch("sma.live.__main__._build_alpaca", return_value=alpaca_mock),
        patch("sma.live.__main__._build_rails", return_value=RiskRails(stop_loss_pct=0.0)),
        patch("sma.live.__main__.load_settings", return_value=MagicMock()),
        patch("sma.live.__main__.writer_lock", _make_writer_lock_factory(lock_path)),
    ):
        result = runner.invoke(
            stop_loss_sweep_cmd,
            [
                "--asof-date",
                asof.isoformat(),
                "--db",
                str(db_path),
                "--config",
                str(config_path),
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, f"stop-loss-sweep CLI failed:\n{result.output}"

    sentinel = read_sentinel(label="com.sma.live.stop-loss.weekday", asof=asof)
    assert sentinel is not None, "sentinel was not written"
    assert sentinel["label"] == "com.sma.live.stop-loss.weekday"
    assert sentinel["asof"] == asof.isoformat()
    assert "completed_at" in sentinel
    assert sentinel["completed_at"].endswith("Z"), "completed_at must be UTC with Z suffix"
    assert "positions_checked" in sentinel
    assert "positions_triggered" in sentinel
    assert "sells_submitted" in sentinel
