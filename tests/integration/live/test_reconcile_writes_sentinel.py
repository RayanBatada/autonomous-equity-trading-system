"""Task 12: reconcile adopts writer_lock + writes sentinel."""

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
    alpaca.get_orders_for_date.return_value = []
    account = MagicMock()
    account.equity = "100000.0"
    account.cash = "50000.0"
    account.buying_power = "100000.0"
    account.long_market_value = "50000.0"
    account.trading_blocked = False
    account.account_blocked = False
    alpaca.get_account.return_value = {
        "equity": 100_000.0,
        "cash": 50_000.0,
        "buying_power": 100_000.0,
        "long_market_value": 50_000.0,
    }
    alpaca.get_positions.return_value = {}
    # Ancient next-session so reconcile's drift time-guard always passes
    # (this test exercises the sentinel write, not the time gate).
    alpaca.next_session_date.return_value = date(1970, 1, 1)
    return alpaca


def test_reconcile_writes_sentinel(monkeypatch, tmp_path):
    """reconcile CLI writes the sentinel inside writer_lock after work completes."""
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

    from sma.live.__main__ import reconcile_cmd

    runner = CliRunner()

    with (
        patch("sma.live.__main__._build_alpaca", return_value=alpaca_mock),
        patch("sma.live.__main__.load_settings", return_value=MagicMock()),
        patch("sma.live.__main__.writer_lock", _make_writer_lock_factory(lock_path)),
        # _today_et gates the explicit-asof future check; pin it past `asof`.
        patch("sma.live.__main__._today_et", return_value=date(2026, 12, 31)),
    ):
        result = runner.invoke(
            reconcile_cmd,
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

    assert result.exit_code == 0, f"reconcile CLI failed:\n{result.output}"

    sentinel = read_sentinel(label="com.sma.live.reconcile.daily", asof=asof)
    assert sentinel is not None, "sentinel was not written"
    assert sentinel["label"] == "com.sma.live.reconcile.daily"
    assert sentinel["asof"] == asof.isoformat()
    assert "completed_at" in sentinel
    assert sentinel["completed_at"].endswith("Z"), "completed_at must be UTC with Z suffix"
    assert "fills_persisted" in sentinel
    assert "account_equity" in sentinel
    assert "positions_count" in sentinel
