"""Task 13: stop-loss-sweep adopts writer_lock + writes sentinel."""

from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from sma.config import LivePreOpenGuard
from sma.sentinels import read_sentinel

ET = ZoneInfo("America/New_York")
# Both tests exercise the REAL sweep path, which is now gated by a market-hours
# guard (_sweep_skip_reason). Pin the wall clock inside the band around the
# session open or the sweep correctly skips and these assertions test nothing.
IN_WINDOW = datetime(2026, 5, 1, 9, 25, tzinfo=ET)


def _settings_with_guard(**guard_kwargs):
    """Real preopen_guard config so the 09:25 divergence guard uses real
    thresholds (not MagicMock attributes) in these CLI-wiring tests."""
    return SimpleNamespace(
        live=SimpleNamespace(preopen_guard=LivePreOpenGuard(**guard_kwargs))
    )


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
    alpaca.list_open_orders.return_value = []  # pre-open guard reads this
    # Real session window: the market-hours guard treats a None session as a
    # holiday and skips the sweep, and anchors its band to the open.
    alpaca.session_window.return_value = (
        datetime(2026, 5, 1, 9, 30, tzinfo=ET),
        datetime(2026, 5, 1, 16, 0, tzinfo=ET),
    )
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
        patch("sma.live.__main__.load_settings", return_value=_settings_with_guard()),
        patch("sma.live.__main__.writer_lock", _make_writer_lock_factory(lock_path)),
        patch("sma.live.__main__._now_et", return_value=IN_WINDOW),
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
    assert sentinel["halted_preopen_divergence"] is False  # empty ledger → no halt


def test_stop_loss_sweep_halts_on_preopen_divergence(monkeypatch, tmp_path):
    """The 09:25 guard: when the live broker book has lost most of the positions
    our ledger holds (the 7/07 wipe), the sweep HALTS — cancels the day's queued
    orders, pages, writes a halted sentinel, and does NOT run the stop sweep."""
    import sma.locks as _locks
    from sma.ingest.store import Store

    lock_path = tmp_path / ".sma-writer.lock"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", lock_path)
    asof = date(2026, 5, 1)
    db_path = tmp_path / "test.duckdb"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")

    # Seed a ledger of 11 held positions + a prior equity snapshot (the wipe halt
    # needs equity to have collapsed vs the snapshot). Store.connect needs the lock.
    from sma.locks import writer_lock
    with writer_lock(lock_path=lock_path, label="seed"):
        s = Store(path=str(db_path)).connect()
        for i in range(11):
            s.conn.execute(
                "INSERT INTO paper_fills (alpaca_order_id, asof_date, ticker, side, "
                " filled_shares, fill_price, status, submitted_at, filled_at, run_id) "
                "VALUES (?, DATE '2026-05-01', ?, 'BUY', 100, 100.0, 'filled', "
                " TIMESTAMP '2026-05-01 15:00:00', TIMESTAMP '2026-05-01 15:00:00', 1)",
                [f"o{i}", f"T{i}"],
            )
        s.conn.execute(
            "INSERT INTO account_snapshots (asof_date, equity, cash, buying_power, "
            " long_market_value, position_count, total_unrealized_pnl, run_id) "
            "VALUES (DATE '2026-05-01', 110000, 0, 0, 110000, 11, 0, 1)"
        )
        s.conn.close()

    alpaca = _make_alpaca_mock()
    alpaca.get_positions.return_value = {}  # broker WIPED
    alpaca.get_account.return_value = {"equity": "6969.00"}  # equity collapsed
    # A queued QCOM-style oversell order (the 7/07 phantom short) to be cancelled.
    alpaca.list_open_orders.return_value = [
        {"id": "q", "symbol": "T0", "side": "SELL", "qty": 100}
    ]

    pages = []
    from click.testing import CliRunner

    from sma.live.__main__ import stop_loss_sweep_cmd
    from sma.risk.rails import RiskRails

    with (
        patch("sma.live.__main__._build_alpaca", return_value=alpaca),
        patch("sma.live.__main__._build_rails", return_value=RiskRails(stop_loss_pct=0.0)),
        patch("sma.live.__main__.load_settings", return_value=_settings_with_guard()),
        patch("sma.live.__main__.notify_failure",
              lambda title, message: pages.append((title, message))),
        patch("sma.live.__main__.writer_lock", _make_writer_lock_factory(lock_path)),
        patch("sma.live.__main__._now_et", return_value=IN_WINDOW),
    ):
        result = CliRunner().invoke(
            stop_loss_sweep_cmd,
            ["--asof-date", asof.isoformat(), "--db", str(db_path),
             "--config", str(config_path)],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    alpaca.cancel_all_open_orders.assert_called_once()
    assert any("HALTED" in t for t, _ in pages)
    sentinel = read_sentinel(label="com.sma.live.stop-loss.weekday", asof=asof)
    assert sentinel["halted_preopen_divergence"] is True
    assert len(sentinel["missing_names"]) == 11
