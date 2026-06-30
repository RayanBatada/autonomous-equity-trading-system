"""Task 9: decide writes a sentinel after submission, inside writer_lock."""

from contextlib import contextmanager
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from sma.risk.rails import RiskRails
from sma.sentinels import read_sentinel, write_sentinel


def _real_rails() -> RiskRails:
    """Real RiskRails with safe defaults (stop_loss disabled, standard caps)."""
    return RiskRails(stop_loss_pct=0.0)


def _upstream_sentinels(asof: date, sentinel_dir: Path) -> None:
    """Pre-write the three upstream sentinels so preflight passes."""
    for label in (
        "com.sma.ingest.daily",
        "com.sma.model.predict.daily",
        "com.sma.agents.daily",
    ):
        write_sentinel(
            label=label,
            asof=asof,
            payload={
                "label": label,
                "asof": asof.isoformat(),
                "run_id": 1,
                "quality": {"passed": True, "blocking_failures": []},
            },
        )


def _make_alpaca_mock(asof: date) -> MagicMock:
    alpaca = MagicMock()
    # next_session_date is called by run_preflight calendar check.
    # asof=2026-04-30 (Thursday) -> next trading day is 2026-05-01 (Friday).
    from datetime import timedelta

    expected_next = asof + timedelta(days=1)
    while expected_next.weekday() >= 5:
        expected_next += timedelta(days=1)
    alpaca.next_session_date.return_value = expected_next
    alpaca.get_account.return_value = {
        "equity": 100_000.0,
        "cash": 100_000.0,
        "blocked_count": 0,
    }
    alpaca.get_positions.return_value = {}
    alpaca.submit_day_opg_buy.return_value = "alpaca_id_123"
    alpaca.submit_day_sell.return_value = "alpaca_id_456"
    return alpaca


def _make_writer_lock_factory(lock_path: Path):
    """Return a drop-in for writer_lock that uses a test-scoped lock_path.

    The real writer_lock captures DEFAULT_LOCK_PATH as a default-arg value at
    import time, so monkeypatching the module attribute does not affect the
    default. This factory explicitly passes the tmp lock_path so the pid file
    lands in tmp_path and the Store.connect() check (which reads
    sma.locks.DEFAULT_LOCK_PATH at call time, after monkeypatch) agrees.
    """
    from sma.locks import writer_lock as _real_writer_lock

    @contextmanager
    def _writer_lock_with_path(*, label: str, **kwargs):
        with _real_writer_lock(lock_path=lock_path, label=label, **kwargs):
            yield

    return _writer_lock_with_path


def test_decide_writes_sentinel_after_submit(tmp_path, monkeypatch):
    """A complete decide run (dry-run), verifies the sentinel lands inside writer_lock."""
    lock_path = tmp_path / ".sma-writer.lock"
    sentinel_dir = tmp_path / "sentinels"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))
    # Patch both the module attr (read by Store.connect at call time) and
    # pass the same path explicitly to writer_lock via _make_writer_lock_factory.
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", lock_path)

    asof = date(2026, 4, 30)
    _upstream_sentinels(asof, sentinel_dir)

    db_path = tmp_path / "test.duckdb"
    # Create stub config/universe files so click path-exists validation passes.
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text("tickers: []\n")

    alpaca_mock = _make_alpaca_mock(asof)

    from click.testing import CliRunner

    from sma.live.__main__ import decide as decide_cmd

    runner = CliRunner()

    # Patch _build_alpaca to inject our mock (avoids needing real env vars).
    # Patch _build_strategy to inject a trivial strategy (avoids needing models).
    # Patch run_preflight to bypass it (sentinel-based preflight already tested separately).
    # Patch writer_lock to explicitly pass the test-scoped lock_path (default arg
    # is captured at import time, so monkeypatching the module attr is not enough).
    class FakeStrategy:
        def decide(self, *, asof_date, prices):
            return []

    with (
        patch("sma.live.__main__._build_alpaca", return_value=alpaca_mock),
        patch("sma.live.__main__._build_strategy", return_value=FakeStrategy()),
        patch("sma.live.__main__._build_rails", return_value=_real_rails()),
        patch("sma.live.__main__.load_settings", return_value=MagicMock()),
        patch("sma.live.__main__.load_universe", return_value=["AAPL", "MSFT"]),
        patch("sma.live.__main__.run_preflight"),
        patch("sma.live.__main__.writer_lock", _make_writer_lock_factory(lock_path)),
    ):
        result = runner.invoke(
            decide_cmd,
            [
                "--asof-date",
                asof.isoformat(),
                "--dry-run",
                "--db",
                str(db_path),
                "--config",
                str(config_path),
                "--universe",
                str(universe_path),
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, f"decide CLI failed:\n{result.output}"

    sentinel = read_sentinel(label="com.sma.live.decide.daily", asof=asof)
    assert sentinel is not None, "sentinel was not written"
    assert sentinel["label"] == "com.sma.live.decide.daily"
    assert sentinel["asof"] == asof.isoformat()
    assert sentinel["dry_run"] is True
    assert "completed_at" in sentinel
    assert sentinel["completed_at"].endswith("Z"), "completed_at must be UTC with Z suffix"
    assert "submitted_count" in sentinel
    assert "failed_count" in sentinel
    assert "skipped_count" in sentinel
    assert "decisions_total" in sentinel
    assert "canary_ticker" in sentinel
    assert sentinel["canary_ticker"] is None
    # dry-run: submitted=0, failed=0, skipped=0
    assert sentinel["submitted_count"] == 0
    assert sentinel["failed_count"] == 0
    assert sentinel["skipped_count"] == 0


def test_decide_sentinel_canary_ticker_propagates(tmp_path, monkeypatch):
    """canary_ticker is populated in the sentinel when --canary is passed."""
    lock_path = tmp_path / ".sma-writer.lock"
    sentinel_dir = tmp_path / "sentinels"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", lock_path)

    asof = date(2026, 4, 30)
    _upstream_sentinels(asof, sentinel_dir)

    db_path = tmp_path / "test.duckdb"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text("tickers: []\n")

    alpaca_mock = _make_alpaca_mock(asof)

    from click.testing import CliRunner

    from sma.live.__main__ import decide as decide_cmd

    runner = CliRunner()

    class FakeStrategy:
        def decide(self, *, asof_date, prices):
            from sma.backtest.strategies.base import StrategyDecision

            return [StrategyDecision(asof_date=asof_date, ticker="AAPL", target_weight=0.05)]

    with (
        patch("sma.live.__main__._build_alpaca", return_value=alpaca_mock),
        patch("sma.live.__main__._build_strategy", return_value=FakeStrategy()),
        patch("sma.live.__main__._build_rails", return_value=_real_rails()),
        patch("sma.live.__main__.load_settings", return_value=MagicMock()),
        patch("sma.live.__main__.load_universe", return_value=["AAPL", "MSFT"]),
        patch("sma.live.__main__.run_preflight"),
        patch("sma.live.__main__.writer_lock", _make_writer_lock_factory(lock_path)),
    ):
        result = runner.invoke(
            decide_cmd,
            [
                "--asof-date",
                asof.isoformat(),
                "--dry-run",
                "--canary",
                "AAPL",
                "--db",
                str(db_path),
                "--config",
                str(config_path),
                "--universe",
                str(universe_path),
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, f"decide CLI failed:\n{result.output}"

    sentinel = read_sentinel(label="com.sma.live.decide.daily", asof=asof)
    assert sentinel is not None
    assert sentinel["canary_ticker"] == "AAPL"
    assert sentinel["dry_run"] is True
