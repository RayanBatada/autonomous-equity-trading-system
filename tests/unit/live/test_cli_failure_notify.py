"""Live CLI jobs must not fail silently (2026-06-09: decide's preflight
refusal and stop-loss's 9:25 DNS crash both exited 1 with only a log line;
the no-trade night surfaced only via the 22:30 monitoring sweep).

- decide: ANY failure on a real (non-dry-run) invocation notifies a human.
- stop-loss: the opening Alpaca call retries transient errors; a final
  failure notifies and still exits nonzero.
"""

from datetime import date, datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner

from sma.live.preflight import UpstreamMissedDeadline, UpstreamReadinessFailed

# Inside the sweep's 09:20-16:05 ET market-hours guard, on the same day the
# stop-loss tests below use as their asof.
_IN_SWEEP_WINDOW = datetime(2026, 4, 30, 9, 25, tzinfo=ZoneInfo("America/New_York"))


def _stub_paths(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    universe = tmp_path / "universe.yaml"
    universe.write_text("tickers: []\n")
    return config, universe


def _decide_args(tmp_path, config, universe, *, dry_run=False):
    args = [
        "--asof-date", date(2026, 4, 30).isoformat(),
        "--db", str(tmp_path / "t.duckdb"),
        "--config", str(config),
        "--universe", str(universe),
    ]
    if dry_run:
        args.append("--dry-run")
    return args


def test_decide_preflight_refusal_notifies_human(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    config, universe = _stub_paths(tmp_path)
    from sma.live.__main__ import decide as decide_cmd

    notifications = []
    with (
        patch("sma.live.__main__._build_alpaca", return_value=MagicMock()),
        patch("sma.live.__main__._build_rails", return_value=MagicMock()),
        patch("sma.live.__main__.load_settings", return_value=MagicMock()),
        patch("sma.live.__main__.load_universe", return_value=["AAPL"]),
        patch(
            "sma.live.__main__.run_preflight",
            side_effect=UpstreamReadinessFailed("ingest has unwaived blocking failures"),
        ),
        patch(
            "sma.live.__main__.notify_failure",
            side_effect=lambda title, message: notifications.append((title, message)),
        ),
    ):
        result = CliRunner().invoke(
            decide_cmd, _decide_args(tmp_path, config, universe)
        )
    assert result.exit_code != 0
    assert len(notifications) == 1
    title, message = notifications[0]
    assert "did not trade" in (title + message).lower()
    assert "unwaived blocking failures" in message


def test_decide_upstream_missed_deadline_notify_includes_rekick_hint(tmp_path, monkeypatch):
    """2026-08-05 post-mortem (8/4 no-trade outage): when decide's preflight
    times out waiting on a stalled upstream (predict never wrote a sentinel),
    the page must be actionable from the notification alone — name the
    upstream in plain English and give the exact launchctl commands to heal
    it, not just 'decide failed'."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    # Pin the launchd adapter: this asserts launchd-specific recovery
    # commands regardless of host OS (CI runs ubuntu-latest, which would
    # otherwise default to SystemdAdapter). See sma/sched_adapter.py.
    monkeypatch.setenv("SMA_SCHED_ADAPTER", "launchd")
    config, universe = _stub_paths(tmp_path)
    from sma.live.__main__ import decide as decide_cmd

    notifications = []
    with (
        patch("sma.live.__main__._build_alpaca", return_value=MagicMock()),
        patch("sma.live.__main__._build_rails", return_value=MagicMock()),
        patch("sma.live.__main__.load_settings", return_value=MagicMock()),
        patch("sma.live.__main__.load_universe", return_value=["AAPL"]),
        patch(
            "sma.live.__main__.run_preflight",
            side_effect=UpstreamMissedDeadline(
                "com.sma.model.predict.daily sentinel not ready by deadline "
                "2026-08-04 20:00:00-04:00: no sentinel",
                label="com.sma.model.predict.daily",
            ),
        ),
        patch(
            "sma.live.__main__.notify_failure",
            side_effect=lambda title, message: notifications.append((title, message)),
        ),
    ):
        result = CliRunner().invoke(
            decide_cmd, _decide_args(tmp_path, config, universe)
        )
    assert result.exit_code != 0
    assert len(notifications) == 1
    title, message = notifications[0]
    assert "did not trade" in (title + message).lower()
    # Names the stalled upstream in plain English.
    assert "predict" in message
    # Gives copy-pasteable recovery commands for both the stalled upstream
    # and decide itself — actionable from the notification alone at 1am.
    assert "launchctl kickstart -p gui/$UID/com.sma.model.predict.daily" in message
    assert "launchctl kickstart -p gui/$UID/com.sma.live.decide.daily" in message


def test_rekick_hint_uses_systemctl_on_systemd_adapter(monkeypatch):
    """Same hint on a systemd host prints the systemctl equivalent (per
    host-migration-runbook.md Section 2c) instead of a launchctl command
    that would not work there."""
    monkeypatch.setenv("SMA_SCHED_ADAPTER", "systemd")
    from sma.live.__main__ import _rekick_hint

    hint = _rekick_hint("com.sma.model.predict.daily")
    assert "predict" in hint
    assert "systemctl --user start sma-model.predict.service" in hint
    assert "systemctl --user start sma-live.decide.service" in hint
    assert "launchctl" not in hint


def test_decide_dry_run_failure_does_not_notify(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    config, universe = _stub_paths(tmp_path)
    from sma.live.__main__ import decide as decide_cmd

    notifications = []
    with (
        patch("sma.live.__main__._build_alpaca", return_value=MagicMock()),
        patch("sma.live.__main__._build_rails", return_value=MagicMock()),
        patch("sma.live.__main__.load_settings", return_value=MagicMock()),
        patch("sma.live.__main__.load_universe", return_value=["AAPL"]),
        patch(
            "sma.live.__main__.run_preflight",
            side_effect=UpstreamReadinessFailed("nope"),
        ),
        patch(
            "sma.live.__main__.notify_failure",
            side_effect=lambda title, message: notifications.append((title, message)),
        ),
    ):
        result = CliRunner().invoke(
            decide_cmd, _decide_args(tmp_path, config, universe, dry_run=True)
        )
    assert result.exit_code != 0
    assert notifications == []  # a manual dry-run failure must not page anyone


def test_stop_loss_transient_get_positions_is_retried(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    config, _ = _stub_paths(tmp_path)
    from sma.live import __main__ as live_main

    alpaca = MagicMock()
    alpaca.get_positions.side_effect = [
        ConnectionError("DNS down"),
        ConnectionError("DNS down"),
        {},
    ]
    alpaca.session_window.return_value = (
        datetime(2026, 4, 30, 9, 30, tzinfo=ZoneInfo("America/New_York")),
        datetime(2026, 4, 30, 16, 0, tzinfo=ZoneInfo("America/New_York")),
    )
    # This test exercises the get_positions retry wrapper, not the pre-open guard;
    # disable the guard so it doesn't add its own broker reads.
    from types import SimpleNamespace

    from sma.config import LivePreOpenGuard
    guard_off = SimpleNamespace(live=SimpleNamespace(preopen_guard=LivePreOpenGuard(enabled=False)))
    sweep_result = MagicMock(triggered=0, submitted=0, failed=0)
    # The sentinel payload stamps the sweep's run_id, which must be JSON-safe.
    store_cls = MagicMock()
    store_cls.return_value.connect.return_value.allocate_run_id.return_value = 7
    sleeps = []
    with (
        patch("sma.live.__main__._build_alpaca", return_value=alpaca),
        patch("sma.live.__main__._build_rails", return_value=MagicMock()),
        patch("sma.live.__main__.load_settings", return_value=guard_off),
        patch("sma.live.__main__.stop_loss_sweep", return_value=sweep_result),
        patch("sma.live.__main__.Store", store_cls),
        patch("sma.live.retry._SLEEP", side_effect=sleeps.append),
        # The market-hours guard runs before get_positions; pin the clock inside
        # the band around the session open so the retry path is actually reached.
        patch("sma.live.__main__._now_et", return_value=_IN_SWEEP_WINDOW),
    ):
        result = CliRunner().invoke(
            live_main.stop_loss_sweep_cmd,
            [
                "--asof-date", date(2026, 4, 30).isoformat(),
                "--db", str(tmp_path / "t.duckdb"),
                "--config", str(config),
            ],
        )
    assert result.exit_code == 0, result.output
    assert alpaca.get_positions.call_count == 3
    assert len(sleeps) == 2


def test_stop_loss_permanent_failure_notifies_and_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    config, _ = _stub_paths(tmp_path)
    from sma.live import __main__ as live_main

    alpaca = MagicMock()
    alpaca.get_positions.side_effect = ConnectionError("DNS still down")
    alpaca.session_window.return_value = (
        datetime(2026, 4, 30, 9, 30, tzinfo=ZoneInfo("America/New_York")),
        datetime(2026, 4, 30, 16, 0, tzinfo=ZoneInfo("America/New_York")),
    )
    notifications = []
    with (
        patch("sma.live.__main__._build_alpaca", return_value=alpaca),
        patch("sma.live.__main__._build_rails", return_value=MagicMock()),
        patch("sma.live.__main__.load_settings", return_value=MagicMock()),
        patch("sma.live.retry._SLEEP", side_effect=lambda s: None),
        patch("sma.live.__main__._now_et", return_value=_IN_SWEEP_WINDOW),
        patch(
            "sma.live.__main__.notify_failure",
            side_effect=lambda title, message: notifications.append((title, message)),
        ),
    ):
        result = CliRunner().invoke(
            live_main.stop_loss_sweep_cmd,
            [
                "--asof-date", date(2026, 4, 30).isoformat(),
                "--db", str(tmp_path / "t.duckdb"),
                "--config", str(config),
            ],
        )
    assert result.exit_code != 0
    assert alpaca.get_positions.call_count == 3  # initial + 2 retries
    assert len(notifications) == 1
    assert "stop-loss" in notifications[0][0].lower()


def test_decide_acquires_lock_with_evening_chain_patience():
    """2026-06-12 first wide-universe night: agents (267 tickers of theses)
    held the writer lock past decide's default 30s patience → decide DIED on
    WriterLockTimeoutError and the night needed a manual kick. Evening-chain
    jobs legitimately queue on each other; decide must wait minutes, not
    seconds."""
    import inspect

    from sma.live import __main__ as live_main

    src = inspect.getsource(live_main._decide_impl)
    assert 'writer_lock(label="decide", timeout_s=' in src, (
        "decide must pass an explicit multi-minute timeout_s"
    )
