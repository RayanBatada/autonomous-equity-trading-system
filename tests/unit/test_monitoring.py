"""Tests for sma.monitoring.check_critical_jobs_fired.

Empirical 2026-05-22 failure mode: decide.daily silently didn't fire for a
trading day (preflight bug), no alert surfaced. The monitor closes that
visibility gap by checking sentinels at 22:30 ET.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from sma.monitoring import CRITICAL_LABELS, check_critical_jobs_fired
from sma.sentinels import write_sentinel


def _trading_day_alpaca(asof: date) -> MagicMock:
    alpaca = MagicMock()
    alpaca.sessions_between.return_value = [asof]  # asof itself is a trading day
    return alpaca


def _holiday_alpaca() -> MagicMock:
    alpaca = MagicMock()
    alpaca.sessions_between.return_value = []  # no trading session = holiday
    return alpaca


def test_no_alert_when_all_critical_sentinels_present(monkeypatch, tmp_path):
    """All critical jobs have sentinels for today → no notification."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))

    asof = date(2026, 5, 26)
    for label in CRITICAL_LABELS:
        write_sentinel(
            label=label,
            asof=asof,
            payload={"label": label, "asof": asof.isoformat()},
        )

    notifies: list[tuple[str, str]] = []
    missing = check_critical_jobs_fired(
        asof=asof,
        alpaca=_trading_day_alpaca(asof),
        notify_fn=lambda title, message: notifies.append((title, message)),
    )

    assert missing == []
    assert notifies == []


def test_alert_when_decide_sentinel_missing(monkeypatch, tmp_path):
    """Trading day with no decide sentinel → exactly one notification."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))

    asof = date(2026, 5, 26)
    # Don't write any sentinels.

    notifies: list[tuple[str, str]] = []
    missing = check_critical_jobs_fired(
        asof=asof,
        alpaca=_trading_day_alpaca(asof),
        notify_fn=lambda title, message: notifies.append((title, message)),
    )

    assert missing == list(CRITICAL_LABELS)
    assert len(notifies) == len(CRITICAL_LABELS)
    title, message = notifies[0]
    assert "missed" in title.lower()
    assert "com.sma.live.decide.daily" in message
    assert asof.isoformat() in message


def test_no_alert_on_market_holiday(monkeypatch, tmp_path):
    """Memorial Day, Christmas, etc. — decide is correctly blocked by
    preflight quality.failed (no Mon prices). Don't alert on those."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))

    asof = date(2026, 5, 25)  # Memorial Day

    notifies: list[tuple[str, str]] = []
    missing = check_critical_jobs_fired(
        asof=asof,
        alpaca=_holiday_alpaca(),
        notify_fn=lambda title, message: notifies.append((title, message)),
    )

    assert missing == []
    assert notifies == []


def test_calendar_failure_alerts_and_still_checks(monkeypatch, tmp_path):
    """If the Alpaca calendar lookup raises, monitoring must NOT crash before
    alerting (the 2026-06-05 audit finding). It notifies that market status is
    unknown and still checks the critical sentinels, so a real missed decide on
    a weekday still surfaces."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))

    asof = date(2026, 4, 30)  # Thursday
    notifies: list[tuple[str, str]] = []
    alpaca = MagicMock()
    alpaca.sessions_between.side_effect = RuntimeError("calendar api down")

    # No sentinels written -> decide is missing.
    missing = check_critical_jobs_fired(
        asof=asof,
        alpaca=alpaca,
        notify_fn=lambda title, message: notifies.append((title, message)),
    )

    assert "com.sma.live.decide.daily" in missing  # still checked despite calendar failure
    blob = " ".join(t + " " + m for t, m in notifies).lower()
    assert "market status" in blob or "calendar" in blob  # degraded-calendar alert sent
    assert "decide" in blob  # and the missing-job alert still fired


# --- model staleness (2026-07-20: retrain missed 7/13 entirely; the model served
# week-old weights with only a single same-day watchdog page as signal) ---------


def _touch_model(models_dir, train_date: str):
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / f"xgb_ret_30d_forward_{train_date}_abcd1234.pkl").write_bytes(b"x")


def _write_retrain_sentinel(train_date: str):
    from datetime import date as date_cls

    from sma.sentinels import write_sentinel

    write_sentinel(
        label="com.sma.model.retrain.weekly",
        asof=date_cls.fromisoformat(train_date),
        payload={"run_id": 1, "completed_at": "x"},
    )


def test_model_staleness_pages_same_evening_as_the_missed_retrain(tmp_path, monkeypatch):
    """Mon 7/13 retrain missed → the 22:30 monitoring run THAT EVENING pages
    (based on the missing retrain sentinel, not an aged artifact — review
    2026-07-20 #6: the old 8-day artifact-age rule stayed quiet until Wednesday)."""
    from datetime import date as date_cls

    from sma.monitoring import check_model_staleness

    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    _touch_model(tmp_path / "m", "2026-07-06")
    _write_retrain_sentinel("2026-07-06")  # last Monday ran fine
    notes = []
    stale = check_model_staleness(
        asof=date_cls(2026, 7, 13),  # miss-Monday evening: no 7/13 sentinel
        models_dir=tmp_path / "m",
        notify_fn=lambda title, message: notes.append((title, message)),
    )
    assert stale is True
    assert any("stale" in (t + m).lower() for t, m in notes)


def test_model_staleness_page_rekick_command_is_launchd_on_macos_adapter(tmp_path, monkeypatch):
    """The stale-retrain page's recovery command comes from
    sma.sched_adapter.get_adapter() (see host-migration-runbook.md Section
    2c) -- pinned to launchd here reproduces today's hardcoded text
    byte-for-byte."""
    from datetime import date as date_cls

    from sma.monitoring import check_model_staleness

    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("SMA_SCHED_ADAPTER", "launchd")
    _touch_model(tmp_path / "m", "2026-07-06")
    _write_retrain_sentinel("2026-07-06")
    notes = []
    check_model_staleness(
        asof=date_cls(2026, 7, 13),
        models_dir=tmp_path / "m",
        notify_fn=lambda title, message: notes.append((title, message)),
    )
    message = notes[0][1]
    assert "launchctl kickstart -p gui/$UID/com.sma.model.retrain.weekly" in message


def test_model_staleness_page_rekick_command_is_systemctl_on_systemd_adapter(tmp_path, monkeypatch):
    """Same page on a systemd host prints the systemctl equivalent instead
    of a launchctl command that would not work there."""
    from datetime import date as date_cls

    from sma.monitoring import check_model_staleness

    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("SMA_SCHED_ADAPTER", "systemd")
    _touch_model(tmp_path / "m", "2026-07-06")
    _write_retrain_sentinel("2026-07-06")
    notes = []
    check_model_staleness(
        asof=date_cls(2026, 7, 13),
        models_dir=tmp_path / "m",
        notify_fn=lambda title, message: notes.append((title, message)),
    )
    message = notes[0][1]
    assert "systemctl --user start sma-model.retrain.service" in message
    assert "launchctl" not in message


def test_model_staleness_quiet_when_retrain_ran_today(tmp_path, monkeypatch):
    from datetime import date as date_cls

    from sma.monitoring import check_model_staleness

    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    _touch_model(tmp_path / "m", "2026-07-13")
    _write_retrain_sentinel("2026-07-13")
    notes = []
    stale = check_model_staleness(
        asof=date_cls(2026, 7, 13),
        models_dir=tmp_path / "m",
        notify_fn=lambda title, message: notes.append((title, message)),
    )
    assert stale is False and notes == []


def test_model_staleness_quiet_midweek_when_last_monday_ran(tmp_path, monkeypatch):
    from datetime import date as date_cls

    from sma.monitoring import check_model_staleness

    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    _touch_model(tmp_path / "m", "2026-07-13")
    _write_retrain_sentinel("2026-07-13")
    notes = []
    stale = check_model_staleness(
        asof=date_cls(2026, 7, 16),  # Thursday; Monday ran → healthy cadence
        models_dir=tmp_path / "m",
        notify_fn=lambda title, message: notes.append((title, message)),
    )
    assert stale is False and notes == []


def test_model_staleness_quarantine_week_is_not_a_missed_retrain_page(tmp_path, monkeypatch):
    """Review 2026-07-20 #5: retrain RAN but its artifact was quarantined by the
    deploy gate (incumbent keeps serving BY DESIGN). The old artifact-age rule
    paged 'retrain missed — Mac asleep?' every evening with a wrong diagnosis.
    Sentinel present → not a miss → no page."""
    from datetime import date as date_cls

    from sma.monitoring import check_model_staleness

    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    _touch_model(tmp_path / "m", "2026-07-06")  # old incumbent still serving
    _write_retrain_sentinel("2026-07-13")  # retrain ran; gate quarantined
    notes = []
    stale = check_model_staleness(
        asof=date_cls(2026, 7, 16),
        models_dir=tmp_path / "m",
        notify_fn=lambda title, message: notes.append((title, message)),
    )
    assert stale is False
    assert not any("missed" in (t + m).lower() for t, m in notes)


def test_model_staleness_pages_when_no_model_at_all(tmp_path, monkeypatch):
    from datetime import date as date_cls

    from sma.monitoring import check_model_staleness

    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    _write_retrain_sentinel("2026-07-13")
    notes = []
    stale = check_model_staleness(
        asof=date_cls(2026, 7, 16),
        models_dir=tmp_path / "empty",
        notify_fn=lambda title, message: notes.append((title, message)),
    )
    assert stale is True and notes


def test_check_cli_writes_its_own_completion_sentinel(tmp_path, monkeypatch):
    """2026-08-29: com.sma.monitoring.daily never wrote a completion sentinel
    (found while building the weekly digest, which reads exactly this
    sentinel for its ops section -- without this, the job showed up
    'missing' every single night despite running fine, and sma.watchdog
    re-kicked it past deadline every evening too).

    Sentinel is keyed by ET's today, NOT the runner's local/system date
    (`check()` computes `asof = datetime.now(ET).date()` -- CI runs in UTC,
    where the two dates can differ for hours around US midnight)."""
    from datetime import datetime

    from click.testing import CliRunner

    from sma.monitoring.__main__ import ET, cli
    from sma.sentinels import read_sentinel

    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text("placeholder: true\n")

    monkeypatch.setattr("sma.monitoring.__main__.load_settings", lambda config_path: object())
    monkeypatch.setattr("sma.monitoring.__main__._build_alpaca", lambda settings: object())
    monkeypatch.setattr("sma.monitoring.__main__.check_critical_jobs_fired", lambda **k: [])
    monkeypatch.setattr("sma.monitoring.__main__.check_model_staleness", lambda **k: False)
    monkeypatch.setattr("sma.monitoring.__main__.check_regime_turn", lambda **k: None)

    runner = CliRunner()
    result = runner.invoke(cli, ["check", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    today_et = datetime.now(ET).date()
    sentinel = read_sentinel(label="com.sma.monitoring.daily", asof=today_et)
    assert sentinel is not None
    assert sentinel["missing_critical_jobs"] == []
    assert sentinel["model_stale"] is False
    assert sentinel["regime_turn"] is None
