import configparser
from pathlib import Path

from ops.systemd.render_units import render_all


def _read_unit(path: Path) -> configparser.RawConfigParser:
    """Parse a systemd unit file with configparser.

    Unit files are INI-shaped (section headers + key=value), except that
    systemd explicitly allows a key to repeat within a section (e.g. two
    OnCalendar= lines). configparser rejects that by default, so callers that
    need the repeated-key lines (the watchdog's two OnCalendar=) should read
    the raw text instead — this helper is for single-valued lookups.
    """
    cp = configparser.RawConfigParser(strict=False)
    cp.optionxform = str  # preserve case (systemd keys are case-sensitive)
    cp.read_string(path.read_text())
    return cp


def _render(tmp_path: Path) -> Path:
    out_dir = tmp_path / "rendered"
    render_all(
        out_dir=out_dir,
        repo_root=Path("/REPO"),
        venv=Path("/VENV"),
        home_dir=Path("/HOME"),
    )
    return out_dir


def test_renders_35_unit_files(tmp_path):
    """(13 SCHEDULE + 3 OPTIONAL_SCHEDULE) jobs x (.service + .timer) = 32,
    + watchdog (.service + .timer) = 34, + dashboard (.service only) = 35."""
    out_dir = _render(tmp_path)
    files = sorted(p.name for p in out_dir.glob("sma-*"))
    assert len(files) == 35


def test_weekly_digest_sunday_1800(tmp_path):
    """2026-08-29: week-in-review digest picked up automatically since this
    generator reads the same src/sma/schedule.py manifest as the launchd
    renderer -- no per-job wiring needed here."""
    out_dir = _render(tmp_path)
    timer_text = (out_dir / "sma-weekly-digest.timer").read_text()
    assert "OnCalendar=Sun 18:00:00" in timer_text
    service_text = (out_dir / "sma-weekly-digest.service").read_text()
    assert "sma.monitoring" in service_text
    assert "weekly-digest" in service_text


def test_every_schedule_job_has_a_service_and_timer(tmp_path):
    from sma.schedule import SCHEDULE

    out_dir = _render(tmp_path)
    from ops.launchd.render_plists import _short_name

    for job in SCHEDULE:
        short = _short_name(job.label)
        assert (out_dir / f"sma-{short}.service").exists(), f"missing service for {job.label}"
        assert (out_dir / f"sma-{short}.timer").exists(), f"missing timer for {job.label}"


def test_ingest_calendar_mon_fri_1830(tmp_path):
    out_dir = _render(tmp_path)
    timer_text = (out_dir / "sma-ingest.timer").read_text()
    assert "OnCalendar=Mon..Fri 18:30:00" in timer_text


def test_decide_uses_theses_flag(tmp_path):
    out_dir = _render(tmp_path)
    service_text = (out_dir / "sma-live.decide.service").read_text()
    assert "--use-theses" in service_text
    timer_text = (out_dir / "sma-live.decide.timer").read_text()
    assert "OnCalendar=Mon..Fri 20:00:00" in timer_text


def test_retrain_monday_0400_keeps_demean_labels(tmp_path):
    """Weekly retrain MUST train demeaned (alpha) labels; see the identical
    guard in tests/unit/ops/test_render_plists.py — a dropped
    --demean-labels here would silently regress the Linux host the same way
    it nearly did on the Mac (2026-06-19)."""
    out_dir = _render(tmp_path)
    timer_text = (out_dir / "sma-model.retrain.timer").read_text()
    assert "OnCalendar=Mon 04:00:00" in timer_text
    service_text = (out_dir / "sma-model.retrain.service").read_text()
    assert "--demean-labels" in service_text


def test_autoresearch_monday_0700(tmp_path):
    out_dir = _render(tmp_path)
    # short name keeps ".nightly" — _short_name only strips .daily/.weekday/
    # .weekly, and autoresearch is none of those.
    timer_text = (out_dir / "sma-autoresearch.nightly.timer").read_text()
    assert "OnCalendar=Mon 07:00:00" in timer_text
    service_text = (out_dir / "sma-autoresearch.nightly.service").read_text()
    assert "sma.autoresearch" in service_text
    assert "search" in service_text
    assert "--n-configs" in service_text


def test_backup_fires_every_day(tmp_path):
    """The only job that runs all 7 days; a dropped weekday drops a whole
    day's backup silently (mirrors test_backup_fires_2200_every_day in the
    launchd suite)."""
    out_dir = _render(tmp_path)
    timer_text = (out_dir / "sma-backup.timer").read_text()
    assert "OnCalendar=*-*-* 22:00:00" in timer_text


def test_senate_and_house_sunday(tmp_path):
    out_dir = _render(tmp_path)
    senate = (out_dir / "sma-senate-ingest.timer").read_text()
    house = (out_dir / "sma-house-ingest.timer").read_text()
    assert "OnCalendar=Sun 10:00:00" in senate
    assert "OnCalendar=Sun 11:00:00" in house


def test_watchdog_hourly_and_overnight_checkpoints(tmp_path):
    """Multi-hour watchdog schedule: hourly checkpoints (10,13,16,19-23) plus
    the two overnight half-past slots, as two separate OnCalendar= lines
    (systemd allows the key to repeat) — mirrors the launchd plist's flat
    list of StartCalendarInterval dicts."""
    out_dir = _render(tmp_path)
    timer_text = (out_dir / "sma-watchdog.timer").read_text()
    lines = [ln.strip() for ln in timer_text.splitlines() if ln.strip().startswith("OnCalendar=")]
    assert "OnCalendar=*-*-* 10,13,16,19,20,21,22,23:00:00" in lines
    assert "OnCalendar=*-*-* 00,01:30:00" in lines
    assert len(lines) == 2


def test_watchdog_has_onbootsec_for_fast_recovery_at_boot(tmp_path):
    """launchd's watchdog plist sets RunAtLoad=true for a fast recovery pass
    at boot/wake. systemd's timer analog is OnBootSec=, not a service-level
    flag — a oneshot .service has no "run at unit-file-load" concept, only
    the timer can fire it. Confirm the twin doesn't silently drop this."""
    out_dir = _render(tmp_path)
    timer_text = (out_dir / "sma-watchdog.timer").read_text()
    assert "OnBootSec=" in timer_text


def test_dashboard_is_restart_always_with_no_timer(tmp_path):
    out_dir = _render(tmp_path)
    assert not (out_dir / "sma-dashboard.timer").exists()
    service_text = (out_dir / "sma-dashboard.service").read_text()
    assert "Restart=always" in service_text
    assert "Type=simple" in service_text
    assert "streamlit" in service_text
    assert "caffeinate" not in service_text  # no Linux equivalent; see README


def test_no_unit_references_caffeinate(tmp_path):
    """caffeinate is macOS-only; render_plists.py bakes it into every
    ProgramArguments, but there is nothing to keep awake on a host that never
    sleeps (host-migration-runbook.md Section 2b)."""
    out_dir = _render(tmp_path)
    for f in out_dir.glob("sma-*"):
        assert "caffeinate" not in f.read_text(), f"{f.name} still references caffeinate"


def test_every_service_sets_working_directory_and_environment_file(tmp_path):
    out_dir = _render(tmp_path)
    for f in out_dir.glob("sma-*.service"):
        text = f.read_text()
        assert "WorkingDirectory=/REPO" in text
        assert "EnvironmentFile=/REPO/.env" in text
        assert "Environment=TZ=America/New_York" in text


def test_every_timer_sets_persistent_true(tmp_path):
    """Persistent=true reproduces launchd's missed-job catch-up at next
    boot/wake — the exact mechanism that fired all 14 jobs at once during
    the 2026-08-12 boot storm. The application-layer guards that mechanism
    stress-tested (dead-zone checks, sentinel idempotency, writer/heavy
    locks) are what make this safe, not the absence of catch-up."""
    out_dir = _render(tmp_path)
    for f in out_dir.glob("sma-*.timer"):
        assert "Persistent=true" in f.read_text(), f"{f.name} missing Persistent=true"


def test_oneshot_jobs_are_type_oneshot(tmp_path):
    out_dir = _render(tmp_path)
    for f in out_dir.glob("sma-*.service"):
        if f.stem == "sma-dashboard":
            continue
        assert "Type=oneshot" in f.read_text()


def test_generated_header_documents_timezone_assumption(tmp_path):
    """Task requirement: the ET-wall-clock assumption (host system timezone
    must be America/New_York; OnCalendar= does not read TZ=) must be
    documented prominently in every generated unit's header, not just the
    README."""
    out_dir = _render(tmp_path)
    for f in out_dir.glob("sma-*"):
        text = f.read_text()
        assert "America/New_York" in text
        assert "GENERATED" in text.upper()


def test_check_no_drift_when_installed_matches_rendered(tmp_path):
    from ops.systemd.render_units import _check_drift

    installed_dir = tmp_path / "installed"
    render_all(
        out_dir=installed_dir, repo_root=Path("/REPO"), venv=Path("/VENV"), home_dir=Path("/HOME")
    )
    rc = _check_drift(
        repo_root=Path("/REPO"),
        venv=Path("/VENV"),
        home_dir=Path("/HOME"),
        installed_dir=installed_dir,
    )
    assert rc == 0


def test_check_detects_drift(tmp_path):
    from ops.systemd.render_units import _check_drift

    installed_dir = tmp_path / "installed"
    render_all(
        out_dir=installed_dir, repo_root=Path("/REPO"), venv=Path("/VENV"), home_dir=Path("/HOME")
    )
    drifted = installed_dir / "sma-ingest.timer"
    drifted.write_text(drifted.read_text().replace("18:30:00", "19:00:00"))
    rc = _check_drift(
        repo_root=Path("/REPO"),
        venv=Path("/VENV"),
        home_dir=Path("/HOME"),
        installed_dir=installed_dir,
    )
    assert rc == 1


def test_check_reports_missing_installed_dir(tmp_path):
    from ops.systemd.render_units import _check_drift

    rc = _check_drift(
        repo_root=Path("/REPO"),
        venv=Path("/VENV"),
        home_dir=Path("/HOME"),
        installed_dir=tmp_path / "does-not-exist",
    )
    assert rc == 1
