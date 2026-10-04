"""The 09:25 sweep must refuse to act on out-of-hours marks.

2026-08-12 incident: the Mac rebooted at 20:11 ET. launchd re-fires a MISSED
StartCalendarInterval at the next boot, so the 09:25 stop-loss sweep ran at
20:15 — 10.8h late, against after-hours marks — even though the watchdog had
explicitly logged "too late to safely kick" and kicked nothing (kicked=0).
RunAtLoad was already false on every scheduled job, so no plist change can
prevent this: the guard has to live in the job.

Defense in depth: no kick mechanism (launchd catch-up, watchdog, manual) may
make the sweep evaluate positions or submit orders outside a sane window.
"""

from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from sma.config import LivePreOpenGuard
from sma.sentinels import read_sentinel

ET = ZoneInfo("America/New_York")
ASOF = date(2026, 8, 12)  # a Wednesday


def _settings_with_guard(**guard_kwargs):
    return SimpleNamespace(
        live=SimpleNamespace(preopen_guard=LivePreOpenGuard(**guard_kwargs))
    )


def _make_writer_lock_factory(lock_path: Path):
    from sma.locks import writer_lock as _real_writer_lock

    @contextmanager
    def _writer_lock_with_path(*, label: str, **kwargs):
        with _real_writer_lock(lock_path=lock_path, label=label, **kwargs):
            yield

    return _writer_lock_with_path


def _session(day: date, *, close_hour: int = 16, close_minute: int = 0):
    """(open, close) ET-aware, the shape AlpacaClient.session_window returns."""
    return (
        datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET),
        datetime(day.year, day.month, day.day, close_hour, close_minute, tzinfo=ET),
    )


def _make_alpaca_mock(*, trading_day: bool = True, session=None) -> MagicMock:
    alpaca = MagicMock()
    alpaca.get_positions.return_value = {}
    alpaca.list_open_orders.return_value = []
    alpaca.get_account.return_value = {
        "equity": 100_000.0,
        "cash": 50_000.0,
        "buying_power": 100_000.0,
        "long_market_value": 50_000.0,
    }
    if not trading_day:
        alpaca.session_window.return_value = None
    else:
        alpaca.session_window.return_value = session or _session(ASOF)
    return alpaca


def _run_sweep(
    *, monkeypatch, tmp_path, now_et: datetime, trading_day: bool = True,
    session=None, asof: date = ASOF,
):
    """Invoke the stop-loss CLI with the wall clock pinned to `now_et`."""
    import sma.locks as _locks

    lock_path = tmp_path / ".sma-writer.lock"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", lock_path)

    db_path = tmp_path / "test.duckdb"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")

    alpaca = _make_alpaca_mock(trading_day=trading_day, session=session)
    pages: list[tuple[str, str]] = []

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
        patch("sma.live.__main__._now_et", return_value=now_et),
    ):
        result = CliRunner().invoke(
            stop_loss_sweep_cmd,
            ["--asof-date", asof.isoformat(), "--db", str(db_path),
             "--config", str(config_path)],
            catch_exceptions=False,
        )
    return result, alpaca, pages


def test_sweep_refuses_to_run_after_the_close(monkeypatch, tmp_path):
    """The exact 2026-08-12 case: a boot-triggered catch-up at 20:15 ET."""
    result, alpaca, pages = _run_sweep(
        monkeypatch=monkeypatch, tmp_path=tmp_path,
        now_et=datetime(2026, 8, 12, 20, 15, tzinfo=ET),
    )

    assert result.exit_code == 0, result.output
    assert "out of hours" in result.output.lower()
    # It must not have touched the broker book at all.
    alpaca.get_positions.assert_not_called()
    alpaca.submit_market_sell.assert_not_called()
    alpaca.cancel_all_open_orders.assert_not_called()
    # One page so the skipped sweep reaches a human.
    assert len(pages) == 1
    assert "stop-loss" in pages[0][0].lower()


def test_sweep_refuses_to_run_before_the_window_opens(monkeypatch, tmp_path):
    """A 03:00 ET recovery run is just as unsafe as a 20:15 one."""
    result, alpaca, _ = _run_sweep(
        monkeypatch=monkeypatch, tmp_path=tmp_path,
        now_et=datetime(2026, 8, 12, 3, 0, tzinfo=ET),
    )

    assert result.exit_code == 0, result.output
    alpaca.get_positions.assert_not_called()
    alpaca.submit_market_sell.assert_not_called()


def test_skipped_sweep_writes_a_flagged_sentinel(monkeypatch, tmp_path):
    """Terminate cleanly: the watchdog re-kicks a job with no sentinel inside its
    late-kick window, and pages at every checkpoint past it. A sentinel flagged
    `skipped_out_of_hours` stops both without pretending the sweep ran."""
    _run_sweep(
        monkeypatch=monkeypatch, tmp_path=tmp_path,
        now_et=datetime(2026, 8, 12, 20, 15, tzinfo=ET),
    )

    sentinel = read_sentinel(label="com.sma.live.stop-loss.weekday", asof=ASOF)
    assert sentinel is not None, "a skipped sweep must still terminate the watchdog"
    assert sentinel["skipped_out_of_hours"] is True
    assert sentinel["positions_triggered"] == 0
    assert sentinel["sells_submitted"] == 0
    # No run_id: sentinels._is_strictly_newer refuses to let this skip-sentinel
    # overwrite a REAL run's sentinel that already carries one.
    assert sentinel.get("run_id") is None


def test_skip_sentinel_never_overwrites_a_real_run(monkeypatch, tmp_path):
    """Order matters: a real 09:25 run followed by a stray 20:15 catch-up must
    leave the real sentinel intact."""
    import sma.locks as _locks

    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", tmp_path / ".sma-writer.lock")

    from sma.sentinels import write_sentinel

    write_sentinel(
        label="com.sma.live.stop-loss.weekday", asof=ASOF,
        payload={"label": "com.sma.live.stop-loss.weekday", "run_id": 42,
                 "completed_at": "2026-08-12T13:25:00Z", "sells_submitted": 1},
    )

    _run_sweep(
        monkeypatch=monkeypatch, tmp_path=tmp_path,
        now_et=datetime(2026, 8, 12, 20, 15, tzinfo=ET),
    )

    sentinel = read_sentinel(label="com.sma.live.stop-loss.weekday", asof=ASOF)
    assert sentinel["run_id"] == 42, "the real morning run's sentinel was clobbered"
    assert sentinel["sells_submitted"] == 1
    assert "skipped_out_of_hours" not in sentinel


def test_successful_sweep_sentinel_carries_a_run_id(monkeypatch, tmp_path):
    """The forensic record's only protection is the run_id: without one, the
    skip payload's completed_at wins the _is_strictly_newer fallback."""
    _run_sweep(
        monkeypatch=monkeypatch, tmp_path=tmp_path,
        now_et=datetime(2026, 8, 12, 9, 25, tzinfo=ET),
    )

    sentinel = read_sentinel(label="com.sma.live.stop-loss.weekday", asof=ASOF)
    assert isinstance(sentinel.get("run_id"), int), (
        "a real sweep must stamp a run_id, on every path"
    )


def test_a_real_morning_run_survives_a_later_catch_up_skip(monkeypatch, tmp_path):
    """End to end, in the order it actually happens: a genuine 09:25 sweep,
    then a 20:15 launchd catch-up after a reboot. Before the success payload
    carried a run_id both sentinels were run_id-less, so _is_strictly_newer
    fell through to completed_at and the LATER skip always won — erasing the
    morning run's halt flags from the pre-open guard."""
    real, _, _ = _run_sweep(
        monkeypatch=monkeypatch, tmp_path=tmp_path,
        now_et=datetime(2026, 8, 12, 9, 25, tzinfo=ET),
    )
    assert real.exit_code == 0, real.output
    morning = read_sentinel(label="com.sma.live.stop-loss.weekday", asof=ASOF)
    assert "skipped_out_of_hours" not in morning

    catch_up, _, _ = _run_sweep(
        monkeypatch=monkeypatch, tmp_path=tmp_path,
        now_et=datetime(2026, 8, 12, 20, 15, tzinfo=ET),
    )
    assert catch_up.exit_code == 0, catch_up.output

    after = read_sentinel(label="com.sma.live.stop-loss.weekday", asof=ASOF)
    assert after == morning, "the catch-up skip overwrote the real morning run"
    assert "skipped_out_of_hours" not in after
    assert "halted_preopen_divergence" in after


def test_non_trading_day_skips_quietly(monkeypatch, tmp_path):
    """A holiday sweep has no work. Skip WITHOUT paging — decide's holiday path
    is a quiet no-op for the same reason, and a page every holiday is noise."""
    result, alpaca, pages = _run_sweep(
        monkeypatch=monkeypatch, tmp_path=tmp_path,
        now_et=datetime(2026, 8, 12, 9, 25, tzinfo=ET),
        trading_day=False,
    )

    assert result.exit_code == 0, result.output
    alpaca.get_positions.assert_not_called()
    assert pages == [], "a holiday skip must not page"
    sentinel = read_sentinel(label="com.sma.live.stop-loss.weekday", asof=ASOF)
    assert sentinel["holiday_skipped"] is True


def test_sweep_runs_normally_inside_the_window(monkeypatch, tmp_path):
    """The guard must not break the real 09:25 fire — tomorrow's live run."""
    result, alpaca, pages = _run_sweep(
        monkeypatch=monkeypatch, tmp_path=tmp_path,
        now_et=datetime(2026, 8, 12, 9, 25, tzinfo=ET),
    )

    assert result.exit_code == 0, result.output
    alpaca.get_positions.assert_called()
    assert pages == []
    sentinel = read_sentinel(label="com.sma.live.stop-loss.weekday", asof=ASOF)
    assert sentinel.get("skipped_out_of_hours") is not True
    assert "positions_checked" in sentinel


def test_guard_window_is_anchored_to_the_session_open(monkeypatch, tmp_path):
    """Boundaries: open-10min and open+15min are IN, a minute outside is OUT.

    The sweep PROXIES THE OPEN (stop_loss.py module contract) — it prices the
    exit off the pre-market/last-close mark as a stand-in for the auction
    print — so the band is tight around 09:30, not the whole session.
    """
    for hh, mm, should_run in [(9, 19, False), (9, 20, True), (9, 25, True),
                               (9, 45, True), (9, 46, False)]:
        case_dir = tmp_path / f"{hh:02d}{mm:02d}"
        case_dir.mkdir()
        result, alpaca, _ = _run_sweep(
            monkeypatch=monkeypatch, tmp_path=case_dir,
            now_et=datetime(2026, 8, 12, hh, mm, tzinfo=ET),
        )
        assert result.exit_code == 0, result.output
        assert alpaca.get_positions.called is should_run, (
            f"{hh:02d}:{mm:02d} should {'run' if should_run else 'skip'}"
        )


def test_midday_late_kick_skips(monkeypatch, tmp_path):
    """A watchdog/launchd late-kick at 15:30 passed the old 09:20-16:05 band
    and ran an OPEN proxy six hours after the open. It must skip."""
    result, alpaca, pages = _run_sweep(
        monkeypatch=monkeypatch, tmp_path=tmp_path,
        now_et=datetime(2026, 8, 12, 15, 30, tzinfo=ET),
    )

    assert result.exit_code == 0, result.output
    alpaca.get_positions.assert_not_called()
    alpaca.submit_market_sell.assert_not_called()
    assert len(pages) == 1


def test_half_day_afternoon_skips(monkeypatch, tmp_path):
    """2026-11-27 (day after Thanksgiving) closes at 13:00 ET. Under the flat
    09:20-16:05 band a 13:30 invocation swept POST-CLOSE marks and called them
    live. Probed against the live Alpaca calendar 2026-08-16: open 09:30,
    close 13:00."""
    half_day = date(2026, 11, 27)
    result, alpaca, pages = _run_sweep(
        monkeypatch=monkeypatch, tmp_path=tmp_path,
        now_et=datetime(2026, 11, 27, 13, 30, tzinfo=ET),
        asof=half_day,
        session=_session(half_day, close_hour=13, close_minute=0),
    )

    assert result.exit_code == 0, result.output
    alpaca.get_positions.assert_not_called()
    alpaca.submit_market_sell.assert_not_called()
    assert len(pages) == 1
    assert "13:00" in pages[0][1], "the page should name the real session close"


def test_half_day_morning_still_runs(monkeypatch, tmp_path):
    """The half-day OPEN is still 09:30, so the normal 09:25 fire is valid —
    the shortened close must not suppress the sweep that day."""
    half_day = date(2026, 12, 24)
    result, alpaca, pages = _run_sweep(
        monkeypatch=monkeypatch, tmp_path=tmp_path,
        now_et=datetime(2026, 12, 24, 9, 25, tzinfo=ET),
        asof=half_day,
        session=_session(half_day, close_hour=13, close_minute=0),
    )

    assert result.exit_code == 0, result.output
    alpaca.get_positions.assert_called()
    assert pages == []


def test_calendar_failure_falls_back_to_the_fixed_band(monkeypatch, tmp_path):
    """A calendar blip must not BYPASS the guard: the fallback band around the
    standard 09:30 open still applies. 09:25 runs, 15:30 skips."""
    for hh, mm, should_run in [(9, 25, True), (15, 30, False)]:
        case_dir = tmp_path / f"{hh:02d}{mm:02d}"
        case_dir.mkdir()
        import sma.locks as _locks

        monkeypatch.setenv("SMA_SENTINEL_DIR", str(case_dir / "sentinels"))
        monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", case_dir / ".lock")
        config_path = case_dir / "config.yaml"
        config_path.write_text("{}\n")

        alpaca = _make_alpaca_mock()
        alpaca.session_window.side_effect = RuntimeError("calendar 503")
        pages: list[tuple[str, str]] = []

        from click.testing import CliRunner

        from sma.live.__main__ import stop_loss_sweep_cmd
        from sma.risk.rails import RiskRails

        with (
            patch("sma.live.__main__._build_alpaca", return_value=alpaca),
            patch("sma.live.__main__._build_rails", return_value=RiskRails(stop_loss_pct=0.0)),
            patch("sma.live.__main__.load_settings", return_value=_settings_with_guard()),
            patch("sma.live.__main__.notify_failure",
                  lambda title, message, _p=pages: _p.append((title, message))),
            patch("sma.live.__main__.writer_lock",
                  _make_writer_lock_factory(case_dir / ".lock")),
            patch("sma.live.__main__._now_et",
                  return_value=datetime(2026, 8, 12, hh, mm, tzinfo=ET)),
        ):
            result = CliRunner().invoke(
                stop_loss_sweep_cmd,
                ["--asof-date", ASOF.isoformat(), "--db", str(case_dir / "t.duckdb"),
                 "--config", str(config_path)],
                catch_exceptions=False,
            )
        assert result.exit_code == 0, result.output
        assert alpaca.get_positions.called is should_run, (
            f"{hh:02d}:{mm:02d} should {'run' if should_run else 'skip'} on a "
            f"calendar failure"
        )
