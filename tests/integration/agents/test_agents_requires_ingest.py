"""Agents must not build theses on yesterday's news.

2026-08-12: the reboot at 20:11 ET fired the whole manifest at once, so agents
started at 20:15 while ingest was still running (it did not finish until 20:23).
Every thesis that night was built on the PREVIOUS day's news and filings.

decide already refuses to run on a missing upstream (preflight). agents had no
such gate — it is upstream of decide's thesis overlay, so a stale-news thesis
night silently degrades the trade decisions that follow.

The defer contract: exit 0 writing NO agents sentinel. The watchdog re-kicks any
job that is past its deadline with no sentinel, inside late_kick_max_hours
(agents: deadline 20:15 + 6h default = kickable to 02:15, with checkpoints at
21/22/23/00:30/01:30), so a deferred run self-heals once ingest lands.
"""

from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner

from sma.agents.__main__ import cli
from sma.sentinels import read_sentinel, write_sentinel

ET = ZoneInfo("America/New_York")
ASOF = date(2026, 8, 12)


def _make_writer_lock_factory(lock_path: Path):
    from sma.locks import writer_lock as _real_writer_lock

    @contextmanager
    def _writer_lock_with_path(*, label: str, **kwargs):
        with _real_writer_lock(lock_path=lock_path, label=label, **kwargs):
            yield

    return _writer_lock_with_path


def _fake_settings():
    settings = MagicMock()
    settings.agents.daily_budget_usd = 0.5
    settings.agents.warn_threshold_pct = 80
    settings.agents.haiku_model_id = "claude-haiku-4-5"
    settings.agents.triggers.price_move_pct = 0.05
    settings.agents.triggers.material_filing_types = ["8-K", "10-K", "10-Q"]
    settings.secrets.anthropic_api_key = "test-key"
    return settings


def _run_agents(*, tmp_path, monkeypatch, now_et: datetime, asof: date = ASOF):
    lock_path = tmp_path / ".sma-writer.lock"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", lock_path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text("tickers: [AAPL, MSFT]\n")

    fake_pipeline = MagicMock()
    fake_pipeline.run.return_value = MagicMock(conviction="bullish", score=0.6)

    with (
        patch("sma.agents.__main__.load_settings", return_value=_fake_settings()),
        patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline),
        patch("sma.agents.__main__._build_context", return_value=MagicMock()),
        patch("sma.agents.__main__.load_universe", return_value=["AAPL", "MSFT"]),
        patch("sma.agents.__main__.writer_lock", _make_writer_lock_factory(lock_path)),
        patch("sma.agents.__main__._now_et", return_value=now_et),
    ):
        result = CliRunner().invoke(
            cli,
            ["run", "--asof-date", asof.isoformat(), "--force-full",
             "--db", str(tmp_path / "test.duckdb"), "--config", str(config_path),
             "--universe", str(universe_path)],
            catch_exceptions=False,
        )
    return result, fake_pipeline


def _write_ingest_sentinel(asof: date = ASOF, **extra):
    payload = {
        "label": "com.sma.ingest.daily",
        "asof": asof.isoformat(),
        "run_id": 7,
        "completed_at": "2026-08-12T22:23:00Z",
        "quality": {"passed": True, "blocking_failures": []},
    }
    payload.update(extra)
    write_sentinel(label="com.sma.ingest.daily", asof=asof, payload=payload)


def test_defers_when_todays_ingest_has_not_completed(tmp_path, monkeypatch):
    """The 2026-08-12 case: agents fired at 20:15 with ingest still running."""
    result, pipeline = _run_agents(
        tmp_path=tmp_path, monkeypatch=monkeypatch,
        now_et=datetime(2026, 8, 12, 20, 15, tzinfo=ET),
    )

    assert result.exit_code == 0, result.output
    assert pipeline.run.call_count == 0, "no thesis may be built on stale news"
    assert "ingest" in result.output.lower()


def test_deferred_run_writes_no_sentinel_so_the_watchdog_rekicks(tmp_path, monkeypatch):
    """Writing an agents sentinel here would tell the watchdog the job is DONE
    and strand the night with zero fresh theses."""
    _run_agents(
        tmp_path=tmp_path, monkeypatch=monkeypatch,
        now_et=datetime(2026, 8, 12, 20, 15, tzinfo=ET),
    )

    assert read_sentinel(label="com.sma.agents.daily", asof=ASOF) is None


def test_runs_once_todays_ingest_sentinel_lands(tmp_path, monkeypatch):
    """The normal 19:45 fire, with ingest already done, must proceed untouched.

    Historically this needed to land before a flat 19:58 ET wall-clock cutoff
    that stopped agents from starting ANY ticker past that time. Replaced
    2026-08-13 by a decide-sentinel gate (_deadline_reached in
    sma/agents/__main__.py) so a legitimately late re-kick can still do real
    work for as long as decide has not yet run for the day.
    """
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    _write_ingest_sentinel()

    result, pipeline = _run_agents(
        tmp_path=tmp_path, monkeypatch=monkeypatch,
        now_et=datetime(2026, 8, 12, 19, 50, tzinfo=ET),
    )

    assert result.exit_code == 0, result.output
    assert pipeline.run.call_count == 2, "both universe tickers should be processed"
    assert read_sentinel(label="com.sma.agents.daily", asof=ASOF) is not None


def test_historical_asof_is_not_gated(tmp_path, monkeypatch):
    """Manual/backfill runs for a PAST asof are unbudgeted and ungated, matching
    the same-day-only rule the deadline budget already uses (ingest doctrine)."""
    result, pipeline = _run_agents(
        tmp_path=tmp_path, monkeypatch=monkeypatch,
        # Wall clock is a later day than the asof being rebuilt.
        now_et=datetime(2026, 8, 20, 11, 0, tzinfo=ET),
        asof=date(2026, 8, 12),
    )

    assert result.exit_code == 0, result.output
    assert pipeline.run.call_count == 2, "a historical rebuild must not be gated"
