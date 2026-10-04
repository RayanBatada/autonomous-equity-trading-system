"""Agents' per-ticker deadline budget: start a same-day ticker only while
(now_ET < 19:58 on asof) OR (decide's sentinel for asof already exists), with
a 02:30 ET hard floor the next morning.

2026-08-13: da322f7 made agents DEFER (no sentinel) when today's ingest
sentinel is missing, so a post-outage watchdog re-kick can land well after
19:58 with ZERO tickers processed. The flat wall-clock check tripped on the
FIRST ticker of any such re-kick and skipped the entire universe.

91e0baa over-corrected into a pure decide-gate — "keep going until decide's
sentinel exists" — which is CIRCULAR: agents holds the single global
writer_lock (sma.locks) across its entire ticker loop, and decide needs that
same lock before it can write the sentinel the gate waits on. The gate waited
on a value it prevented, producing two deterministic no-trade paths:

  (a) a Friday full refresh from 19:45 holds the lock past 20:00; decide
      times out at 20:15 (timeout_s=900) and pages "bot did NOT trade";
  (b) a post-outage 01:30 watchdog pass kickstarts agents BEFORE decide
      (SCHEDULE iteration order); agents takes the lock, decide dies at
      01:45, and no checkpoint remains before decide's 03:00 window closes.

The union rule fixes both: agents yields the lock whenever decide is still
pending, and only keeps grinding once decide has actually run (a late
re-kick then does real work for tomorrow's overlay instead of tripping on
ticker one). Theses are ADVISORY and the stale-thesis fallback already
covers the deferred night.
"""

from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner

from sma.agents.__main__ import cli
from sma.sentinels import read_sentinel, write_sentinel

ET = ZoneInfo("America/New_York")
ASOF = date(2026, 8, 13)


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


def _write_decide_sentinel(asof: date = ASOF, **extra):
    payload = {
        "label": "com.sma.live.decide.daily",
        "asof": asof.isoformat(),
        "decisions_total": 11,
        "submitted_count": 0,
        "completed_at": "2026-08-13T00:24:32Z",
    }
    payload.update(extra)
    write_sentinel(label="com.sma.live.decide.daily", asof=asof, payload=payload)


def _write_ingest_sentinel(asof: date = ASOF, **extra):
    payload = {
        "label": "com.sma.ingest.daily",
        "asof": asof.isoformat(),
        "run_id": 7,
        "completed_at": "2026-08-13T18:39:00Z",
        "quality": {"passed": True, "blocking_failures": []},
    }
    payload.update(extra)
    write_sentinel(label="com.sma.ingest.daily", asof=asof, payload=payload)


def _run_agents(
    *, tmp_path, monkeypatch, now_et: datetime | None = None, asof: date = ASOF,
    pipeline=None, tickers=("AAPL", "MSFT", "NVDA"), now_sequence=None,
):
    """`now_sequence` pins successive _now_et() reads (ingest gate first, then
    one per ticker iteration) so a run can straddle the 19:58 cutoff."""
    lock_path = tmp_path / ".sma-writer.lock"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", lock_path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text(f"tickers: [{', '.join(tickers)}]\n")

    fake_pipeline = pipeline if pipeline is not None else MagicMock()
    if pipeline is None:
        fake_pipeline.run.return_value = MagicMock(conviction="bullish", score=0.6)

    if now_sequence is not None:
        now_patch = mock_patch(
            "sma.agents.__main__._now_et", side_effect=list(now_sequence)
        )
    else:
        now_patch = mock_patch("sma.agents.__main__._now_et", return_value=now_et)

    with (
        mock_patch("sma.agents.__main__.load_settings", return_value=_fake_settings()),
        mock_patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline),
        mock_patch("sma.agents.__main__._build_context", return_value=MagicMock()),
        mock_patch("sma.agents.__main__.load_universe", return_value=list(tickers)),
        mock_patch("sma.agents.__main__.writer_lock", _make_writer_lock_factory(lock_path)),
        now_patch,
    ):
        result = CliRunner().invoke(
            cli,
            ["run", "--asof-date", asof.isoformat(), "--force-full",
             "--db", str(tmp_path / "test.duckdb"), "--config", str(config_path),
             "--universe", str(universe_path)],
            catch_exceptions=False,
        )
    return result, fake_pipeline


def test_evening_run_before_the_cutoff_processes_everything(tmp_path, monkeypatch):
    """The normal 19:45 fire: well before 19:58, decide hasn't run yet, so the
    whole list is worked through."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    _write_ingest_sentinel()
    result, pipeline = _run_agents(
        tmp_path=tmp_path, monkeypatch=monkeypatch,
        now_et=datetime(2026, 8, 13, 19, 46, tzinfo=ET),
    )

    assert result.exit_code == 0, result.output
    assert pipeline.run.call_count == 3
    sentinel = read_sentinel(label="com.sma.agents.daily", asof=ASOF)
    assert sentinel["tickers_skipped_deadline"] == 0


def test_friday_full_refresh_yields_the_lock_at_1958(tmp_path, monkeypatch):
    """Failure path (a): a Friday full refresh started 19:45 is still grinding
    at 20:05 with decide not yet run. Under the pure decide-gate it kept the
    writer lock, decide timed out at 20:15 (timeout_s=900) and paged "bot did
    NOT trade". agents must instead stop starting tickers and release."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    _write_ingest_sentinel()
    result, pipeline = _run_agents(
        tmp_path=tmp_path, monkeypatch=monkeypatch,
        now_et=datetime(2026, 8, 13, 20, 5, tzinfo=ET),
    )

    assert result.exit_code == 0, result.output
    assert pipeline.run.call_count == 0, "agents must yield the lock to decide"
    sentinel = read_sentinel(label="com.sma.agents.daily", asof=ASOF)
    assert sentinel["tickers_skipped_deadline"] == 3


def test_overnight_watchdog_kick_defers_to_a_pending_decide(tmp_path, monkeypatch):
    """Failure path (b): the 01:30 post-outage checkpoint kickstarts agents
    BEFORE decide (SCHEDULE order). decide's late-kick window closes at 03:00
    and there is no checkpoint after 01:30, so agents holding the lock here is
    the whole night's trading. It must defer instead."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    _write_ingest_sentinel()
    result, pipeline = _run_agents(
        tmp_path=tmp_path, monkeypatch=monkeypatch,
        now_et=datetime(2026, 8, 14, 1, 30, tzinfo=ET),
    )

    assert result.exit_code == 0, result.output
    assert pipeline.run.call_count == 0
    sentinel = read_sentinel(label="com.sma.agents.daily", asof=ASOF)
    assert sentinel["tickers_skipped_deadline"] == 3


def test_late_rekick_after_decide_has_run_does_real_work(tmp_path, monkeypatch):
    """The da322f7 case the decide-gate existed for: an ingest heal pushes
    agents to a 21:15 re-kick. decide already ran at 20:00, so nothing is
    queued behind us on the lock — grind the universe for tomorrow's overlay
    rather than tripping on ticker one (the 2026-08-13 zero-thesis "success")."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    _write_ingest_sentinel()
    _write_decide_sentinel()
    result, pipeline = _run_agents(
        tmp_path=tmp_path, monkeypatch=monkeypatch,
        now_et=datetime(2026, 8, 13, 21, 15, tzinfo=ET),
    )

    assert result.exit_code == 0, result.output
    assert pipeline.run.call_count == 3
    sentinel = read_sentinel(label="com.sma.agents.daily", asof=ASOF)
    assert sentinel["tickers_skipped_deadline"] == 0


def test_overnight_continuation_after_decide_ran_keeps_going(tmp_path, monkeypatch):
    """01:00 ET the morning after, decide's sentinel is present: the lock is
    uncontended, so the continuation runs right up to the 02:30 floor."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    _write_ingest_sentinel()
    _write_decide_sentinel()
    result, pipeline = _run_agents(
        tmp_path=tmp_path, monkeypatch=monkeypatch,
        now_et=datetime(2026, 8, 14, 1, 0, tzinfo=ET),
    )

    assert result.exit_code == 0, result.output
    assert pipeline.run.call_count == 3


def test_hard_floor_stops_at_0240_even_with_a_decide_sentinel(tmp_path, monkeypatch):
    """The 02:30 floor is independent of the decide gate in BOTH directions:
    even with decide's sentinel present (so the lock is free), agents must not
    still be grinding into decide's 03:00 late-kick tail."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    _write_ingest_sentinel()
    _write_decide_sentinel()
    result, pipeline = _run_agents(
        tmp_path=tmp_path, monkeypatch=monkeypatch,
        now_et=datetime(2026, 8, 14, 2, 40, tzinfo=ET),
    )

    assert result.exit_code == 0, result.output
    assert pipeline.run.call_count == 0
    sentinel = read_sentinel(label="com.sma.agents.daily", asof=ASOF)
    assert sentinel["tickers_skipped_deadline"] == 3


def test_hard_floor_stops_at_0240_without_a_decide_sentinel(tmp_path, monkeypatch):
    """decide has NOT run (no sentinel) and the wall clock is 02:40 ET the
    morning after — agents cannot run forever if decide never completes."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    _write_ingest_sentinel()
    result, pipeline = _run_agents(
        tmp_path=tmp_path, monkeypatch=monkeypatch,
        now_et=datetime(2026, 8, 14, 2, 40, tzinfo=ET),
    )

    assert result.exit_code == 0, result.output
    assert pipeline.run.call_count == 0
    sentinel = read_sentinel(label="com.sma.agents.daily", asof=ASOF)
    assert sentinel["tickers_skipped_deadline"] == 3


def test_clock_crossing_1958_mid_run_stops_starting_new_tickers(tmp_path, monkeypatch):
    """The cutoff is evaluated per ticker: a run that starts at 19:50 keeps
    going until the clock passes 19:58, then skips the tail."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    _write_ingest_sentinel()
    # One read for the ingest gate, then one per ticker iteration.
    result, pipeline = _run_agents(
        tmp_path=tmp_path, monkeypatch=monkeypatch,
        now_sequence=[
            datetime(2026, 8, 13, 19, 50, tzinfo=ET),  # ingest gate
            datetime(2026, 8, 13, 19, 50, tzinfo=ET),  # AAPL — runs
            datetime(2026, 8, 13, 19, 57, tzinfo=ET),  # MSFT — runs
            datetime(2026, 8, 13, 19, 58, tzinfo=ET),  # NVDA — cutoff
        ],
    )

    assert result.exit_code == 0, result.output
    assert pipeline.run.call_count == 2
    sentinel = read_sentinel(label="com.sma.agents.daily", asof=ASOF)
    assert sentinel["tickers_skipped_deadline"] == 1


def test_decide_landing_mid_run_does_not_stop_agents(tmp_path, monkeypatch):
    """decide completing mid-loop only ever makes the lock LESS contended, so
    it must not cut the run short (the old gate did exactly that)."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    _write_ingest_sentinel()

    def _side_effect(ctx, *, run_id):
        # Simulate decide completing right after the first ticker starts.
        _write_decide_sentinel()
        return MagicMock(conviction="bullish", score=0.6)

    fake_pipeline = MagicMock()
    fake_pipeline.run.side_effect = _side_effect

    result, pipeline = _run_agents(
        tmp_path=tmp_path, monkeypatch=monkeypatch,
        now_et=datetime(2026, 8, 13, 19, 50, tzinfo=ET),
        pipeline=fake_pipeline,
    )

    assert result.exit_code == 0, result.output
    assert pipeline.run.call_count == 3
    sentinel = read_sentinel(label="com.sma.agents.daily", asof=ASOF)
    assert sentinel["tickers_skipped_deadline"] == 0


def test_historical_asof_ignores_both_the_gate_and_the_floor(tmp_path, monkeypatch):
    """A manual backfill for a long-past asof must not be gated by either the
    decide-sentinel check or the hard floor (matches ingest doctrine)."""
    result, pipeline = _run_agents(
        tmp_path=tmp_path, monkeypatch=monkeypatch,
        now_et=datetime(2026, 8, 20, 11, 0, tzinfo=ET),
        asof=date(2026, 8, 13),
    )

    assert result.exit_code == 0, result.output
    assert pipeline.run.call_count == 3
