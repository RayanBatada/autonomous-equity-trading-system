"""Task 11: agents pipeline adopts writer_lock + writes sentinel."""

from contextlib import contextmanager
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from sma.agents.__main__ import cli
from sma.agents.base import StrategistOutput
from sma.agents.pipeline import CachedThesisFallback
from sma.sentinels import read_sentinel


def _make_writer_lock_factory(lock_path: Path):
    """Return a drop-in for writer_lock that uses a test-scoped lock_path.

    The real writer_lock captures DEFAULT_LOCK_PATH as a default-arg value at
    import time, so monkeypatching the module attribute does not affect the
    default. This factory explicitly passes the tmp lock_path so the pid file
    lands in tmp_path and the Store.connect() check agrees.
    """
    from sma.locks import writer_lock as _real_writer_lock

    @contextmanager
    def _writer_lock_with_path(*, label: str, **kwargs):
        with _real_writer_lock(lock_path=lock_path, label=label, **kwargs):
            yield

    return _writer_lock_with_path


def _fake_settings():
    """Minimal settings mock so load_settings does not require a real config.yaml."""
    settings = MagicMock()
    settings.agents.daily_budget_usd = 0.5
    settings.agents.warn_threshold_pct = 80
    settings.agents.haiku_model_id = "claude-haiku-4-5"
    settings.agents.triggers.price_move_pct = 0.05
    settings.agents.triggers.material_filing_types = ["8-K", "10-K", "10-Q"]
    settings.secrets.anthropic_api_key = "test-key"
    return settings


def test_run_writes_sentinel(tmp_path, monkeypatch):
    """agents run CLI writes com.sma.agents.daily sentinel inside writer_lock."""
    lock_path = tmp_path / ".sma-writer.lock"
    sentinel_dir = tmp_path / "sentinels"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", lock_path)

    asof = date(2026, 4, 30)
    db_path = tmp_path / "test.duckdb"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text("tickers: [AAPL, MSFT]\n")

    fake_pipeline = MagicMock()
    fake_pipeline.run.return_value = MagicMock(conviction="bullish", score=0.6)

    runner = CliRunner()

    with (
        patch("sma.agents.__main__.load_settings", return_value=_fake_settings()),
        patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline),
        patch("sma.agents.__main__._build_context", return_value=MagicMock()),
        patch(
            "sma.agents.__main__.load_universe",
            return_value=["AAPL", "MSFT"],
        ),
        patch(
            "sma.agents.__main__.writer_lock",
            _make_writer_lock_factory(lock_path),
        ),
    ):
        # 2026-04-30 is a Thursday; use --force-full to avoid trigger logic
        result = runner.invoke(
            cli,
            [
                "run",
                "--asof-date",
                asof.isoformat(),
                "--force-full",
                "--db",
                str(db_path),
                "--config",
                str(config_path),
                "--universe",
                str(universe_path),
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, f"agents run CLI failed:\n{result.output}"

    sentinel = read_sentinel(label="com.sma.agents.daily", asof=asof)
    assert sentinel is not None, "sentinel was not written"
    assert sentinel["label"] == "com.sma.agents.daily"
    assert sentinel["asof"] == asof.isoformat()
    assert "completed_at" in sentinel
    assert sentinel["completed_at"].endswith("Z"), "completed_at must be UTC with Z suffix"
    assert "tickers_processed" in sentinel
    assert "tickers_skipped_no_trigger" in sentinel
    assert "tickers_skipped_budget" in sentinel
    assert "budget_spent_usd" in sentinel
    assert "force_full" in sentinel
    assert sentinel["force_full"] is True
    # With fake pipeline returning success for both tickers:
    assert sentinel["tickers_processed"] == 2
    assert sentinel["tickers_skipped_budget"] == 0
    # quality block lets live_readiness() gate decide on this sentinel. Success
    # path → passed=True, no blocking failures, and a real run_id for monotonicity.
    assert sentinel["run_id"] is not None
    assert sentinel["quality"]["passed"] is True
    assert sentinel["quality"]["blocking_failures"] == []


def test_run_all_tickers_failing_blocks_sentinel(tmp_path, monkeypatch):
    """If EVERY ticker raises (LLM down / code bug), the sentinel must write
    quality.passed=False with an all_tickers_failed blocking failure. Pre-fix,
    per-ticker exceptions were swallowed and _budget_exhausted was only true when
    skipped_budget>0 — so a totally broken run green-lit decide via
    live_readiness()."""
    lock_path = tmp_path / ".sma-writer.lock"
    sentinel_dir = tmp_path / "sentinels"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", lock_path)

    asof = date(2026, 4, 30)
    db_path = tmp_path / "test.duckdb"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text("tickers: [AAPL, MSFT]\n")

    fake_pipeline = MagicMock()
    fake_pipeline.run.side_effect = RuntimeError("boom")  # every ticker fails

    runner = CliRunner()
    with (
        patch("sma.agents.__main__.load_settings", return_value=_fake_settings()),
        patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline),
        patch("sma.agents.__main__._build_context", return_value=MagicMock()),
        patch("sma.agents.__main__.load_universe", return_value=["AAPL", "MSFT"]),
        patch("sma.agents.__main__.writer_lock", _make_writer_lock_factory(lock_path)),
    ):
        result = runner.invoke(
            cli,
            [
                "run", "--asof-date", asof.isoformat(), "--force-full",
                "--db", str(db_path), "--config", str(config_path),
                "--universe", str(universe_path),
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, f"agents run CLI failed:\n{result.output}"
    sentinel = read_sentinel(label="com.sma.agents.daily", asof=asof)
    assert sentinel["tickers_processed"] == 0
    assert sentinel["tickers_failed"] == 2
    assert sentinel["quality"]["passed"] is False
    assert "all_tickers_failed" in sentinel["quality"]["blocking_failures"]


def test_run_tracks_budget_skipped(tmp_path, monkeypatch):
    """tickers_skipped_budget is incremented when pipeline.run returns None (budget)."""
    lock_path = tmp_path / ".sma-writer.lock"
    sentinel_dir = tmp_path / "sentinels"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", lock_path)

    asof = date(2026, 4, 30)
    db_path = tmp_path / "test.duckdb"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text("tickers: [AAPL, MSFT]\n")

    # First ticker succeeds, second returns None (budget exhausted / no cached fallback)
    fake_pipeline = MagicMock()
    fake_pipeline.run.side_effect = [
        MagicMock(conviction="bullish", score=0.6),
        None,
    ]

    runner = CliRunner()

    with (
        patch("sma.agents.__main__.load_settings", return_value=_fake_settings()),
        patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline),
        patch("sma.agents.__main__._build_context", return_value=MagicMock()),
        patch(
            "sma.agents.__main__.load_universe",
            return_value=["AAPL", "MSFT"],
        ),
        patch(
            "sma.agents.__main__.writer_lock",
            _make_writer_lock_factory(lock_path),
        ),
    ):
        result = runner.invoke(
            cli,
            [
                "run",
                "--asof-date",
                asof.isoformat(),
                "--force-full",
                "--db",
                str(db_path),
                "--config",
                str(config_path),
                "--universe",
                str(universe_path),
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, f"agents run CLI failed:\n{result.output}"

    sentinel = read_sentinel(label="com.sma.agents.daily", asof=asof)
    assert sentinel is not None
    # None return means budget-skip (no thesis produced)
    assert sentinel["tickers_processed"] == 1
    assert sentinel["tickers_skipped_budget"] == 1


def test_run_counts_budget_cached_fallback_separately(tmp_path, monkeypatch):
    """A budget-exhausted cached thesis is advisory context, not fresh work."""
    lock_path = tmp_path / ".sma-writer.lock"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", lock_path)

    asof = date(2026, 4, 30)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text("tickers: [AAPL, MSFT]\n")

    fake_pipeline = MagicMock()
    fake_pipeline.run.side_effect = [
        CachedThesisFallback(
            conviction="neutral",
            score=0.0,
            flags=[],
            action_hint="hold",
            reasoning="cached",
        ),
        None,
    ]

    with (
        patch("sma.agents.__main__.load_settings", return_value=_fake_settings()),
        patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline),
        patch("sma.agents.__main__._build_context", return_value=MagicMock()),
        patch("sma.agents.__main__.load_universe", return_value=["AAPL", "MSFT"]),
        patch("sma.agents.__main__.writer_lock", _make_writer_lock_factory(lock_path)),
    ):
        result = CliRunner().invoke(
            cli,
            [
                "run", "--asof-date", asof.isoformat(), "--force-full",
                "--db", str(tmp_path / "test.duckdb"), "--config", str(config_path),
                "--universe", str(universe_path),
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    sentinel = read_sentinel(label="com.sma.agents.daily", asof=asof)
    assert sentinel["tickers_processed"] == 0
    assert sentinel["tickers_cached_fallback"] == 1
    assert sentinel["tickers_skipped_budget"] == 1
    assert sentinel["quality"]["passed"] is False
    assert "budget_exhausted" in sentinel["quality"]["blocking_failures"]


def test_run_does_not_treat_normal_strategist_output_as_cached(tmp_path, monkeypatch):
    lock_path = tmp_path / ".sma-writer.lock"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", lock_path)

    asof = date(2026, 4, 30)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text("tickers: [AAPL]\n")

    fake_pipeline = MagicMock()
    fake_pipeline.run.return_value = StrategistOutput(
        conviction="bullish",
        score=0.6,
        flags=[],
        action_hint="enter",
        reasoning="fresh",
    )

    with (
        patch("sma.agents.__main__.load_settings", return_value=_fake_settings()),
        patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline),
        patch("sma.agents.__main__._build_context", return_value=MagicMock()),
        patch("sma.agents.__main__.load_universe", return_value=["AAPL"]),
        patch("sma.agents.__main__.writer_lock", _make_writer_lock_factory(lock_path)),
    ):
        result = CliRunner().invoke(
            cli,
            [
                "run", "--asof-date", asof.isoformat(), "--force-full",
                "--db", str(tmp_path / "test.duckdb"), "--config", str(config_path),
                "--universe", str(universe_path),
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    sentinel = read_sentinel(label="com.sma.agents.daily", asof=asof)
    assert sentinel["tickers_processed"] == 1
    assert sentinel["tickers_cached_fallback"] == 0
    assert sentinel["quality"]["passed"] is True


def test_run_no_trigger_tickers_counted(tmp_path, monkeypatch):
    """tickers_skipped_no_trigger is the gap between universe and triggered tickers."""
    lock_path = tmp_path / ".sma-writer.lock"
    sentinel_dir = tmp_path / "sentinels"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", lock_path)

    # 2026-04-29 is a Wednesday (Mon-Thu path, triggers used)
    asof = date(2026, 4, 29)
    db_path = tmp_path / "test.duckdb"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text("tickers: [AAPL, MSFT, GOOGL]\n")

    fake_pipeline = MagicMock()
    fake_pipeline.run.return_value = MagicMock(conviction="bullish", score=0.6)

    runner = CliRunner()

    with (
        patch("sma.agents.__main__.load_settings", return_value=_fake_settings()),
        patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline),
        patch("sma.agents.__main__._build_context", return_value=MagicMock()),
        patch(
            "sma.agents.__main__.load_universe",
            return_value=["AAPL", "MSFT", "GOOGL"],
        ),
        # Only AAPL triggered; MSFT + GOOGL are skipped
        patch(
            "sma.agents.__main__.tickers_needing_refresh",
            return_value=["AAPL"],
        ),
        patch(
            "sma.agents.__main__.writer_lock",
            _make_writer_lock_factory(lock_path),
        ),
    ):
        result = runner.invoke(
            cli,
            [
                "run",
                "--asof-date",
                asof.isoformat(),
                "--db",
                str(db_path),
                "--config",
                str(config_path),
                "--universe",
                str(universe_path),
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, f"agents run CLI failed:\n{result.output}"

    sentinel = read_sentinel(label="com.sma.agents.daily", asof=asof)
    assert sentinel is not None
    assert sentinel["tickers_processed"] == 1
    # universe has 3; only 1 triggered; 2 skipped due to no trigger
    assert sentinel["tickers_skipped_no_trigger"] == 2
    assert sentinel["force_full"] is False


def test_prefill_acquires_lock_per_ticker(tmp_path, monkeypatch):
    """prefill acquires writer_lock for each ticker-day write (option b)."""
    lock_path = tmp_path / ".sma-writer.lock"
    sentinel_dir = tmp_path / "sentinels"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", lock_path)

    db_path = tmp_path / "test.duckdb"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text("tickers: [AAPL, MSFT]\n")

    fake_pipeline = MagicMock()
    fake_pipeline.run.return_value = MagicMock(conviction="bullish", score=0.6)

    lock_acquire_count = []

    real_wl_factory = _make_writer_lock_factory(lock_path)

    @contextmanager
    def counting_writer_lock(*, label: str, **kwargs):
        lock_acquire_count.append(label)
        with real_wl_factory(label=label, **kwargs):
            yield

    runner = CliRunner()

    with (
        patch("sma.agents.__main__.load_settings", return_value=_fake_settings()),
        patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline),
        patch("sma.agents.__main__._build_context", return_value=MagicMock()),
        patch(
            "sma.agents.__main__.load_universe",
            return_value=["AAPL", "MSFT"],
        ),
        patch(
            "sma.agents.__main__.writer_lock",
            counting_writer_lock,
        ),
    ):
        # 2025-07-04 is a Friday: full universe (2 ticker-days)
        result = runner.invoke(
            cli,
            [
                "prefill",
                "--start",
                "2025-07-04",
                "--end",
                "2025-07-04",
                "--max-cost-usd",
                "10.0",
                "--db",
                str(db_path),
                "--config",
                str(config_path),
                "--universe",
                str(universe_path),
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, f"prefill CLI failed:\n{result.output}"
    # Per-ticker option b: writer_lock acquired at least once per ticker-day.
    # There may also be a brief setup lock (allocate_run_id) and a final cost
    # query lock, so the count is >= 2 (2 ticker-days). All labels are "prefill".
    assert len(lock_acquire_count) >= 2, (
        f"expected >= 2 writer_lock calls (per-ticker option b); got {len(lock_acquire_count)}"
    )
    assert all(label == "prefill" for label in lock_acquire_count)


def test_run_majority_failures_block_sentinel(tmp_path, monkeypatch):
    """audit (MED): a run where MOST tickers errored still wrote passed=True
    (_all_failed only fired at processed==0). A majority-failed run is a
    systemic problem (LLM flaking, quota, code bug) and must not read as a
    healthy advisory; decide degrades gracefully either way (agents is an
    advisory dep)."""
    lock_path = tmp_path / ".sma-writer.lock"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", lock_path)

    asof = date(2026, 4, 30)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text("tickers: [AAPL, MSFT, GOOG]\n")

    fake_pipeline = MagicMock()
    fake_pipeline.run.side_effect = [
        MagicMock(conviction="bullish", score=0.6),  # AAPL ok
        RuntimeError("boom"),                        # MSFT fails
        RuntimeError("boom"),                        # GOOG fails
    ]

    with (
        patch("sma.agents.__main__.load_settings", return_value=_fake_settings()),
        patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline),
        patch("sma.agents.__main__._build_context", return_value=MagicMock()),
        patch("sma.agents.__main__.load_universe", return_value=["AAPL", "MSFT", "GOOG"]),
        patch("sma.agents.__main__.writer_lock", _make_writer_lock_factory(lock_path)),
    ):
        result = CliRunner().invoke(
            cli,
            [
                "run", "--asof-date", asof.isoformat(), "--force-full",
                "--db", str(tmp_path / "test.duckdb"), "--config", str(config_path),
                "--universe", str(universe_path),
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    sentinel = read_sentinel(label="com.sma.agents.daily", asof=asof)
    assert sentinel["tickers_processed"] == 1
    assert sentinel["tickers_failed"] == 2
    assert sentinel["quality"]["passed"] is False
    assert "majority_tickers_failed" in sentinel["quality"]["blocking_failures"]


def test_run_stops_starting_tickers_past_deadline(tmp_path, monkeypatch):
    """2026-06-12 first wide-universe night: 81 new tickers had no theses, so
    agents processed ~everything at ~2.5min/ticker and held the WRITER LOCK
    for hours — decide died on lock timeout at 20:00 (manual kill+kick saved
    the night). Agents is advisory: past the cutoff it must stop STARTING
    tickers, count them as deadline-skipped, and release the lock."""
    from datetime import datetime as _dt

    lock_path = tmp_path / ".sma-writer.lock"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", lock_path)

    asof = date(2026, 4, 30)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text("tickers: [AAPL, MSFT, GOOG]\n")

    fake_pipeline = MagicMock()
    fake_pipeline.run.return_value = MagicMock(conviction="bullish", score=0.6)
    late = _dt(2026, 4, 30, 20, 30)  # SAME day as asof, past the 19:58 cutoff

    with (
        patch("sma.agents.__main__.load_settings", return_value=_fake_settings()),
        patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline),
        patch("sma.agents.__main__._build_context", return_value=MagicMock()),
        patch("sma.agents.__main__.load_universe", return_value=["AAPL", "MSFT", "GOOG"]),
        patch("sma.agents.__main__.writer_lock", _make_writer_lock_factory(lock_path)),
        patch("sma.agents.__main__._now_et", return_value=late),
    ):
        result = CliRunner().invoke(
            cli,
            ["run", "--asof-date", asof.isoformat(), "--force-full",
             "--db", str(tmp_path / "test.duckdb"), "--config", str(config_path),
             "--universe", str(universe_path)],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    assert fake_pipeline.run.call_count == 0, "no ticker may START past the cutoff"
    sentinel = read_sentinel(label="com.sma.agents.daily", asof=asof)
    assert sentinel["tickers_skipped_deadline"] == 3
    assert sentinel["quality"]["passed"] is True  # advisory partial is honest+ok
