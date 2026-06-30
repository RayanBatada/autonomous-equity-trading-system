"""Integration tests for the `prefill` subcommand on the agents CLI."""

import threading
import time
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from sma.agents.__main__ import cli


def test_prefill_help_shows_options():
    runner = CliRunner()
    result = runner.invoke(cli, ["prefill", "--help"])
    assert result.exit_code == 0
    assert "--start" in result.output
    assert "--end" in result.output
    assert "--dry-run" in result.output
    assert "--max-cost-usd" in result.output
    assert "--daily-budget-usd" in result.output
    assert "--max-workers" in result.output


def test_prefill_dry_run_estimates_count_and_cost(tmp_path):
    """--dry-run prints estimate and exits 0 without calling Anthropic."""
    runner = CliRunner()
    fake_pipeline = MagicMock()
    with patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline), \
         patch("sma.agents.__main__.load_universe", return_value=["AAPL", "MSFT"]):
        result = runner.invoke(cli, [
            "prefill",
            "--start", "2025-07-04",  # Friday
            "--end", "2025-07-04",
            "--dry-run",
            "--db", str(tmp_path / "smoke.duckdb"),
        ])
    assert result.exit_code == 0
    # Friday → both tickers refreshed: 2 ticker-days
    assert "2 ticker-days" in result.output or "2" in result.output
    assert "$" in result.output  # cost estimate printed
    # Pipeline never called
    assert fake_pipeline.run.call_count == 0


def test_prefill_skips_weekends(tmp_path):
    """A Saturday → 0 ticker-days enqueued."""
    runner = CliRunner()
    fake_pipeline = MagicMock()
    with patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline), \
         patch("sma.agents.__main__.load_universe", return_value=["AAPL"]):
        result = runner.invoke(cli, [
            "prefill",
            "--start", "2025-07-05",  # Saturday
            "--end", "2025-07-06",  # Sunday
            "--dry-run",
            "--db", str(tmp_path / "smoke.duckdb"),
        ])
    assert result.exit_code == 0
    assert "0 ticker-days" in result.output


def test_prefill_friday_enqueues_full_universe(tmp_path):
    """Friday-only date range enqueues all universe tickers."""
    runner = CliRunner()
    fake_pipeline = MagicMock()
    with patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline), \
         patch("sma.agents.__main__.load_universe", return_value=["AAPL", "MSFT", "GOOGL"]):
        result = runner.invoke(cli, [
            "prefill",
            "--start", "2025-07-04",  # Friday
            "--end", "2025-07-04",
            "--dry-run",
            "--db", str(tmp_path / "smoke.duckdb"),
        ])
    assert result.exit_code == 0
    assert "3 ticker-days" in result.output


def test_prefill_thursday_uses_triggers(tmp_path):
    """Mon-Thu uses tickers_needing_refresh; with empty triggers, 0 ticker-days."""
    runner = CliRunner()
    fake_pipeline = MagicMock()
    with patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline), \
         patch("sma.agents.__main__.load_universe", return_value=["AAPL", "MSFT"]), \
         patch("sma.agents.__main__.tickers_needing_refresh", return_value=[]):
        result = runner.invoke(cli, [
            "prefill",
            "--start", "2025-07-03",  # Thursday
            "--end", "2025-07-03",
            "--dry-run",
            "--db", str(tmp_path / "smoke.duckdb"),
        ])
    assert result.exit_code == 0
    assert "0 ticker-days" in result.output


def test_prefill_thursday_with_triggered_ticker(tmp_path):
    """If trigger module returns 1 ticker for Thu, 1 ticker-day enqueued."""
    runner = CliRunner()
    fake_pipeline = MagicMock()
    with patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline), \
         patch("sma.agents.__main__.load_universe", return_value=["AAPL", "MSFT"]), \
         patch("sma.agents.__main__.tickers_needing_refresh", return_value=["AAPL"]):
        result = runner.invoke(cli, [
            "prefill",
            "--start", "2025-07-03",
            "--end", "2025-07-03",
            "--dry-run",
            "--db", str(tmp_path / "smoke.duckdb"),
        ])
    assert result.exit_code == 0
    assert "1 ticker-days" in result.output


def test_prefill_aborts_when_estimate_exceeds_max_cost(tmp_path):
    """If estimated cost exceeds --max-cost-usd, abort with exit code != 0."""
    runner = CliRunner()
    # Big universe + many days = high estimate
    big_universe = [f"T{i}" for i in range(100)]  # 100 tickers
    with patch("sma.agents.__main__._build_pipeline", return_value=MagicMock()), \
         patch("sma.agents.__main__.load_universe", return_value=big_universe):
        result = runner.invoke(cli, [
            "prefill",
            "--start", "2025-07-04",  # Friday
            "--end", "2025-07-25",  # 4 Fridays = 400 ticker-days * $0.005 = $2
            "--max-cost-usd", "0.50",  # below the $2 estimate
            "--dry-run",
            "--db", str(tmp_path / "smoke.duckdb"),
        ])
    assert result.exit_code != 0
    assert "exceeds" in result.output.lower() or "max" in result.output.lower()


def test_prefill_real_run_calls_pipeline_per_ticker_day(tmp_path):
    """Without --dry-run, pipeline.run is called for each ticker-day in queue."""
    runner = CliRunner()
    fake_pipeline = MagicMock()
    fake_pipeline.run.return_value = MagicMock(conviction="bullish")

    with patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline), \
         patch("sma.agents.__main__._build_context",
               return_value=MagicMock()), \
         patch("sma.agents.__main__.load_universe", return_value=["AAPL", "MSFT"]):
        result = runner.invoke(cli, [
            "prefill",
            "--start", "2025-07-04",  # Friday
            "--end", "2025-07-04",
            "--max-cost-usd", "10.0",
            "--db", str(tmp_path / "smoke.duckdb"),
        ])

    assert result.exit_code == 0, result.output
    assert fake_pipeline.run.call_count == 2  # 2 universe tickers on a Friday


def test_prefill_daily_budget_override_propagates(tmp_path):
    """--daily-budget-usd, when present, overrides settings.agents.daily_budget_usd
    in the cost tracker. Without override the smoke run capped at $0.50; the
    flag exists so a one-off prefill can raise the ceiling without editing
    config.yaml.
    """
    runner = CliRunner()
    fake_pipeline = MagicMock()
    fake_pipeline.run.return_value = MagicMock(conviction="bullish")
    captured_budgets: list[float] = []

    def capture_pipeline(settings, store):
        captured_budgets.append(settings.agents.daily_budget_usd)
        return fake_pipeline

    with patch("sma.agents.__main__._build_pipeline", side_effect=capture_pipeline), \
         patch("sma.agents.__main__._build_context", return_value=MagicMock()), \
         patch("sma.agents.__main__.load_universe", return_value=["AAPL"]):
        result = runner.invoke(cli, [
            "prefill",
            "--start", "2025-07-04",
            "--end", "2025-07-04",
            "--max-cost-usd", "100.0",
            "--daily-budget-usd", "50.0",
            "--db", str(tmp_path / "smoke.duckdb"),
        ])

    assert result.exit_code == 0, result.output
    assert captured_budgets == [50.0], (
        f"expected daily_budget_usd=50.0 to propagate; got {captured_budgets}"
    )


def test_prefill_daily_budget_default_uses_config(tmp_path):
    """Without --daily-budget-usd flag, the config value (typically $0.50) wins."""
    runner = CliRunner()
    fake_pipeline = MagicMock()
    fake_pipeline.run.return_value = MagicMock(conviction="bullish")
    captured_budgets: list[float] = []

    def capture_pipeline(settings, store):
        captured_budgets.append(settings.agents.daily_budget_usd)
        return fake_pipeline

    with patch("sma.agents.__main__._build_pipeline", side_effect=capture_pipeline), \
         patch("sma.agents.__main__._build_context", return_value=MagicMock()), \
         patch("sma.agents.__main__.load_universe", return_value=["AAPL"]):
        result = runner.invoke(cli, [
            "prefill",
            "--start", "2025-07-04",
            "--end", "2025-07-04",
            "--max-cost-usd", "100.0",
            "--db", str(tmp_path / "smoke.duckdb"),
        ])

    assert result.exit_code == 0, result.output
    # Config default in this repo is 0.50; we don't hardcode the exact value
    # in the test (config may evolve), just that no override was applied.
    assert len(captured_budgets) == 1
    assert captured_budgets[0] != 50.0  # not the override-test sentinel
    assert captured_budgets[0] > 0  # but is a real budget


def test_prefill_cost_estimate_constant_matches_observed_rate():
    """The PREFILL_USD_PER_TICKER_DAY constant should track real cost.
    Smoke A on 2026-04-27 measured $0.011/thesis end-to-end (3 LLM calls
    each); the constant was 0.005 (under-estimate). Bump to 0.012 so dry-run
    estimates don't lull users into expecting half the real cost.
    """
    from sma.agents.__main__ import PREFILL_USD_PER_TICKER_DAY

    assert 0.010 <= PREFILL_USD_PER_TICKER_DAY <= 0.020, (
        f"PREFILL_USD_PER_TICKER_DAY={PREFILL_USD_PER_TICKER_DAY} outside "
        "the empirically observed range; update if cost-per-thesis changes."
    )


def test_prefill_max_workers_runs_concurrently(tmp_path):
    """With --max-workers 5, pipeline.run() executes from multiple threads.
    Verifies (a) all ticker-days complete, (b) at least 2 distinct threads
    actually ran ticker-days (proving the executor is real, not collapsed
    to sequential).
    """
    runner = CliRunner()
    fake_pipeline = MagicMock()
    thread_ids: set[int] = set()
    lock = threading.Lock()

    def slow_run(ctx, run_id):
        # Sleep just long enough that 5 workers actually overlap.
        with lock:
            thread_ids.add(threading.get_ident())
        time.sleep(0.05)
        return MagicMock(conviction="bullish")

    fake_pipeline.run.side_effect = slow_run

    universe = [f"T{i}" for i in range(20)]  # 20 tickers
    with patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline), \
         patch("sma.agents.__main__._build_context", return_value=MagicMock()), \
         patch("sma.agents.__main__.load_universe", return_value=universe):
        result = runner.invoke(cli, [
            "prefill",
            "--start", "2025-07-04",  # Friday → 20 ticker-days
            "--end", "2025-07-04",
            "--max-cost-usd", "10.0",
            "--max-workers", "5",
            "--db", str(tmp_path / "smoke.duckdb"),
        ])

    assert result.exit_code == 0, result.output
    assert fake_pipeline.run.call_count == 20
    assert len(thread_ids) >= 2, (
        f"expected concurrent execution; got {len(thread_ids)} thread(s)"
    )


def test_prefill_max_workers_default_is_one(tmp_path):
    """Without --max-workers, run sequentially in a single thread (back-compat)."""
    runner = CliRunner()
    fake_pipeline = MagicMock()
    thread_ids: set[int] = set()
    lock = threading.Lock()

    def capture_thread(ctx, run_id):
        with lock:
            thread_ids.add(threading.get_ident())
        return MagicMock(conviction="bullish")

    fake_pipeline.run.side_effect = capture_thread

    with patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline), \
         patch("sma.agents.__main__._build_context", return_value=MagicMock()), \
         patch("sma.agents.__main__.load_universe", return_value=["A", "B", "C"]):
        result = runner.invoke(cli, [
            "prefill",
            "--start", "2025-07-04",
            "--end", "2025-07-04",
            "--max-cost-usd", "10.0",
            "--db", str(tmp_path / "smoke.duckdb"),
        ])

    assert result.exit_code == 0, result.output
    assert len(thread_ids) == 1, (
        f"sequential default should use exactly 1 thread; saw {len(thread_ids)}"
    )


def test_prefill_max_workers_propagates_per_task_exception(tmp_path):
    """One ticker-day raising must not abort other workers; failure is logged
    per-task and the run continues."""
    runner = CliRunner()
    fake_pipeline = MagicMock()

    def maybe_fail(ctx, run_id):
        # ctx is a MagicMock per-call; we use call count to force failure
        # on a specific call without depending on ctx contents
        n = fake_pipeline.run.call_count
        if n == 2:
            raise RuntimeError("synthetic LLM failure")
        return MagicMock(conviction="bullish")

    fake_pipeline.run.side_effect = maybe_fail

    universe = [f"T{i}" for i in range(5)]
    with patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline), \
         patch("sma.agents.__main__._build_context", return_value=MagicMock()), \
         patch("sma.agents.__main__.load_universe", return_value=universe):
        result = runner.invoke(cli, [
            "prefill",
            "--start", "2025-07-04",
            "--end", "2025-07-04",
            "--max-cost-usd", "10.0",
            "--max-workers", "3",
            "--db", str(tmp_path / "smoke.duckdb"),
        ])

    assert result.exit_code == 0, result.output
    # All 5 ticker-days attempted; one raised, four returned.
    assert fake_pipeline.run.call_count == 5


def test_prefill_skips_already_persisted_theses(tmp_path):
    """If a thesis for (ticker, asof) already exists, don't re-run the pipeline for it."""
    runner = CliRunner()
    db_path = tmp_path / "smoke.duckdb"
    fake_pipeline = MagicMock()
    fake_pipeline.run.return_value = MagicMock(conviction="bullish")

    # Pre-seed a thesis for AAPL on 2025-07-04
    from sma.ingest.store import Store
    store = Store(path=str(db_path)).connect()
    store.conn.execute(
        """
        INSERT INTO theses (ticker, asof_date, run_id, news_summary, key_developments,
            notable_filings, bull_case, bear_case, asymmetric_risks, catalyst_window,
            conviction, score, flags, action_hint, reasoning)
        VALUES ('AAPL', DATE '2025-07-04', 0, '', '[]', '[]', '', '', '[]', 'far',
                'neutral', 0.0, '[]', 'hold', '')
        """
    )
    store.close()

    with patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline), \
         patch("sma.agents.__main__._build_context", return_value=MagicMock()), \
         patch("sma.agents.__main__.load_universe", return_value=["AAPL", "MSFT"]):
        result = runner.invoke(cli, [
            "prefill",
            "--start", "2025-07-04",
            "--end", "2025-07-04",
            "--max-cost-usd", "10.0",
            "--db", str(db_path),
        ])

    assert result.exit_code == 0
    # AAPL skipped (already persisted), MSFT runs
    assert fake_pipeline.run.call_count == 1
