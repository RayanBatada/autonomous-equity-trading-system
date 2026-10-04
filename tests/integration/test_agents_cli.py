"""Integration tests for the Phase 4 LLM agent CLI (`python -m sma.agents`)."""

from datetime import date, datetime
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from sma.agents.__main__ import _build_context, cli
from sma.ingest.store import Store


def test_cli_help_shows_run_subcommand():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output.lower()


def test_run_help_lists_options():
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--asof-date" in result.output
    assert "--force-full" in result.output


def test_run_friday_processes_full_universe(tmp_path, monkeypatch):
    """On a Friday asof, the pipeline runs against every universe ticker."""
    runner = CliRunner()
    fake_pipeline = MagicMock()
    fake_pipeline.run.return_value = MagicMock(conviction="bullish", score=0.5)

    with (
        patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline),
        patch("sma.agents.__main__._build_context", return_value=MagicMock()),
        patch("sma.agents.__main__.load_universe", return_value=["AAPL", "MSFT", "GOOGL"]),
    ):
        # 2026-04-24 is a Friday
        result = runner.invoke(
            cli, ["run", "--asof-date", "2026-04-24", "--db", str(tmp_path / "t.duckdb")]
        )

    assert result.exit_code == 0, result.output
    # Pipeline.run called for each of the 3 tickers
    assert fake_pipeline.run.call_count == 3
    assert "Friday full refresh" in result.output


def test_run_thursday_processes_only_triggered_tickers(tmp_path, monkeypatch):
    """On a Thu asof, only tickers from tickers_needing_refresh are processed."""
    runner = CliRunner()
    fake_pipeline = MagicMock()
    fake_pipeline.run.return_value = MagicMock()

    with (
        patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline),
        patch("sma.agents.__main__._build_context", return_value=MagicMock()),
        patch("sma.agents.__main__.load_universe", return_value=["AAPL", "MSFT", "GOOGL"]),
        patch("sma.agents.__main__.tickers_needing_refresh", return_value=["AAPL"]),
    ):
        # 2026-04-23 is a Thursday
        result = runner.invoke(
            cli, ["run", "--asof-date", "2026-04-23", "--db", str(tmp_path / "t.duckdb")]
        )

    assert result.exit_code == 0, result.output
    # Only AAPL processed
    assert fake_pipeline.run.call_count == 1
    assert "triggered refresh" in result.output.lower() or "1 tickers" in result.output


def test_run_force_full_overrides_weekday_logic(tmp_path, monkeypatch):
    """--force-full on a Tuesday processes the full universe anyway."""
    runner = CliRunner()
    fake_pipeline = MagicMock()
    fake_pipeline.run.return_value = MagicMock()

    with (
        patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline),
        patch("sma.agents.__main__._build_context", return_value=MagicMock()),
        patch("sma.agents.__main__.load_universe", return_value=["AAPL", "MSFT"]),
    ):
        # 2026-04-21 is a Tuesday
        result = runner.invoke(
            cli,
            [
                "run", "--asof-date", "2026-04-21", "--force-full",
                "--db", str(tmp_path / "t.duckdb"),
            ],
        )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.run.call_count == 2


def test_run_swallows_per_ticker_pipeline_failures(tmp_path, monkeypatch):
    """If pipeline.run raises for one ticker, others continue."""
    runner = CliRunner()
    fake_pipeline = MagicMock()
    fake_pipeline.run.side_effect = [
        RuntimeError("AAPL crashed"),
        MagicMock(conviction="bullish", score=0.5),
    ]

    with (
        patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline),
        patch("sma.agents.__main__._build_context", return_value=MagicMock()),
        patch("sma.agents.__main__.load_universe", return_value=["AAPL", "MSFT"]),
    ):
        result = runner.invoke(
            cli, ["run", "--asof-date", "2026-04-24", "--db", str(tmp_path / "t.duckdb")]
        )

    assert result.exit_code == 0  # the runner should NOT die from one ticker's failure
    assert fake_pipeline.run.call_count == 2


def test_run_fails_fast_when_anthropic_key_missing(tmp_path, monkeypatch):
    """No ANTHROPIC_API_KEY -> fail fast with a clear error."""
    runner = CliRunner()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from click.exceptions import ClickException

    err = ClickException("ANTHROPIC_API_KEY not set in .env; Phase 4 requires it")
    with patch("sma.agents.__main__._build_pipeline", side_effect=err):
        result = runner.invoke(
            cli, ["run", "--asof-date", "2026-04-24", "--db", str(tmp_path / "t.duckdb")]
        )

    assert result.exit_code != 0
    assert "ANTHROPIC_API_KEY" in result.output


def test_thesis_help_shows_options():
    runner = CliRunner()
    result = runner.invoke(cli, ["thesis", "--help"])
    assert result.exit_code == 0
    assert "--ticker" in result.output
    assert "--asof" in result.output


def test_thesis_prints_all_three_agent_outputs(tmp_path):
    """Pipeline runs, thesis row persisted, output prints researcher/analyst/strategist sections."""
    runner = CliRunner()
    db_path = tmp_path / "smoke.duckdb"

    captured = {}

    def fake_build_pipeline(settings, store):
        # Capture the CLI's store so the fake pipeline can seed the theses row
        # using the same connection the CLI later reads from.
        captured["store"] = store
        fake = MagicMock()

        def fake_run(ctx, run_id):
            store.conn.execute(
                """
                INSERT INTO theses (ticker, asof_date, run_id,
                    news_summary, key_developments, notable_filings,
                    bull_case, bear_case, asymmetric_risks, catalyst_window,
                    conviction, score, flags, action_hint, reasoning)
                VALUES (?, ?, ?, 'apple had a strong q', '["beat eps","raised guidance"]', '[]',
                        'demand strong', 'supply chain risk', '[]', 'far',
                        'bullish', 0.6, '["earnings_beat"]', 'enter', 'beat + raised guide')
                """,
                [ctx.ticker, ctx.asof_date, run_id],
            )
            return MagicMock(conviction="bullish")

        fake.run.side_effect = fake_run
        return fake

    with (
        patch("sma.agents.__main__._build_pipeline", side_effect=fake_build_pipeline),
        patch(
            "sma.agents.__main__._build_context",
            return_value=MagicMock(
                ticker="AAPL", asof_date=date(2026, 4, 25), news_rows=[], filing_rows=[]
            ),
        ),
    ):
        result = runner.invoke(
            cli,
            [
                "thesis",
                "--ticker",
                "AAPL",
                "--asof",
                "2026-04-25",
                "--db",
                str(db_path),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Researcher" in result.output
    assert "apple had a strong q" in result.output
    assert "Analyst" in result.output
    assert "demand strong" in result.output
    assert "Strategist" in result.output
    assert "bullish" in result.output
    assert "earnings_beat" in result.output


# ---------------------------------------------------------------------------
# Fix #4 regression: thesis CLI must call writer_lock before opening Store.
# We spy on sma.agents.__main__.writer_lock to confirm it is invoked with
# label="agents-thesis". If the wrapper were absent, the spy would record
# zero calls with that label.
# ---------------------------------------------------------------------------


def test_thesis_acquires_writer_lock_itself(tmp_path):
    """thesis CLI must call writer_lock(label='agents-thesis') before opening Store."""
    import sma.agents.__main__ as agents_main
    from sma.locks import writer_lock as real_writer_lock

    lock_calls: list[str] = []

    from contextlib import contextmanager

    @contextmanager
    def spy_writer_lock(*, label, **kwargs):
        lock_calls.append(label)
        with real_writer_lock(label=label, **kwargs):
            yield

    db_path = tmp_path / "smoke.duckdb"
    runner = CliRunner()

    def fake_build_pipeline(settings, store):
        fake = MagicMock()

        def fake_run(ctx, run_id):
            store.conn.execute(
                """
                INSERT INTO theses (ticker, asof_date, run_id,
                    news_summary, key_developments, notable_filings,
                    bull_case, bear_case, asymmetric_risks, catalyst_window,
                    conviction, score, flags, action_hint, reasoning)
                VALUES (?, ?, ?, 'lock test', '[]', '[]',
                        'bull', 'bear', '[]', 'far',
                        'neutral', 0.5, '[]', 'hold', 'tested lock')
                """,
                [ctx.ticker, ctx.asof_date, run_id],
            )
            return MagicMock(conviction="neutral")

        fake.run.side_effect = fake_run
        return fake

    with (
        patch.object(agents_main, "writer_lock", spy_writer_lock),
        patch("sma.agents.__main__._build_pipeline", side_effect=fake_build_pipeline),
        patch(
            "sma.agents.__main__._build_context",
            return_value=MagicMock(
                ticker="AAPL", asof_date=date(2026, 4, 25), news_rows=[], filing_rows=[]
            ),
        ),
    ):
        result = runner.invoke(
            cli,
            [
                "thesis",
                "--ticker",
                "AAPL",
                "--asof",
                "2026-04-25",
                "--db",
                str(db_path),
            ],
        )

    assert result.exit_code == 0, f"thesis CLI failed:\n{result.output}"
    assert "agents-thesis" in lock_calls, (
        "thesis CLI did not call writer_lock(label='agents-thesis'); "
        f"observed lock labels: {lock_calls}"
    )


# ---------------------------------------------------------------------------
# Look-ahead leak regression: _build_context must NEVER return news or filings
# published after asof. Codex adversarial review (2026-04-28) caught this as a
# CRITICAL leak in the historical prefill path; the fix added an upper-bound
# predicate to both queries.
# ---------------------------------------------------------------------------


def test_build_context_excludes_news_published_after_asof(tmp_path):
    """News rows with published_at > asof must NOT appear in the context."""
    db_path = tmp_path / "leak.duckdb"
    store = Store(path=str(db_path)).connect()
    rid = store.allocate_run_id()
    asof = date(2025, 7, 4)

    def _row(dt: datetime, headline: str, hash_: str, src: str = "finnhub") -> tuple:
        return ("AAPL", dt, src, headline, f"u-{hash_}", None, hash_, rid)

    rows = [
        # PAST: should appear
        _row(datetime(2025, 7, 1, 9, 0), "past-1", "h-past-1"),
        _row(datetime(2025, 7, 4, 14, 0), "asof-day", "h-asof", src="alpaca_news"),
        # FUTURE: must be filtered
        _row(datetime(2025, 7, 5, 9, 0), "next-day", "h-future-1"),
        _row(datetime(2025, 12, 31, 9, 0), "5-months-future", "h-future-2"),
        _row(datetime(2026, 4, 28, 9, 0), "9-months-future", "h-future-3"),
    ]
    for r in rows:
        store.conn.execute("INSERT INTO news VALUES (?, ?, ?, ?, ?, ?, ?, ?)", list(r))

    ctx = _build_context(store, "AAPL", asof)
    headlines = {n.headline for n in ctx.news_rows}

    assert "past-1" in headlines, "past news should be present"
    assert "asof-day" in headlines, "same-day-as-asof news should be present"
    assert "next-day" not in headlines, "next-day news must be filtered (LEAK)"
    assert "5-months-future" not in headlines, "future news must be filtered (LEAK)"
    assert "9-months-future" not in headlines, "future news must be filtered (LEAK)"


def test_build_context_excludes_filings_filed_after_asof(tmp_path):
    """Filing rows with filed_at > asof must NOT appear in the context."""
    db_path = tmp_path / "leak.duckdb"
    store = Store(path=str(db_path)).connect()
    rid = store.allocate_run_id()
    asof = date(2025, 7, 4)

    rows = [
        # PAST: should appear
        (
            "AAPL",
            "10-Q",
            datetime(2025, 5, 1, 16, 0),
            "acc-past-1",
            "https://sec/past1",
            "summary past",
            rid,
        ),
        (
            "AAPL",
            "8-K",
            datetime(2025, 7, 4, 8, 0),
            "acc-asof",
            "https://sec/asof",
            "summary asof",
            rid,
        ),
        # FUTURE: must be filtered
        (
            "AAPL",
            "8-K",
            datetime(2025, 7, 5, 8, 0),
            "acc-future-1",
            "https://sec/f1",
            "future summary 1",
            rid,
        ),
        (
            "AAPL",
            "10-K",
            datetime(2026, 1, 1, 8, 0),
            "acc-future-2",
            "https://sec/f2",
            "future summary 2",
            rid,
        ),
    ]
    for r in rows:
        store.conn.execute("INSERT INTO filings VALUES (?, ?, ?, ?, ?, ?, ?)", list(r))

    ctx = _build_context(store, "AAPL", asof)
    accessions = {f.url.rsplit("/", 1)[-1] for f in ctx.filing_rows}

    assert any("past1" in a for a in accessions), "past filing should be present"
    assert any("asof" in a for a in accessions), "asof-day filing should be present"
    assert not any("f1" in a for a in accessions), "next-day filing must be filtered (LEAK)"
    assert not any("f2" in a for a in accessions), "future filing must be filtered (LEAK)"


def test_run_skips_tickers_with_existing_same_day_thesis(tmp_path, monkeypatch):
    """A same-day rerun must NOT re-spend on tickers that already have a thesis
    (2026-06-05 audit: double-spend + duplicate theses)."""
    db = tmp_path / "t.duckdb"
    seed = Store(path=str(db)).connect()
    seed.conn.execute(
        """
        INSERT INTO theses (ticker, asof_date, run_id, news_summary, key_developments,
            notable_filings, bull_case, bear_case, asymmetric_risks, catalyst_window,
            conviction, score, flags, action_hint, reasoning)
        VALUES ('AAPL', DATE '2026-04-24', 1, 'x', '[]', '[]', 'b', 'b', '[]', 'far',
                'neutral', 0.0, '[]', 'hold', 'existing')
        """
    )
    seed.close()

    fake_pipeline = MagicMock()
    fake_pipeline.run.return_value = MagicMock(conviction="bullish", score=0.5)
    with (
        patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline),
        patch("sma.agents.__main__._build_context", return_value=MagicMock()),
        patch("sma.agents.__main__.load_universe", return_value=["AAPL", "MSFT"]),
    ):
        result = CliRunner().invoke(
            cli, ["run", "--asof-date", "2026-04-24", "--db", str(db)]
        )

    assert result.exit_code == 0, result.output
    # AAPL already had a thesis → skipped; only MSFT is processed.
    assert fake_pipeline.run.call_count == 1
