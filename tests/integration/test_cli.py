from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from sma.ingest.__main__ import cli


def _write_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({
        "ingest": {
            "default_lookback_days": 1,
            "rate_limits": {
                "finnhub": {"requests_per_minute": 55},
                "newsapi": {"requests_per_day": 90},
                "edgar": {"requests_per_second": 9},
            },
            "retries": {"max": 3, "base_delay": 0.0, "jitter": 0.0},
            "circuit_breaker": {"failures_to_open": 5, "cooldown_minutes": 60},
        },
        "sources_enabled": ["yfinance"],
    }))
    return cfg


def _write_universe(tmp_path: Path) -> Path:
    p = tmp_path / "universe.yaml"
    p.write_text(yaml.safe_dump({
        "universe": {"refresh_policy": "manual", "tickers": ["AAPL"]},
    }))
    return p


def test_cli_run_with_no_sources_succeeds(tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch):
    cfg = _write_config(tmp_path)
    universe = _write_universe(tmp_path)
    db = tmp_path / "test.duckdb"

    monkeypatch.setenv("FINNHUB_API_KEY", "x")
    monkeypatch.setenv("NEWSAPI_KEY", "x")
    monkeypatch.setenv("ALPACA_API_KEY", "x")
    monkeypatch.setenv("ALPACA_API_SECRET", "x")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("EDGAR_USER_AGENT", "T (t@e.com)")

    runner = CliRunner()
    result = runner.invoke(cli, [
        "run",
        "--config", str(cfg),
        "--universe", str(universe),
        "--db", str(db),
        "--sources", "",
        "--asof-date", "2026-04-23",
        "--skip-quality",
    ])
    assert result.exit_code == 0, result.output


def test_cli_run_no_deadline_flag_disables_the_budget(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch):
    """2026-09-17: `--no-deadline` must reach ingest_run() as deadline=None,
    on TOP of the same-day-only exemption IngestRunner.run() now applies on
    its own (test_runner_retry.py). Uses --asof-date 2026-09-16 (a Wednesday
    com.sma.ingest.daily would normally get a real scheduled deadline for)
    so the flag is doing the work, not the date."""
    from sma.ingest import __main__ as ingest_main
    from sma.ingest.quality import QualityReport
    from sma.ingest.runner import IngestRunResult

    cfg = _write_config(tmp_path)
    universe = _write_universe(tmp_path)
    db = tmp_path / "test.duckdb"

    monkeypatch.setenv("FINNHUB_API_KEY", "x")
    monkeypatch.setenv("NEWSAPI_KEY", "x")
    monkeypatch.setenv("ALPACA_API_KEY", "x")
    monkeypatch.setenv("ALPACA_API_SECRET", "x")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("EDGAR_USER_AGENT", "T (t@e.com)")
    monkeypatch.chdir(tmp_path)  # keep logs/quality/*.txt out of the real repo

    captured: list[dict] = []

    def _fake_ingest_run(**kwargs):
        captured.append(kwargs)
        return IngestRunResult(
            run_id=1,
            quality_report=QualityReport(asof_date=kwargs["asof"], run_id=1, checks=[]),
            dead_frozen_flags=[],
        )

    monkeypatch.setattr(ingest_main, "ingest_run", _fake_ingest_run)

    runner = CliRunner()
    common_args = [
        "run",
        "--config", str(cfg),
        "--universe", str(universe),
        "--db", str(db),
        "--sources", "",
        "--asof-date", "2026-09-16",
    ]

    result = runner.invoke(cli, [*common_args, "--no-deadline"])
    assert result.exit_code == 0, result.output
    assert captured[-1]["deadline"] is None, "explicit --no-deadline must force deadline=None"

    result = runner.invoke(cli, common_args)
    assert result.exit_code == 0, result.output
    assert captured[-1]["deadline"] is not None, (
        "without the flag, a scheduled weekday still gets a real deadline "
        "(IngestRunner itself is what exempts non-today asofs)"
    )
