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
