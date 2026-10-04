from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml
from click.testing import CliRunner

from sma.ingest.__main__ import cli


def _seed_env(monkeypatch):
    for k in ("FINNHUB_API_KEY", "NEWSAPI_KEY", "ALPACA_API_KEY",
              "ALPACA_API_SECRET"):
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("EDGAR_USER_AGENT", "T (t@e.com)")


def _config_yaml(tmp_path: Path, lookback: int) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({
        "ingest": {
            "default_lookback_days": lookback,
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
    return p


def _universe_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "universe.yaml"
    p.write_text(yaml.safe_dump({
        "universe": {"refresh_policy": "manual", "tickers": ["AAPL"]},
    }))
    return p


def _fake_yf():
    return pd.DataFrame(
        {"Open": [100.0], "High": [101.0], "Low": [99.0],
         "Close": [100.5], "Adj Close": [100.5], "Volume": [1_000_000]},
        index=pd.to_datetime(["2026-04-23"]),
    )


def test_first_run_uses_long_lookback(tmp_path, monkeypatch):
    _seed_env(monkeypatch)
    cfg = _config_yaml(tmp_path, lookback=1095)
    uni = _universe_yaml(tmp_path)
    db = tmp_path / "sma.duckdb"
    seen_lookback = {}
    real_init = __import__("sma.ingest.sources.yfinance_prices",
                           fromlist=["YFinancePricesSource"]).YFinancePricesSource

    class Spy(real_init):
        def __init__(self, lookback_days: int):
            seen_lookback["v"] = lookback_days
            super().__init__(lookback_days=lookback_days)

    with patch("sma.ingest.__main__.YFinancePricesSource", Spy), \
         patch("sma.ingest.sources.yfinance_prices.yf.download",
               return_value=_fake_yf()):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run", "--config", str(cfg), "--universe", str(uni),
            "--db", str(db), "--sources", "yfinance",
            "--asof-date", "2026-04-23", "--skip-quality",
        ])
    assert result.exit_code == 0, result.output
    assert seen_lookback["v"] == 1095


def test_subsequent_run_uses_45_day_lookback(tmp_path, monkeypatch):
    """Incremental nights use 45d (NOT 1d): the adj-drift self-heal needs a
    real overlap window — the 1d clamp silently defeated the source's split
    re-sync design (KLAC 10x for 3 weeks; review 2026-07-01)."""
    _seed_env(monkeypatch)
    cfg = _config_yaml(tmp_path, lookback=1095)
    uni = _universe_yaml(tmp_path)
    db = tmp_path / "sma.duckdb"
    real_init = __import__("sma.ingest.sources.yfinance_prices",
                           fromlist=["YFinancePricesSource"]).YFinancePricesSource

    seen_lookbacks = []

    class Spy(real_init):
        def __init__(self, lookback_days: int):
            seen_lookbacks.append(lookback_days)
            super().__init__(lookback_days=lookback_days)

    with patch("sma.ingest.__main__.YFinancePricesSource", Spy), \
         patch("sma.ingest.sources.yfinance_prices.yf.download",
               return_value=_fake_yf()):
        runner = CliRunner()
        for asof in ("2026-04-23", "2026-04-24"):
            result = runner.invoke(cli, [
                "run", "--config", str(cfg), "--universe", str(uni),
                "--db", str(db), "--sources", "yfinance",
                "--asof-date", asof, "--skip-quality",
            ])
            assert result.exit_code == 0, result.output

    assert seen_lookbacks[0] == 1095
    assert seen_lookbacks[1] == 45


def test_explicit_lookback_overrides_auto_detect(tmp_path, monkeypatch):
    _seed_env(monkeypatch)
    cfg = _config_yaml(tmp_path, lookback=1095)
    uni = _universe_yaml(tmp_path)
    db = tmp_path / "sma.duckdb"
    real_init = __import__("sma.ingest.sources.yfinance_prices",
                           fromlist=["YFinancePricesSource"]).YFinancePricesSource

    seen_lookbacks = []

    class Spy(real_init):
        def __init__(self, lookback_days: int):
            seen_lookbacks.append(lookback_days)
            super().__init__(lookback_days=lookback_days)

    with patch("sma.ingest.__main__.YFinancePricesSource", Spy), \
         patch("sma.ingest.sources.yfinance_prices.yf.download",
               return_value=_fake_yf()):
        runner = CliRunner()
        # First run: empty DB, but explicit --lookback-days 7 overrides.
        result = runner.invoke(cli, [
            "run", "--config", str(cfg), "--universe", str(uni),
            "--db", str(db), "--sources", "yfinance",
            "--asof-date", "2026-04-23", "--skip-quality",
            "--lookback-days", "7",
        ])
        assert result.exit_code == 0, result.output
        # Second run: DB now has data, explicit --lookback-days 7 still wins.
        result = runner.invoke(cli, [
            "run", "--config", str(cfg), "--universe", str(uni),
            "--db", str(db), "--sources", "yfinance",
            "--asof-date", "2026-04-24", "--skip-quality",
            "--lookback-days", "7",
        ])
        assert result.exit_code == 0, result.output

    assert seen_lookbacks == [7, 7]
    assert "explicit lookback (7d)" in result.output
