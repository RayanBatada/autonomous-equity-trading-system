from pathlib import Path

import pytest
import yaml

from sma.config import Settings, load_settings


def test_load_settings_reads_yaml_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "ingest": {
            "default_lookback_days": 100,
            "rate_limits": {
                "finnhub": {"requests_per_minute": 55},
                "newsapi": {"requests_per_day": 90},
                "edgar": {"requests_per_second": 9},
            },
            "retries": {"max": 3, "base_delay": 1.0, "jitter": 0.5},
            "circuit_breaker": {"failures_to_open": 5, "cooldown_minutes": 60},
        },
        "sources_enabled": ["yfinance", "finnhub_news"],
    }))

    monkeypatch.setenv("FINNHUB_API_KEY", "fake-fh")
    monkeypatch.setenv("NEWSAPI_KEY", "fake-news")
    monkeypatch.setenv("ALPACA_API_KEY", "fake-alp")
    monkeypatch.setenv("ALPACA_API_SECRET", "fake-alp-sec")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("EDGAR_USER_AGENT", "Test User (t@example.com)")

    s: Settings = load_settings(config_path=cfg_path)

    assert s.ingest.default_lookback_days == 100
    assert s.ingest.rate_limits.finnhub.requests_per_minute == 55
    assert s.sources_enabled == ["yfinance", "finnhub_news"]
    assert s.secrets.finnhub_api_key == "fake-fh"
    assert s.secrets.alpaca_base_url == "https://paper-api.alpaca.markets"


def test_load_settings_raises_when_required_secret_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "ingest": {
            "default_lookback_days": 100,
            "rate_limits": {
                "finnhub": {"requests_per_minute": 55},
                "newsapi": {"requests_per_day": 90},
                "edgar": {"requests_per_second": 9},
            },
            "retries": {"max": 3, "base_delay": 1.0, "jitter": 0.5},
            "circuit_breaker": {"failures_to_open": 5, "cooldown_minutes": 60},
        },
        "sources_enabled": ["yfinance"],
    }))
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
    monkeypatch.delenv("EDGAR_USER_AGENT", raising=False)
    # Isolate from any real .env file in the repo (pydantic-settings reads it
    # by default, which would defeat the purpose of this test once the user
    # has filled in their real .env).
    monkeypatch.chdir(tmp_path)

    with pytest.raises(Exception):
        load_settings(config_path=cfg_path)
