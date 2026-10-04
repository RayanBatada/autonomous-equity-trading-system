"""live.sessions / live.execution / intraday config blocks (2026-09-26).

Every default is OFF or inert: a config.yaml with none of these blocks must
load, and nothing it produces may arm a session or change the open path."""

from datetime import time
from pathlib import Path

import pytest
import yaml

from sma.config import IntradayConfig, LiveExecution, LiveSessions, load_settings

_BASE = {
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
}


@pytest.fixture
def _env(monkeypatch):
    for k, v in {
        "FINNHUB_API_KEY": "x", "NEWSAPI_KEY": "x", "ALPACA_API_KEY": "x",
        "ALPACA_API_SECRET": "x", "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
        "EDGAR_USER_AGENT": "t (t@example.com)",
    }.items():
        monkeypatch.setenv(k, v)


def _load(tmp_path: Path, extra: dict):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({**_BASE, **extra}))
    return load_settings(config_path=p)


def test_defaults_are_off_when_blocks_absent(tmp_path, _env):
    s = _load(tmp_path, {})
    assert s.live.sessions.midday.enabled is False
    assert s.live.sessions.close.enabled is False
    assert s.live.sessions.midday.bounds() == (time(10, 30), time(11, 0))
    assert s.live.sessions.close.bounds() == (time(15, 40), time(15, 55))
    assert s.live.execution.limit_offset_bps == 10.0
    assert s.live.execution.fallback == "market"
    assert s.intraday.tickers[0] == "SPY"
    assert len(s.intraday.tickers) == 12
    assert s.intraday.include_held is True


def test_blocks_parse(tmp_path, _env):
    s = _load(tmp_path, {
        "live": {
            "sessions": {"midday": {"enabled": True, "window": "10:45-11:15"}},
            "execution": {"limit_offset_bps": 5, "fallback": "leave"},
        },
        "intraday": {"tickers": ["SPY"], "include_held": False},
    })
    assert s.live.sessions.midday.enabled is True
    assert s.live.sessions.midday.bounds() == (time(10, 45), time(11, 15))
    assert s.live.sessions.close.enabled is False
    assert s.live.execution.limit_offset_bps == 5
    assert s.live.execution.fallback == "leave"
    assert s.intraday.tickers == ["SPY"]


def test_bad_fallback_and_window_rejected():
    with pytest.raises(ValueError):
        LiveExecution(fallback="yolo")
    with pytest.raises(ValueError):
        LiveSessions(midday={"window": "11:00-10:30"})
    with pytest.raises(ValueError):
        IntradayConfig(tickers=[])
