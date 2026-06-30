"""HIGH 2 + HIGH 3: backfill-news and backfill-earnings acquire writer_lock.

Without the wrap, Store.connect(read_only=False) raises WriterLockNotHeld.
With the wrap, the CLI runs successfully.

Strategy: spy on writer_lock using a counting wrapper so we can assert it
was called with the correct label, without replacing the real implementation
(the real lock still runs, satisfying Store's assertion).
"""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml
from click.testing import CliRunner

import sma.ingest.__main__ as ingest_main
from sma.ingest.__main__ import cli
from sma.locks import writer_lock as real_writer_lock

# ---------------------------------------------------------------------------
# Helpers shared by both tests
# ---------------------------------------------------------------------------


def _seed_env(monkeypatch):
    for k in ("FINNHUB_API_KEY", "NEWSAPI_KEY", "ALPACA_API_KEY", "ALPACA_API_SECRET"):
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("EDGAR_USER_AGENT", "T (t@e.com)")


def _write_config(tmp_path: Path, sources_enabled=None) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
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
                "sources_enabled": sources_enabled or ["finnhub_news"],
            }
        )
    )
    return cfg


def _write_universe(tmp_path: Path) -> Path:
    p = tmp_path / "universe.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "universe": {
                    "refresh_policy": "manual",
                    "tickers": ["AAPL"],
                },
            }
        )
    )
    return p


def _make_lock_spy(lock_path: Path) -> tuple:
    """Return (spy_factory, calls_list).

    spy_factory is a drop-in for writer_lock that records each invocation's
    label and still acquires the real lock so Store.connect() is satisfied.
    """
    calls: list[str] = []

    @contextmanager
    def _spy(*, label: str, **kwargs):
        calls.append(label)
        with real_writer_lock(lock_path=lock_path, label=label, **kwargs):
            yield

    return _spy, calls


# ---------------------------------------------------------------------------
# HIGH 2: backfill-news acquires writer_lock
# ---------------------------------------------------------------------------


def test_backfill_news_acquires_writer_lock(tmp_path, monkeypatch):
    """backfill-news must wrap its Store.connect() inside writer_lock.

    The conftest autouse fixture holds the default lock. This test uses its
    own lock path and a counting spy to verify writer_lock is called with
    label='backfill-news'.
    """
    lock_path = tmp_path / ".sma-writer.lock"
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", lock_path)

    _seed_env(monkeypatch)
    cfg = _write_config(tmp_path, sources_enabled=["finnhub_news"])
    uni = _write_universe(tmp_path)
    db = tmp_path / "sma.duckdb"

    spy, calls = _make_lock_spy(lock_path)

    # Stub _build_source with a proper fake news source that does real DB inserts.
    # Using MagicMock.fetch would cause DuckDB to choke on mock return types; use
    # the same FakeNewsSource pattern as the existing backfill_news integration tests.
    from sma.ingest.sources.base import IngestResult

    class _FakeNewsSource:
        def __init__(self, name, lookback_days=7):
            self.name = name

        def fetch(self, tickers, asof_date, store, run_id):
            return IngestResult(self.name, 0, "ok", None)

    def _fake_build(name, settings, finnhub_limiter=None, lookback_days_override=None):
        return _FakeNewsSource(name=name)

    with (
        patch.object(ingest_main, "writer_lock", spy),
        patch.object(ingest_main, "_build_source", _fake_build),
    ):
        result = CliRunner().invoke(
            cli,
            [
                "backfill-news",
                "--config",
                str(cfg),
                "--universe",
                str(uni),
                "--db",
                str(db),
                "--start",
                "2026-01-01",
                "--end",
                "2026-01-07",
                "--batch-days",
                "7",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "backfill-news" in calls, (
        f"writer_lock was not called with label='backfill-news'; calls={calls}"
    )


# ---------------------------------------------------------------------------
# HIGH 3: backfill-earnings acquires writer_lock
# ---------------------------------------------------------------------------


def test_backfill_earnings_acquires_writer_lock(tmp_path, monkeypatch):
    """backfill-earnings must wrap its Store.connect() inside writer_lock.

    Same spy pattern as the news test above.
    """
    lock_path = tmp_path / ".sma-writer.lock"
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", lock_path)

    _seed_env(monkeypatch)
    cfg = _write_config(tmp_path, sources_enabled=["yfinance"])
    uni = _write_universe(tmp_path)
    db = tmp_path / "sma.duckdb"

    spy, calls = _make_lock_spy(lock_path)

    fake_client = MagicMock()
    fake_client.company_earnings.return_value = []

    with (
        patch.object(ingest_main, "writer_lock", spy),
        patch(
            "sma.ingest.earnings_backfill.finnhub.Client",
            lambda api_key: fake_client,
        ),
    ):
        result = CliRunner().invoke(
            cli,
            [
                "backfill-earnings",
                "--config",
                str(cfg),
                "--universe",
                str(uni),
                "--db",
                str(db),
                "--quarters",
                "1",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "backfill-earnings" in calls, (
        f"writer_lock was not called with label='backfill-earnings'; calls={calls}"
    )
