"""Integration tests for the `backfill-news` CLI subcommand.

The tests mock _build_source so we never hit a real news API. The fake
news source records the (asof_date, lookback_days, tickers) tuples it was
called with and inserts a deterministic fake article per chunk so we can
verify dedup behavior against the real DuckDB news table.
"""

from datetime import date as date_cls
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from sma.ingest.__main__ import cli
from sma.ingest.sources.base import IngestResult


def _seed_env(monkeypatch):
    for k in ("FINNHUB_API_KEY", "NEWSAPI_KEY", "ALPACA_API_KEY",
              "ALPACA_API_SECRET"):
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("EDGAR_USER_AGENT", "T (t@e.com)")


def _write_config(tmp_path: Path, sources_enabled=None) -> Path:
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
        "sources_enabled": sources_enabled
        if sources_enabled is not None
        else ["yfinance", "alpaca", "finnhub_news", "alpaca_news",
              "newsapi", "edgar"],
    }))
    return cfg


def _write_universe(tmp_path: Path, tickers=None) -> Path:
    p = tmp_path / "universe.yaml"
    p.write_text(yaml.safe_dump({
        "universe": {
            "refresh_policy": "manual",
            "tickers": tickers or ["AAPL", "MSFT", "GOOG"],
        },
    }))
    return p


class FakeNewsSource:
    """In-memory news source that records calls and inserts one fake row.

    `name` is set per-instance to whatever name was used to construct it,
    so tests can distinguish e.g. finnhub_news vs alpaca_news.
    """

    # Class-level registry so tests can inspect calls across instances.
    calls: list[dict] = []
    constructed: list[str] = []

    def __init__(self, name: str, lookback_days: int):
        self.name = name
        self.lookback_days = lookback_days
        FakeNewsSource.constructed.append(name)

    def fetch(self, tickers, asof_date, store, run_id):
        FakeNewsSource.calls.append({
            "name": self.name,
            "asof_date": asof_date,
            "lookback_days": self.lookback_days,
            "tickers": list(tickers),
        })
        # Insert one deterministic row per ticker so we can test dedup.
        rows = []
        for t in tickers:
            published_at = datetime.combine(asof_date, datetime.min.time())
            headline = f"fake-{self.name}-{asof_date.isoformat()}-{t}"
            url = f"https://example.test/{self.name}/{asof_date}/{t}"
            body = "fake body"
            h = f"hash-{self.name}-{asof_date.isoformat()}-{t}"
            rows.append((
                t, published_at, self.name, headline, url, body, h, run_id,
            ))
        if rows:
            store.conn.executemany(
                "INSERT OR REPLACE INTO news "
                "(ticker, published_at, source, headline, url, body_excerpt, "
                "hash, run_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return IngestResult(self.name, len(rows), "ok", None)


@pytest.fixture(autouse=True)
def _reset_fake():
    FakeNewsSource.calls = []
    FakeNewsSource.constructed = []
    yield
    FakeNewsSource.calls = []
    FakeNewsSource.constructed = []


def _patch_build_source(monkeypatch, news_only: bool = True):
    """Replace _build_source to return FakeNewsSource for known names.

    If news_only=True, raises if a non-news source is requested (helps
    test 3 verify that backfill never builds price/edgar/fundamentals
    sources). Otherwise returns FakeNewsSource for news names and a
    minimal stub for any other name.
    """
    news_names = {"finnhub_news", "alpaca_news", "newsapi"}

    def fake_build(name, settings, finnhub_limiter=None,
                   lookback_days_override=None):
        if name in news_names:
            return FakeNewsSource(
                name=name,
                lookback_days=(
                    lookback_days_override
                    if lookback_days_override is not None else 7
                ),
            )
        if news_only:
            raise AssertionError(
                f"backfill-news must not construct non-news source: {name}"
            )
        # Fallback stub (not exercised for backfill, but kept defensive).
        class _Stub:
            name = "stub"

            def fetch(self, tickers, asof_date, store, run_id):
                return IngestResult(self.name, 0, "ok", None)
        return _Stub()

    monkeypatch.setattr(
        "sma.ingest.__main__._build_source", fake_build,
    )


def test_backfill_walks_chunks_from_start_to_end(tmp_path, monkeypatch):
    _seed_env(monkeypatch)
    cfg = _write_config(
        tmp_path,
        sources_enabled=["finnhub_news"],
    )
    uni = _write_universe(tmp_path)
    db = tmp_path / "sma.duckdb"
    _patch_build_source(monkeypatch)

    result = CliRunner().invoke(cli, [
        "backfill-news",
        "--config", str(cfg),
        "--universe", str(uni),
        "--db", str(db),
        "--start", "2026-01-01",
        "--end", "2026-01-31",
        "--batch-days", "7",
    ])
    assert result.exit_code == 0, result.output

    finnhub_calls = [c for c in FakeNewsSource.calls
                     if c["name"] == "finnhub_news"]
    assert len(finnhub_calls) == 5, finnhub_calls
    asof_dates = [c["asof_date"] for c in finnhub_calls]
    # Each chunk starts the day after previous chunk_end. Verify no gaps
    # and full coverage of [start, end].
    expected = [
        date_cls(2026, 1, 7),
        date_cls(2026, 1, 14),
        date_cls(2026, 1, 21),
        date_cls(2026, 1, 28),
        date_cls(2026, 1, 31),
    ]
    assert asof_dates == expected
    # Verify no gaps: chunk i+1's start (asof - lookback) is exactly
    # 1 day after chunk i's asof.
    for i in range(len(finnhub_calls) - 1):
        cur_asof = finnhub_calls[i]["asof_date"]
        next_lookback = finnhub_calls[i + 1]["lookback_days"]
        next_asof = finnhub_calls[i + 1]["asof_date"]
        next_start = next_asof - timedelta(days=next_lookback)
        assert next_start == cur_asof + timedelta(days=1), (
            f"gap between chunk {i} and {i + 1}"
        )


def test_backfill_idempotent(tmp_path, monkeypatch):
    _seed_env(monkeypatch)
    cfg = _write_config(tmp_path, sources_enabled=["finnhub_news"])
    uni = _write_universe(tmp_path, tickers=["AAPL"])
    db = tmp_path / "sma.duckdb"
    _patch_build_source(monkeypatch)

    args = [
        "backfill-news",
        "--config", str(cfg),
        "--universe", str(uni),
        "--db", str(db),
        "--start", "2026-01-01",
        "--end", "2026-01-14",
        "--batch-days", "7",
    ]
    runner = CliRunner()
    r1 = runner.invoke(cli, args)
    assert r1.exit_code == 0, r1.output

    # Count rows after first run.
    import duckdb
    conn = duckdb.connect(str(db))
    n1 = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    conn.close()
    assert n1 > 0

    # Second run with the same range and same fake source.
    r2 = runner.invoke(cli, args)
    assert r2.exit_code == 0, r2.output

    conn = duckdb.connect(str(db))
    n2 = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    conn.close()
    assert n2 == n1, (
        f"second run added {n2 - n1} rows; expected idempotent (0 net new)"
    )


def test_backfill_filters_to_news_sources_only(tmp_path, monkeypatch):
    _seed_env(monkeypatch)
    # Enable a mix of news and non-news sources.
    cfg = _write_config(
        tmp_path,
        sources_enabled=[
            "yfinance", "alpaca", "edgar", "finnhub_fundamentals",
            "finnhub_news", "alpaca_news", "newsapi",
        ],
    )
    uni = _write_universe(tmp_path)
    db = tmp_path / "sma.duckdb"
    # news_only=True -> _build_source raises if ever asked for non-news.
    _patch_build_source(monkeypatch, news_only=True)

    result = CliRunner().invoke(cli, [
        "backfill-news",
        "--config", str(cfg),
        "--universe", str(uni),
        "--db", str(db),
        "--start", "2026-01-01",
        "--end", "2026-01-07",
        "--batch-days", "7",
    ])
    assert result.exit_code == 0, result.output

    constructed = set(FakeNewsSource.constructed)
    assert constructed == {"finnhub_news", "alpaca_news", "newsapi"}, constructed


def test_backfill_respects_sources_flag(tmp_path, monkeypatch):
    _seed_env(monkeypatch)
    cfg = _write_config(
        tmp_path,
        sources_enabled=["finnhub_news", "alpaca_news", "newsapi"],
    )
    uni = _write_universe(tmp_path)
    db = tmp_path / "sma.duckdb"
    _patch_build_source(monkeypatch)

    result = CliRunner().invoke(cli, [
        "backfill-news",
        "--config", str(cfg),
        "--universe", str(uni),
        "--db", str(db),
        "--start", "2026-01-01",
        "--end", "2026-01-07",
        "--batch-days", "7",
        "--sources", "alpaca_news",
    ])
    assert result.exit_code == 0, result.output

    constructed = set(FakeNewsSource.constructed)
    assert constructed == {"alpaca_news"}, constructed


def test_backfill_respects_tickers_flag(tmp_path, monkeypatch):
    _seed_env(monkeypatch)
    cfg = _write_config(tmp_path, sources_enabled=["finnhub_news"])
    uni = _write_universe(tmp_path, tickers=["AAPL", "MSFT", "GOOG", "AMZN"])
    db = tmp_path / "sma.duckdb"
    _patch_build_source(monkeypatch)

    result = CliRunner().invoke(cli, [
        "backfill-news",
        "--config", str(cfg),
        "--universe", str(uni),
        "--db", str(db),
        "--start", "2026-01-01",
        "--end", "2026-01-07",
        "--batch-days", "7",
        "--tickers", "AAPL,MSFT",
    ])
    assert result.exit_code == 0, result.output

    assert FakeNewsSource.calls, "fake source was never called"
    for call in FakeNewsSource.calls:
        assert sorted(call["tickers"]) == ["AAPL", "MSFT"], call


def test_backfill_handles_partial_chunk(tmp_path, monkeypatch):
    """start=Jan 1, end=Jan 10, batch_days=7 -> 2 chunks.

    Chunk 1: Jan 1..Jan 7 inclusive (asof=Jan 7, lookback=6).
    Chunk 2: Jan 8..Jan 10 inclusive (asof=Jan 10, lookback=2 to clamp).
    The final chunk has lookback < batch_days so we don't read past --end
    or before chunk_start.
    """
    _seed_env(monkeypatch)
    cfg = _write_config(tmp_path, sources_enabled=["finnhub_news"])
    uni = _write_universe(tmp_path)
    db = tmp_path / "sma.duckdb"
    _patch_build_source(monkeypatch)

    result = CliRunner().invoke(cli, [
        "backfill-news",
        "--config", str(cfg),
        "--universe", str(uni),
        "--db", str(db),
        "--start", "2026-01-01",
        "--end", "2026-01-10",
        "--batch-days", "7",
    ])
    assert result.exit_code == 0, result.output

    calls = [c for c in FakeNewsSource.calls if c["name"] == "finnhub_news"]
    assert len(calls) == 2, calls
    assert calls[0]["asof_date"] == date_cls(2026, 1, 7)
    assert calls[0]["lookback_days"] == 6
    assert calls[1]["asof_date"] == date_cls(2026, 1, 10)
    # Final chunk lookback clamps to (chunk_end - chunk_start) = 2.
    assert calls[1]["lookback_days"] == 2
