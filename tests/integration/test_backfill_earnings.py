"""Integration tests for the historical earnings backfill module.

The Finnhub client is mocked with MagicMock so we never hit the live API.
Persistence is exercised against a real in-memory DuckDB Store, so the
INSERT OR REPLACE + PK semantics are tested for real.
"""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests
import yaml
from click.testing import CliRunner
from finnhub.exceptions import FinnhubAPIException

from sma.ingest.__main__ import cli
from sma.ingest.earnings_backfill import (
    backfill_earnings,
    backfill_earnings_yfinance,
)
from sma.ingest.ratelimit import TokenBucket
from sma.ingest.store import Store


def _yf_earnings_df(rows: list[tuple]) -> pd.DataFrame:
    """Build a DataFrame matching yfinance's get_earnings_dates output.

    rows: list of (date_str, eps_estimate, reported_eps, surprise_pct).
    Index is timezone-aware DatetimeIndex named 'Earnings Date'.
    Columns: 'EPS Estimate', 'Reported EPS', 'Surprise(%)'.
    Rows are returned in descending date order to match yfinance.
    """
    sorted_rows = sorted(rows, key=lambda r: r[0], reverse=True)
    idx = pd.DatetimeIndex(
        [pd.Timestamp(r[0], tz="America/New_York") for r in sorted_rows],
        name="Earnings Date",
    )
    return pd.DataFrame(
        {
            "EPS Estimate": [r[1] for r in sorted_rows],
            "Reported EPS": [r[2] for r in sorted_rows],
            "Surprise(%)": [r[3] for r in sorted_rows],
        },
        index=idx,
    )


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    yield s
    s.close()


def _earnings_item(symbol: str, period: str, estimate: float | None,
                   actual: float | None, year: int = 2026,
                   quarter: int = 1) -> dict:
    return {
        "symbol": symbol,
        "period": period,
        "actual": actual,
        "estimate": estimate,
        "surprise": (
            None if actual is None or estimate is None else actual - estimate
        ),
        "surprisePercent": None,
        "quarter": quarter,
        "year": year,
    }


def test_backfill_inserts_rows(store):
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_earnings_backfill")

    fake_rows = [
        _earnings_item("AAPL", "2026-03-31", 1.32, 1.39, 2026, 1),
        _earnings_item("AAPL", "2025-12-31", 2.10, 2.18, 2025, 4),
        _earnings_item("AAPL", "2025-09-30", 1.55, 1.64, 2025, 3),
        _earnings_item("AAPL", "2025-06-30", 1.40, 1.42, 2025, 2),
    ]
    fake_client = MagicMock()
    fake_client.company_earnings.return_value = fake_rows

    n = backfill_earnings(
        api_key="fake",
        tickers=["AAPL"],
        store=store,
        run_id=run_id,
        quarters=4,
        client=fake_client,
    )
    assert n == 4

    rows = store.conn.execute(
        "SELECT ticker, report_date, eps_estimate, eps_actual, "
        "revenue_estimate, revenue_actual, source FROM earnings "
        "ORDER BY report_date DESC"
    ).fetchall()
    assert len(rows) == 4
    assert rows[0][0] == "AAPL"
    assert rows[0][1] == date(2026, 3, 31)
    assert rows[0][2] == pytest.approx(1.32)
    assert rows[0][3] == pytest.approx(1.39)
    assert rows[0][4] is None
    assert rows[0][5] is None
    assert all(r[6] == "finnhub_earnings_backfill" for r in rows)
    fake_client.company_earnings.assert_called_once_with(
        symbol="AAPL", limit=4,
    )


def test_backfill_translates_dash_ticker_to_dot_for_request(store):
    """BRK-B must be requested from Finnhub in its dot form (BRK.B) -- the
    same dash-to-dot translation alpaca_prices/alpaca_news/finnhub_news
    already do for this ticker."""
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_earnings_backfill")

    fake_client = MagicMock()
    fake_client.company_earnings.return_value = []

    backfill_earnings(
        api_key="fake", tickers=["BRK-B"], store=store, run_id=run_id,
        quarters=4, client=fake_client,
    )
    fake_client.company_earnings.assert_called_once_with(symbol="BRK.B", limit=4)


def test_backfill_stores_brk_a_response_under_canonical_brk_b(store):
    """Regression (verified live 2026-08-16): Finnhub's company_earnings
    reports Berkshire under symbol="BRK.A" on BOTH endpoints regardless of
    query spelling. Before this fix, `symbol = item.get("symbol") or ticker`
    stored the vendor's response symbol, so all 4 rows landed under ticker
    'BRK.A' -- a symbol outside the universe (canonical is BRK-B), so
    nothing downstream ever read them. This call is per-ticker (no batch
    ambiguity), so the row must always be stored under the canonical
    `ticker` we requested, never the vendor's response symbol.
    """
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_earnings_backfill")

    fake_client = MagicMock()
    fake_client.company_earnings.return_value = [
        _earnings_item("BRK.A", "2026-05-02", 5.1, 5.4, 2026, 1),
    ]

    n = backfill_earnings(
        api_key="fake", tickers=["BRK-B"], store=store, run_id=run_id,
        quarters=4, client=fake_client,
    )
    assert n == 1
    tickers = {r[0] for r in store.conn.execute(
        "SELECT ticker FROM earnings"
    ).fetchall()}
    assert tickers == {"BRK-B"}


def test_backfill_handles_finnhub_exception_per_ticker(store):
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_earnings_backfill")

    def side_effect(symbol, limit):
        if symbol == "BAD":
            raise FinnhubAPIException(MagicMock(status_code=429, text="rate"))
        return [_earnings_item(symbol, "2026-03-31", 1.0, 1.1)]

    fake_client = MagicMock()
    fake_client.company_earnings.side_effect = side_effect

    n = backfill_earnings(
        api_key="fake",
        tickers=["AAPL", "BAD", "MSFT"],
        store=store,
        run_id=run_id,
        quarters=8,
        client=fake_client,
    )
    assert n == 2
    tickers = {r[0] for r in store.conn.execute(
        "SELECT ticker FROM earnings"
    ).fetchall()}
    assert tickers == {"AAPL", "MSFT"}


def test_backfill_read_timeout_then_success_retries_and_inserts(store):
    """The 2026-08-03 backfill hit repeated company_earnings read timeouts
    (transient Finnhub slowness). One retry must recover the ticker instead
    of dropping it. Sleep is injected so the test doesn't actually wait.
    """
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_earnings_backfill")

    calls = {"n": 0}

    def side_effect(symbol, limit):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ReadTimeout("read timed out")
        return [_earnings_item(symbol, "2026-03-31", 1.0, 1.1)]

    fake_client = MagicMock()
    fake_client.company_earnings.side_effect = side_effect

    n = backfill_earnings(
        api_key="fake", tickers=["AAPL"], store=store, run_id=run_id,
        quarters=8, client=fake_client, sleep_fn=lambda s: None,
    )
    assert calls["n"] == 2
    assert n == 1
    tickers = {r[0] for r in store.conn.execute(
        "SELECT ticker FROM earnings"
    ).fetchall()}
    assert tickers == {"AAPL"}


def test_backfill_read_timeout_twice_gives_up_with_warning(store):
    """Only ONE retry on a read timeout: two consecutive timeouts give up
    and the ticker is skipped, but the run continues for other tickers.
    """
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_earnings_backfill")

    calls = {"n": 0}

    def side_effect(symbol, limit):
        calls["n"] += 1
        if symbol == "SLOW":
            raise requests.exceptions.ReadTimeout("read timed out")
        return [_earnings_item(symbol, "2026-03-31", 1.0, 1.1)]

    fake_client = MagicMock()
    fake_client.company_earnings.side_effect = side_effect

    n = backfill_earnings(
        api_key="fake", tickers=["SLOW", "AAPL"], store=store, run_id=run_id,
        quarters=8, client=fake_client, sleep_fn=lambda s: None,
    )
    # SLOW: initial + 1 retry = 2 calls; AAPL: 1 call.
    assert calls["n"] == 3
    assert n == 1
    tickers = {r[0] for r in store.conn.execute(
        "SELECT ticker FROM earnings"
    ).fetchall()}
    assert tickers == {"AAPL"}


def test_backfill_non_timeout_exception_not_retried(store):
    """A non-timeout exception (e.g. a 429) is not retried -- single call,
    ticker skipped, matching pre-existing behavior (and what the
    `--provider both` CLI path relies on for the yfinance gap-fill).
    """
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_earnings_backfill")

    calls = {"n": 0}

    def side_effect(symbol, limit):
        calls["n"] += 1
        raise RuntimeError("connection reset")

    fake_client = MagicMock()
    fake_client.company_earnings.side_effect = side_effect

    n = backfill_earnings(
        api_key="fake", tickers=["AAPL"], store=store, run_id=run_id,
        quarters=8, client=fake_client, sleep_fn=lambda s: None,
    )
    assert calls["n"] == 1
    assert n == 0


def test_backfill_raises_client_timeout_to_30s(store, monkeypatch):
    """The finnhub SDK hardcodes a 10s DEFAULT_TIMEOUT. When backfill_earnings
    constructs its own client (client=None, the production path), it must
    raise that to 30s to give real responses room to land.
    """
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_earnings_backfill")

    fake_client = MagicMock()
    fake_client.company_earnings.return_value = []
    monkeypatch.setattr(
        "sma.ingest.earnings_backfill.finnhub.Client",
        lambda api_key: fake_client,
    )

    backfill_earnings(
        api_key="fake", tickers=["AAPL"], store=store, run_id=run_id,
        quarters=1,
    )
    assert fake_client.DEFAULT_TIMEOUT == 30


def test_backfill_idempotent(store):
    run_id1 = store.allocate_run_id()
    store.log_run_start(run_id1, source="finnhub_earnings_backfill")

    fake_rows = [
        _earnings_item("AAPL", "2026-03-31", 1.32, 1.39),
        _earnings_item("AAPL", "2025-12-31", 2.10, 2.18, 2025, 4),
    ]
    fake_client = MagicMock()
    fake_client.company_earnings.return_value = fake_rows

    backfill_earnings(
        api_key="fake", tickers=["AAPL"], store=store,
        run_id=run_id1, quarters=2, client=fake_client,
    )
    n_first = store.conn.execute(
        "SELECT COUNT(*) FROM earnings"
    ).fetchone()[0]
    assert n_first == 2

    run_id2 = store.allocate_run_id()
    store.log_run_start(run_id2, source="finnhub_earnings_backfill")
    backfill_earnings(
        api_key="fake", tickers=["AAPL"], store=store,
        run_id=run_id2, quarters=2, client=fake_client,
    )
    n_second = store.conn.execute(
        "SELECT COUNT(*) FROM earnings"
    ).fetchone()[0]
    assert n_second == n_first, (
        f"second run added {n_second - n_first} rows; expected idempotent"
    )


def test_backfill_handles_partial_data(store):
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_earnings_backfill")

    fake_client = MagicMock()
    fake_client.company_earnings.return_value = [
        _earnings_item("AAPL", "2026-03-31", None, None),
    ]
    n = backfill_earnings(
        api_key="fake", tickers=["AAPL"], store=store,
        run_id=run_id, quarters=1, client=fake_client,
    )
    assert n == 1
    row = store.conn.execute(
        "SELECT ticker, report_date, eps_estimate, eps_actual FROM earnings"
    ).fetchone()
    assert row == ("AAPL", date(2026, 3, 31), None, None)


def test_backfill_uses_rate_limiter(store):
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_earnings_backfill")

    fake_client = MagicMock()
    fake_client.company_earnings.side_effect = lambda symbol, limit: [
        _earnings_item(symbol, "2026-03-31", 1.0, 1.1),
    ]
    rl = MagicMock(spec=TokenBucket)
    backfill_earnings(
        api_key="fake",
        tickers=["AAPL", "MSFT", "GOOG"],
        store=store,
        run_id=run_id,
        quarters=1,
        rate_limiter=rl,
        client=fake_client,
    )
    assert rl.acquire.call_count == 3


def test_backfill_invalid_period_skipped(store):
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_earnings_backfill")

    fake_client = MagicMock()
    fake_client.company_earnings.return_value = [
        _earnings_item("AAPL", "bad-date-string", 1.0, 1.1),
        _earnings_item("AAPL", "2026-03-31", 1.32, 1.39),
    ]
    n = backfill_earnings(
        api_key="fake", tickers=["AAPL"], store=store,
        run_id=run_id, quarters=2, client=fake_client,
    )
    assert n == 1
    rows = store.conn.execute(
        "SELECT ticker, report_date FROM earnings"
    ).fetchall()
    assert rows == [("AAPL", date(2026, 3, 31))]


def test_backfill_missing_period_skipped(store):
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_earnings_backfill")

    fake_client = MagicMock()
    fake_client.company_earnings.return_value = [
        {"symbol": "AAPL", "actual": 1.0, "estimate": 1.0},  # no period
        _earnings_item("AAPL", "2026-03-31", 1.32, 1.39),
    ]
    n = backfill_earnings(
        api_key="fake", tickers=["AAPL"], store=store,
        run_id=run_id, quarters=2, client=fake_client,
    )
    assert n == 1


def test_backfill_handles_empty_response(store):
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_earnings_backfill")

    fake_client = MagicMock()
    fake_client.company_earnings.return_value = []
    n = backfill_earnings(
        api_key="fake", tickers=["NEWCO"], store=store,
        run_id=run_id, quarters=8, client=fake_client,
    )
    assert n == 0
    assert store.conn.execute(
        "SELECT COUNT(*) FROM earnings"
    ).fetchone()[0] == 0


# ---------- CLI ----------


def _seed_env(monkeypatch):
    for k in ("FINNHUB_API_KEY", "NEWSAPI_KEY", "ALPACA_API_KEY",
              "ALPACA_API_SECRET"):
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("EDGAR_USER_AGENT", "T (t@e.com)")


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


def _write_universe(tmp_path: Path, tickers=None) -> Path:
    p = tmp_path / "universe.yaml"
    p.write_text(yaml.safe_dump({
        "universe": {
            "refresh_policy": "manual",
            "tickers": tickers or ["AAPL", "MSFT"],
        },
    }))
    return p


def test_cli_backfill_earnings_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["backfill-earnings", "--help"])
    assert result.exit_code == 0, result.output
    assert "--quarters" in result.output
    assert "--tickers" in result.output
    assert "--config" in result.output
    assert "--universe" in result.output
    assert "--db" in result.output


def test_cli_backfill_earnings_runs(tmp_path, monkeypatch):
    _seed_env(monkeypatch)
    cfg = _write_config(tmp_path)
    uni = _write_universe(tmp_path, tickers=["AAPL", "MSFT"])
    db = tmp_path / "sma.duckdb"

    fake_rows_aapl = [
        _earnings_item("AAPL", "2026-03-31", 1.32, 1.39),
        _earnings_item("AAPL", "2025-12-31", 2.10, 2.18, 2025, 4),
    ]
    fake_rows_msft = [
        _earnings_item("MSFT", "2026-03-31", 3.10, 3.22),
    ]

    def side_effect(symbol, limit):
        return {"AAPL": fake_rows_aapl, "MSFT": fake_rows_msft}.get(symbol, [])

    fake_client = MagicMock()
    fake_client.company_earnings.side_effect = side_effect

    monkeypatch.setattr(
        "sma.ingest.earnings_backfill.finnhub.Client",
        lambda api_key: fake_client,
    )

    result = CliRunner().invoke(cli, [
        "backfill-earnings",
        "--config", str(cfg),
        "--universe", str(uni),
        "--db", str(db),
        "--quarters", "2",
    ])
    assert result.exit_code == 0, result.output
    assert "backfill complete: 3 rows across 2 tickers" in result.output

    import duckdb
    conn = duckdb.connect(str(db))
    n = conn.execute("SELECT COUNT(*) FROM earnings").fetchone()[0]
    assert n == 3
    sources = {r[0] for r in conn.execute(
        "SELECT DISTINCT source FROM earnings"
    ).fetchall()}
    assert sources == {"finnhub_earnings_backfill"}
    conn.close()


def test_cli_backfill_earnings_respects_tickers_flag(tmp_path, monkeypatch):
    _seed_env(monkeypatch)
    cfg = _write_config(tmp_path)
    uni = _write_universe(tmp_path, tickers=["AAPL", "MSFT", "GOOG"])
    db = tmp_path / "sma.duckdb"

    called = []

    def side_effect(symbol, limit):
        called.append(symbol)
        return [_earnings_item(symbol, "2026-03-31", 1.0, 1.1)]

    fake_client = MagicMock()
    fake_client.company_earnings.side_effect = side_effect

    monkeypatch.setattr(
        "sma.ingest.earnings_backfill.finnhub.Client",
        lambda api_key: fake_client,
    )

    result = CliRunner().invoke(cli, [
        "backfill-earnings",
        "--config", str(cfg),
        "--universe", str(uni),
        "--db", str(db),
        "--quarters", "1",
        "--tickers", "AAPL,GOOG",
    ])
    assert result.exit_code == 0, result.output
    assert sorted(called) == ["AAPL", "GOOG"]


# ---------- yfinance backfill ----------


def _yf_factory(by_ticker: dict[str, pd.DataFrame | Exception]):
    """Build a ticker_factory that returns a MagicMock per ticker.

    Each MagicMock's .get_earnings_dates() either returns the provided
    DataFrame or raises the provided exception.
    """
    def _factory(ticker: str):
        m = MagicMock()
        item = by_ticker.get(ticker)
        if isinstance(item, Exception):
            m.get_earnings_dates.side_effect = item
        else:
            m.get_earnings_dates.return_value = item
        return m
    return _factory


def test_yf_backfill_inserts_rows(store):
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance_earnings_backfill")

    df = _yf_earnings_df([
        ("2026-03-31", 1.94, None, None),  # future, NaN actual — filtered
        ("2026-01-29", 2.67, 2.84, 6.34),
        ("2025-10-30", 1.77, 1.85, 4.52),
        ("2025-07-31", 1.43, 1.57, 9.48),
        ("2025-05-01", 1.63, 1.65, 1.50),
    ])
    factory = _yf_factory({"AAPL": df})

    n = backfill_earnings_yfinance(
        tickers=["AAPL"],
        store=store,
        run_id=run_id,
        quarters=4,
        ticker_factory=factory,
    )
    assert n == 4

    rows = store.conn.execute(
        "SELECT ticker, report_date, eps_estimate, eps_actual, "
        "revenue_estimate, revenue_actual, source FROM earnings "
        "ORDER BY report_date DESC"
    ).fetchall()
    assert len(rows) == 4
    assert rows[0][0] == "AAPL"
    assert rows[0][1] == date(2026, 1, 29)
    assert rows[0][2] == pytest.approx(2.67)
    assert rows[0][3] == pytest.approx(2.84)
    assert rows[0][4] is None and rows[0][5] is None
    assert all(r[6] == "yfinance_earnings_backfill" for r in rows)


def test_yf_backfill_drops_unreported_rows(store):
    """Rows with NaN 'Reported EPS' (not yet reported) must be skipped."""
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance_earnings_backfill")

    df = _yf_earnings_df([
        ("2026-04-30", 1.94, None, None),
        ("2026-01-29", 2.67, 2.84, 6.34),
    ])
    factory = _yf_factory({"AAPL": df})

    n = backfill_earnings_yfinance(
        tickers=["AAPL"], store=store, run_id=run_id, quarters=8,
        ticker_factory=factory,
    )
    assert n == 1
    row = store.conn.execute(
        "SELECT report_date, eps_actual FROM earnings"
    ).fetchone()
    assert row == (date(2026, 1, 29), pytest.approx(2.84))


def test_yf_backfill_caps_at_quarters(store):
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance_earnings_backfill")

    df = _yf_earnings_df([
        (f"202{y}-{m:02d}-15", 1.0, 1.1, 10.0)
        for y in range(0, 6) for m in (3, 6, 9, 12)
    ])
    factory = _yf_factory({"AAPL": df})

    n = backfill_earnings_yfinance(
        tickers=["AAPL"], store=store, run_id=run_id, quarters=4,
        ticker_factory=factory,
    )
    assert n == 4


def test_yf_backfill_handles_per_ticker_failure(store):
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance_earnings_backfill")

    good_df = _yf_earnings_df([("2026-01-29", 2.67, 2.84, 6.34)])
    factory = _yf_factory({
        "AAPL": good_df,
        "BAD": RuntimeError("yahoo 429"),
        "MSFT": _yf_earnings_df([("2026-01-25", 3.10, 3.22, 3.87)]),
    })

    n = backfill_earnings_yfinance(
        tickers=["AAPL", "BAD", "MSFT"],
        store=store, run_id=run_id, quarters=4,
        ticker_factory=factory,
    )
    assert n == 2
    tickers = {r[0] for r in store.conn.execute(
        "SELECT ticker FROM earnings"
    ).fetchall()}
    assert tickers == {"AAPL", "MSFT"}


def test_yf_backfill_handles_empty_df(store):
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance_earnings_backfill")

    factory = _yf_factory({"NEWCO": pd.DataFrame()})
    n = backfill_earnings_yfinance(
        tickers=["NEWCO"], store=store, run_id=run_id, quarters=4,
        ticker_factory=factory,
    )
    assert n == 0


def test_yf_backfill_handles_none_df(store):
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance_earnings_backfill")

    factory = _yf_factory({"NEWCO": None})
    n = backfill_earnings_yfinance(
        tickers=["NEWCO"], store=store, run_id=run_id, quarters=4,
        ticker_factory=factory,
    )
    assert n == 0


def test_yf_backfill_only_missing_skips_existing(store):
    """only_missing=True must skip tickers that already have rows."""
    run_id1 = store.allocate_run_id()
    store.log_run_start(run_id1, source="finnhub_earnings_backfill")
    fake_client = MagicMock()
    fake_client.company_earnings.return_value = [
        _earnings_item("AAPL", "2026-01-29", 2.67, 2.84),
    ]
    backfill_earnings(
        api_key="fake", tickers=["AAPL"], store=store,
        run_id=run_id1, quarters=1, client=fake_client,
    )

    run_id2 = store.allocate_run_id()
    store.log_run_start(run_id2, source="yfinance_earnings_backfill")
    called: list[str] = []

    def factory(t):
        called.append(t)
        return MagicMock(
            get_earnings_dates=MagicMock(return_value=_yf_earnings_df([
                ("2025-10-25", 3.10, 3.22, 3.87),
            ]))
        )

    n = backfill_earnings_yfinance(
        tickers=["AAPL", "MSFT"],
        store=store, run_id=run_id2, quarters=1,
        ticker_factory=factory,
        only_missing=True,
    )
    assert called == ["MSFT"], "AAPL should be skipped, only MSFT fetched"
    assert n == 1


def test_yf_backfill_handles_unexpected_schema(store):
    """Bail per-ticker if columns don't include 'Reported EPS'."""
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance_earnings_backfill")

    bogus = pd.DataFrame(
        {"foo": [1, 2]},
        index=pd.DatetimeIndex(["2026-01-01", "2025-01-01"], name="x"),
    )
    factory = _yf_factory({"WEIRD": bogus})

    n = backfill_earnings_yfinance(
        tickers=["WEIRD"], store=store, run_id=run_id, quarters=4,
        ticker_factory=factory,
    )
    assert n == 0


def test_yf_backfill_idempotent(store):
    run_id1 = store.allocate_run_id()
    store.log_run_start(run_id1, source="yfinance_earnings_backfill")

    df = _yf_earnings_df([
        ("2026-01-29", 2.67, 2.84, 6.34),
        ("2025-10-30", 1.77, 1.85, 4.52),
    ])

    def factory(t):
        return MagicMock(
            get_earnings_dates=MagicMock(return_value=df)
        )

    backfill_earnings_yfinance(
        tickers=["AAPL"], store=store, run_id=run_id1, quarters=2,
        ticker_factory=factory,
    )
    n_first = store.conn.execute(
        "SELECT COUNT(*) FROM earnings"
    ).fetchone()[0]

    run_id2 = store.allocate_run_id()
    store.log_run_start(run_id2, source="yfinance_earnings_backfill")
    backfill_earnings_yfinance(
        tickers=["AAPL"], store=store, run_id=run_id2, quarters=2,
        ticker_factory=factory,
    )
    n_second = store.conn.execute(
        "SELECT COUNT(*) FROM earnings"
    ).fetchone()[0]
    assert n_second == n_first


# ---------- CLI: --provider flag ----------


def test_cli_backfill_earnings_provider_help():
    result = CliRunner().invoke(cli, ["backfill-earnings", "--help"])
    assert result.exit_code == 0, result.output
    assert "--provider" in result.output
    assert "finnhub" in result.output and "yfinance" in result.output
    assert "both" in result.output


def test_cli_provider_yfinance_only(tmp_path, monkeypatch):
    _seed_env(monkeypatch)
    cfg = _write_config(tmp_path)
    uni = _write_universe(tmp_path, tickers=["AAPL"])
    db = tmp_path / "sma.duckdb"

    df = _yf_earnings_df([("2026-01-29", 2.67, 2.84, 6.34)])
    monkeypatch.setattr(
        "sma.ingest.earnings_backfill.yf.Ticker",
        lambda t: MagicMock(
            get_earnings_dates=MagicMock(return_value=df)
        ),
    )

    # Finnhub client should NOT be called when provider=yfinance.
    finnhub_called = MagicMock()
    monkeypatch.setattr(
        "sma.ingest.earnings_backfill.finnhub.Client",
        lambda api_key: finnhub_called,
    )

    result = CliRunner().invoke(cli, [
        "backfill-earnings",
        "--config", str(cfg),
        "--universe", str(uni),
        "--db", str(db),
        "--quarters", "1",
        "--provider", "yfinance",
    ])
    assert result.exit_code == 0, result.output
    finnhub_called.company_earnings.assert_not_called()

    import duckdb
    conn = duckdb.connect(str(db))
    sources = {r[0] for r in conn.execute(
        "SELECT DISTINCT source FROM earnings"
    ).fetchall()}
    assert sources == {"yfinance_earnings_backfill"}
    conn.close()


def test_cli_provider_both_finnhub_then_yfinance_gap_fill(
    tmp_path, monkeypatch,
):
    """Finnhub fills AAPL; yfinance gap-fills MSFT (Finnhub 429'd it)."""
    _seed_env(monkeypatch)
    cfg = _write_config(tmp_path)
    uni = _write_universe(tmp_path, tickers=["AAPL", "MSFT"])
    db = tmp_path / "sma.duckdb"

    def finnhub_side_effect(symbol, limit):
        if symbol == "AAPL":
            return [_earnings_item("AAPL", "2026-01-29", 2.67, 2.84)]
        raise FinnhubAPIException(MagicMock(status_code=429, text="quota"))

    fake_client = MagicMock()
    fake_client.company_earnings.side_effect = finnhub_side_effect
    monkeypatch.setattr(
        "sma.ingest.earnings_backfill.finnhub.Client",
        lambda api_key: fake_client,
    )

    yf_called: list[str] = []

    def yf_factory(ticker):
        yf_called.append(ticker)
        return MagicMock(
            get_earnings_dates=MagicMock(return_value=_yf_earnings_df([
                ("2026-01-25", 3.10, 3.22, 3.87),
            ]))
        )
    monkeypatch.setattr(
        "sma.ingest.earnings_backfill.yf.Ticker", yf_factory,
    )

    result = CliRunner().invoke(cli, [
        "backfill-earnings",
        "--config", str(cfg),
        "--universe", str(uni),
        "--db", str(db),
        "--quarters", "1",
        "--provider", "both",
    ])
    assert result.exit_code == 0, result.output
    assert yf_called == ["MSFT"], (
        "yfinance should only fill MSFT (gap), not AAPL"
    )

    import duckdb
    conn = duckdb.connect(str(db))
    rows = conn.execute(
        "SELECT ticker, source FROM earnings ORDER BY ticker"
    ).fetchall()
    assert rows == [
        ("AAPL", "finnhub_earnings_backfill"),
        ("MSFT", "yfinance_earnings_backfill"),
    ]
    conn.close()
