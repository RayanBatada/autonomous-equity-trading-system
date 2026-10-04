"""sma.ingest.intraday: 1-minute IEX bars. Data client is a MagicMock."""

import uuid
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml
from click.testing import CliRunner

from sma.ingest import intraday as mod
from sma.ingest.store import Store
from sma.sentinels import read_sentinel

ASOF = date(2026, 9, 25)


def _bar(minute, close=100.0):
    ts = datetime(2026, 9, 25, 13, 30, tzinfo=UTC) + timedelta(minutes=minute)
    return SimpleNamespace(timestamp=ts, open=close, high=close + 1, low=close - 1,
                           close=close, volume=10)


def _client(bars_by_sym):
    c = MagicMock()
    c.get_stock_bars.side_effect = lambda req: SimpleNamespace(data={
        s: bars_by_sym.get(s, []) for s in req.symbol_or_symbols})
    return c


def _store(tmp_path):
    return Store(path=str(tmp_path / "i.duckdb")).connect()


def test_fetch_upserts_and_is_idempotent(tmp_path):
    s = _store(tmp_path)
    c = _client({"SPY": [_bar(0), _bar(1), _bar(2, close=float("nan"))],
                 "BRK.B": [_bar(0, 400.0)]})
    r1 = mod.fetch_intraday(tickers=["SPY", "BRK-B"], asof_date=ASOF, store=s, run_id=1, client=c)
    r2 = mod.fetch_intraday(tickers=["SPY", "BRK-B"], asof_date=ASOF, store=s, run_id=2, client=c)
    assert r1.rows_inserted == r2.rows_inserted == 3
    rows = s.conn.execute(
        "SELECT ticker, ts, close, source, run_id FROM prices_intraday ORDER BY ticker, ts"
    ).fetchall()
    assert len(rows) == 3
    assert rows[0][0] == "BRK-B" and rows[0][2] == 400.0  # mapped back to canonical
    assert rows[1][1] == datetime(2026, 9, 25, 13, 30)    # naive UTC bar start
    assert {r[3] for r in rows} == {"alpaca_iex"} and {r[4] for r in rows} == {2}
    req = c.get_stock_bars.call_args[0][0]
    assert req.start.replace(tzinfo=None) == datetime(2026, 9, 25, 4, 0)  # ET midnight, in UTC
    assert str(getattr(req.feed, "value", req.feed)).lower() == "iex"


def test_held_tickers_nets_the_ledger(tmp_path):
    s = _store(tmp_path)
    for oid, t, side, q in [("1", "AAPL", "BUY", 5), ("2", "AAPL", "SELL", 5),
                            ("3", "MSFT", "BUY", 2), ("4", "NVDA", "BUY", 1)]:
        s.conn.execute(
            "INSERT INTO paper_fills (alpaca_order_id, intended_order_id, asof_date, ticker, "
            "side, filled_shares, fill_price, status, submitted_at, run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, 1, 'filled', ?, 1)",
            [oid, str(uuid.uuid4()), ASOF, t, side, q, datetime(2026, 9, 25)])
    assert mod.held_tickers(s.conn) == ["MSFT", "NVDA"]
    assert mod.resolve_tickers(["SPY", "msft"], ["MSFT", "NVDA"]) == ["SPY", "MSFT", "NVDA"]


def test_cli_writes_log_and_sentinel(tmp_path, monkeypatch):
    for k in ("FINNHUB_API_KEY", "NEWSAPI_KEY", "ALPACA_API_KEY", "ALPACA_API_SECRET",
              "EDGAR_USER_AGENT"):
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({
        "ingest": {"default_lookback_days": 1,
                   "rate_limits": {"finnhub": {"requests_per_minute": 1},
                                   "newsapi": {"requests_per_day": 1},
                                   "edgar": {"requests_per_second": 1}},
                   "retries": {"max": 1, "base_delay": 1.0, "jitter": 0.0},
                   "circuit_breaker": {"failures_to_open": 1, "cooldown_minutes": 1}},
        "sources_enabled": [],
        "intraday": {"tickers": ["SPY", "XLK"], "include_held": False},
    }))
    fake = _client({"SPY": [_bar(0), _bar(1)], "XLK": [_bar(0)]})
    monkeypatch.setattr(mod, "_build_data_client", lambda settings: fake)
    db = str(tmp_path / "c.duckdb")
    res = CliRunner().invoke(mod.intraday_cmd, ["--asof-date", "2026-09-25", "--db", db,
                                                "--config", str(cfg)])
    assert res.exit_code == 0, res.output
    assert "3 bars fetched for 2 tickers" in res.output
    sent = read_sentinel(label="com.sma.ingest.intraday", asof=ASOF)
    assert sent["rows_for_day"] == 3 and sent["tickers"] == ["SPY", "XLK"]
    s = Store(path=db).connect()
    assert s.conn.execute(
        "SELECT status, rows_inserted FROM ingest_log WHERE source='alpaca_intraday'"
    ).fetchone() == ("ok", 3)
    s.close()


def test_ingest_cli_exposes_intraday():
    from sma.ingest.__main__ import cli
    assert "intraday" in cli.commands


def test_fetch_error_logged_and_raised(tmp_path, monkeypatch):
    c = MagicMock()
    c.get_stock_bars.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        mod.fetch_intraday(tickers=["SPY"], asof_date=ASOF, store=_store(tmp_path),
                           run_id=1, client=c)
