"""sma.live.session: the intraday session job. Broker is a MagicMock."""

import json
from datetime import date, datetime, time
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
import yaml

from sma.ingest.store import Store
from sma.live import session as mod
from sma.live.alpaca_client import LimitOrderResult, QuoteUnavailableError, RefPrice
from sma.sentinels import read_sentinel

ET = ZoneInfo("America/New_York")
DAY = date(2026, 9, 28)  # a Monday


@pytest.fixture
def env(monkeypatch):
    for k in ("FINNHUB_API_KEY", "NEWSAPI_KEY", "ALPACA_API_KEY", "ALPACA_API_SECRET",
              "EDGAR_USER_AGENT"):
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


def _config(tmp_path, *, enabled=True, fallback="market"):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({
        "ingest": {"default_lookback_days": 1,
                   "rate_limits": {"finnhub": {"requests_per_minute": 1},
                                   "newsapi": {"requests_per_day": 1},
                                   "edgar": {"requests_per_second": 1}},
                   "retries": {"max": 1, "base_delay": 1.0, "jitter": 0.0},
                   "circuit_breaker": {"failures_to_open": 1, "cooldown_minutes": 1}},
        "sources_enabled": [],
        "live": {"sessions": {"midday": {"enabled": enabled}},
                 "execution": {"fallback": fallback}},
    }))
    return str(p)


def _alpaca(positions=None, cash=10_000.0, quotes=None):
    a = MagicMock()
    a.session_window.side_effect = lambda *, day: (
        datetime.combine(day, time(9, 30), tzinfo=ET),
        datetime.combine(day, time(16, 0), tzinfo=ET))
    a.get_account.return_value = {"cash": cash, "equity": 50_000.0,
                                  "trading_blocked": False, "account_blocked": False}
    a.get_positions.return_value = positions or {}
    a.list_open_orders.return_value = []
    quotes = quotes or {}

    def ref(t, max_spread_bps=50.0):
        if t not in quotes:
            raise QuoteUnavailableError(t)
        return RefPrice(bid=quotes[t] - 0.01, ask=quotes[t] + 0.01, source="quote")
    a.reference_price.side_effect = ref
    n = iter(range(100))

    def submit(ticker, side, qty, **kw):
        return LimitOrderResult(order_id=f"o{next(n)}", ticker=ticker, side=side, qty=qty,
                                limit_price=kw["ref"].ask, ref=kw["ref"],
                                client_order_id=kw.get("client_order_id"))
    a.submit_marketable_limit.side_effect = submit
    a.sweep_unfilled.side_effect = lambda working, **kw: working
    return a


def _now(h=10, m=35):
    return lambda: datetime.combine(DAY, time(h, m), tzinfo=ET)


def _targets(tmp_path, data, name="midday"):
    p = tmp_path / f"session_targets-{name}-{DAY.isoformat()}.json"
    p.write_text(json.dumps(data))


def _run(tmp_path, alpaca, *, dry=False, now=None, cfg=None, name="midday"):
    return mod.run_session(name=name, dry_run=dry, db=str(tmp_path / "s.duckdb"),
                           config=cfg or _config(tmp_path), alpaca=alpaca,
                           now_fn=now or _now(), state_dir=tmp_path, echo=lambda *_: None)


def test_no_targets_is_the_default_and_exits_clean(tmp_path, env):
    a = _alpaca()
    out = _run(tmp_path, a)
    assert out["status"] == "no_targets"
    assert "no targets for session" in out["detail"]
    assert read_sentinel(label="com.sma.live.session.midday", asof=DAY)["status"] == "no_targets"
    a.submit_marketable_limit.assert_not_called()


def test_outside_window_and_market_closed_skip(tmp_path, env):
    _targets(tmp_path, {"AAPL": 1000})
    a = _alpaca(quotes={"AAPL": 100})
    assert _run(tmp_path, a, now=_now(12, 0))["status"] == "out_of_window"
    a.session_window.side_effect = lambda *, day: None
    assert _run(tmp_path, a)["status"] == "market_closed"
    a.submit_marketable_limit.assert_not_called()


def test_half_day_clips_close_window(tmp_path, env):
    _targets(tmp_path, {"AAPL": 1000}, name="close")
    a = _alpaca(quotes={"AAPL": 100})
    a.session_window.side_effect = lambda *, day: (
        datetime.combine(day, time(9, 30), tzinfo=ET),
        datetime.combine(day, time(13, 0), tzinfo=ET))
    out = _run(tmp_path, a, name="close", now=_now(15, 45))
    assert out["status"] == "out_of_window"


def test_disabled_config_does_not_trade(tmp_path, env):
    _targets(tmp_path, {"AAPL": 1000})
    a = _alpaca(quotes={"AAPL": 100})
    out = _run(tmp_path, a, cfg=_config(tmp_path, enabled=False))
    assert out["status"] == "disabled"
    a.submit_marketable_limit.assert_not_called()


def test_dry_run_plans_and_writes_nothing(tmp_path, env):
    _targets(tmp_path, {"AAPL": 1000})
    a = _alpaca(quotes={"AAPL": 100})
    out = _run(tmp_path, a, dry=True, cfg=_config(tmp_path, enabled=False), now=_now(20, 0))
    assert out["status"] == "dry_run" and out["orders"] == [("BUY", "AAPL", 10.0)]
    a.submit_marketable_limit.assert_not_called()
    assert read_sentinel(label="com.sma.live.session.midday", asof=DAY) is None
    assert not (tmp_path / "s.duckdb").exists()


def test_live_run_submits_sells_first_records_audit_and_sentinel(tmp_path, env):
    _targets(tmp_path, {"AAPL": 2000, "MSFT": 0, "NVDA": 500, "XLK": 30})
    a = _alpaca(positions={"MSFT": {"shares": 5, "cost_basis": 1}, "NVDA": {"shares": 1}},
                quotes={"AAPL": 100, "MSFT": 400, "NVDA": 200, "XLK": 50})
    out = _run(tmp_path, a)
    assert out["status"] == "ok" and out["submitted"] == 3
    calls = [(c.args[0], c.args[1], c.args[2]) for c in a.submit_marketable_limit.call_args_list]
    assert calls == [("MSFT", "SELL", 5.0), ("AAPL", "BUY", 20.0), ("NVDA", "BUY", 1.0)]
    assert a.submit_marketable_limit.call_args_list[0].kwargs["client_order_id"] == (
        "sma-2026-09-28-MSFT-session-midday-SELL")
    assert {s["ticker"] for s in out["skipped"]} == {"XLK"}   # $30 delta under $50 minimum
    kw = a.sweep_unfilled.call_args.kwargs
    assert kw["deadline"] == datetime.combine(DAY, time(11, 0), tzinfo=ET)
    assert kw["fallback"] == "market" and kw["reprice_once"] is True
    s = Store(path=str(tmp_path / "s.duckdb")).connect()
    rows = s.conn.execute(
        "SELECT ticker, side, source, session, order_type, last_price, limit_price, "
        "alpaca_order_id FROM intended_orders ORDER BY ticker").fetchall()
    s.close()
    assert rows[0] == ("AAPL", "BUY", "session-midday", "midday", "limit", 100.0, 100.01, "o1")
    assert len(rows) == 3
    sent = read_sentinel(label="com.sma.live.session.midday", asof=DAY)
    assert sent["status"] == "ok" and sent["submitted"] == 3


def test_second_run_same_day_refuses(tmp_path, env):
    _targets(tmp_path, {"AAPL": 2000})
    a = _alpaca(quotes={"AAPL": 100})
    assert _run(tmp_path, a)["status"] == "ok"
    assert _run(tmp_path, a)["status"] == "already_ran"
    assert a.submit_marketable_limit.call_count == 1


def test_buys_scaled_to_cash_plus_sells_no_margin(tmp_path, env):
    _targets(tmp_path, {"AAPL": 10_000})
    a = _alpaca(cash=1_000.0, quotes={"AAPL": 100})
    _run(tmp_path, a)
    assert a.submit_marketable_limit.call_args.args[2] == 9.0  # 1000 / 100.00 mid, floored


def test_replacement_rows_written_by_sweep_callback(tmp_path, env):
    from sma.live.alpaca_client import WorkingOrder
    _targets(tmp_path, {"AAPL": 2000})
    a = _alpaca(quotes={"AAPL": 100})

    def sweep(working, **kw):
        w = working[0]
        new = WorkingOrder(order_id="m1", ticker=w.ticker, side=w.side, qty=w.qty, stage="market")
        assert kw["coid_fn"](w, "market") == "sma-2026-09-28-AAPL-session-midday-market-BUY"
        kw["on_replace"](w, new, {"limit_price": None})
        return [w, new]
    a.sweep_unfilled.side_effect = sweep
    _run(tmp_path, a)
    s = Store(path=str(tmp_path / "s.duckdb")).connect()
    rows = s.conn.execute("SELECT source, order_type, last_price, alpaca_order_id "
                          "FROM intended_orders ORDER BY source").fetchall()
    s.close()
    assert rows == [("session-midday", "limit", 100.0, "o0"),
                    ("session-midday-market", "market", 100.0, "m1")]


def test_open_order_and_bad_quote_names_are_skipped(tmp_path, env):
    _targets(tmp_path, {"AAPL": 2000, "ZZZZ": 1000})
    a = _alpaca(quotes={"AAPL": 100})
    a.list_open_orders.return_value = [{"symbol": "AAPL"}]
    out = _run(tmp_path, a)
    assert {s["ticker"] for s in out["skipped"]} == {"AAPL", "ZZZZ"}
    a.submit_marketable_limit.assert_not_called()


def test_malformed_targets_raise(tmp_path, env):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"AAPL": -5}))
    with pytest.raises(ValueError):
        mod.load_targets(p)
    p.write_text("[1,2]")
    with pytest.raises(ValueError):
        mod.load_targets(p)
    assert mod.load_targets(tmp_path / "missing.json") is None


def test_cli_registered_on_live_group():
    from sma.live.__main__ import cli
    assert "session" in cli.commands
