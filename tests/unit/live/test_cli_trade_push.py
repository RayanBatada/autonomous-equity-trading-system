"""CLI-level tests for the nightly trade-push notification (Rayan's ask,
2026-08-31): after decide submits orders, push a summary to his phone via
ntfy so he can manually mirror the trades on his own account. Wired at the
very end of _decide_impl, after write_sentinel and the failed-order page —
see src/sma/live/__main__.py.

CRITICAL: every test here that reaches a non-dry-run submit patches
sma.live.__main__.send_ntfy. This repo's real .env carries a real
SMA_NTFY_TOPIC (tests/conftest.py's autouse _no_real_ntfy_pushes fixture
also unsets it as a second line of defense) -- a real, unmocked send_ntfy
call here would page Rayan's actual phone during `pytest`.
"""

from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from sma.backtest.strategies.base import StrategyDecision
from sma.risk.rails import RiskRails


def _real_rails() -> RiskRails:
    return RiskRails(stop_loss_pct=0.0, min_hold_days=0)


def _make_writer_lock_factory(lock_path: Path):
    """Same rationale as tests/integration/live/test_decide_writes_sentinel.py:
    writer_lock's lock_path default is captured at import time, so a bare
    monkeypatch of sma.locks.DEFAULT_LOCK_PATH does not redirect it -- the
    CLI's own writer_lock(label="decide", ...) call must be pointed at the
    SAME tmp lock_path explicitly or it falls through to the real repo lock
    file."""
    from sma.locks import writer_lock as _real_writer_lock

    @contextmanager
    def _writer_lock_with_path(*, label: str, **kwargs):
        with _real_writer_lock(lock_path=lock_path, label=label, **kwargs):
            yield

    return _writer_lock_with_path


def _alpaca_mock(*, equity=100_000.0, positions=None):
    alpaca = MagicMock()
    alpaca.get_account.return_value = {
        "equity": equity, "cash": equity, "blocked_count": 0,
    }
    alpaca.get_positions.return_value = positions or {}
    submitted: list[str] = []

    def _buy(ticker, shares, *, client_order_id=None):
        oid = f"ord-buy-{len(submitted)}"
        submitted.append(oid)
        return oid

    def _sell(ticker, shares, *, client_order_id=None):
        oid = f"ord-sell-{len(submitted)}"
        submitted.append(oid)
        return oid

    alpaca.submit_day_market_buy.side_effect = _buy
    alpaca.submit_day_sell.side_effect = _sell
    return alpaca


class _BuyAAPLStrategy:
    def decide(self, *, asof_date, prices):
        return [StrategyDecision(asof_date=asof_date, ticker="AAPL", target_weight=0.05)]


class _NoDecisionsStrategy:
    def decide(self, *, asof_date, prices):
        return []


def _seed_prices(db_path, asof):
    from sma.ingest.store import Store

    store = Store(path=str(db_path)).connect()
    rid = store.allocate_run_id()
    for offset in range(5):
        d = asof - timedelta(days=offset)
        store.conn.execute(
            "INSERT INTO prices (ticker, date, open, high, low, close, "
            " adj_close, volume, source, run_id) "
            "VALUES ('AAPL', ?, 200, 200, 200, 200, 200, 1000000, 'yfinance', ?)",
            [d, rid],
        )
    store.conn.close()


def _make_settings(*, trade_pushes: bool):
    settings = MagicMock()
    settings.notify.trade_pushes = trade_pushes
    return settings


def _invoke_decide(
    tmp_path, *, asof, dry_run, alpaca, strategy, settings, send_ntfy_mock=None,
):
    from click.testing import CliRunner

    from sma.live.__main__ import decide as decide_cmd

    tmp_path.mkdir(parents=True, exist_ok=True)
    lock_path = tmp_path / ".sma-writer.lock"
    db_path = tmp_path / "test.duckdb"
    _seed_prices(db_path, asof)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text("tickers: []\n")

    args = [
        "--asof-date", asof.isoformat(),
        "--db", str(db_path),
        "--config", str(config_path),
        "--universe", str(universe_path),
    ]
    if dry_run:
        args.append("--dry-run")

    sent: list[tuple] = []
    if send_ntfy_mock is None:
        send_ntfy_mock = MagicMock(
            side_effect=lambda text, *, title=None, **kw: (
                sent.append((title, text)) or True
            )
        )

    with (
        patch("sma.live.__main__._build_alpaca", return_value=alpaca),
        patch("sma.live.__main__._build_strategy", return_value=strategy),
        patch("sma.live.__main__._build_rails", return_value=_real_rails()),
        patch("sma.live.__main__.load_settings", return_value=settings),
        patch("sma.live.__main__.load_universe", return_value=["AAPL"]),
        patch("sma.live.__main__.run_preflight"),
        patch("sma.live.__main__.writer_lock", _make_writer_lock_factory(lock_path)),
        patch("sma.live.__main__.send_ntfy", send_ntfy_mock),
    ):
        result = CliRunner().invoke(decide_cmd, args, catch_exceptions=False)
    return result, sent, send_ntfy_mock, db_path


def _intended_orders_rows(db_path):
    from sma.ingest.store import Store

    store = Store(path=str(db_path)).connect()
    rows = store.conn.execute(
        "SELECT ticker, side, target_shares, target_weight, last_price, status "
        "FROM intended_orders ORDER BY ticker"
    ).fetchall()
    store.conn.close()
    return rows


def test_push_sent_after_real_submit_with_flag_on(tmp_path):
    asof = date(2026, 8, 28)
    alpaca = _alpaca_mock(equity=100_000.0)
    result, sent, mock_send, _ = _invoke_decide(
        tmp_path, asof=asof, dry_run=False, alpaca=alpaca,
        strategy=_BuyAAPLStrategy(), settings=_make_settings(trade_pushes=True),
    )
    assert result.exit_code == 0, result.output
    assert len(sent) == 1
    title, text = sent[0]
    assert title == "SMA trades - Fri 8/28"
    assert "BUY AAPL — 5.0% of equity" in text.split("\n")
    assert text.splitlines()[-1] == "Equity $100,000 | 0 positions"


def test_push_not_sent_when_flag_off_but_trading_unaffected(tmp_path):
    asof = date(2026, 8, 28)
    alpaca = _alpaca_mock(equity=100_000.0)
    result, sent, mock_send, db_path = _invoke_decide(
        tmp_path, asof=asof, dry_run=False, alpaca=alpaca,
        strategy=_BuyAAPLStrategy(), settings=_make_settings(trade_pushes=False),
    )
    assert result.exit_code == 0, result.output
    mock_send.assert_not_called()
    assert sent == []
    rows = _intended_orders_rows(db_path)
    assert rows == [("AAPL", "BUY", 25, 0.05, 200.0, "submitted")]


def test_push_not_sent_on_dry_run_even_with_flag_on(tmp_path):
    asof = date(2026, 8, 28)
    alpaca = _alpaca_mock(equity=100_000.0)
    result, sent, mock_send, _ = _invoke_decide(
        tmp_path, asof=asof, dry_run=True, alpaca=alpaca,
        strategy=_BuyAAPLStrategy(), settings=_make_settings(trade_pushes=True),
    )
    assert result.exit_code == 0, result.output
    mock_send.assert_not_called()
    assert sent == []


def test_zero_orders_message(tmp_path):
    asof = date(2026, 8, 28)
    alpaca = _alpaca_mock(equity=100_000.0, positions={})
    result, sent, mock_send, _ = _invoke_decide(
        tmp_path, asof=asof, dry_run=False, alpaca=alpaca,
        strategy=_NoDecisionsStrategy(), settings=_make_settings(trade_pushes=True),
    )
    assert result.exit_code == 0, result.output
    assert len(sent) == 1
    _, text = sent[0]
    assert text == (
        f"{asof.isoformat()}\n"
        "No trades tonight — book unchanged (0 holds)\n"
        "Equity $100,000 | 0 positions"
    )


def test_send_ntfy_failure_does_not_break_the_cli(tmp_path):
    asof = date(2026, 8, 28)
    alpaca = _alpaca_mock(equity=100_000.0)
    boom = MagicMock(side_effect=RuntimeError("ntfy down"))
    result, sent, mock_send, db_path = _invoke_decide(
        tmp_path, asof=asof, dry_run=False, alpaca=alpaca,
        strategy=_BuyAAPLStrategy(), settings=_make_settings(trade_pushes=True),
        send_ntfy_mock=boom,
    )
    assert result.exit_code == 0, result.output
    assert boom.called
    # The order itself was still submitted -- a notify failure must never
    # unwind or affect trading that already happened.
    rows = _intended_orders_rows(db_path)
    assert rows == [("AAPL", "BUY", 25, 0.05, 200.0, "submitted")]


def test_trading_path_is_bit_identical_regardless_of_the_flag(tmp_path):
    """The ONLY difference flipping notify.trade_pushes makes is whether
    send_ntfy gets called -- decisions, orders, DB rows, sentinel and exit
    code must be identical either way."""
    asof = date(2026, 8, 28)

    result_on, sent_on, _, db_on = _invoke_decide(
        tmp_path / "on", asof=asof, dry_run=False,
        alpaca=_alpaca_mock(equity=100_000.0), strategy=_BuyAAPLStrategy(),
        settings=_make_settings(trade_pushes=True),
    )
    result_off, sent_off, _, db_off = _invoke_decide(
        tmp_path / "off", asof=asof, dry_run=False,
        alpaca=_alpaca_mock(equity=100_000.0), strategy=_BuyAAPLStrategy(),
        settings=_make_settings(trade_pushes=False),
    )

    assert result_on.exit_code == result_off.exit_code == 0
    assert result_on.output == result_off.output
    assert _intended_orders_rows(db_on) == _intended_orders_rows(db_off)
    assert len(sent_on) == 1
    assert sent_off == []


# ---------------------------------------------------------------------------
# record_trade_push persistence (2026-09-04): the CLI must persist a
# com.sma.trade-push-<asof> sentinel right after the send attempt, delivered
# or not, so sma.live.reconcile._detect_trade_push_drift can verify it
# tomorrow. tests/conftest.py's autouse _isolate_sentinel_dir fixture points
# SMA_SENTINEL_DIR at a per-test temp dir, so read_sentinel here reads the
# same isolated file the CLI (invoked in-process via CliRunner) just wrote.
# ---------------------------------------------------------------------------


def test_push_persisted_with_delivered_true_and_correct_orders_payload(tmp_path):
    from sma.live.trade_push import TRADE_PUSH_SENTINEL_LABEL
    from sma.sentinels import read_sentinel

    asof = date(2026, 8, 28)
    alpaca = _alpaca_mock(equity=100_000.0)
    result, sent, mock_send, _ = _invoke_decide(
        tmp_path, asof=asof, dry_run=False, alpaca=alpaca,
        strategy=_BuyAAPLStrategy(), settings=_make_settings(trade_pushes=True),
    )
    assert result.exit_code == 0, result.output
    assert len(sent) == 1

    sentinel = read_sentinel(label=TRADE_PUSH_SENTINEL_LABEL, asof=asof)
    assert sentinel is not None
    assert sentinel["delivered"] is True
    assert sentinel["equity"] == 100_000.0
    assert sentinel["title"] == "SMA trades - Fri 8/28"
    assert sentinel["orders"] == [{
        "ticker": "AAPL", "side": "BUY", "shares": 25,
        "decide_price": 200.0, "order_notional": 5_000.0,
        "order_pct_of_equity": 0.05, "full_exit": False,
    }]


def test_push_persisted_with_delivered_false_when_send_ntfy_raises(tmp_path):
    from sma.live.trade_push import TRADE_PUSH_SENTINEL_LABEL
    from sma.sentinels import read_sentinel

    asof = date(2026, 8, 28)
    alpaca = _alpaca_mock(equity=100_000.0)
    boom = MagicMock(side_effect=RuntimeError("ntfy down"))
    result, sent, mock_send, _ = _invoke_decide(
        tmp_path, asof=asof, dry_run=False, alpaca=alpaca,
        strategy=_BuyAAPLStrategy(), settings=_make_settings(trade_pushes=True),
        send_ntfy_mock=boom,
    )
    assert result.exit_code == 0, result.output
    assert boom.called

    sentinel = read_sentinel(label=TRADE_PUSH_SENTINEL_LABEL, asof=asof)
    assert sentinel is not None
    assert sentinel["delivered"] is False
    assert sentinel["orders"]  # the payload is still recorded, send failure aside


def test_no_push_file_written_when_trade_pushes_flag_off(tmp_path):
    from sma.live.trade_push import TRADE_PUSH_SENTINEL_LABEL
    from sma.sentinels import read_sentinel

    asof = date(2026, 8, 28)
    alpaca = _alpaca_mock(equity=100_000.0)
    result, sent, mock_send, _ = _invoke_decide(
        tmp_path, asof=asof, dry_run=False, alpaca=alpaca,
        strategy=_BuyAAPLStrategy(), settings=_make_settings(trade_pushes=False),
    )
    assert result.exit_code == 0, result.output
    assert sent == []
    assert read_sentinel(label=TRADE_PUSH_SENTINEL_LABEL, asof=asof) is None


def test_no_push_file_written_on_dry_run(tmp_path):
    from sma.live.trade_push import TRADE_PUSH_SENTINEL_LABEL
    from sma.sentinels import read_sentinel

    asof = date(2026, 8, 28)
    alpaca = _alpaca_mock(equity=100_000.0)
    result, sent, mock_send, _ = _invoke_decide(
        tmp_path, asof=asof, dry_run=True, alpaca=alpaca,
        strategy=_BuyAAPLStrategy(), settings=_make_settings(trade_pushes=True),
    )
    assert result.exit_code == 0, result.output
    assert sent == []
    assert read_sentinel(label=TRADE_PUSH_SENTINEL_LABEL, asof=asof) is None
