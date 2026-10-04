"""decide CLI + sleeves: every enabled sleeve's book lands in sleeve_targets on
a real run, nothing is written on a dry run, and an attribution failure can
never fail a trade night. `status` gains a Sleeves: section."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

from sma.ingest.store import Store
from tests.unit.live.test_cli_trade_push import (
    _alpaca_mock,
    _BuyAAPLStrategy,
    _intended_orders_rows,
    _invoke_decide,
    _make_settings,
)

ASOF = date(2026, 8, 28)


def _targets(db_path):
    s = Store(path=str(db_path)).connect()
    try:
        return s.conn.execute(
            "SELECT asof_date, session, sleeve, ticker, weight, mode, capital_fraction "
            "FROM sleeve_targets ORDER BY ticker").fetchall()
    finally:
        s.close()


def test_real_decide_persists_incumbent_sleeve_targets(tmp_path):
    result, _, _, db = _invoke_decide(
        tmp_path, asof=ASOF, dry_run=False, alpaca=_alpaca_mock(),
        strategy=_BuyAAPLStrategy(), settings=_make_settings(trade_pushes=False),
    )
    assert result.exit_code == 0, result.output
    assert _targets(db) == [(ASOF, "open", "xgb_momentum", "AAPL", 0.05, "live", 1.0)]
    assert [r[0] for r in _intended_orders_rows(db)] == ["AAPL"]


def test_dry_run_writes_no_sleeve_targets(tmp_path):
    result, _, _, db = _invoke_decide(
        tmp_path, asof=ASOF, dry_run=True, alpaca=_alpaca_mock(),
        strategy=_BuyAAPLStrategy(), settings=_make_settings(trade_pushes=False),
    )
    assert result.exit_code == 0, result.output
    assert _targets(db) == []


def test_attribution_failure_never_fails_decide(tmp_path):
    with patch(
        "sma.strategies.attribution.persist_sleeve_targets",
        side_effect=RuntimeError("disk on fire"),
    ), patch(
        "sma.strategies.attribution.score_pending", side_effect=RuntimeError("also"),
    ):
        result, _, _, db = _invoke_decide(
            tmp_path, asof=ASOF, dry_run=False, alpaca=_alpaca_mock(),
            strategy=_BuyAAPLStrategy(), settings=_make_settings(trade_pushes=False),
        )
    assert result.exit_code == 0, result.output
    assert [r[0] for r in _intended_orders_rows(db)] == ["AAPL"]


def test_status_prints_sleeves_section(tmp_path):
    from click.testing import CliRunner

    from sma.live.__main__ import status

    db = tmp_path / "s.duckdb"
    s = Store(path=str(db)).connect()
    s.conn.execute(
        "INSERT INTO sleeve_targets (asof_date, session, sleeve, ticker, weight, mode, "
        "capital_fraction, run_id) VALUES (?, 'open', 'xgb_momentum', 'AAPL', 0.1, 'live', 1, 1)",
        [ASOF],
    )
    s.close()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "strategies:\n  sleeves:\n"
        "    - {name: xgb_momentum, capital_fraction: 1.0, mode: live}\n"
        "    - {name: reversal, capital_fraction: 0.0, mode: shadow}\n"
    )
    out = CliRunner().invoke(status, ["--db", str(db), "--config", str(cfg)])
    assert out.exit_code == 0, out.output
    lines = out.output.splitlines()
    i = lines.index("Sleeves:")
    assert "xgb_momentum" in lines[i + 1] and "2026-08-28" in lines[i + 1]
    assert "reversal" in lines[i + 2] and "shadow" in lines[i + 2] and "never" in lines[i + 2]


def test_status_survives_bad_strategies_config(tmp_path):
    from click.testing import CliRunner

    from sma.live.__main__ import status

    db = tmp_path / "s.duckdb"
    Store(path=str(db)).connect().close()
    cfg = tmp_path / "config.yaml"
    cfg.write_text("strategies:\n  sleeves:\n    - {name: a, capital_fraction: 0.9}\n"
                   "    - {name: b, capital_fraction: 0.9}\n")
    out = CliRunner().invoke(status, ["--db", str(db), "--config", str(cfg)])
    assert out.exit_code == 0, out.output
    assert "Sleeves: unavailable" in out.output
