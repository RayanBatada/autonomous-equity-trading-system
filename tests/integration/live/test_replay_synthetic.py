"""Replay and sleeve-golden guarantees on a GENERATED fixture.

tests/fixtures/replay_synthetic.duckdb is built by
scripts/build_synthetic_replay_fixture.py from a seeded generator: no row
comes from production or the broker. It carries the same guarantees as the
real 2026-08-28 fixture (test_replay.py, test_sleeve_golden.py), so they still
run in the public copy of this repo, where the real fixture is left out:

1. replay reproduces a pinned set of orders exactly (a byte-level regression:
   any change to scoring, rails, sizing or min-hold moves them);
2. the sleeve path and the raw incumbent path give identical orders and
   decisions, on the reconstructed book and on an empty one;
3. a shadow sleeve changes no orders;
4. replay never submits, never mutates the database, never writes a sentinel.

If a deliberate change moves the pinned orders, rebuild nothing: re-pin
EXPECTED_* from `python -m sma.live replay --asof-date 2026-08-28 --book
asof|empty --db tests/fixtures/replay_synthetic.duckdb` and say why in the
commit.
"""

from __future__ import annotations

import hashlib
import shutil
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.integration.live.test_sleeve_golden import _GreedyShadow, _replay, _setup

REPO_ROOT = Path(__file__).resolve().parents[3]
SYNTH = REPO_ROOT / "tests" / "fixtures" / "replay_synthetic.duckdb"
ASOF = date(2026, 8, 28)

EXPECTED_ASOF = {
    ("SELL", "MAR", 4.0), ("SELL", "PEP", 113.0), ("SELL", "FANG", 118.0),
    ("SELL", "FTNT", 33.0), ("SELL", "TSM", 234.0),
    ("BUY", "SPG", 26.0), ("BUY", "NEM", 203.0), ("BUY", "PNC", 142.0), ("BUY", "KDP", 248.0),
}
EXPECTED_EMPTY = {
    ("BUY", "SPG", 26.0), ("BUY", "NEM", 203.0), ("BUY", "PNC", 142.0),
    ("BUY", "BRK-B", 136.0), ("BUY", "NOC", 98.0), ("BUY", "MS", 84.0),
    ("BUY", "NVDA", 245.0), ("BUY", "BKR", 31.0), ("BUY", "MCD", 107.0),
    ("BUY", "ROK", 128.0),
}
HELD_MIN_HOLD_BLOCKED = ("ETN", "SHOP", "SRE", "TFC", "TMUS")


def test_synthetic_fixture_is_committed_and_marked_synthetic():
    import duckdb

    assert SYNTH.exists()
    con = duckdb.connect(str(SYNTH), read_only=True)
    try:
        ids = [r[0] for r in con.execute("SELECT alpaca_order_id FROM paper_fills").fetchall()]
        models = {r[0] for r in con.execute("SELECT DISTINCT model_id FROM predictions").fetchall()}
    finally:
        con.close()
    assert ids and all(i.startswith("synthetic-") for i in ids)
    assert models == {"synthetic_model_for_tests"}


@pytest.mark.parametrize("book,expected", [("asof", EXPECTED_ASOF), ("empty", EXPECTED_EMPTY)])
def test_replay_reproduces_pinned_synthetic_orders(book, expected):
    settings, universe, rails, sizing = _setup()
    orders, _ = _replay(ASOF, SYNTH, book, settings, universe, rails, sizing)
    assert {(s, t, n) for t, s, n, _, _ in orders} == expected


@pytest.mark.parametrize("book", ["asof", "empty"])
def test_sleeve_path_matches_incumbent_path_on_synthetic(book, monkeypatch):
    import sma.live.__main__ as main

    settings, universe, rails, sizing = _setup()
    sleeve = _replay(ASOF, SYNTH, book, settings, universe, rails, sizing)
    monkeypatch.setattr(main, "_build_decide_strategy", main._build_strategy)
    incumbent = _replay(ASOF, SYNTH, book, settings, universe, rails, sizing)
    assert sleeve[0], "a real run happened"
    assert sleeve == incumbent


def test_shadow_sleeve_changes_no_orders_on_synthetic(monkeypatch):
    from sma.config import StrategiesSettings
    from sma.strategies import registry

    settings, universe, rails, sizing = _setup()
    base = _replay(ASOF, SYNTH, "empty", settings, universe, rails, sizing)
    monkeypatch.setitem(registry._REGISTRY, "golden_test_shadow", _GreedyShadow)
    shadowed_settings = settings.model_copy(update={"strategies": StrategiesSettings(sleeves=[
        {"name": "xgb_momentum", "capital_fraction": 1.0, "mode": "live"},
        {"name": "golden_test_shadow", "capital_fraction": 0.5, "mode": "shadow"},
    ])})
    shadowed = _replay(ASOF, SYNTH, "empty", shadowed_settings, universe, rails, sizing)
    assert shadowed[0] == base[0]


def test_min_hold_blocks_are_reported_as_holds():
    from sma.live.replay import replay_once

    settings, universe, rails, sizing = _setup()
    r = replay_once(asof=ASOF, db_path=SYNTH, universe=universe, rails=rails, sizing=sizing,
                    settings=settings, book="asof", use_theses=True)
    by = {d.ticker: d for d in r.decisions}
    for t in HELD_MIN_HOLD_BLOCKED:
        assert by[t].action == "HOLD", t


def test_replay_never_submits_on_synthetic():
    from sma.live.replay import replay_once

    settings, universe, rails, sizing = _setup()
    spy = MagicMock()
    spy.get_account.return_value = {
        "equity": 100_000.0, "cash": 10_000.0, "buying_power": 10_000.0,
        "long_market_value": 90_000.0, "trading_blocked": False, "account_blocked": False,
    }
    spy.get_positions.return_value = {}
    for m in ("submit_day_opg_buy", "submit_day_market_buy", "submit_day_sell",
              "submit_market_sell", "get_order_by_client_order_id"):
        getattr(spy, m).side_effect = AssertionError("replay must never submit")
    r = replay_once(asof=ASOF, db_path=SYNTH, universe=universe, rails=rails, sizing=sizing,
                    settings=settings, book="current", use_theses=True, alpaca=spy)
    assert r.aborted is False
    assert spy.get_account.call_count >= 1


def test_replay_never_mutates_synthetic_db(tmp_path, monkeypatch):
    from sma.live.replay import replay_once

    db = tmp_path / "copy.duckdb"
    shutil.copyfile(SYNTH, db)
    before = hashlib.sha256(db.read_bytes()).hexdigest()

    def _boom(*_a, **_kw):
        raise AssertionError("replay must never write a sentinel")

    monkeypatch.setattr("sma.sentinels.write_sentinel", _boom)
    settings, universe, rails, sizing = _setup()
    r = replay_once(asof=ASOF, db_path=db, universe=universe, rails=rails, sizing=sizing,
                    settings=settings, book="asof", use_theses=True)
    assert r.orders
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_replay_cli_on_synthetic():
    from click.testing import CliRunner

    from sma.live.__main__ import replay_cmd

    result = CliRunner().invoke(
        replay_cmd,
        ["--asof-date", "2026-08-28", "--book", "asof", "--db", str(SYNTH),
         "--config", str(REPO_ROOT / "config.yaml"),
         "--universe", str(REPO_ROOT / "src" / "sma" / "universe.yaml")],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "9 order(s)" in result.output
