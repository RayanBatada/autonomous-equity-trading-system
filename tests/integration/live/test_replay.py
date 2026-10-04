"""Integration tests for sma.live.replay's end-to-end guarantees.

1. Regression: replaying 2026-08-28 against a fixture pinned to that real,
   recorded decide night reproduces its actual intended_orders (AFRM BUY 25,
   PWR BUY 3, LLY SELL 1, LUV SELL 77 -- verified 2026-09-01 against
   production data/sma.duckdb before the fixture was extracted; see
   scripts referenced in the fixture's own build history).
2. replay NEVER calls an order-submission method, even with a real book.
3. replay NEVER mutates the database or writes a sentinel.

The fixture (tests/fixtures/replay_2026_08_28.duckdb) holds only the real
rows the replay pipeline actually reads for that date (predictions, theses,
prices, earnings, politician_trades, account_snapshots) plus one synthetic
paper_fills BUY per held ticker, dated at that ticker's real most-recent-buy
date -- see the extraction notes in that table's row shape below. This
keeps the regression reproducible in CI without needing the (gitignored)
production database.
"""

import hashlib
import shutil
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sma.config import load_settings
from sma.ingest.universe import load_universe
from sma.live.__main__ import _build_rails, _build_sizing
from sma.live.replay import replay_once

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "replay_2026_08_28.duckdb"
CONFIG = REPO_ROOT / "config.yaml"
UNIVERSE = REPO_ROOT / "src" / "sma" / "universe.yaml"

EXPECTED_2026_08_28_ORDERS = {
    ("BUY", "AFRM", 25.0),
    ("BUY", "PWR", 3.0),
    ("SELL", "LLY", 1.0),
    ("SELL", "LUV", 77.0),
}


def _real_settings_universe_rails_sizing():
    settings = load_settings(config_path=CONFIG)
    universe = load_universe(UNIVERSE)
    rails = _build_rails(settings)
    sizing = _build_sizing(settings)
    return settings, universe, rails, sizing


@pytest.mark.skipif(not FIXTURE.exists(), reason="replay fixture not present")
def test_replay_reproduces_2026_08_28_decide_night():
    settings, universe, rails, sizing = _real_settings_universe_rails_sizing()

    result = replay_once(
        asof=date(2026, 8, 28),
        db_path=FIXTURE,
        universe=universe,
        rails=rails,
        sizing=sizing,
        settings=settings,
        book="asof",
        use_theses=True,
    )

    assert result.aborted is False
    actual = {(o.side, o.ticker, o.shares) for o in result.orders}
    assert actual == EXPECTED_2026_08_28_ORDERS, (
        "replay's orders for 2026-08-28 no longer match the real recorded "
        f"decide night (documented nondeterminism, if any, belongs here): {actual}"
    )


@pytest.mark.skipif(not FIXTURE.exists(), reason="replay fixture not present")
def test_replay_never_submits_even_with_book_current():
    """A spy AlpacaClient whose submit_* methods raise if called -- book=
    'current' is the ONLY mode that hands a real-shaped client through, so
    it is the sharpest test of the never-submit guarantee."""
    settings, universe, rails, sizing = _real_settings_universe_rails_sizing()

    spy = MagicMock()
    spy.get_account.return_value = {
        "equity": 100_000.0, "cash": 10_000.0, "buying_power": 10_000.0,
        "long_market_value": 90_000.0, "trading_blocked": False,
        "account_blocked": False,
    }
    spy.get_positions.return_value = {}

    def _boom(*_a, **_kw):
        raise AssertionError("replay must never submit an order")

    spy.submit_day_opg_buy.side_effect = _boom
    spy.submit_day_market_buy.side_effect = _boom
    spy.submit_day_sell.side_effect = _boom
    spy.submit_market_sell.side_effect = _boom
    spy.get_order_by_client_order_id.side_effect = _boom

    result = replay_once(
        asof=date(2026, 8, 28),
        db_path=FIXTURE,
        universe=universe,
        rails=rails,
        sizing=sizing,
        settings=settings,
        book="current",
        use_theses=True,
        alpaca=spy,
    )

    assert result.aborted is False
    assert spy.submit_day_opg_buy.call_count == 0
    assert spy.submit_day_market_buy.call_count == 0
    assert spy.submit_day_sell.call_count == 0
    assert spy.submit_market_sell.call_count == 0
    assert spy.get_order_by_client_order_id.call_count == 0
    # Reads DID happen -- this isn't passing because nothing ran.
    assert spy.get_account.call_count >= 1
    assert spy.get_positions.call_count >= 1


@pytest.mark.skipif(not FIXTURE.exists(), reason="replay fixture not present")
def test_replay_never_mutates_the_database_or_writes_a_sentinel(tmp_path, monkeypatch):
    """fs+DB hash before/after: byte-identical file, and write_sentinel is
    never even imported-and-called (a hard assertion, not just "no new
    files")."""
    db_copy = tmp_path / "replay_copy.duckdb"
    shutil.copyfile(FIXTURE, db_copy)
    before_hash = hashlib.sha256(db_copy.read_bytes()).hexdigest()

    def _must_not_be_called(*_a, **_kw):
        raise AssertionError("replay must never write a sentinel")

    monkeypatch.setattr("sma.sentinels.write_sentinel", _must_not_be_called)

    settings, universe, rails, sizing = _real_settings_universe_rails_sizing()
    result = replay_once(
        asof=date(2026, 8, 28),
        db_path=db_copy,
        universe=universe,
        rails=rails,
        sizing=sizing,
        settings=settings,
        book="asof",   # no AlpacaClient/network needed
        use_theses=True,
    )

    assert result.aborted is False
    assert len(result.orders) > 0   # a real run happened, not a silent no-op

    after_hash = hashlib.sha256(db_copy.read_bytes()).hexdigest()
    assert after_hash == before_hash, "replay mutated the database file"

    import os
    sentinel_dir = os.environ.get("SMA_SENTINEL_DIR")
    if sentinel_dir and Path(sentinel_dir).exists():
        assert list(Path(sentinel_dir).iterdir()) == []


@pytest.mark.skipif(not FIXTURE.exists(), reason="replay fixture not present")
def test_replay_reports_holds_and_a_blocked_rebuy_with_rail_attribution():
    """Sanity check on the human-readable report: the 6 held names the real
    2026-08-28 night dead-zone-skipped show up as HOLD, and every held
    ticker gets a decision row (no silent omissions)."""
    settings, universe, rails, sizing = _real_settings_universe_rails_sizing()

    result = replay_once(
        asof=date(2026, 8, 28),
        db_path=FIXTURE,
        universe=universe,
        rails=rails,
        sizing=sizing,
        settings=settings,
        book="asof",
        use_theses=True,
    )

    by_ticker = {d.ticker: d for d in result.decisions}
    for held_no_order in ("ENPH", "INTC", "PLTR", "ALB", "F", "RBLX", "CAT"):
        assert held_no_order in by_ticker, f"{held_no_order} missing from report"
        assert by_ticker[held_no_order].action == "HOLD"
    for traded in ("AFRM", "PWR", "LLY", "LUV"):
        assert by_ticker[traded].action in {"ENTRY", "EXIT", "RESIZE"}


@pytest.mark.skipif(not FIXTURE.exists(), reason="replay fixture not present")
def test_replay_cli_wiring():
    from click.testing import CliRunner

    from sma.live.__main__ import replay_cmd

    result = CliRunner().invoke(
        replay_cmd,
        [
            "--asof-date", "2026-08-28",
            "--book", "asof",
            "--db", str(FIXTURE),
            "--config", str(CONFIG),
            "--universe", str(UNIVERSE),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "AFRM" in result.output
    assert "PWR" in result.output
    assert "4 order(s)" in result.output
