"""Golden test: the sleeve path reproduces the incumbent's orders exactly.

Two layers:

1. CI (always runs): against the committed 2026-08-28 replay fixture, the real
   decide builder (`_build_decide_strategy`, a SleeveBook over the default
   config) and the raw incumbent (`_build_strategy`, XGBoostTopKStrategy
   handed straight to decide_once, the pre-sleeve path) produce identical
   orders (ticker, side, shares, type, last price) and identical post-rails
   decisions, on both the reconstructed book and an empty book.

2. Production (opt-in, SMA_GOLDEN_PROD=1 and data/sma.duckdb present):
   tests/fixtures/sleeve_golden_prod_2026_09.json was captured from
   `replay --book asof` and `--book empty` on 9 real decide nights
   (2026-08-24 .. 2026-09-24, 128 orders) at HEAD 2e4e21e, BEFORE the sleeve
   refactor. Re-running replay (now through the sleeve path) must match it
   exactly. Opt-in because the prod DB is gitignored and a later price
   revision could legitimately move a past night. 2026-10-01: the 2026-09-02
   [asof] case was re-pinned when replay stopped seeing fills after asof (the
   capture had leaked LUV's 9/14 buy into that night's min_hold; see the
   fixture's "repinned" note).
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pytest

from sma.config import StrategiesSettings, load_settings
from sma.ingest.universe import load_universe
from sma.live.__main__ import _build_rails, _build_sizing
from sma.live.replay import replay_once
from sma.strategies import registry
from sma.strategies.allocator import SleeveBook
from sma.strategies.base import Strategy, TargetBook

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "replay_2026_08_28.duckdb"
PROD_GOLDEN = REPO_ROOT / "tests" / "fixtures" / "sleeve_golden_prod_2026_09.json"
PROD_DB = REPO_ROOT / "data" / "sma.duckdb"
CONFIG = REPO_ROOT / "config.yaml"
UNIVERSE = REPO_ROOT / "src" / "sma" / "universe.yaml"


def _setup():
    settings = load_settings(config_path=CONFIG)
    return settings, load_universe(UNIVERSE), _build_rails(settings), _build_sizing(settings)


def _replay(asof, db, book, settings, universe, rails, sizing):
    r = replay_once(
        asof=asof, db_path=db, universe=universe, rails=rails, sizing=sizing,
        settings=settings, book=book, use_theses=True,
    )
    assert not r.aborted, r.aborted_reason
    return (
        [[o.ticker, o.side, o.shares, o.type, o.last_price] for o in r.orders],
        [[d.ticker, d.action, d.rail, d.prior_weight, d.target_weight] for d in r.decisions],
    )


def test_default_config_is_incumbent_alone_live_at_full_capital():
    settings = load_settings(config_path=CONFIG)
    assert [s.model_dump() for s in settings.strategies.sleeves] == [
        {"name": "xgb_momentum", "capital_fraction": 1.0, "mode": "live", "enabled": True}
    ]
    assert StrategiesSettings().model_dump() == settings.strategies.model_dump()


@pytest.mark.skipif(not FIXTURE.exists(), reason="replay fixture not present")
def test_decide_builder_is_the_sleeve_path(monkeypatch):
    import sma.live.__main__ as main

    built = {}
    real = main._build_decide_strategy

    def spy(*a, **kw):
        built["s"] = real(*a, **kw)
        return built["s"]

    monkeypatch.setattr(main, "_build_decide_strategy", spy)
    settings, universe, rails, sizing = _setup()
    _replay(date(2026, 8, 28), FIXTURE, "asof", settings, universe, rails, sizing)
    assert isinstance(built["s"], SleeveBook)
    assert [spec.name for spec, _ in built["s"].sleeves] == ["xgb_momentum"]
    assert [p.name for p in built["s"].proposals] == ["xgb_momentum"]


@pytest.mark.skipif(not FIXTURE.exists(), reason="replay fixture not present")
@pytest.mark.parametrize("book", ["asof", "empty"])
def test_sleeve_path_matches_incumbent_path_exactly(book, monkeypatch):
    import sma.live.__main__ as main

    settings, universe, rails, sizing = _setup()
    asof = date(2026, 8, 28)
    sleeve_orders, sleeve_decisions = _replay(
        asof, FIXTURE, book, settings, universe, rails, sizing,
    )

    monkeypatch.setattr(main, "_build_decide_strategy", main._build_strategy)
    inc_orders, inc_decisions = _replay(asof, FIXTURE, book, settings, universe, rails, sizing)

    assert sleeve_orders, "a real run happened"
    assert sleeve_orders == inc_orders
    assert sleeve_decisions == inc_decisions
    if book == "asof":
        # The real recorded 2026-08-28 decide night (see test_replay.py).
        assert {(s, t, n) for t, s, n, _, _ in sleeve_orders} == {
            ("BUY", "AFRM", 25.0), ("BUY", "PWR", 3.0),
            ("SELL", "LLY", 1.0), ("SELL", "LUV", 77.0),
        }


class _GreedyShadow(Strategy):
    """Wants 100% of one name the incumbent never touches."""

    name = "golden_test_shadow"
    session = "open"

    @classmethod
    def from_build(cls, build):
        return cls()

    def propose(self, asof, ctx):
        return TargetBook(weights={"NANC": 1.0})


@pytest.mark.skipif(not FIXTURE.exists(), reason="replay fixture not present")
def test_adding_a_shadow_sleeve_changes_no_orders(monkeypatch):
    settings, universe, rails, sizing = _setup()
    asof = date(2026, 8, 28)
    base = _replay(asof, FIXTURE, "empty", settings, universe, rails, sizing)

    monkeypatch.setitem(registry._REGISTRY, "golden_test_shadow", _GreedyShadow)
    with_shadow = settings.model_copy(update={"strategies": StrategiesSettings(sleeves=[
        {"name": "xgb_momentum", "capital_fraction": 1.0, "mode": "live"},
        {"name": "golden_test_shadow", "capital_fraction": 0.5, "mode": "shadow"},
    ])})
    shadowed = _replay(asof, FIXTURE, "empty", with_shadow, universe, rails, sizing)
    assert shadowed[0] == base[0]
    assert all(t != "NANC" for t, *_ in shadowed[0])


@pytest.mark.skipif(
    os.environ.get("SMA_GOLDEN_PROD") != "1" or not PROD_DB.exists(),
    reason="prod golden is opt-in: SMA_GOLDEN_PROD=1 with data/sma.duckdb present",
)
def test_sleeve_path_matches_pre_refactor_prod_capture():
    golden = json.loads(PROD_GOLDEN.read_text())
    settings, universe, rails, sizing = _setup()
    for case in golden["cases"]:
        orders, decisions = _replay(
            date.fromisoformat(case["asof"]), PROD_DB, case["book"],
            settings, universe, rails, sizing,
        )
        assert orders == case["orders"], f"{case['asof']} [{case['book']}] orders drifted"
        assert decisions == case["decisions"], f"{case['asof']} [{case['book']}] decisions drifted"
