"""Sleeve framework: TargetBook, registry, allocator validation + combiner,
shadow isolation through decide_once."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from sma.backtest.strategies.base import StrategyDecision
from sma.ingest.store import Store
from sma.live.decide import decide_once
from sma.risk.rails import RiskRails
from sma.strategies import registry
from sma.strategies.allocator import (
    SleeveBook,
    build_sleeve_book,
    combine_notional,
    combine_weights,
    validate_sleeves,
)
from sma.strategies.base import (
    SleeveContext,
    Strategy,
    TargetBook,
    book_from_decisions,
    call_strategy_decide,
)

ASOF = date(2026, 4, 30)


@dataclass
class Spec:
    name: str
    capital_fraction: float
    mode: str = "live"
    enabled: bool = True


class FixedSleeve(Strategy):
    name = "fixed_test"
    session = "open"
    horizon_days = 1

    def __init__(self, weights, *, raise_exc=None):
        self.weights = weights
        self.raise_exc = raise_exc
        self.calls = 0

    def propose(self, asof, ctx):
        self.calls += 1
        if self.raise_exc:
            raise self.raise_exc
        return TargetBook(weights=dict(self.weights))


# ---- TargetBook -----------------------------------------------------------


def test_target_book_gross_and_validate():
    b = TargetBook(weights={"A": 0.6, "B": 0.4})
    assert b.gross == pytest.approx(1.0)
    b.validate()


@pytest.mark.parametrize("weights", [{"A": -0.1}, {"A": float("nan")}, {"A": 0.7, "B": 0.4}])
def test_target_book_rejects_bad_weights(weights):
    with pytest.raises(ValueError):
        TargetBook(weights=weights).validate()


def test_target_book_max_gross_none_skips_gross_check():
    TargetBook(weights={"A": 0.7, "B": 0.6}).validate(max_gross=None)


def test_target_book_limit_hint_must_reference_a_weight():
    with pytest.raises(ValueError):
        TargetBook(weights={"A": 0.5}, limit_hint={"B": 10.0}).validate()


def test_book_from_decisions_preserves_order_and_rejects_duplicates():
    ds = [StrategyDecision(ASOF, t, w) for t, w in [("Z", 0.1), ("A", 0.2), ("M", 0.0)]]
    assert list(book_from_decisions(ds).weights.items()) == [("Z", 0.1), ("A", 0.2), ("M", 0.0)]
    with pytest.raises(ValueError):
        book_from_decisions(ds + [StrategyDecision(ASOF, "A", 0.3)])


def test_call_strategy_decide_passes_holdings_only_when_supported():
    class New:
        def decide(self, *, asof_date, prices, current_holdings=None):
            return current_holdings

    class Old:
        def decide(self, *, asof_date, prices):
            return "old"

    assert call_strategy_decide(New(), asof=ASOF, prices=None, current_holdings={"X"}) == {"X"}
    assert call_strategy_decide(Old(), asof=ASOF, prices=None, current_holdings={"X"}) == "old"


# ---- registry -------------------------------------------------------------


def test_registry_knows_xgb_momentum():
    assert "xgb_momentum" in registry.names()
    cls = registry.get("xgb_momentum")
    assert cls.session == "open"


def test_registry_unknown_name_raises():
    with pytest.raises(KeyError, match="unknown sleeve"):
        registry.get("no_such_sleeve")


def test_registry_rejects_unknown_session():
    class Bad(Strategy):
        name = "bad_session_test"
        session = "lunch"

        def propose(self, asof, ctx):
            return TargetBook({})

    with pytest.raises(ValueError, match="session"):
        registry.register(Bad)


# ---- allocator validation -------------------------------------------------


def test_validate_default_single_live_sleeve_ok():
    validate_sleeves([Spec("xgb_momentum", 1.0)])


def test_validate_live_fractions_over_one_fail():
    with pytest.raises(ValueError, match="sum to"):
        validate_sleeves([Spec("a", 0.7), Spec("b", 0.4)])


def test_validate_shadow_and_disabled_do_not_count_toward_one():
    validate_sleeves([
        Spec("a", 1.0), Spec("b", 1.0, mode="shadow"), Spec("c", 1.0, enabled=False),
    ])


@pytest.mark.parametrize(
    "specs",
    [
        [Spec("a", 1.0), Spec("a", 0.0, mode="shadow")],  # duplicate name
        [Spec("a", 1.0, mode="paper")],                   # unknown mode
        [Spec("a", 1.5)],                                 # fraction > 1
        [Spec("a", 1.0, mode="shadow")],                  # no live sleeve
    ],
)
def test_validate_rejects(specs):
    with pytest.raises(ValueError):
        validate_sleeves(specs)


# ---- combiner -------------------------------------------------------------


def test_combine_single_sleeve_at_one_is_exact_identity():
    w = {"A": 0.1, "B": 0.0971428571428571, "C": 0.057142857142857}
    out = combine_weights([(1.0, TargetBook(weights=w))])
    assert list(out.items()) == list(w.items())  # exact floats, same order


def test_combine_nets_overlapping_sleeves():
    out = combine_weights([
        (0.6, TargetBook(weights={"A": 0.5, "B": 0.5})),
        (0.4, TargetBook(weights={"B": 0.25, "C": 0.75})),
    ])
    assert list(out) == ["A", "B", "C"]
    assert out["A"] == pytest.approx(0.30)
    assert out["B"] == pytest.approx(0.40)
    assert out["C"] == pytest.approx(0.30)


def test_combine_notional_uses_equity():
    out = combine_notional([(0.5, TargetBook(weights={"A": 0.5}))], equity=100_000.0)
    assert out == {"A": pytest.approx(25_000.0)}


# ---- SleeveBook -----------------------------------------------------------


def test_sleeve_book_shadow_contributes_nothing_and_is_recorded():
    live = FixedSleeve({"A": 0.2})
    shadow = FixedSleeve({"S": 0.9})
    sb = SleeveBook([(Spec("live1", 1.0), live), (Spec("sh", 1.0, mode="shadow"), shadow)])
    ds = sb.decide(asof_date=ASOF, prices=pd.DataFrame(), current_holdings=set())
    assert [(d.ticker, d.target_weight) for d in ds] == [("A", 0.2)]
    assert shadow.calls == 1
    by_name = {p.name: p for p in sb.proposals}
    assert by_name["sh"].mode == "shadow"
    assert by_name["sh"].book.weights == {"S": 0.9}


def test_sleeve_book_shadow_failure_never_breaks_live():
    sb = SleeveBook([
        (Spec("live1", 1.0), FixedSleeve({"A": 0.2})),
        (Spec("sh", 0.0, mode="shadow"), FixedSleeve({}, raise_exc=RuntimeError("boom"))),
    ])
    ds = sb.decide(asof_date=ASOF, prices=pd.DataFrame(), current_holdings=set())
    assert [d.ticker for d in ds] == ["A"]
    sh = next(p for p in sb.proposals if p.name == "sh")
    assert sh.book is None and "boom" in sh.error


def test_sleeve_book_live_failure_propagates():
    sb = SleeveBook([(Spec("live1", 1.0), FixedSleeve({}, raise_exc=RuntimeError("boom")))])
    with pytest.raises(RuntimeError):
        sb.decide(asof_date=ASOF, prices=pd.DataFrame(), current_holdings=set())


def test_sleeve_book_rejects_oversized_new_sleeve_book():
    sb = SleeveBook([(Spec("live1", 1.0), FixedSleeve({"A": 0.8, "B": 0.8}))])
    with pytest.raises(ValueError, match="gross"):
        sb.decide(asof_date=ASOF, prices=pd.DataFrame(), current_holdings=set())


def test_sleeve_book_passes_holdings_to_sleeves():
    seen = {}

    class Spy(FixedSleeve):
        def propose(self, asof, ctx: SleeveContext):
            seen["h"] = ctx.current_holdings
            return TargetBook({"A": 0.1})

    SleeveBook([(Spec("live1", 1.0), Spy({}))]).decide(
        asof_date=ASOF, prices=pd.DataFrame(), current_holdings={"H1", "H2"},
    )
    assert seen["h"] == frozenset({"H1", "H2"})


def test_sleeve_book_skips_other_sessions():
    class Close(FixedSleeve):
        session = "close"

    close = Close({"C": 0.5})
    sb = SleeveBook([(Spec("live1", 1.0), FixedSleeve({"A": 0.2})),
                     (Spec("cl", 0.0, mode="shadow"), close)])
    sb.decide(asof_date=ASOF, prices=pd.DataFrame(), current_holdings=set())
    assert close.calls == 0


def test_build_sleeve_book_refuses_live_sleeve_outside_open_session(monkeypatch):
    class Close(FixedSleeve):
        name = "close_test_sleeve"
        session = "close"

        @classmethod
        def from_build(cls, build):
            return cls({"C": 0.5})

    monkeypatch.setitem(registry._REGISTRY, "close_test_sleeve", Close)

    @dataclass
    class Cfg:
        sleeves: list

    with pytest.raises(ValueError, match="open session"):
        build_sleeve_book(Cfg([Spec("close_test_sleeve", 1.0)]), build_ctx=None)


def test_build_sleeve_book_skips_disabled(monkeypatch):
    built = []

    class Fx(FixedSleeve):
        name = "fx_test_sleeve"

        @classmethod
        def from_build(cls, build):
            built.append(1)
            return cls({"A": 0.1})

    monkeypatch.setitem(registry._REGISTRY, "fx_test_sleeve", Fx)

    @dataclass
    class Cfg:
        sleeves: list

    sb = build_sleeve_book(
        Cfg([Spec("fx_test_sleeve", 1.0, enabled=True)]), build_ctx=None,
    )
    assert len(sb.sleeves) == 1 and built == [1]


# ---- through decide_once: a shadow sleeve produces zero orders -------------


def _alpaca():
    a = MagicMock()
    a.get_account.return_value = {"equity": 100_000.0, "cash": 100_000.0,
                                  "trading_blocked": False, "account_blocked": False}
    a.get_positions.return_value = {}
    return a


def _seed_prices(store, tickers):
    for t in tickers:
        store.conn.execute(
            "INSERT INTO prices (ticker, date, open, high, low, close, adj_close, volume, "
            "source, run_id) VALUES (?, ?, 10, 10, 10, 10, 10, 1000000, 'yfinance', 1)",
            [t, ASOF],
        )


def _run(tmp_path, strategy):
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = Store(path=tmp_path / "t.duckdb").connect()
    try:
        _seed_prices(store, ["AAA", "BBB", "SHD"])
        return decide_once(
            asof=ASOF, store=store, alpaca=_alpaca(), universe=["AAA", "BBB", "SHD"],
            strategy=strategy, sector_for=lambda t: t, rails=RiskRails(), dry_run=True,
        )
    finally:
        store.close()


def test_shadow_sleeve_generates_zero_orders_through_decide(tmp_path):
    live_only = SleeveBook([(Spec("live1", 1.0), FixedSleeve({"AAA": 0.04, "BBB": 0.03}))])
    with_shadow = SleeveBook([
        (Spec("live1", 1.0), FixedSleeve({"AAA": 0.04, "BBB": 0.03})),
        (Spec("sh", 1.0, mode="shadow"), FixedSleeve({"SHD": 0.05})),
    ])
    r1 = _run(tmp_path / "a", live_only)
    r2 = _run(tmp_path / "b", with_shadow)
    o1 = [(o.ticker, o.side, o.shares) for o in r1.orders]
    o2 = [(o.ticker, o.side, o.shares) for o in r2.orders]
    assert o1 and o1 == o2
    assert all(o.ticker != "SHD" for o in r2.orders)
    assert any(p.name == "sh" and p.book.weights == {"SHD": 0.05} for p in with_shadow.proposals)
