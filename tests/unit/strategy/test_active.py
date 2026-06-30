"""Tests for sma.strategy.active — the Phase 6 autoresearch surface.

These tests pin the contract of `tilt()` and `TiltContext`. The autoresearch
loop runner relies on this signature being stable across iterations; if the
agent breaks it, these tests catch the break before any candidate gets
considered for promotion.
"""

from datetime import date
from inspect import signature

import pytest

from sma.backtest.strategies.base import StrategyDecision
from sma.strategy.active import TiltContext, tilt


def _ctx(**overrides) -> TiltContext:
    defaults = dict(
        quant_scores={"AAPL": 0.05, "MSFT": 0.04},
        theses=None,
        portfolio_dollars={},
        sector_exposure={},
        sector_for=lambda _t: "TECH",
        account_equity=100_000.0,
        cash=100_000.0,
    )
    defaults.update(overrides)
    return TiltContext(**defaults)


def test_default_tilt_is_identity():
    """v0 default body: return decisions unchanged."""
    decisions = [
        StrategyDecision(asof_date=date(2026, 5, 9), ticker="AAPL", target_weight=0.05),
        StrategyDecision(asof_date=date(2026, 5, 9), ticker="MSFT", target_weight=0.05),
    ]
    out = tilt(asof_date=date(2026, 5, 9), decisions=decisions, ctx=_ctx())
    assert out == decisions


def test_tilt_signature_is_stable():
    """The autoresearch loop runner enforces the signature stays put.
    Lock it here so any future edit that drifts the signature breaks
    a unit test before the loop runner has a chance to catch it."""
    sig = signature(tilt)
    params = sig.parameters
    # All three params are keyword-only, in this exact order.
    assert list(params.keys()) == ["asof_date", "decisions", "ctx"]
    assert all(p.kind == p.KEYWORD_ONLY for p in params.values()), (
        "tilt() params must remain keyword-only"
    )


def test_tilt_context_is_frozen():
    """Mutating ctx between iterations would let an agent leak state."""
    ctx = _ctx()
    with pytest.raises((AttributeError, Exception)):  # noqa: PT011 — frozen dataclass
        ctx.cash = 0  # type: ignore[misc]


def test_tilt_context_required_fields():
    """Required keys per spec; if they ever drift, the loop's prompt
    template breaks. Pin the field set here."""
    expected = {
        "quant_scores", "theses", "portfolio_dollars",
        "sector_exposure", "sector_for",
        "account_equity", "cash",
    }
    actual = set(TiltContext.__dataclass_fields__.keys())
    assert actual == expected, (
        f"TiltContext fields drifted: missing={expected - actual}, "
        f"extra={actual - expected}"
    )
