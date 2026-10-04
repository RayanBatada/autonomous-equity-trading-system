from datetime import date
from unittest.mock import Mock

import pandas as pd
import pytest

from sma.backtest.strategies.xgb_top_k import XGBoostTopKStrategy
from sma.model.predictor import Predictor


def _fake_predictor(scores: dict) -> Mock:
    p = Mock(spec=Predictor)
    p.predict_for = Mock(return_value=scores)
    return p


UNIVERSE = ["AAA", "BBB", "CCC", "DDD", "EEE"]
SCORES = {"AAA": 0.10, "BBB": 0.05, "CCC": 0.02, "DDD": -0.01, "EEE": -0.05}
ASOF = date(2024, 6, 1)
EMPTY_PRICES = pd.DataFrame()


def test_picks_top_k_by_predicted_score():
    # weight_tilt=False to preserve the legacy flat-weight assertion below.
    # The tilted variant is exercised by test_weight_tilt_scales_by_rank.
    strat = XGBoostTopKStrategy(
        predictor=_fake_predictor(SCORES), universe=UNIVERSE, k=3,
        target_weight_per_position=0.05, weight_tilt=False,
    )
    decisions = strat.decide(ASOF, EMPTY_PRICES)
    assert len(decisions) == 3
    tickers = [d.ticker for d in decisions]
    assert tickers == ["AAA", "BBB", "CCC"]
    for d in decisions:
        assert d.target_weight == 0.05


def test_returns_fewer_than_k_when_predictor_returns_fewer():
    scores = {"AAA": 0.10, "BBB": 0.05}
    strat = XGBoostTopKStrategy(predictor=_fake_predictor(scores), universe=["AAA", "BBB"], k=10)
    decisions = strat.decide(ASOF, EMPTY_PRICES)
    assert len(decisions) == 2


def test_returns_empty_when_predictor_returns_empty():
    strat = XGBoostTopKStrategy(predictor=_fake_predictor({}), universe=UNIVERSE, k=3)
    decisions = strat.decide(ASOF, EMPTY_PRICES)
    assert decisions == []


def test_returns_empty_when_no_model_available():
    p = Mock(spec=Predictor)
    p.predict_for = Mock(side_effect=FileNotFoundError("no model"))
    strat = XGBoostTopKStrategy(predictor=p, universe=UNIVERSE, k=3)
    decisions = strat.decide(ASOF, EMPTY_PRICES)
    assert decisions == []


def test_tilt_error_swallowed_by_default_keeps_live_safe(monkeypatch):
    """Live safety (default): a crashing tilt() must NOT take down decide() —
    it logs and falls back to the untilted decisions."""
    import sma.strategy.active as active

    def boom(*, asof_date, decisions, ctx):
        raise TypeError(
            "StrategyDecision.__init__() got an unexpected keyword argument "
            "'signal_metadata'"
        )

    monkeypatch.setattr(active, "tilt", boom)
    strat = XGBoostTopKStrategy(
        predictor=_fake_predictor(SCORES), universe=UNIVERSE, k=3,
        target_weight_per_position=0.05, weight_tilt=False,
    )
    decisions = strat.decide(ASOF, EMPTY_PRICES)  # must not raise
    assert [d.ticker for d in decisions] == ["AAA", "BBB", "CCC"]


def test_tilt_error_propagates_when_strict(monkeypatch):
    """Autoresearch eval (tilt_strict=True): a crashing tilt() must propagate
    so the experiment is recorded as eval_error — NOT silently swallowed and
    scored as the untilted baseline (which made every broken proposal look
    like a valid no-improvement result)."""
    import sma.strategy.active as active

    def boom(*, asof_date, decisions, ctx):
        raise TypeError(
            "StrategyDecision.__init__() got an unexpected keyword argument "
            "'signal_metadata'"
        )

    monkeypatch.setattr(active, "tilt", boom)
    strat = XGBoostTopKStrategy(
        predictor=_fake_predictor(SCORES), universe=UNIVERSE, k=3,
        target_weight_per_position=0.05, weight_tilt=False,
        tilt_strict=True,
    )
    with pytest.raises(TypeError, match="signal_metadata"):
        strat.decide(ASOF, EMPTY_PRICES)


def test_decision_asof_date_matches_input():
    strat = XGBoostTopKStrategy(predictor=_fake_predictor(SCORES), universe=UNIVERSE, k=3)
    asof = date(2024, 9, 15)
    decisions = strat.decide(asof, EMPTY_PRICES)
    for d in decisions:
        assert d.asof_date == asof


def test_decision_target_weight_uses_construction_value():
    # weight_tilt=False so the assertion against a flat 0.10 holds.
    strat = XGBoostTopKStrategy(
        predictor=_fake_predictor(SCORES),
        universe=UNIVERSE,
        k=3,
        target_weight_per_position=0.10,
        weight_tilt=False,
    )
    decisions = strat.decide(ASOF, EMPTY_PRICES)
    for d in decisions:
        assert d.target_weight == 0.10


def test_init_rejects_invalid_k():
    with pytest.raises(ValueError):
        XGBoostTopKStrategy(predictor=Mock(spec=Predictor), universe=["AAA"], k=0)
    with pytest.raises(ValueError):
        XGBoostTopKStrategy(predictor=Mock(spec=Predictor), universe=["AAA"], k=-5)


def test_init_rejects_invalid_weight():
    for bad in [0.0, -0.01, 1.01]:
        with pytest.raises(ValueError):
            XGBoostTopKStrategy(
                predictor=Mock(spec=Predictor),
                universe=["AAA"],
                k=1,
                target_weight_per_position=bad,
            )


def test_decisions_sum_within_one_for_default_k_and_weight():
    # 25 tickers, k=20, weight=0.05 -> top 20 selected, sum = 1.0
    # weight_tilt=False to preserve the flat-weight sum assertion.
    scores = {f"T{i:02d}": float(i) for i in range(25)}
    universe = list(scores.keys())
    strat = XGBoostTopKStrategy(
        predictor=_fake_predictor(scores),
        universe=universe,
        k=20,
        target_weight_per_position=0.05,
        weight_tilt=False,
    )
    decisions = strat.decide(ASOF, EMPTY_PRICES)
    assert len(decisions) == 20
    total = sum(d.target_weight for d in decisions)
    assert abs(total - 1.0) < 1e-9


def test_weight_tilt_scales_by_rank():
    """v1 manual tilt, gross-preserving (audit xgb_top_k:415): the rank shape
    is linear from target down to floor_ratio*target, renormalized toward
    sum == k*target and CLAMPED at target (pre-fix the mean was ~0.7x target
    → ~70% gross; the risk pipeline rejects above-target weights)."""
    strat = XGBoostTopKStrategy(
        predictor=_fake_predictor({"AAA": 0.10, "BBB": 0.05, "CCC": 0.01}),
        universe=["AAA", "BBB", "CCC"],
        k=3,
        target_weight_per_position=0.10,
        weight_tilt=True,
        weight_tilt_floor_ratio=0.4,
    )
    decisions = strat.decide(ASOF, EMPTY_PRICES)
    by_ticker = {d.ticker: d.target_weight for d in decisions}
    scale = 0.30 / 0.21
    assert by_ticker["AAA"] == pytest.approx(0.10)  # clamped at target
    assert by_ticker["CCC"] == pytest.approx(0.04 * scale)
    assert by_ticker["BBB"] == pytest.approx(min(0.07 * scale, 0.10))


def test_weight_tilt_with_single_pick_uses_full_weight():
    """Edge case: only one pick → no rank to interpolate. Use target_weight."""
    strat = XGBoostTopKStrategy(
        predictor=_fake_predictor({"AAA": 0.05}),
        universe=["AAA"],
        k=1,
        target_weight_per_position=0.10,
        weight_tilt=True,
    )
    decisions = strat.decide(ASOF, EMPTY_PRICES)
    assert decisions[0].target_weight == pytest.approx(0.10)


def test_politician_flow_overlay_promotes_accumulated_demotes_distributed():
    """When the politician_trades table shows net positive flow on AAA
    and net negative flow on BBB, the overlay multiplies AAA's score up
    and BBB's score down. With initially-equal scores, AAA should rank
    higher after the overlay is applied."""
    from unittest.mock import MagicMock

    # Equal raw quant scores so ranking is determined by the overlay.
    pred = _fake_predictor({"AAA": 0.05, "BBB": 0.05, "CCC": 0.05})

    # Mock store.conn returning a flow table:
    #   AAA: +$50,000 (full +20% tilt)
    #   BBB: -$50,000 (full -20% tilt)
    #   CCC: $0 (no tilt)
    store = MagicMock()
    store.conn.execute.return_value.fetchall.return_value = [
        ("AAA", 50_000.0),
        ("BBB", -50_000.0),
    ]

    strat = XGBoostTopKStrategy(
        predictor=pred,
        universe=["AAA", "BBB", "CCC"],
        k=3,
        target_weight_per_position=0.10,
        weight_tilt=True,
        weight_tilt_floor_ratio=0.4,
        use_politician_flow=True,
        politician_flow_max_tilt=0.20,
        politician_flow_normalizer=50_000.0,
        store=store,
    )

    decisions = strat.decide(ASOF, EMPTY_PRICES)
    tickers = [d.ticker for d in decisions]
    # AAA's score got tilted up (+20%), BBB down (-20%); ranking flips:
    # AAA > CCC (no tilt) > BBB.
    assert tickers == ["AAA", "CCC", "BBB"]
    # Gross-recovering tilt (audit :415): renorm by 0.30/0.21 then clamp at target.
    by_ticker = {d.ticker: d.target_weight for d in decisions}
    assert by_ticker["AAA"] == pytest.approx(0.10)  # clamped
    assert by_ticker["BBB"] == pytest.approx(0.04 * 0.30 / 0.21)


def test_politician_flow_overlay_disabled_when_store_missing():
    """If `store` is None or `use_politician_flow=False`, ranking is
    unaffected by politician data (legacy behavior)."""
    pred = _fake_predictor({"AAA": 0.05, "BBB": 0.04, "CCC": 0.03})
    strat = XGBoostTopKStrategy(
        predictor=pred,
        universe=["AAA", "BBB", "CCC"],
        k=3,
        target_weight_per_position=0.10,
        weight_tilt=False,
        use_politician_flow=False,
        store=None,
    )
    decisions = strat.decide(ASOF, EMPTY_PRICES)
    assert [d.ticker for d in decisions] == ["AAA", "BBB", "CCC"]


def test_hysteresis_keeps_held_buffer_name_instead_of_churning():
    """A held name still ranked in the buffer (k < rank < hold_rank) is KEPT,
    not churned for a marginally-better new name. ranked A>B>C>D>E, k=2,
    hold_rank=4: plain top-k is {A,B}; holding C (rank 2, in buffer) keeps C and
    fills the other slot with A — NOT B. Cuts turnover on a 30-day signal."""
    scores = {"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0}
    strat = XGBoostTopKStrategy(
        predictor=_fake_predictor(scores), universe=["A", "B", "C", "D", "E"],
        k=2, hold_rank=4,
    )
    base = {d.ticker for d in strat.decide(ASOF, EMPTY_PRICES)}
    assert base == {"A", "B"}  # no holdings -> plain top-k
    held = {d.ticker for d in strat.decide(ASOF, EMPTY_PRICES, current_holdings={"C"})}
    assert held == {"A", "C"}  # keep C (buffer), buy A, NOT B


def test_weight_tilt_preserves_total_gross():
    """audit xgb_top_k:415 — the tilt averaged ~0.7x target so a full top-K
    deployed only ~70% gross (live cash drag). The tilt must preserve
    sum == k * target_weight (rails clip any single name at max_position_pct)."""
    strat = XGBoostTopKStrategy(
        predictor=_fake_predictor({"AAA": 0.10, "BBB": 0.05, "CCC": 0.01}),
        universe=["AAA", "BBB", "CCC"],
        k=3,
        target_weight_per_position=0.10,
        weight_tilt=True,
        weight_tilt_floor_ratio=0.4,
    )
    decisions = strat.decide(ASOF, EMPTY_PRICES)
    weights = [d.target_weight for d in decisions]
    # renorm-then-clamp: raw (0.10, 0.07, 0.04) * 0.30/0.21, clamped at target
    # -> (0.10, 0.10, 0.0571): ~86% gross (vs 70% pre-fix), never above target
    # (the risk pipeline REJECTS, not clips, above-cap decisions).
    assert sum(weights) == pytest.approx(0.10 + 0.10 + 0.04 * 0.30 / 0.21, abs=1e-9)
    assert sum(weights) > 0.21, "must deploy more gross than the un-normalized tilt"
    assert weights[0] >= weights[1] >= weights[2], "rank shape is non-increasing"
    assert max(weights) <= 0.10 + 1e-12, "never exceed target (rails would reject)"


def test_bullish_tilt_improves_negative_scores():
    """audit xgb_top_k:242 — base_score * 1.05 made NEGATIVE scores MORE
    negative, demoting bullish names. The tilt must move any score UP."""
    strat = XGBoostTopKStrategy(
        predictor=_fake_predictor({}), universe=["AAA"], k=1, use_theses=True,
    )
    strat.store = Mock()
    strat._recent_thesis = Mock(return_value={"conviction": "bullish"})
    assert strat._tilt_score("AAA", -0.02, ASOF) > -0.02
    assert strat._tilt_score("AAA", 0.02, ASOF) > 0.02


def test_politician_flow_tilt_is_sign_safe():
    """Same bug class as the bullish tilt: score *= (1+signal) inverts the
    overlay's meaning on negative scores (buy-flow made them MORE negative)."""
    strat = XGBoostTopKStrategy(
        predictor=_fake_predictor({"AAA": -0.045, "BBB": -0.05}),
        universe=["AAA", "BBB"],
        k=1,
        weight_tilt=False,
        use_politician_flow=True,
        store=Mock(),
    )
    # Strong BUY flow on AAA must lift it above BBB even though both are negative.
    strat._fetch_politician_flow = Mock(return_value={"AAA": 1e12})
    decisions = strat.decide(ASOF, EMPTY_PRICES)
    assert decisions[0].ticker == "AAA", (
        "positive politician flow on a negative score must IMPROVE the score"
    )


def test_politician_flow_overlay_windows_on_filing_date_and_excludes_options():
    """Codex module review (2026-06-11 HIGH): the overlay windowed on
    transaction_date (trades disclose ~58d later → non-public info in
    backtests) and didn't exclude options — diverging from the model's
    Predictor._fetch_politician_flows contract. The SQL must window on
    filing_date and filter asset_type."""
    import inspect

    from sma.backtest.strategies import xgb_top_k as mod

    src = inspect.getsource(mod.XGBoostTopKStrategy._fetch_politician_flow)
    assert "filing_date >= ?" in src and "filing_date <= ?" in src
    assert "transaction_date >= ?" not in src
    assert "'Stock Option'" in src


def test_tilt_context_gets_real_sector_mapping():
    """Strategy review 2026-06-11: the Phase-6 TiltContext was built with
    sector_for=lambda: 'Unknown', blinding every autoresearch tilt() proposal
    to sectors — one reason that search surface never found anything. The
    context must carry the real sma.sectors mapping."""
    captured = {}

    def fake_tilt(*, asof_date, decisions, ctx):
        captured["sector_nvda"] = ctx.sector_for("NVDA")
        return decisions

    import sma.strategy.active as active_mod
    orig = active_mod.tilt
    active_mod.tilt = fake_tilt
    try:
        strat = XGBoostTopKStrategy(
            predictor=_fake_predictor({"NVDA": 0.05}), universe=["NVDA"], k=1,
            weight_tilt=False,
        )
        strat.decide(ASOF, EMPTY_PRICES)
    finally:
        active_mod.tilt = orig
    assert captured["sector_nvda"] not in (None, "Unknown"), (
        "tilt() must see real sectors, not the 'Unknown' stub"
    )


# ---------------------------------------------------------------------------
# Entry conviction floor (min_score) — 2026-07-01. Default None = off.
# ---------------------------------------------------------------------------
def test_conviction_floor_drops_subfloor_names_and_holds_cash():
    """min_score keeps only new-buy candidates whose RAW score clears the bar,
    so weak names are skipped (fewer decisions = cash held)."""
    strat = XGBoostTopKStrategy(
        predictor=_fake_predictor(SCORES), universe=UNIVERSE, k=5,
        target_weight_per_position=0.05, min_score=0.03,
    )
    decisions = strat.decide(ASOF, EMPTY_PRICES)
    tickers = {d.ticker for d in decisions}
    # Only AAA (0.10) and BBB (0.05) clear 0.03; CCC/DDD/EEE are held out → cash.
    assert tickers == {"AAA", "BBB"}, tickers
    assert len(decisions) == 2


def test_conviction_floor_none_is_a_noop():
    """Default min_score=None buys the full top-K (no names filtered)."""
    strat = XGBoostTopKStrategy(
        predictor=_fake_predictor(SCORES), universe=UNIVERSE, k=5,
        target_weight_per_position=0.05, min_score=None,
    )
    assert len(strat.decide(ASOF, EMPTY_PRICES)) == 5


def test_conviction_floor_above_all_holds_full_cash():
    """A floor above every score buys nothing (holds cash), no crash."""
    strat = XGBoostTopKStrategy(
        predictor=_fake_predictor(SCORES), universe=UNIVERSE, k=5,
        target_weight_per_position=0.05, min_score=0.5,
    )
    assert strat.decide(ASOF, EMPTY_PRICES) == []


def test_conviction_floor_rejects_non_finite_scores():
    """Bug: a bare `>= min_score` rejects NaN by accident (NaN compares False)
    but lets a +inf score THROUGH and buys it. The gate must require a FINITE
    score: +inf and NaN are both rejected; a finite score above the floor still
    passes."""
    scores = {
        "AAA": float("inf"),   # degenerate model output → must be rejected
        "BBB": float("nan"),   # must be rejected (as before)
        "CCC": 0.10,           # finite, above the floor → kept
        "DDD": 0.01,           # finite, below the floor → dropped
    }
    strat = XGBoostTopKStrategy(
        predictor=_fake_predictor(scores), universe=["AAA", "BBB", "CCC", "DDD"],
        k=4, target_weight_per_position=0.05, min_score=0.03,
    )
    tickers = {d.ticker for d in strat.decide(ASOF, EMPTY_PRICES)}
    assert tickers == {"CCC"}, f"+inf/NaN leaked through the min_score gate; {tickers}"


def test_conviction_floor_none_lets_inf_through_unchanged():
    """The isfinite guard is scoped to the min_score gate: with min_score=None
    the gate is skipped entirely, so an +inf score is NOT filtered here (default
    behavior unchanged — inf handling is only the floor's concern)."""
    scores = {"AAA": float("inf"), "BBB": 0.05}
    strat = XGBoostTopKStrategy(
        predictor=_fake_predictor(scores), universe=["AAA", "BBB"],
        k=2, target_weight_per_position=0.05, min_score=None,
    )
    tickers = {d.ticker for d in strat.decide(ASOF, EMPTY_PRICES)}
    assert tickers == {"AAA", "BBB"}, tickers


def test_conviction_floor_is_entry_only_holds_are_exempt():
    """A HELD name below the floor is retained (this is an entry gate, not an
    exit): CCC (0.02, sub-floor) stays because it's already held."""
    strat = XGBoostTopKStrategy(
        predictor=_fake_predictor(SCORES), universe=UNIVERSE, k=5,
        target_weight_per_position=0.05, min_score=0.03,
    )
    decisions = strat.decide(ASOF, EMPTY_PRICES, current_holdings={"CCC"})
    tickers = {d.ticker for d in decisions}
    assert "CCC" in tickers, f"held sub-floor name was wrongly dropped; {tickers}"
    # New buys still floor-gated: only AAA + BBB join CCC.
    assert tickers == {"AAA", "BBB", "CCC"}, tickers
