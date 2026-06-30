"""Phase 4 Task 11: XGBoostTopKStrategy theses-integration tests.

Verifies the four asymmetric C-rules layered on top of the Phase 2/3
quant strategy when ``use_theses=True``:

1. Bearish veto on buy candidates.
2. Bullish tilt: 1.05x score multiplier.
3. Outside-top-30 override: strong_bullish + catalyst flag only.
4. Strong-bearish exit trigger for held positions.

When ``use_theses=False`` (Phase 3 baseline) NONE of these rules fire.
"""

import json
from datetime import date, timedelta
from unittest.mock import Mock

import pytest

from sma.backtest.strategies.xgb_top_k import XGBoostTopKStrategy
from sma.ingest.store import Store
from sma.model.predictor import Predictor


def _fake_predictor() -> Mock:
    p = Mock(spec=Predictor)
    p.predict_for = Mock(return_value={})
    return p


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    yield s
    s.close()


def _seed_thesis(store, ticker, conviction, flags=None,
                 asof=date(2026, 4, 26), days_old=0):
    eff_asof = asof - timedelta(days=days_old)
    store.conn.execute(
        """
        INSERT INTO theses (ticker, asof_date, run_id,
            news_summary, key_developments, notable_filings,
            bull_case, bear_case, asymmetric_risks, catalyst_window,
            conviction, score, flags, action_hint, reasoning)
        VALUES (?, ?, 1, '', '[]', '[]', '', '', '[]', 'far',
                ?, 0.0, ?, 'hold', '')
        """,
        [ticker, eff_asof, conviction, json.dumps(flags or [])],
    )


def _make_strategy(use_theses, store=None):
    return XGBoostTopKStrategy(
        predictor=_fake_predictor(),
        universe=["AAPL", "MSFT", "GOOGL", "ZZZ", "YYY", "XXX", "WWW"],
        k=20,
        use_theses=use_theses,
        store=store,
    )


def test_bearish_veto_drops_top_30_buy_candidate(store):
    """Rule 1: bearish thesis on a top-30 candidate skips the buy."""
    _seed_thesis(store, "AAPL", conviction="bearish")
    strat = _make_strategy(use_theses=True, store=store)
    candidates_in = ["AAPL", "MSFT", "GOOGL"]
    out = strat._apply_thesis_buy_filter(
        candidates=candidates_in,
        asof_date=date(2026, 4, 26),
    )
    assert "AAPL" not in out
    assert "MSFT" in out
    assert "GOOGL" in out


def test_bullish_tilt_boosts_quant_score(store):
    """Rule 2: bullish thesis multiplies quant score by 1.05."""
    _seed_thesis(store, "AAPL", conviction="bullish")
    strat = _make_strategy(use_theses=True, store=store)
    base = 0.10
    tilted = strat._tilt_score("AAPL", base, asof_date=date(2026, 4, 26))
    assert tilted == pytest.approx(base * 1.05)


def test_strong_bullish_also_tilts(store):
    """Rule 2 (continued): strong_bullish also gets the same 1.05 tilt."""
    _seed_thesis(store, "AAPL", conviction="strong_bullish")
    strat = _make_strategy(use_theses=True, store=store)
    assert strat._tilt_score("AAPL", 0.10, asof_date=date(2026, 4, 26)) == pytest.approx(0.105)


def test_neutral_or_bearish_no_tilt(store):
    """Bearish/neutral thesis does NOT tilt (avoid double penalty after veto)."""
    _seed_thesis(store, "AAPL", conviction="bearish")
    _seed_thesis(store, "MSFT", conviction="neutral")
    strat = _make_strategy(use_theses=True, store=store)
    assert strat._tilt_score("AAPL", 0.10, asof_date=date(2026, 4, 26)) == 0.10
    assert strat._tilt_score("MSFT", 0.10, asof_date=date(2026, 4, 26)) == 0.10


def test_outside_top_30_override_strong_bullish_with_catalyst(store):
    """Rule 3: ticker outside top-30 enters via override if strong_bullish + catalyst flag."""
    _seed_thesis(store, "ZZZ", conviction="strong_bullish", flags=["earnings_beat"])
    _seed_thesis(store, "YYY", conviction="bullish", flags=["earnings_beat"])  # not strong
    _seed_thesis(store, "XXX", conviction="strong_bullish", flags=[])  # no catalyst
    # bad flag (not in catalyst-allow set)
    _seed_thesis(store, "WWW", conviction="strong_bullish", flags=["litigation_material"])
    strat = _make_strategy(use_theses=True, store=store)
    eligible = strat._candidates_via_override(
        outside_top_30=["ZZZ", "YYY", "XXX", "WWW"],
        asof_date=date(2026, 4, 26),
    )
    assert "ZZZ" in eligible
    assert "YYY" not in eligible
    assert "XXX" not in eligible
    assert "WWW" not in eligible


def test_strong_bearish_on_held_triggers_thesis_exit(store):
    """Rule 4: held + strong_bearish -> 'thesis_exit' decision."""
    _seed_thesis(store, "AAPL", conviction="strong_bearish")
    strat = _make_strategy(use_theses=True, store=store)
    sells = strat._thesis_exit_decisions(held={"AAPL", "MSFT"}, asof_date=date(2026, 4, 26))
    actions = {d["ticker"]: d["action"] for d in sells}
    assert actions.get("AAPL") == "thesis_exit"
    assert "MSFT" not in actions  # no thesis = no exit


def test_held_with_only_bearish_does_not_exit(store):
    """Rule 4: only strong_bearish triggers exit; plain bearish does not."""
    _seed_thesis(store, "AAPL", conviction="bearish")
    strat = _make_strategy(use_theses=True, store=store)
    sells = strat._thesis_exit_decisions(held={"AAPL"}, asof_date=date(2026, 4, 26))
    assert sells == []


def test_stale_thesis_falls_back_to_neutral(store):
    """If thesis is >7 calendar days old, no LLM rules fire."""
    _seed_thesis(store, "AAPL", conviction="strong_bearish", days_old=10)
    strat = _make_strategy(use_theses=True, store=store)
    # 1) bearish veto NOT triggered -> AAPL stays in candidates
    out = strat._apply_thesis_buy_filter(["AAPL", "MSFT"], asof_date=date(2026, 4, 26))
    assert "AAPL" in out
    # 2) sell trigger NOT triggered for held ticker
    sells = strat._thesis_exit_decisions(held={"AAPL"}, asof_date=date(2026, 4, 26))
    assert sells == []
    # 3) tilt NOT applied
    assert strat._tilt_score("AAPL", 0.10, asof_date=date(2026, 4, 26)) == 0.10


def test_use_theses_false_disables_all_rules(store):
    """use_theses=False is the Phase 3 baseline -- no thesis logic fires."""
    _seed_thesis(store, "AAPL", conviction="strong_bearish")
    strat = _make_strategy(use_theses=False, store=store)
    # No thesis_exit ever
    assert strat._thesis_exit_decisions(held={"AAPL"}, asof_date=date(2026, 4, 26)) == []
    # Tilt is identity
    assert strat._tilt_score("AAPL", 0.10, asof_date=date(2026, 4, 26)) == 0.10
    # Buy filter is identity
    assert strat._apply_thesis_buy_filter(["AAPL"], asof_date=date(2026, 4, 26)) == ["AAPL"]


def test_no_thesis_at_all_acts_neutral(store):
    """When no thesis row exists for a ticker, treat as neutral conviction."""
    strat = _make_strategy(use_theses=True, store=store)
    # No theses seeded; ticker passes through filters unchanged.
    assert strat._apply_thesis_buy_filter(["AAPL"], asof_date=date(2026, 4, 26)) == ["AAPL"]
    assert strat._tilt_score("AAPL", 0.10, asof_date=date(2026, 4, 26)) == 0.10
    assert strat._thesis_exit_decisions(held={"AAPL"}, asof_date=date(2026, 4, 26)) == []


# ---- 2026-06-05 audit: bearish veto must not force-sell merely-bearish holds ---


def _scoring_predictor(scores):
    p = Mock(spec=Predictor)
    p.predict_for = Mock(return_value=scores)
    return p


def _empty_prices():
    import pandas as pd
    return pd.DataFrame()


def _theses_strategy(store, scores):
    return XGBoostTopKStrategy(
        predictor=_scoring_predictor(scores),
        universe=["AAPL", "MSFT", "GOOGL"],
        k=20,
        use_theses=True,
        store=store,
    )


_SCORES = {"AAPL": 0.5, "MSFT": 0.4, "GOOGL": 0.3}


def test_decide_keeps_merely_bearish_held_name(store):
    """A HELD name with a merely-`bearish` thesis must be RETAINED — only
    strong_bearish triggers an exit. (Old code force-sold it via the buy filter.)"""
    _seed_thesis(store, "AAPL", conviction="bearish")
    out = _theses_strategy(store, _SCORES).decide(
        date(2026, 4, 26), _empty_prices(), current_holdings={"AAPL"}
    )
    assert "AAPL" in {d.ticker for d in out}


def test_decide_drops_strong_bearish_held_name(store):
    """A HELD name with strong_bearish DOES exit (dropped from decisions)."""
    _seed_thesis(store, "AAPL", conviction="strong_bearish")
    out = _theses_strategy(store, _SCORES).decide(
        date(2026, 4, 26), _empty_prices(), current_holdings={"AAPL"}
    )
    assert "AAPL" not in {d.ticker for d in out}


def test_decide_drops_bearish_new_buy(store):
    """A NEW buy with a (merely) bearish thesis is still vetoed — don't buy weakness."""
    _seed_thesis(store, "AAPL", conviction="bearish")
    out = _theses_strategy(store, _SCORES).decide(
        date(2026, 4, 26), _empty_prices(), current_holdings=set()
    )
    assert "AAPL" not in {d.ticker for d in out}
