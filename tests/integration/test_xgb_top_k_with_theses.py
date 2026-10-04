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
import logging
from datetime import date, timedelta
from unittest.mock import Mock

import pytest
from loguru import logger as _loguru_logger

from sma.backtest.strategies.xgb_top_k import XGBoostTopKStrategy
from sma.ingest.store import Store
from sma.model.predictor import Predictor


class _PropagateHandler(logging.Handler):
    """Forward loguru records into stdlib ``logging`` so pytest's ``caplog``
    can see them. xgb_top_k.py logs via loguru, which does not propagate to
    stdlib logging by default (loguru's documented caplog-interop recipe)."""

    def emit(self, record: logging.LogRecord) -> None:
        logging.getLogger(record.name).handle(record)


@pytest.fixture
def caplog(caplog):
    handler_id = _loguru_logger.add(_PropagateHandler(), format="{message}")
    caplog.set_level(logging.INFO)
    yield caplog
    _loguru_logger.remove(handler_id)


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


def test_bearish_veto_logs_thesis_veto_line(store, caplog):
    """Vetoing a buy candidate logs a grep-friendly thesis_veto: line with
    ticker + conviction, and notes the slot goes unfilled (no backfill —
    2026-08 study: thesis interventions were unlogged, costing hours to
    reconstruct after the fact)."""
    _seed_thesis(store, "AAPL", conviction="bearish")
    strat = _make_strategy(use_theses=True, store=store)
    strat._apply_thesis_buy_filter(
        candidates=["AAPL", "MSFT"], asof_date=date(2026, 4, 26)
    )
    assert "thesis_veto:" in caplog.text
    assert "AAPL" in caplog.text
    assert "bearish" in caplog.text
    assert "unfilled" in caplog.text or "no backfill" in caplog.text
    # MSFT was not vetoed -> must not appear on a thesis_veto line.
    veto_lines = [line for line in caplog.text.splitlines() if "thesis_veto:" in line]
    assert all("MSFT" not in line for line in veto_lines)


def test_no_veto_no_thesis_veto_log(store, caplog):
    """A candidate with no bearish thesis does not fire thesis_veto:."""
    strat = _make_strategy(use_theses=True, store=store)
    strat._apply_thesis_buy_filter(candidates=["AAPL"], asof_date=date(2026, 4, 26))
    assert "thesis_veto:" not in caplog.text


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


def test_held_strong_bearish_exit_logs_thesis_exit_line(store, caplog):
    """``_held_exits_on_thesis`` is the live-fired exit path inside decide() that
    force-sells a held name. Logs a grep-friendly thesis_exit: line with ticker +
    conviction so the study can reconstruct which held names were exited by a
    thesis without hours of log archaeology."""
    _seed_thesis(store, "AAPL", conviction="strong_bearish")
    strat = _make_strategy(use_theses=True, store=store)
    result = strat._held_exits_on_thesis("AAPL", asof_date=date(2026, 4, 26))
    assert result is True
    assert "thesis_exit:" in caplog.text
    assert "AAPL" in caplog.text
    assert "strong_bearish" in caplog.text


def test_held_without_a_thesis_does_not_exit(store):
    """No thesis for a held name means neutral: it is retained, not churned."""
    _seed_thesis(store, "AAPL", conviction="strong_bearish")
    strat = _make_strategy(use_theses=True, store=store)
    assert strat._held_exits_on_thesis("MSFT", asof_date=date(2026, 4, 26)) is False


def test_held_only_bearish_no_thesis_exit_log(store, caplog):
    """A merely-bearish held name is retained (rule 4), so no thesis_exit:
    line fires -- only strong_bearish force-sells."""
    _seed_thesis(store, "AAPL", conviction="bearish")
    strat = _make_strategy(use_theses=True, store=store)
    result = strat._held_exits_on_thesis("AAPL", asof_date=date(2026, 4, 26))
    assert result is False
    assert "thesis_exit:" not in caplog.text


def test_stale_thesis_falls_back_to_neutral(store):
    """If thesis is >7 calendar days old, no LLM rules fire."""
    _seed_thesis(store, "AAPL", conviction="strong_bearish", days_old=10)
    strat = _make_strategy(use_theses=True, store=store)
    # 1) bearish veto NOT triggered -> AAPL stays in candidates
    out = strat._apply_thesis_buy_filter(["AAPL", "MSFT"], asof_date=date(2026, 4, 26))
    assert "AAPL" in out
    # 2) sell trigger NOT triggered for held ticker
    assert strat._held_exits_on_thesis("AAPL", asof_date=date(2026, 4, 26)) is False
    # 3) tilt NOT applied
    assert strat._tilt_score("AAPL", 0.10, asof_date=date(2026, 4, 26)) == 0.10


def test_use_theses_false_disables_all_rules(store):
    """use_theses=False is the Phase 3 baseline -- no thesis logic fires."""
    _seed_thesis(store, "AAPL", conviction="strong_bearish")
    strat = _make_strategy(use_theses=False, store=store)
    # No thesis_exit ever
    assert strat._held_exits_on_thesis("AAPL", asof_date=date(2026, 4, 26)) is False
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
    assert strat._held_exits_on_thesis("AAPL", asof_date=date(2026, 4, 26)) is False


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
