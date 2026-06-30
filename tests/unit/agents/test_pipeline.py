import json
from datetime import date, timedelta
from unittest.mock import MagicMock

import pytest

from sma.agents.base import (
    AnalystOutput,
    PriceContext,
    ResearcherOutput,
    StrategistOutput,
)
from sma.agents.cost_tracker import BudgetExhausted
from sma.agents.pipeline import CachedThesisFallback, ThesisContext, ThesisPipeline
from sma.ingest.store import Store


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    yield s
    s.close()


def _make_mocks():
    r = MagicMock()
    r.run.return_value = ResearcherOutput(
        news_summary="summary",
        key_developments=["dev1", "dev2"],
        notable_filings=["filing1"],
    )
    a = MagicMock()
    a.run.return_value = AnalystOutput(
        bull_case="bull text",
        bear_case="bear text",
        asymmetric_risks=["risk1"],
        catalyst_window="far",
    )
    s = MagicMock()
    s.run.return_value = StrategistOutput(
        conviction="bullish",
        score=0.5,
        flags=["earnings_beat"],
        action_hint="enter",
        reasoning="strong q",
    )
    return r, a, s


def _ctx(ticker="AAPL", asof=None):
    return ThesisContext(
        ticker=ticker,
        asof_date=asof or date(2026, 4, 26),
        news_rows=[],
        filing_rows=[],
        price_ctx=PriceContext(total_return_30d=0.05),
        sector="Tech",
        next_earnings_date=None,
        held_shares=0,
        quant_predicted_return=0.01,
        quant_universe_rank=5,
        portfolio_sector_exposure_pct=10.0,
    )


def test_pipeline_runs_all_three_agents(store):
    r, a, s = _make_mocks()
    p = ThesisPipeline(researcher=r, analyst=a, strategist=s, store=store)
    p.run(_ctx(), run_id=1)
    r.run.assert_called_once()
    a.run.assert_called_once()
    s.run.assert_called_once()


def test_pipeline_persists_complete_thesis(store):
    r, a, s = _make_mocks()
    p = ThesisPipeline(researcher=r, analyst=a, strategist=s, store=store)
    p.run(_ctx(), run_id=1)
    rows = store.conn.execute(
        """
        SELECT ticker, news_summary, bull_case, conviction, score,
               key_developments, flags, action_hint, reasoning
        FROM theses WHERE ticker = 'AAPL'
        """
    ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row[0] == "AAPL"
    assert row[1] == "summary"
    assert row[2] == "bull text"
    assert row[3] == "bullish"
    assert row[4] == 0.5
    # JSON columns: parse as JSON-strings (DuckDB's JSON type returns str on read)
    key_devs = json.loads(row[5]) if isinstance(row[5], str) else row[5]
    flags = json.loads(row[6]) if isinstance(row[6], str) else row[6]
    assert key_devs == ["dev1", "dev2"]
    assert flags == ["earnings_beat"]
    assert row[7] == "enter"
    assert row[8] == "strong q"


def test_pipeline_returns_cached_thesis_on_budget_exhausted_at_researcher(store):
    """Pre-seed older thesis. Researcher raises BudgetExhausted. Pipeline returns cached row."""
    store.conn.execute(
        """
        INSERT INTO theses (ticker, asof_date, run_id, news_summary, key_developments,
            notable_filings, bull_case, bear_case, asymmetric_risks, catalyst_window,
            conviction, score, flags, action_hint, reasoning)
        VALUES ('AAPL', DATE '2026-04-25', 999, 'old summary', '[]', '[]',
                'old bull', 'old bear', '[]', 'far', 'neutral', 0.0, '[]', 'hold', 'cached')
        """
    )
    r, a, s = _make_mocks()
    r.run.side_effect = BudgetExhausted("test")
    p = ThesisPipeline(researcher=r, analyst=a, strategist=s, store=store)

    out = p.run(_ctx(), run_id=1)

    assert out is not None
    assert out.conviction == "neutral"
    assert out.reasoning == "cached"
    assert isinstance(out, CachedThesisFallback)
    a.run.assert_not_called()
    s.run.assert_not_called()


def test_pipeline_returns_cached_on_budget_at_analyst(store):
    """Cached fallback works regardless of which agent raises."""
    store.conn.execute(
        """
        INSERT INTO theses (ticker, asof_date, run_id, news_summary, key_developments,
            notable_filings, bull_case, bear_case, asymmetric_risks, catalyst_window,
            conviction, score, flags, action_hint, reasoning)
        VALUES ('AAPL', DATE '2026-04-25', 999, 'old', '[]', '[]', 'b', 'b', '[]', 'far',
                'bullish', 0.4, '[]', 'enter', 'cached')
        """
    )
    r, a, s = _make_mocks()
    a.run.side_effect = BudgetExhausted("budget hit at analyst")
    p = ThesisPipeline(researcher=r, analyst=a, strategist=s, store=store)
    out = p.run(_ctx(), run_id=1)
    assert out is not None
    assert out.reasoning == "cached"
    s.run.assert_not_called()


def test_pipeline_returns_none_when_budget_exhausted_and_no_cached_thesis(store):
    """No prior thesis exists — pipeline returns None."""
    r, a, s = _make_mocks()
    r.run.side_effect = BudgetExhausted("test")
    p = ThesisPipeline(researcher=r, analyst=a, strategist=s, store=store)
    out = p.run(_ctx(ticker="ZZZZ"), run_id=1)
    assert out is None


def test_pipeline_picks_most_recent_cached_thesis(store):
    """If multiple prior theses exist, the most recent (by asof_date) is returned."""
    asof_today = date(2026, 4, 26)
    for d_offset, conv, reasoning in [
        (10, "bullish", "ten_days_ago"),
        (3, "neutral", "three_days_ago"),
        (1, "bearish", "yesterday"),
    ]:
        store.conn.execute(
            """
            INSERT INTO theses (ticker, asof_date, run_id, news_summary, key_developments,
                notable_filings, bull_case, bear_case, asymmetric_risks, catalyst_window,
                conviction, score, flags, action_hint, reasoning)
            VALUES ('AAPL', ?, 100, 'x', '[]', '[]', 'b', 'b', '[]', 'far',
                    ?, 0.0, '[]', 'hold', ?)
            """,
            [asof_today - timedelta(days=d_offset), conv, reasoning],
        )
    r, a, s = _make_mocks()
    r.run.side_effect = BudgetExhausted("test")
    p = ThesisPipeline(researcher=r, analyst=a, strategist=s, store=store)
    out = p.run(_ctx(asof=asof_today), run_id=1)
    assert out.conviction == "bearish"
    assert out.reasoning == "yesterday"


def test_pipeline_does_not_return_stale_cached_thesis(store):
    """A cached fallback OLDER than the freshness window must NOT be returned —
    it would masquerade as fresh research and the strategy wouldn't use it
    anyway (2026-06-05 audit). Returns None so the run counts it as skipped."""
    store.conn.execute(
        """
        INSERT INTO theses (ticker, asof_date, run_id, news_summary, key_developments,
            notable_filings, bull_case, bear_case, asymmetric_risks, catalyst_window,
            conviction, score, flags, action_hint, reasoning)
        VALUES ('AAPL', DATE '2026-03-27', 999, 'x', '[]', '[]', 'b', 'b', '[]', 'far',
                'bullish', 0.4, '[]', 'enter', 'stale')
        """
    )
    r, a, s = _make_mocks()
    r.run.side_effect = BudgetExhausted("test")
    p = ThesisPipeline(researcher=r, analyst=a, strategist=s, store=store)
    out = p.run(_ctx(asof=date(2026, 4, 26)), run_id=1)  # 2026-03-27 is ~30d stale
    assert out is None
