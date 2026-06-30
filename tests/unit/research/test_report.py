"""Tests for the on-demand single-ticker research report renderer.

We seed a fresh DuckDB with rows in each relevant table and assert that
`build_report` produces the markdown sections we expect, with the right
data and the right empty-state strings.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from sma.ingest.store import Store
from sma.research.report import build_report


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(path=tmp_path / "test.duckdb").connect()
    yield s
    s.close()


def _insert_prediction(store: Store, ticker: str, asof: date, value: float, model_id: str = "m1"):
    store.conn.execute(
        "INSERT INTO predictions (asof_date, ticker, target, predicted_value, model_id) "
        "VALUES (?, ?, 'ret_5d', ?, ?)",
        [asof, ticker, value, model_id],
    )


def _insert_price(store: Store, ticker: str, d: date, close: float):
    store.conn.execute(
        "INSERT INTO prices (ticker, date, open, high, low, close, adj_close, "
        "volume, source, run_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 1000, 'test', 1)",
        [ticker, d, close, close, close, close, close],
    )


def _insert_news(store: Store, ticker: str, when: datetime, headline: str, source: str = "Reuters"):
    store.conn.execute(
        "INSERT INTO news (ticker, published_at, source, headline, url, "
        "body_excerpt, hash, run_id) "
        "VALUES (?, ?, ?, ?, 'http://x', NULL, ?, 1)",
        [ticker, when, source, headline, f"{ticker}-{when.isoformat()}"],
    )


def _insert_thesis(
    store: Store, ticker: str, asof: date, *,
    conviction="bullish", score=0.55, run_id: int = 1, reasoning: str | None = None,
):
    store.conn.execute(
        """
        INSERT INTO theses (
            ticker, asof_date, run_id, news_summary, key_developments,
            notable_filings, bull_case, bear_case, asymmetric_risks,
            catalyst_window, conviction, score, flags, action_hint, reasoning
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ticker, asof, run_id, "Strong news cluster.",
            json.dumps(["beat earnings"]), json.dumps([]),
            "Cloud TAM expansion.", "Macro headwind.",
            json.dumps(["concentration risk"]),
            "near", conviction, score, json.dumps(["earnings_beat"]),
            "enter", reasoning or "Combined quant + qualitative.",
        ],
    )


def _insert_politician_trade(store: Store, ticker: str, transaction_date: date):
    store.conn.execute(
        """
        INSERT INTO politician_trades (
            doc_id, chamber, last_name, first_name, state_dst, filing_date,
            transaction_date, ticker, asset_description, asset_type,
            transaction_type, amount_min, amount_max, run_id
        ) VALUES (?, 'house', 'Doe', 'John', 'CA-12', ?, ?, ?,
                  'NVIDIA Corp', 'common_stock', 'purchase', 1001, 15000, 1)
        """,
        [f"d{transaction_date.toordinal()}", transaction_date + timedelta(days=20),
         transaction_date, ticker],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_report_header_and_universe_membership(store):
    rep = build_report(
        store=store, ticker="nvda",  # lowercase input
        asof=date(2026, 4, 30),
        universe=["NVDA", "AAPL"],
        sector="Information Technology",
        position=None,
    )
    md = rep.to_markdown()
    assert "# NVDA — research report" in md  # uppercase normalisation
    assert "**As of**: 2026-04-30" in md
    assert "**Sector**: Information Technology" in md
    assert "**In universe**: yes" in md


def test_report_marks_ticker_outside_universe(store):
    rep = build_report(
        store=store, ticker="ZZZZ",
        asof=date(2026, 4, 30),
        universe=["NVDA"],
        sector=None,
        position=None,
    )
    md = rep.to_markdown()
    assert "**In universe**: no" in md
    assert "**Sector**: unknown" in md


def test_quant_section_uses_latest_prediction_and_price_ctx(store):
    asof = date(2026, 4, 30)
    _insert_prediction(store, "NVDA", asof - timedelta(days=2), 0.034, model_id="old")
    _insert_prediction(store, "NVDA", asof, 0.0123, model_id="new")
    # Two prices that produce a +5% return.
    _insert_price(store, "NVDA", asof - timedelta(days=20), 100.0)
    _insert_price(store, "NVDA", asof, 105.0)
    rep = build_report(
        store=store, ticker="NVDA", asof=asof,
        universe=["NVDA"], sector="Tech", position=None,
    )
    quant = next(s for s in rep.sections if s.title == "Quant signal")
    assert "+0.0123" in quant.body  # latest prediction value
    assert "model `new`" in quant.body
    assert "**Last close**" in quant.body
    assert "$105.00" in quant.body
    assert "+5.00%" in quant.body  # 100 -> 105


def test_quant_section_handles_no_prediction(store):
    asof = date(2026, 4, 30)
    rep = build_report(
        store=store, ticker="ZZZZ", asof=asof,
        universe=[], sector=None, position=None,
    )
    quant = next(s for s in rep.sections if s.title == "Quant signal")
    assert "No XGBoost prediction" in quant.body
    assert "No price rows" in quant.body


def test_news_section_lists_recent_headlines(store):
    asof = date(2026, 4, 30)
    _insert_news(store, "NVDA", datetime(2026, 4, 29, 10, 0), "Blackwell ramps")
    _insert_news(store, "NVDA", datetime(2026, 4, 25, 12, 0), "Cloud capex up")
    _insert_news(store, "NVDA", datetime(2026, 3, 1, 10, 0), "Old, should not appear")
    rep = build_report(
        store=store, ticker="NVDA", asof=asof,
        universe=["NVDA"], sector="Tech", position=None,
    )
    news = next(s for s in rep.sections if s.title.startswith("Recent news"))
    assert "Blackwell ramps" in news.body
    assert "Cloud capex up" in news.body
    assert "should not appear" not in news.body


def test_news_section_empty_state(store):
    rep = build_report(
        store=store, ticker="NVDA", asof=date(2026, 4, 30),
        universe=["NVDA"], sector="Tech", position=None,
    )
    news = next(s for s in rep.sections if s.title.startswith("Recent news"))
    assert "No news rows" in news.body


def test_theses_section_renders_latest_full_plus_older_headlines(store):
    asof = date(2026, 4, 30)
    _insert_thesis(store, "NVDA", asof, conviction="bullish", score=0.62)
    _insert_thesis(store, "NVDA", asof - timedelta(days=7), conviction="neutral", score=0.05)
    rep = build_report(
        store=store, ticker="NVDA", asof=asof,
        universe=["NVDA"], sector="Tech", position=None,
    )
    theses = next(s for s in rep.sections if s.title == "LLM theses")
    assert "Latest thesis" in theses.body
    assert "**Conviction**: `bullish`" in theses.body
    assert "Cloud TAM expansion" in theses.body
    assert "Earlier theses" in theses.body
    assert "neutral" in theses.body


def test_theses_section_empty_state_suggests_refresh(store):
    rep = build_report(
        store=store, ticker="NVDA", asof=date(2026, 4, 30),
        universe=["NVDA"], sector="Tech", position=None,
    )
    theses = next(s for s in rep.sections if s.title == "LLM theses")
    assert "--refresh" in theses.body


def test_politician_section_lists_recent_trades(store):
    asof = date(2026, 4, 30)
    _insert_politician_trade(store, "NVDA", asof - timedelta(days=10))
    _insert_politician_trade(store, "NVDA", asof - timedelta(days=200))  # outside window
    rep = build_report(
        store=store, ticker="NVDA", asof=asof,
        universe=["NVDA"], sector="Tech", position=None,
    )
    politicians = next(s for s in rep.sections if s.title.startswith("Politician"))
    assert "Doe, John" in politicians.body
    assert "purchase" in politicians.body
    # 200 days back was filtered out — only the 10-day-old row should appear.
    assert politicians.body.count("Doe, John") == 1


def test_politician_section_empty_state(store):
    rep = build_report(
        store=store, ticker="NVDA", asof=date(2026, 4, 30),
        universe=["NVDA"], sector="Tech", position=None,
    )
    politicians = next(s for s in rep.sections if s.title.startswith("Politician"))
    assert "No politician disclosures" in politicians.body


def test_position_section_when_present(store):
    pos = {
        "qty": 50, "avg_entry_price": 100.0, "market_value": 5250.0,
        "unrealized_pl": 250.0, "unrealized_plpc": 0.05,
    }
    rep = build_report(
        store=store, ticker="NVDA", asof=date(2026, 4, 30),
        universe=["NVDA"], sector="Tech", position=pos,
    )
    position = next(s for s in rep.sections if s.title == "Current position")
    assert "**Shares**: `50`" in position.body
    assert "$100.00" in position.body
    assert "+$250.00" in position.body
    assert "+5.00%" in position.body


def test_position_section_empty_when_no_position(store):
    rep = build_report(
        store=store, ticker="NVDA", asof=date(2026, 4, 30),
        universe=["NVDA"], sector="Tech", position=None,
    )
    position = next(s for s in rep.sections if s.title == "Current position")
    assert "No open position" in position.body


def test_theses_section_picks_newest_run_id_for_same_asof(store):
    """Regression for codex finding LOW 3: --refresh creates a new (ticker,
    asof_date, run_id) row, not a replacement. The renderer must show the
    LATEST run, not an arbitrary one. Tie-break by run_id DESC."""
    asof = date(2026, 4, 30)
    _insert_thesis(
        store, "NVDA", asof, run_id=1, conviction="neutral", score=0.05,
        reasoning="STALE: original nightly thesis",
    )
    _insert_thesis(
        store, "NVDA", asof, run_id=99, conviction="bullish", score=0.7,
        reasoning="FRESH: --refresh override",
    )
    rep = build_report(
        store=store, ticker="NVDA", asof=asof,
        universe=["NVDA"], sector="Tech", position=None,
    )
    theses = next(s for s in rep.sections if s.title == "LLM theses")
    # Latest (run_id=99) is FRESH bullish, not stale neutral.
    assert "FRESH: --refresh override" in theses.body
    assert "STALE" not in theses.body
    assert "**Conviction**: `bullish`" in theses.body


def test_to_markdown_produces_all_five_sections(store):
    rep = build_report(
        store=store, ticker="NVDA", asof=date(2026, 4, 30),
        universe=["NVDA"], sector="Tech", position=None,
    )
    md = rep.to_markdown()
    for title in [
        "## Quant signal",
        "## Current position",
        "## Recent news",
        "## LLM theses",
        "## Politician trades",
    ]:
        assert title in md, f"missing section: {title}"
