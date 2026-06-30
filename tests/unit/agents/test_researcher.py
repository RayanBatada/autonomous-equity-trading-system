from datetime import date, datetime
from unittest.mock import MagicMock

import pytest

from sma.agents.base import NewsRow, ResearcherInput
from sma.agents.cost_tracker import CostTracker
from sma.agents.researcher import Researcher
from sma.ingest.store import Store


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    yield s
    s.close()


def _stub_response(news_summary="apple had a quiet week",
                    developments=None, filings=None,
                    input_tok=1500, output_tok=200, cache_read_tok=300):
    msg = MagicMock()
    msg.usage.input_tokens = input_tok
    msg.usage.output_tokens = output_tok
    msg.usage.cache_read_input_tokens = cache_read_tok
    tu = MagicMock()
    tu.type = "tool_use"
    tu.name = "submit_research"
    tu.input = {
        "news_summary": news_summary,
        "key_developments": developments if developments is not None else ["dev one"],
        "notable_filings": filings if filings is not None else [],
    }
    msg.content = [tu]
    msg.stop_reason = "tool_use"
    return msg


def _make_researcher(store, response):
    client = MagicMock()
    client.messages.create.return_value = response
    ct = CostTracker(client=client, store=store, daily_budget_usd=1.0)
    return Researcher(cost_tracker=ct, model="claude-haiku-4-5-20251001"), client


def test_researcher_parses_tool_use_response(store):
    r, client = _make_researcher(store, _stub_response())
    out = r.run(ResearcherInput(
        ticker="AAPL",
        asof_date=date(2026, 4, 26),
        news_rows=[NewsRow(
            published_at=datetime(2026, 4, 25),
            source="alpaca:benzinga",
            headline="Apple posts strong quarter",
            body_excerpt="strong holiday sales",
        )],
        filing_rows=[],
    ), run_id=1)
    assert out.news_summary == "apple had a quiet week"
    assert out.key_developments == ["dev one"]


def test_researcher_uses_correct_tool_definition(store):
    r, client = _make_researcher(store, _stub_response())
    r.run(ResearcherInput(
        ticker="AAPL", asof_date=date(2026, 4, 26),
        news_rows=[], filing_rows=[],
    ), run_id=1)
    kwargs = client.messages.create.call_args.kwargs
    tools = kwargs["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "submit_research"
    assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_research"}
    assert kwargs["model"] == "claude-haiku-4-5-20251001"


def test_researcher_uses_prompt_caching_on_system(store):
    r, client = _make_researcher(store, _stub_response())
    r.run(ResearcherInput(ticker="AAPL", asof_date=date(2026, 4, 26),
                            news_rows=[], filing_rows=[]), run_id=1)
    kwargs = client.messages.create.call_args.kwargs
    sys_block = kwargs["system"]
    assert isinstance(sys_block, list)
    assert sys_block[0]["cache_control"] == {"type": "ephemeral"}
    assert sys_block[0]["type"] == "text"
    assert isinstance(sys_block[0]["text"], str)
    assert len(sys_block[0]["text"]) > 50  # non-trivial prompt


def test_researcher_handles_no_news_gracefully(store):
    r, client = _make_researcher(store, _stub_response(news_summary="(no recent news)",
                                                          developments=[]))
    out = r.run(ResearcherInput(
        ticker="ZZZZ", asof_date=date(2026, 4, 26),
        news_rows=[], filing_rows=[],
    ), run_id=1)
    assert out.news_summary == "(no recent news)"
    assert out.key_developments == []


def test_researcher_truncates_news_to_top_15_recent(store):
    """Only the 15 most-recent headlines are sent to the model."""
    r, client = _make_researcher(store, _stub_response())
    rows = [
        NewsRow(
            published_at=datetime(2026, 4, 25, h, 0),
            source="x",
            headline=f"news {h}",
            body_excerpt="body",
        )
        for h in range(20)
    ]
    r.run(ResearcherInput(ticker="AAPL", asof_date=date(2026, 4, 26),
                            news_rows=rows, filing_rows=[]), run_id=1)
    user_msg = client.messages.create.call_args.kwargs["messages"][0]["content"]
    # At most 15 headlines should appear
    headline_count = user_msg.count("headline:")
    assert headline_count == 15, f"expected 15 headlines, got {headline_count}"
    # The OLDEST 5 entries (hours 0-4) should be DROPPED; newest first
    assert "news 5" in user_msg  # oldest of the kept 15
    assert "news 4" not in user_msg  # dropped
    assert "news 19" in user_msg  # newest


def test_researcher_persists_telemetry_via_cost_tracker(store):
    """The CostTracker writes a row to agent_calls — verify integration."""
    r, _ = _make_researcher(store, _stub_response())
    r.run(ResearcherInput(ticker="AAPL", asof_date=date(2026, 4, 26),
                            news_rows=[], filing_rows=[]), run_id=42)
    rows = store.conn.execute(
        "SELECT agent_role, ticker, run_id, status FROM agent_calls"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0] == ("researcher", "AAPL", 42, "ok")
