from datetime import date
from unittest.mock import MagicMock

import pytest

from sma.agents.analyst import Analyst
from sma.agents.base import AnalystInput, PriceContext, ResearcherOutput
from sma.agents.cost_tracker import CostTracker
from sma.ingest.store import Store


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    yield s
    s.close()


def _stub(bull="bull case here", bear="bear case here", risks=None, window="far"):
    m = MagicMock()
    m.usage.input_tokens = 700
    m.usage.output_tokens = 280
    m.usage.cache_read_input_tokens = 250
    tu = MagicMock()
    tu.type = "tool_use"
    tu.name = "submit_analysis"
    tu.input = {
        "bull_case": bull,
        "bear_case": bear,
        "asymmetric_risks": risks if risks is not None else [],
        "catalyst_window": window,
    }
    m.content = [tu]
    m.stop_reason = "tool_use"
    return m


def _make_analyst(store, response):
    client = MagicMock()
    client.messages.create.return_value = response
    ct = CostTracker(client=client, store=store, daily_budget_usd=1.0)
    return Analyst(cost_tracker=ct, model="claude-haiku-4-5-20251001"), client


def _basic_input(**overrides):
    base = dict(
        ticker="AAPL", asof_date=date(2026, 4, 26),
        research=ResearcherOutput(
            news_summary="strong quarter",
            key_developments=["beat earnings"],
            notable_filings=[],
        ),
        price_ctx=PriceContext(total_return_30d=0.05, max_drawdown_30d=-0.02),
    )
    base.update(overrides)
    return AnalystInput(**base)


def test_analyst_parses_response(store):
    a, _ = _make_analyst(store, _stub())
    out = a.run(_basic_input(), run_id=1)
    assert out.bull_case == "bull case here"
    assert out.bear_case == "bear case here"
    assert out.catalyst_window == "far"


def test_analyst_uses_submit_analysis_tool(store):
    a, client = _make_analyst(store, _stub())
    a.run(_basic_input(), run_id=1)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["tools"][0]["name"] == "submit_analysis"
    assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_analysis"}


def test_analyst_uses_prompt_caching(store):
    a, client = _make_analyst(store, _stub())
    a.run(_basic_input(), run_id=1)
    kwargs = client.messages.create.call_args.kwargs
    sys_block = kwargs["system"]
    assert isinstance(sys_block, list)
    assert sys_block[0]["cache_control"] == {"type": "ephemeral"}


def test_analyst_handles_no_earnings_date(store):
    """next_earnings_date=None should not break prompt assembly."""
    a, _ = _make_analyst(store, _stub())
    out = a.run(_basic_input(next_earnings_date=None), run_id=1)
    assert out is not None


def _stub_no_tool_use():
    m = MagicMock()
    m.usage.input_tokens = 700
    m.usage.output_tokens = 280
    m.usage.cache_read_input_tokens = 250
    tb = MagicMock()
    tb.type = "text"  # model replied in text, no tool_use block
    m.content = [tb]
    m.stop_reason = "end_turn"
    return m


def test_analyst_raises_clear_error_when_no_tool_use(store):
    """If the model returns no tool_use block (stop_reason=end_turn — observed with
    Haiku occasionally despite forced tool_choice), the agent must raise a clear,
    NAMED error — not a bare StopIteration that the pipeline's broad `except`
    swallows, silently dropping the ticker with no thesis and no explanation."""
    a, _ = _make_analyst(store, _stub_no_tool_use())
    with pytest.raises(ValueError, match="tool_use"):
        a.run(_basic_input(), run_id=1)


def test_analyst_passes_research_into_user_message(store):
    """The user message must contain the bullet from researcher's key_developments."""
    a, client = _make_analyst(store, _stub())
    a.run(_basic_input(), run_id=1)
    user_msg = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "beat earnings" in user_msg
    assert "AAPL" in user_msg


def test_analyst_rejects_bad_catalyst_window(store):
    """If LLM returns a bogus catalyst window, AnalystOutput validation raises."""
    a, _ = _make_analyst(store, _stub(window="someday"))
    with pytest.raises(Exception):  # Pydantic ValidationError
        a.run(_basic_input(), run_id=1)


def test_analyst_persists_telemetry(store):
    a, _ = _make_analyst(store, _stub())
    a.run(_basic_input(), run_id=99)
    rows = store.conn.execute(
        "SELECT agent_role, run_id, status FROM agent_calls"
    ).fetchall()
    assert rows[0] == ("analyst", 99, "ok")
