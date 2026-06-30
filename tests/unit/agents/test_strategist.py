from datetime import date
from unittest.mock import MagicMock

import pytest

from sma.agents.base import AnalystOutput, StrategistInput
from sma.agents.cost_tracker import CostTracker
from sma.agents.strategist import Strategist
from sma.ingest.store import Store


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    yield s
    s.close()


def _stub(conviction="neutral", score=0.0, flags=None, action_hint="hold", reasoning="x"):
    m = MagicMock()
    m.usage.input_tokens = 600
    m.usage.output_tokens = 150
    m.usage.cache_read_input_tokens = 250
    tu = MagicMock()
    tu.type = "tool_use"
    tu.name = "submit_strategy"
    tu.input = {
        "conviction": conviction, "score": score,
        "flags": flags if flags is not None else [],
        "action_hint": action_hint, "reasoning": reasoning,
    }
    m.content = [tu]
    m.stop_reason = "tool_use"
    return m


def _make_strategist(store, response):
    client = MagicMock()
    client.messages.create.return_value = response
    ct = CostTracker(client=client, store=store, daily_budget_usd=1.0)
    return Strategist(cost_tracker=ct, model="claude-haiku-4-5-20251001"), client


def _basic_input(held=0, **overrides):
    base = dict(
        ticker="AAPL", asof_date=date(2026, 4, 26),
        analysis=AnalystOutput(
            bull_case="strong q earnings",
            bear_case="weak guidance",
            asymmetric_risks=[],
            catalyst_window="far",
        ),
        held_shares=held,
    )
    base.update(overrides)
    return StrategistInput(**base)


def test_strategist_parses_response(store):
    s_a, _ = _make_strategist(store,
        _stub(conviction="bullish", score=0.6, flags=["earnings_beat"]))
    out = s_a.run(_basic_input(), run_id=1)
    assert out.conviction == "bullish"
    assert out.flags == ["earnings_beat"]
    assert out.score == 0.6


def test_strategist_filters_unknown_flags(store):
    """Pydantic validator on StrategistOutput drops unknown flags."""
    s_a, _ = _make_strategist(store,
        _stub(flags=["earnings_beat", "made_up_flag"]))
    out = s_a.run(_basic_input(), run_id=1)
    assert out.flags == ["earnings_beat"]


def test_strategist_clamps_score(store):
    """Pydantic validator clamps to [-1, +1]."""
    s_a, _ = _make_strategist(store, _stub(score=2.5, conviction="strong_bullish"))
    out = s_a.run(_basic_input(), run_id=1)
    assert out.score == 1.0


def test_strategist_uses_submit_strategy_tool(store):
    s_a, client = _make_strategist(store, _stub())
    s_a.run(_basic_input(), run_id=1)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["tools"][0]["name"] == "submit_strategy"
    assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_strategy"}


def test_strategist_uses_prompt_caching(store):
    s_a, client = _make_strategist(store, _stub())
    s_a.run(_basic_input(), run_id=1)
    sys_block = client.messages.create.call_args.kwargs["system"]
    assert sys_block[0]["cache_control"] == {"type": "ephemeral"}


def test_strategist_user_message_includes_held_shares(store):
    """Strategist needs to know held position size to make exit/reduce hints sensible."""
    s_a, client = _make_strategist(store, _stub())
    s_a.run(_basic_input(held=100), run_id=1)
    user_msg = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "100" in user_msg
    # And quant context too:
    assert "AAPL" in user_msg


def test_strategist_persists_telemetry(store):
    s_a, _ = _make_strategist(store, _stub())
    s_a.run(_basic_input(), run_id=77)
    rows = store.conn.execute(
        "SELECT agent_role, run_id, status FROM agent_calls"
    ).fetchall()
    assert rows[0] == ("strategist", 77, "ok")
