from datetime import date
from unittest.mock import MagicMock

import pytest

from sma.agents.cost_tracker import AgentCallResult, BudgetExhausted, CostTracker, _estimate_cost
from sma.ingest.store import Store


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    yield s
    s.close()


def _stub_anthropic_response(input_tok=100, output_tok=50, cache_read_tok=20):
    """Minimal Anthropic Message stub matching the .usage interface."""
    msg = MagicMock()
    msg.usage.input_tokens = input_tok
    msg.usage.output_tokens = output_tok
    msg.usage.cache_read_input_tokens = cache_read_tok
    msg.usage.cache_creation_input_tokens = 0
    # content with one tool_use block (matches what the agents will use)
    tool_use = MagicMock(type="tool_use", input={"news_summary": "x"})
    msg.content = [tool_use]
    msg.stop_reason = "tool_use"
    return msg


def test_cost_tracker_persists_telemetry(store):
    client = MagicMock()
    client.messages.create.return_value = _stub_anthropic_response()
    ct = CostTracker(client=client, store=store, daily_budget_usd=1.0)

    result = ct.invoke(
        run_id=1, ticker="AAPL", asof_date=date(2026, 4, 26),
        agent_role="researcher", model="claude-haiku-4-5-20251001",
        request_kwargs={"max_tokens": 500, "messages": []},
    )

    rows = store.conn.execute(
        "SELECT agent_role, ticker, status, est_cost_usd FROM agent_calls"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "researcher"
    assert rows[0][1] == "AAPL"
    assert rows[0][2] == "ok"
    assert rows[0][3] > 0
    assert isinstance(result, AgentCallResult)


def test_cost_tracker_blocks_when_budget_exhausted(store):
    """Pre-seed agent_calls with $1.50 today; ceiling is $1.00; invoke raises BudgetExhausted."""
    store.conn.execute(
        """
        INSERT INTO agent_calls (run_id, ticker, asof_date, agent_role, model_id,
            input_tokens, output_tokens, cache_read_tokens, est_cost_usd, latency_ms, status)
        VALUES (1, 'AAPL', CURRENT_DATE, 'researcher', 'claude-haiku-4-5-20251001',
                1000, 100, 0, 1.50, 100, 'ok')
        """
    )
    client = MagicMock()
    ct = CostTracker(client=client, store=store, daily_budget_usd=1.0)

    with pytest.raises(BudgetExhausted):
        ct.invoke(
            run_id=2, ticker="MSFT", asof_date=date(2026, 4, 26),
            agent_role="researcher", model="claude-haiku-4-5-20251001",
            request_kwargs={"max_tokens": 500, "messages": []},
        )

    # Anthropic was NOT called
    assert client.messages.create.call_count == 0
    # An "exhausted" telemetry row was logged
    skip_rows = store.conn.execute(
        "SELECT status FROM agent_calls WHERE status = 'budget_exhausted_skipped'"
    ).fetchall()
    assert len(skip_rows) == 1


def test_cost_tracker_reserves_before_call_when_near_budget(store):
    """A call at $0.49/$0.50 must not start and then push spend over budget."""
    store.conn.execute(
        """
        INSERT INTO agent_calls (run_id, ticker, asof_date, agent_role, model_id,
            input_tokens, output_tokens, cache_read_tokens, est_cost_usd, latency_ms, status)
        VALUES (1, 'AAPL', CURRENT_DATE, 'researcher', 'claude-haiku-4-5-20251001',
                1000, 100, 0, 0.49, 100, 'ok')
        """
    )
    client = MagicMock()
    ct = CostTracker(client=client, store=store, daily_budget_usd=0.50)

    with pytest.raises(BudgetExhausted):
        ct.invoke(
            run_id=2, ticker="MSFT", asof_date=date(2026, 4, 26),
            agent_role="researcher", model="claude-haiku-4-5-20251001",
            request_kwargs={"max_tokens": 500, "messages": []},
        )

    assert client.messages.create.call_count == 0
    rows = store.conn.execute(
        "SELECT status, est_cost_usd FROM agent_calls WHERE status = 'budget_exhausted_skipped'"
    ).fetchall()
    assert rows == [("budget_exhausted_skipped", 0.0)]


def test_cost_tracker_settles_reservation_to_actual_cost(store):
    client = MagicMock()
    client.messages.create.return_value = _stub_anthropic_response(
        input_tok=1000, output_tok=500, cache_read_tok=200
    )
    ct = CostTracker(client=client, store=store, daily_budget_usd=1.0)

    res = ct.invoke(
        run_id=1, ticker="AAPL", asof_date=date(2026, 4, 26),
        agent_role="researcher", model="claude-haiku-4-5-20251001",
        request_kwargs={"max_tokens": 500, "messages": []},
    )

    rows = store.conn.execute(
        "SELECT status, est_cost_usd FROM agent_calls WHERE ticker = 'AAPL'"
    ).fetchall()
    assert rows == [("ok", res.est_cost_usd)]
    assert abs(res.est_cost_usd - 0.00352) < 1e-5


def test_cost_tracker_haiku_pricing_correct(store):
    """1000 input + 200 cache-read + 500 output at Haiku 4.5 prices.

    input_tokens is ALREADY uncached (Anthropic reports the buckets separately),
    so it is NOT reduced by cache_read:
    cost = 1000 * $1/MTok + 200 * $0.10/MTok + 500 * $5/MTok
         = $0.001 + $0.00002 + $0.0025
         = $0.00352
    """
    client = MagicMock()
    client.messages.create.return_value = _stub_anthropic_response(
        input_tok=1000, output_tok=500, cache_read_tok=200
    )
    ct = CostTracker(client=client, store=store, daily_budget_usd=1.0)
    res = ct.invoke(
        run_id=1, ticker="AAPL", asof_date=date(2026, 4, 26),
        agent_role="researcher", model="claude-haiku-4-5-20251001",
        request_kwargs={"max_tokens": 500, "messages": []},
    )
    assert abs(res.est_cost_usd - 0.00352) < 1e-5


def test_cost_tracker_records_error_on_anthropic_exception(store):
    """If Anthropic call raises, status='error' row is persisted and exception re-raised."""
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("boom")
    ct = CostTracker(client=client, store=store, daily_budget_usd=1.0)

    with pytest.raises(RuntimeError):
        ct.invoke(
            run_id=1, ticker="AAPL", asof_date=date(2026, 4, 26),
            agent_role="researcher", model="claude-haiku-4-5-20251001",
            request_kwargs={"max_tokens": 500, "messages": []},
        )

    rows = store.conn.execute("SELECT status, error FROM agent_calls").fetchall()
    assert rows[0][0] == "error"
    assert "boom" in rows[0][1]


def test_cost_tracker_only_counts_today_for_budget(store):
    """Yesterday's spend doesn't count against today's ceiling."""
    store.conn.execute(
        """
        INSERT INTO agent_calls (run_id, ticker, asof_date, agent_role, model_id,
            input_tokens, output_tokens, cache_read_tokens, est_cost_usd, latency_ms,
            status, created_at)
        VALUES (1, 'AAPL', CURRENT_DATE - INTERVAL 1 DAY, 'researcher',
                'claude-haiku-4-5-20251001', 1000, 100, 0, 5.0, 100, 'ok',
                CURRENT_TIMESTAMP - INTERVAL 1 DAY)
        """
    )
    client = MagicMock()
    client.messages.create.return_value = _stub_anthropic_response()
    ct = CostTracker(client=client, store=store, daily_budget_usd=1.0)

    # Should NOT raise - yesterday's $5 spend doesn't count against today's $1 ceiling.
    ct.invoke(
        run_id=2, ticker="MSFT", asof_date=date(2026, 4, 26),
        agent_role="researcher", model="claude-haiku-4-5-20251001",
        request_kwargs={"max_tokens": 500, "messages": []},
    )
    assert client.messages.create.call_count == 1


def test_estimate_cost_helper_haiku():
    """Pure-function pricing helper, separate from the class."""
    cost = _estimate_cost(input_tok=1000, output_tok=500, cache_read_tok=200,
                            cache_write_tok=0, model="claude-haiku-4-5-20251001")
    assert abs(cost - 0.00352) < 1e-5


def test_estimate_cost_zero_inputs():
    assert _estimate_cost(0, 0, 0, 0, "claude-haiku-4-5-20251001") == 0.0


def test_estimate_cost_does_not_subtract_cache_read_from_input():
    """input_tokens is already uncached — billing must NOT subtract cache_read
    (the old bug zeroed input when input==cache_read, under-billing)."""
    # 1M input + 1M cache_read, no output. Correct: $1 + $0.10 = $1.10
    # (old buggy code gave max(0, 1M-1M)=0 input -> $0.10).
    cost = _estimate_cost(1_000_000, 0, 1_000_000, 0, "claude-haiku-4-5-20251001")
    assert abs(cost - 1.10) < 1e-6


def test_estimate_cost_bills_cache_creation():
    """cache_creation_input_tokens (1.25x) was ignored entirely -> under-billed."""
    # 1M cache-write only: 1M * $1.25/MTok = $1.25
    cost = _estimate_cost(0, 0, 0, 1_000_000, "claude-haiku-4-5-20251001")
    assert abs(cost - 1.25) < 1e-6
