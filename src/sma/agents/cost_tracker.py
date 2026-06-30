"""Anthropic SDK wrapper with daily budget ceiling + per-call telemetry."""

import time
import uuid
from dataclasses import dataclass
from datetime import date
from typing import Any

from loguru import logger

from sma.ingest.store import Store

# Haiku 4.5 prices (per million tokens)
HAIKU_INPUT_USD_PER_MTOK = 1.0
HAIKU_OUTPUT_USD_PER_MTOK = 5.0
HAIKU_CACHE_READ_USD_PER_MTOK = 0.10
HAIKU_CACHE_WRITE_USD_PER_MTOK = 1.25  # cache-creation = 1.25x base input
DEFAULT_CALL_RESERVATION_USD = 0.012


class BudgetExhausted(Exception):  # noqa: N818 - public API name fixed by spec
    """Raised when today's running spend has hit the daily ceiling."""


@dataclass
class AgentCallResult:
    response: Any
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    est_cost_usd: float
    latency_ms: int


def _estimate_cost(
    input_tok: int, output_tok: int, cache_read_tok: int,
    cache_write_tok: int, model: str,
) -> float:
    """Haiku 4.5 only for now; extend if other models added.

    Anthropic reports input_tokens / cache_read_input_tokens /
    cache_creation_input_tokens as SEPARATE, mutually-exclusive buckets:
    `input_tokens` is ALREADY the uncached count. The prior code subtracted
    cache_read from input_tokens (double-counting the discount) and ignored
    cache-creation entirely, so it systematically UNDER-billed and could blow
    the daily budget.
    """
    return (
        input_tok * HAIKU_INPUT_USD_PER_MTOK / 1_000_000
        + cache_read_tok * HAIKU_CACHE_READ_USD_PER_MTOK / 1_000_000
        + cache_write_tok * HAIKU_CACHE_WRITE_USD_PER_MTOK / 1_000_000
        + output_tok * HAIKU_OUTPUT_USD_PER_MTOK / 1_000_000
    )


class CostTracker:
    def __init__(
        self,
        client: Any,
        store: Store,
        daily_budget_usd: float,
        warn_threshold_pct: int = 80,
    ):
        self.client = client
        self.store = store
        self.daily_budget_usd = daily_budget_usd
        self.warn_threshold_pct = warn_threshold_pct

    def _today_spend_usd(self) -> float:
        row = self.store.conn.execute(
            "SELECT COALESCE(SUM(est_cost_usd), 0.0) FROM agent_calls "
            "WHERE created_at::DATE = CURRENT_DATE"
        ).fetchone()
        return float(row[0] or 0.0)

    def _reservation_usd(self, request_kwargs: dict, model: str) -> float:
        max_tokens = int(request_kwargs.get("max_tokens") or 0)
        output_reserve = _estimate_cost(
            input_tok=0,
            output_tok=max_tokens,
            cache_read_tok=0,
            cache_write_tok=0,
            model=model,
        )
        return max(DEFAULT_CALL_RESERVATION_USD, output_reserve)

    def invoke(
        self, run_id: int, ticker: str, asof_date: date,
        agent_role: str, model: str, request_kwargs: dict,
    ) -> AgentCallResult:
        spent = self._today_spend_usd()
        reservation = self._reservation_usd(request_kwargs, model)
        if spent + reservation > self.daily_budget_usd:
            self._log_skipped(run_id, ticker, asof_date, agent_role, model, spent)
            raise BudgetExhausted(
                f"today's spend ${spent:.4f} + reserved ${reservation:.4f} "
                f"> ceiling ${self.daily_budget_usd:.2f}"
            )
        if spent + reservation >= self.daily_budget_usd * self.warn_threshold_pct / 100:
            logger.warning(
                "agent budget {}% threshold crossed: ${:.4f} / ${:.2f}",
                self.warn_threshold_pct, spent + reservation, self.daily_budget_usd,
            )

        call_id = self._log_call(
            run_id, ticker, asof_date, agent_role, model,
            input_tok=0, output_tok=0, cache_read_tok=0,
            cost=reservation, latency_ms=0, status="reserved", error=None,
        )
        start = time.monotonic()
        try:
            resp = self.client.messages.create(model=model, **request_kwargs)
        except Exception as e:
            latency = int((time.monotonic() - start) * 1000)
            self._settle_call(
                call_id,
                input_tok=0, output_tok=0, cache_read_tok=0,
                cost=0.0, latency_ms=latency, status="error", error=str(e),
            )
            raise
        latency = int((time.monotonic() - start) * 1000)

        in_tok = int(resp.usage.input_tokens)
        out_tok = int(resp.usage.output_tokens)
        cache_read = int(getattr(resp.usage, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(resp.usage, "cache_creation_input_tokens", 0) or 0)
        cost = _estimate_cost(in_tok, out_tok, cache_read, cache_write, model)

        self._settle_call(
            call_id,
            input_tok=in_tok, output_tok=out_tok, cache_read_tok=cache_read,
            cost=cost, latency_ms=latency, status="ok", error=None,
        )

        return AgentCallResult(
            response=resp, input_tokens=in_tok, output_tokens=out_tok,
            cache_read_tokens=cache_read, est_cost_usd=cost, latency_ms=latency,
        )

    def _log_call(self, run_id, ticker, asof_date, role, model,
                    input_tok, output_tok, cache_read_tok, cost, latency_ms, status, error):
        call_id = str(uuid.uuid4())
        self.store.conn.execute(
            """
            INSERT INTO agent_calls (call_id, run_id, ticker, asof_date,
                agent_role, model_id, input_tokens, output_tokens, cache_read_tokens,
                est_cost_usd, latency_ms, status, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [call_id, run_id, ticker, asof_date, role, model,
              input_tok, output_tok, cache_read_tok, cost, latency_ms, status, error],
        )
        return call_id

    def _settle_call(
        self, call_id, input_tok, output_tok, cache_read_tok, cost, latency_ms, status, error
    ):
        self.store.conn.execute(
            """
            UPDATE agent_calls
            SET input_tokens = ?,
                output_tokens = ?,
                cache_read_tokens = ?,
                est_cost_usd = ?,
                latency_ms = ?,
                status = ?,
                error = ?
            WHERE call_id = ?
            """,
            [input_tok, output_tok, cache_read_tok, cost, latency_ms, status, error, call_id],
        )

    def _log_skipped(self, run_id, ticker, asof_date, role, model, spent):
        self._log_call(
            run_id, ticker, asof_date, role, model,
            input_tok=0, output_tok=0, cache_read_tok=0,
            cost=0.0, latency_ms=0,
            status="budget_exhausted_skipped",
            error=f"spent=${spent:.4f} ceiling=${self.daily_budget_usd:.2f}",
        )
