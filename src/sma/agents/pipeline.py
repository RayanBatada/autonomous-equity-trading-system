"""Pipeline orchestrator: researcher → analyst → strategist for one ticker/day."""

import json
from dataclasses import dataclass
from datetime import date, timedelta

from loguru import logger

from sma.agents.base import (
    AnalystInput,
    FilingRow,
    NewsRow,
    PriceContext,
    ResearcherInput,
    StrategistInput,
    StrategistOutput,
)
from sma.agents.cost_tracker import BudgetExhausted
from sma.ingest.store import Store

# Match the strategy's THESIS_STALE_DAYS: only fall back to a cached thesis the
# strategy would still use, so a stale thesis can't masquerade as fresh research.
_FALLBACK_MAX_AGE_DAYS = 7


@dataclass
class ThesisContext:
    ticker: str
    asof_date: date
    news_rows: list[NewsRow]
    filing_rows: list[FilingRow]
    price_ctx: PriceContext
    sector: str | None
    next_earnings_date: date | None
    held_shares: int
    quant_predicted_return: float | None
    quant_universe_rank: int | None
    portfolio_sector_exposure_pct: float


class CachedThesisFallback(StrategistOutput):
    """Strategist-shaped output loaded from cache after budget exhaustion."""


class ThesisPipeline:
    def __init__(self, researcher, analyst, strategist, store: Store):
        self.researcher = researcher
        self.analyst = analyst
        self.strategist = strategist
        self.store = store

    def run(self, ctx: ThesisContext, run_id: int) -> StrategistOutput | None:
        try:
            r_out = self.researcher.run(
                ResearcherInput(
                    ticker=ctx.ticker,
                    asof_date=ctx.asof_date,
                    news_rows=ctx.news_rows,
                    filing_rows=ctx.filing_rows,
                ),
                run_id=run_id,
            )
        except BudgetExhausted:
            return self._cached_thesis(ctx.ticker, ctx.asof_date)

        try:
            a_out = self.analyst.run(
                AnalystInput(
                    ticker=ctx.ticker,
                    asof_date=ctx.asof_date,
                    research=r_out,
                    price_ctx=ctx.price_ctx,
                    sector=ctx.sector,
                    next_earnings_date=ctx.next_earnings_date,
                ),
                run_id=run_id,
            )
        except BudgetExhausted:
            return self._cached_thesis(ctx.ticker, ctx.asof_date)

        try:
            s_out = self.strategist.run(
                StrategistInput(
                    ticker=ctx.ticker,
                    asof_date=ctx.asof_date,
                    analysis=a_out,
                    held_shares=ctx.held_shares,
                    quant_predicted_return=ctx.quant_predicted_return,
                    quant_universe_rank=ctx.quant_universe_rank,
                    portfolio_sector_exposure_pct=ctx.portfolio_sector_exposure_pct,
                ),
                run_id=run_id,
            )
        except BudgetExhausted:
            return self._cached_thesis(ctx.ticker, ctx.asof_date)

        # Persist complete thesis row.
        self.store.conn.execute(
            """
            INSERT OR REPLACE INTO theses (
                ticker, asof_date, run_id,
                news_summary, key_developments, notable_filings,
                bull_case, bear_case, asymmetric_risks, catalyst_window,
                conviction, score, flags, action_hint, reasoning
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ctx.ticker, ctx.asof_date, run_id,
                r_out.news_summary,
                json.dumps(r_out.key_developments),
                json.dumps(r_out.notable_filings),
                a_out.bull_case, a_out.bear_case,
                json.dumps(a_out.asymmetric_risks),
                a_out.catalyst_window,
                s_out.conviction, s_out.score,
                json.dumps(s_out.flags),
                s_out.action_hint, s_out.reasoning,
            ],
        )
        return s_out

    def _cached_thesis(self, ticker: str, asof_date: date) -> CachedThesisFallback | None:
        cutoff = asof_date - timedelta(days=_FALLBACK_MAX_AGE_DAYS)
        row = self.store.conn.execute(
            """
            SELECT conviction, score, flags, action_hint, reasoning
            FROM theses WHERE ticker = ? AND asof_date < ? AND asof_date >= ?
            ORDER BY asof_date DESC LIMIT 1
            """,
            [ticker, asof_date, cutoff],
        ).fetchone()
        if row is None:
            logger.warning(
                "budget exhausted and no FRESH (<{}d) cached thesis for {}",
                _FALLBACK_MAX_AGE_DAYS, ticker,
            )
            return None
        flags = json.loads(row[2]) if isinstance(row[2], str) else (row[2] or [])
        return CachedThesisFallback(
            conviction=row[0],
            score=row[1],
            flags=flags,
            action_hint=row[3],
            reasoning=row[4],
        )
