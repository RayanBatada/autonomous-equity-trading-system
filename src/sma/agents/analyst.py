"""Analyst agent: reads researcher output + price/sector context, produces bull/bear."""

from importlib import resources

from sma.agents.base import AnalystInput, AnalystOutput
from sma.agents.cost_tracker import CostTracker

ROLE = "analyst"

SUBMIT_ANALYSIS_TOOL = {
    "name": "submit_analysis",
    "description": "Submit your structured bull/bear analysis.",
    "input_schema": {
        "type": "object",
        "properties": {
            "bull_case": {"type": "string"},
            "bear_case": {"type": "string"},
            "asymmetric_risks": {"type": "array", "items": {"type": "string"}},
            "catalyst_window": {
                "type": "string",
                "enum": ["imminent", "near", "far", "none"],
            },
        },
        "required": ["bull_case", "bear_case", "asymmetric_risks", "catalyst_window"],
    },
}


def _load_system_prompt() -> str:
    return (resources.files("sma.agents.prompts") / "analyst.md").read_text()


def _format_user_message(inp: AnalystInput) -> str:
    pc = inp.price_ctx
    earnings_line = (
        f"Next earnings date: {inp.next_earnings_date}"
        if inp.next_earnings_date
        else "Next earnings date: (none scheduled within 14 days)"
    )
    return "\n".join([
        f"Ticker: {inp.ticker}",
        f"As-of date: {inp.asof_date}",
        f"Sector: {inp.sector or '(unknown)'}",
        earnings_line,
        "",
        "RESEARCH INPUT:",
        f"  News summary: {inp.research.news_summary}",
        f"  Key developments: {inp.research.key_developments}",
        f"  Notable filings: {inp.research.notable_filings}",
        "",
        "30-DAY PRICE CONTEXT:",
        f"  Total return: {pc.total_return_30d}",
        f"  Max drawdown: {pc.max_drawdown_30d}",
        f"  Realized vol: {pc.realized_vol_30d}",
        f"  Pct from 52w high: {pc.pct_from_52w_high}",
        f"  Pct from 52w low: {pc.pct_from_52w_low}",
    ])


class Analyst:
    role = ROLE

    def __init__(self, cost_tracker: CostTracker, model: str):
        self.cost_tracker = cost_tracker
        self.model = model
        self._system_prompt = _load_system_prompt()

    def run(self, inp: AnalystInput, run_id: int) -> AnalystOutput:
        result = self.cost_tracker.invoke(
            run_id=run_id,
            ticker=inp.ticker,
            asof_date=inp.asof_date,
            agent_role=ROLE,
            model=self.model,
            request_kwargs={
                "max_tokens": 600,
                "system": [{
                    "type": "text",
                    "text": self._system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                "tools": [SUBMIT_ANALYSIS_TOOL],
                "tool_choice": {"type": "tool", "name": "submit_analysis"},
                "messages": [{"role": "user", "content": _format_user_message(inp)}],
            },
        )
        tool_use_block = next(
            (b for b in result.response.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_use_block is None:
            raise ValueError(
                "Analyst: model returned no tool_use block "
                f"(stop_reason={getattr(result.response, 'stop_reason', '?')})"
            )
        return AnalystOutput.model_validate(tool_use_block.input)
