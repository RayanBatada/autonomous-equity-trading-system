"""Researcher agent: reads news + filings, produces structured summary."""

from importlib import resources

from sma.agents.base import ResearcherInput, ResearcherOutput
from sma.agents.cost_tracker import CostTracker

MAX_NEWS_ROWS = 15
MAX_FILING_ROWS = 5
ROLE = "researcher"

SUBMIT_RESEARCH_TOOL = {
    "name": "submit_research",
    "description": "Submit your structured research summary for the ticker.",
    "input_schema": {
        "type": "object",
        "properties": {
            "news_summary": {"type": "string"},
            "key_developments": {"type": "array", "items": {"type": "string"}},
            "notable_filings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["news_summary", "key_developments", "notable_filings"],
    },
}


def _load_system_prompt() -> str:
    return (resources.files("sma.agents.prompts") / "researcher.md").read_text()


def _format_user_message(inp: ResearcherInput) -> str:
    rows = sorted(inp.news_rows, key=lambda r: r.published_at, reverse=True)[:MAX_NEWS_ROWS]
    news_lines: list[str] = []
    for r in rows:
        line = f"  [{r.published_at.date()}] source: {r.source} | headline: {r.headline}"
        if r.body_excerpt:
            line += f"\n    summary: {r.body_excerpt[:300]}"
        news_lines.append(line)
    filing_lines = [
        f"  [{f.filed_at.date()}] {f.filing_type} | {f.url}"
        for f in inp.filing_rows[:MAX_FILING_ROWS]
    ]
    parts = [
        f"Ticker: {inp.ticker}",
        f"As-of date: {inp.asof_date}",
        "",
        "Recent news (last 7 days, most recent first):"
        if news_lines else "Recent news: (none)",
        *news_lines,
        "",
        "Recent filings (last 90 days):" if filing_lines else "Recent filings: (none)",
        *filing_lines,
    ]
    return "\n".join(parts)


class Researcher:
    role = ROLE

    def __init__(self, cost_tracker: CostTracker, model: str):
        self.cost_tracker = cost_tracker
        self.model = model
        self._system_prompt = _load_system_prompt()

    def run(self, inp: ResearcherInput, run_id: int) -> ResearcherOutput:
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
                "tools": [SUBMIT_RESEARCH_TOOL],
                "tool_choice": {"type": "tool", "name": "submit_research"},
                "messages": [{"role": "user", "content": _format_user_message(inp)}],
            },
        )
        tool_use_block = next(
            (b for b in result.response.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_use_block is None:
            raise ValueError(
                "Researcher: model returned no tool_use block "
                f"(stop_reason={getattr(result.response, 'stop_reason', '?')})"
            )
        return ResearcherOutput.model_validate(tool_use_block.input)
