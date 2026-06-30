"""Strategist agent: reads analysis + portfolio + quant rank, outputs conviction."""

from importlib import resources

from sma.agents.base import StrategistInput, StrategistOutput
from sma.agents.cost_tracker import CostTracker

ROLE = "strategist"

CONVICTION_VALUES = ["strong_bullish", "bullish", "neutral", "bearish", "strong_bearish"]
ACTION_HINTS = ["enter", "hold", "reduce", "exit"]

SUBMIT_STRATEGY_TOOL = {
    "name": "submit_strategy",
    "description": "Submit your structured trading recommendation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "conviction": {"type": "string", "enum": CONVICTION_VALUES},
            "score": {"type": "number"},
            "flags": {"type": "array", "items": {"type": "string"}},
            "action_hint": {"type": "string", "enum": ACTION_HINTS},
            "reasoning": {"type": "string"},
        },
        "required": ["conviction", "score", "flags", "action_hint", "reasoning"],
    },
}


def _load_system_prompt() -> str:
    return (resources.files("sma.agents.prompts") / "strategist.md").read_text()


def _format_user_message(inp: StrategistInput) -> str:
    return "\n".join([
        f"Ticker: {inp.ticker}",
        f"As-of date: {inp.asof_date}",
        "",
        f"BULL CASE: {inp.analysis.bull_case}",
        f"BEAR CASE: {inp.analysis.bear_case}",
        f"ASYMMETRIC RISKS: {inp.analysis.asymmetric_risks}",
        f"CATALYST WINDOW: {inp.analysis.catalyst_window}",
        "",
        "PORTFOLIO CONTEXT:",
        f"  Currently held shares: {inp.held_shares}",
        f"  Quant predicted return: {inp.quant_predicted_return}",
        f"  Quant universe rank: {inp.quant_universe_rank}",
        f"  Portfolio sector exposure: {inp.portfolio_sector_exposure_pct:.1f}%",
    ])


class Strategist:
    role = ROLE

    def __init__(self, cost_tracker: CostTracker, model: str):
        self.cost_tracker = cost_tracker
        self.model = model
        self._system_prompt = _load_system_prompt()

    def run(self, inp: StrategistInput, run_id: int) -> StrategistOutput:
        result = self.cost_tracker.invoke(
            run_id=run_id,
            ticker=inp.ticker,
            asof_date=inp.asof_date,
            agent_role=ROLE,
            model=self.model,
            request_kwargs={
                "max_tokens": 400,
                "system": [{
                    "type": "text",
                    "text": self._system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                "tools": [SUBMIT_STRATEGY_TOOL],
                "tool_choice": {"type": "tool", "name": "submit_strategy"},
                "messages": [{"role": "user", "content": _format_user_message(inp)}],
            },
        )
        tool_use_block = next(
            (b for b in result.response.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_use_block is None:
            raise ValueError(
                "Strategist: model returned no tool_use block "
                f"(stop_reason={getattr(result.response, 'stop_reason', '?')})"
            )
        return StrategistOutput.model_validate(tool_use_block.input)
