"""Pydantic schemas + Agent Protocol for Phase 4 LLM research agents."""

import re
from datetime import date, datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, field_validator

# --- closed flag vocabulary (initial Phase 4 set) ---
ALLOWED_FLAGS: set[str] = {
    "earnings_beat", "earnings_miss", "guidance_raise", "guidance_cut",
    "m&a_announcement", "regulatory_win", "regulatory_risk",
    "fda_approval", "fda_setback",
    "litigation_material", "executive_change", "dividend_change",
    "secular_inflection", "competitive_loss",
    "macro_headwind", "macro_tailwind",
}


# Haiku 4.5 occasionally returns list-of-string fields as a single string with
# XML-style <item>...</item> tags instead of a proper JSON array. Smoke-tested
# on 2026-04-27 — caught before the prefill batch run. Normalize defensively.
_ITEM_TAG_RE = re.compile(r"<item>(.*?)</item>", re.DOTALL | re.IGNORECASE)


def _normalize_str_list(v):
    """Accept lists, XML-tagged strings, or single strings; emit list[str]."""
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        items = [m.strip() for m in _ITEM_TAG_RE.findall(v)]
        if items:
            return [it for it in items if it]
        text = v.strip()
        return [text] if text else []
    return v

# --- type aliases ---
Conviction = Literal["strong_bullish", "bullish", "neutral", "bearish", "strong_bearish"]
ActionHint = Literal["enter", "hold", "reduce", "exit"]
CatalystWindow = Literal["imminent", "near", "far", "none"]
AgentRole = Literal["researcher", "analyst", "strategist"]


# --- shared row types ---
class NewsRow(BaseModel):
    published_at: datetime
    source: str
    headline: str
    body_excerpt: str | None = None


class FilingRow(BaseModel):
    filed_at: datetime
    filing_type: str
    url: str


# --- researcher ---
class ResearcherInput(BaseModel):
    ticker: str
    asof_date: date
    news_rows: list[NewsRow]
    filing_rows: list[FilingRow]


class ResearcherOutput(BaseModel):
    news_summary: str
    # List fields default to [] — Haiku 4.5 occasionally omits these entirely
    # despite the tool schema marking them required.
    key_developments: list[str] = []
    notable_filings: list[str] = []

    @field_validator("key_developments", "notable_filings", mode="before")
    @classmethod
    def normalize_lists(cls, v):
        return _normalize_str_list(v)


# --- analyst ---
class PriceContext(BaseModel):
    total_return_30d: float | None = None
    max_drawdown_30d: float | None = None
    realized_vol_30d: float | None = None
    pct_from_52w_high: float | None = None
    pct_from_52w_low: float | None = None


class AnalystInput(BaseModel):
    ticker: str
    asof_date: date
    research: ResearcherOutput
    price_ctx: PriceContext
    sector: str | None = None
    next_earnings_date: date | None = None


class AnalystOutput(BaseModel):
    bull_case: str
    bear_case: str
    asymmetric_risks: list[str] = []  # Haiku may omit; default empty
    # 'none' is the conservative default if Haiku omits the field — treats
    # the ticker as having no near-term catalyst, which downstream rails
    # interpret as the safest action signal.
    catalyst_window: CatalystWindow = "none"

    @field_validator("asymmetric_risks", mode="before")
    @classmethod
    def normalize_lists(cls, v):
        return _normalize_str_list(v)


# --- strategist ---
class StrategistInput(BaseModel):
    ticker: str
    asof_date: date
    analysis: AnalystOutput
    held_shares: int = 0
    quant_predicted_return: float | None = None
    quant_universe_rank: int | None = None
    portfolio_sector_exposure_pct: float = 0.0


class StrategistOutput(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    conviction: Conviction
    score: float
    flags: list[str] = []  # Haiku may omit; default empty (no catalyst tags)
    action_hint: ActionHint
    reasoning: str

    @field_validator("flags", mode="before")
    @classmethod
    def filter_unknown_flags(cls, v):
        v = _normalize_str_list(v)
        return [f for f in v if f in ALLOWED_FLAGS]

    @field_validator("action_hint", mode="before")
    @classmethod
    def normalize_action_hint(cls, v):
        """Haiku occasionally invents action verbs not in the documented
        vocabulary — 'stay_out' (WFC) and 'avoid' (GE) observed during
        the 2026-04-27 val prefill. Map any unknown to 'hold': the most
        conservative neutral action, which is also what 'stay_out' and
        'avoid' clearly intended.
        """
        if isinstance(v, str) and v in {"enter", "hold", "reduce", "exit"}:
            return v
        return "hold"

    @field_validator("score")
    @classmethod
    def clamp_score(cls, v):
        return max(-1.0, min(1.0, float(v)))


# --- agent protocol ---
class Agent(Protocol):
    role: AgentRole
    model: str

    def run(self, ctx, run_id: int) -> object: ...
