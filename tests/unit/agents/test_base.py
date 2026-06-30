from datetime import date, datetime

import pytest
from pydantic import ValidationError

from sma.agents.base import (
    ALLOWED_FLAGS,
    AnalystInput,  # noqa: F401  (verifies public API exports)
    AnalystOutput,
    FilingRow,  # noqa: F401  (verifies public API exports)
    NewsRow,
    PriceContext,
    ResearcherInput,
    ResearcherOutput,
    StrategistInput,  # noqa: F401  (verifies public API exports)
    StrategistOutput,
    _normalize_str_list,
)


def test_researcher_input_constructs():
    ri = ResearcherInput(
        ticker="AAPL",
        asof_date=date(2026, 4, 26),
        news_rows=[],
        filing_rows=[],
    )
    assert ri.ticker == "AAPL"


def test_researcher_output_validates_required_fields():
    ro = ResearcherOutput(
        news_summary="text",
        key_developments=["one"],
        notable_filings=[],
    )
    assert ro.news_summary == "text"
    assert ro.key_developments == ["one"]


def test_strategist_output_rejects_bad_conviction():
    with pytest.raises(ValidationError):
        StrategistOutput(
            conviction="moonshot",  # not in enum
            score=0.5,
            flags=[],
            action_hint="hold",
            reasoning="x",
        )


def test_strategist_output_filters_unknown_flags():
    """Unknown flags are silently dropped, not allowed through."""
    so = StrategistOutput(
        conviction="bullish",
        score=0.5,
        flags=["earnings_beat", "totally_made_up_flag"],
        action_hint="enter",
        reasoning="x",
    )
    assert "earnings_beat" in so.flags
    assert "totally_made_up_flag" not in so.flags


def test_strategist_output_clamps_score():
    """Out-of-range scores get clamped to [-1, +1]."""
    so = StrategistOutput(
        conviction="strong_bullish", score=2.5,
        flags=[], action_hint="enter", reasoning="x",
    )
    assert so.score == 1.0
    so2 = StrategistOutput(
        conviction="strong_bearish", score=-3.0,
        flags=[], action_hint="exit", reasoning="x",
    )
    assert so2.score == -1.0


def test_strategist_output_normalizes_unknown_action_hint_to_hold():
    """Unknown action_hint values get coerced to 'hold' (conservative
    default), not rejected. Haiku occasionally invents action verbs
    ('stay_out' on WFC, 'avoid' on GE during 2026-04-27 val prefill);
    rejecting would lose the whole ticker-day, while 'hold' preserves
    the rest of the thesis intact.
    """
    so = StrategistOutput(
        conviction="bullish", score=0.5, flags=[],
        action_hint="moonshot",  # not in enum
        reasoning="x",
    )
    assert so.action_hint == "hold"


def test_strategist_output_normalizes_observed_haiku_action_hints():
    """Specific values observed during the 2026-04-27 prefill."""
    for invented in ("stay_out", "avoid"):
        so = StrategistOutput(
            conviction="neutral", score=0.0, flags=[],
            action_hint=invented, reasoning="x",
        )
        assert so.action_hint == "hold", (
            f"expected {invented!r} → 'hold'; got {so.action_hint!r}"
        )


def test_strategist_output_action_hint_passthrough_for_valid_values():
    """All four documented values must NOT be coerced."""
    for valid in ("enter", "hold", "reduce", "exit"):
        so = StrategistOutput(
            conviction="bullish", score=0.5, flags=[],
            action_hint=valid, reasoning="x",
        )
        assert so.action_hint == valid


def test_strategist_output_action_hint_validate_dict_path():
    """The model_validate path must also coerce unknowns."""
    so = StrategistOutput.model_validate({
        "conviction": "bullish", "score": 0.5, "flags": [],
        "action_hint": "stay_out", "reasoning": "x",
    })
    assert so.action_hint == "hold"


def test_analyst_output_rejects_bad_catalyst_window():
    with pytest.raises(ValidationError):
        AnalystOutput(
            bull_case="x", bear_case="y",
            asymmetric_risks=[],
            catalyst_window="someday",  # not in enum
        )


def test_analyst_output_catalyst_window_defaults_to_none():
    """Haiku occasionally omits catalyst_window. Default to 'none' so the
    pipeline doesn't lose the entire ticker-day to a validation error.
    Reasoning: 'none' is the most conservative interpretation (no upcoming
    catalyst), which is also the safest action signal for downstream
    strategy decisions.
    """
    ao = AnalystOutput(
        bull_case="x", bear_case="y", asymmetric_risks=[],
        # catalyst_window omitted entirely
    )
    assert ao.catalyst_window == "none"


def test_analyst_output_validate_dict_without_catalyst_window():
    """model_validate path must also tolerate missing catalyst_window."""
    ao = AnalystOutput.model_validate({
        "bull_case": "b", "bear_case": "r", "asymmetric_risks": [],
    })
    assert ao.catalyst_window == "none"


def test_allowed_flags_includes_canonical_set():
    expected = {
        "earnings_beat", "earnings_miss", "guidance_raise", "guidance_cut",
        "m&a_announcement", "regulatory_win", "regulatory_risk",
        "fda_approval", "fda_setback", "litigation_material",
        "executive_change", "dividend_change", "secular_inflection",
        "competitive_loss", "macro_headwind", "macro_tailwind",
    }
    assert expected.issubset(ALLOWED_FLAGS)


def test_news_row_basic():
    nr = NewsRow(
        published_at=datetime(2026, 4, 26, 14, 0),
        source="alpaca:benzinga",
        headline="Apple reports earnings",
        body_excerpt="strong quarter",
    )
    assert nr.headline == "Apple reports earnings"


def test_price_context_all_optional():
    """All fields default to None to keep prompt assembly simple when prices missing."""
    pc = PriceContext()
    assert pc.total_return_30d is None
    assert pc.max_drawdown_30d is None


# --- list-normalization (defensive against Haiku's <item>...</item> outputs) ---

def test_normalize_str_list_passes_through_proper_lists():
    assert _normalize_str_list(["a", "b"]) == ["a", "b"]
    assert _normalize_str_list([]) == []


def test_normalize_str_list_extracts_xml_items():
    """Haiku 4.5 sometimes wraps list-of-strings as a single XML-tagged string."""
    s = "\n<item>first</item>\n<item>second item</item>\n"
    assert _normalize_str_list(s) == ["first", "second item"]


def test_normalize_str_list_handles_none_and_empty():
    assert _normalize_str_list(None) == []
    assert _normalize_str_list("") == []
    assert _normalize_str_list([""]) == []


def test_normalize_str_list_wraps_plain_string_with_no_tags():
    """If the model returns a free-form string with no <item> tags, wrap it."""
    assert _normalize_str_list("just one thing") == ["just one thing"]


def test_researcher_output_accepts_xml_tagged_lists():
    """End-to-end: ResearcherOutput tolerates Haiku's XML format."""
    ro = ResearcherOutput(
        news_summary="x",
        key_developments="<item>dev1</item><item>dev2</item>",
        notable_filings="<item>8-K filed today</item>",
    )
    assert ro.key_developments == ["dev1", "dev2"]
    assert ro.notable_filings == ["8-K filed today"]


def test_analyst_output_accepts_xml_tagged_risks():
    ao = AnalystOutput(
        bull_case="b", bear_case="r",
        asymmetric_risks="<item>FDA decision binary</item><item>litigation</item>",
        catalyst_window="near",
    )
    assert ao.asymmetric_risks == ["FDA decision binary", "litigation"]


def test_strategist_output_filters_xml_tagged_flags():
    """Even from XML format, unknown flags get filtered."""
    so = StrategistOutput(
        conviction="bullish", score=0.5,
        flags="<item>earnings_beat</item><item>made_up_flag</item>",
        action_hint="enter", reasoning="x",
    )
    assert so.flags == ["earnings_beat"]


# --- missing-field defenses (Haiku omits required list fields sometimes) ---

def test_researcher_output_accepts_missing_list_fields():
    """TSLA bug 2026-04-27: Haiku omitted notable_filings entirely; default to []."""
    ro = ResearcherOutput.model_validate({"news_summary": "x"})
    assert ro.key_developments == []
    assert ro.notable_filings == []


def test_researcher_output_only_news_summary_required():
    """If news_summary is missing, validation should still raise."""
    with pytest.raises(ValidationError):
        ResearcherOutput.model_validate({"key_developments": ["a"]})


def test_analyst_output_accepts_missing_asymmetric_risks():
    ao = AnalystOutput.model_validate({
        "bull_case": "b", "bear_case": "r", "catalyst_window": "near",
    })
    assert ao.asymmetric_risks == []


def test_analyst_output_required_strings_still_enforced():
    with pytest.raises(ValidationError):
        AnalystOutput.model_validate({
            "bear_case": "r", "catalyst_window": "near",
        })


def test_strategist_output_accepts_missing_flags():
    so = StrategistOutput.model_validate({
        "conviction": "neutral", "score": 0.0,
        "action_hint": "hold", "reasoning": "x",
    })
    assert so.flags == []
