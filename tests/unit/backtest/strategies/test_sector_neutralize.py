"""Unit tests for sector-relative (sector-demean) scoring used by xgb_top_k.

Demeaning each ticker's score by its sector mean (strength λ) makes the top-K
rank on within-sector relative strength, so selection spans sectors instead of
piling into whichever sector the model is broadly bullish on.
"""

import math

from sma.backtest.strategies.xgb_top_k import sector_demean_scores


def _sector_of(mapping: dict[str, str]):
    return lambda t: mapping.get(t, "Unknown")


def test_lambda_zero_is_identity():
    scores = {"A1": 1.0, "A2": 3.0, "B1": 0.5}
    sof = _sector_of({"A1": "Tech", "A2": "Tech", "B1": "Energy"})
    assert sector_demean_scores(scores, 0.0, sof) == scores


def test_demean_subtracts_lambda_times_sector_mean():
    scores = {"A1": 1.0, "A2": 3.0, "B1": 0.0, "B2": 0.0}
    sof = _sector_of({"A1": "Tech", "A2": "Tech", "B1": "Energy", "B2": "Energy"})
    out = sector_demean_scores(scores, 1.0, sof)  # Tech mean=2.0, Energy mean=0.0
    assert out["A1"] == -1.0
    assert out["A2"] == 1.0
    assert out["B1"] == 0.0
    assert out["B2"] == 0.0


def test_partial_lambda():
    scores = {"A1": 1.0, "A2": 3.0}  # mean 2.0
    sof = _sector_of({"A1": "Tech", "A2": "Tech"})
    out = sector_demean_scores(scores, 0.5, sof)  # subtract 0.5*2.0 = 1.0
    assert out["A1"] == 0.0
    assert out["A2"] == 2.0


def test_single_member_sector_is_untouched():
    scores = {"A1": 1.0, "A2": 3.0, "SOLO": 5.0}
    sof = _sector_of({"A1": "Tech", "A2": "Tech", "SOLO": "Utilities"})
    out = sector_demean_scores(scores, 1.0, sof)
    # A lone name in its sector must not be collapsed toward 0.
    assert out["SOLO"] == 5.0


def test_unknown_sector_grouped_among_itself():
    scores = {"U1": 2.0, "U2": 4.0, "A1": 1.0, "A2": 1.0}
    sof = _sector_of({"A1": "Tech", "A2": "Tech"})  # U1/U2 -> "Unknown"
    out = sector_demean_scores(scores, 1.0, sof)  # Unknown mean=3.0
    assert out["U1"] == -1.0
    assert out["U2"] == 1.0


def test_empty_scores_returns_empty():
    assert sector_demean_scores({}, 1.0, _sector_of({})) == {}


def test_all_one_sector_preserves_order():
    """If every name is in one sector, demeaning shifts all scores by the same
    constant — selection order (and thus the top-K) is unchanged."""
    scores = {"A1": 1.0, "A2": 3.0, "A3": 2.0}
    sof = _sector_of({"A1": "Tech", "A2": "Tech", "A3": "Tech"})
    out = sector_demean_scores(scores, 1.0, sof)
    raw_order = sorted(scores, key=scores.get, reverse=True)
    out_order = sorted(out, key=out.get, reverse=True)
    assert raw_order == out_order == ["A2", "A3", "A1"]


def test_non_finite_score_does_not_contaminate_sector():
    """A NaN/inf member must not poison the whole sector's mean; finite peers are
    demeaned over finite members only, and the non-finite score passes through."""
    scores = {"A1": float("nan"), "A2": 2.0, "A3": 4.0}  # finite mean = 3.0
    sof = _sector_of({"A1": "Tech", "A2": "Tech", "A3": "Tech"})
    out = sector_demean_scores(scores, 1.0, sof)
    assert out["A2"] == -1.0
    assert out["A3"] == 1.0
    assert math.isnan(out["A1"])  # passed through untouched


def test_sector_with_one_finite_member_is_skipped():
    scores = {"A1": float("inf"), "A2": 5.0}  # only one finite member
    sof = _sector_of({"A1": "Tech", "A2": "Tech"})
    out = sector_demean_scores(scores, 1.0, sof)
    assert out["A2"] == 5.0  # unchanged — fewer than 2 finite members
    assert math.isinf(out["A1"])


def test_neutralization_diversifies_top_k():
    """The reason this feature exists: a sector with uniformly high raw scores
    dominates the raw top-K; after demeaning, other sectors' best names surface."""
    scores = {"T1": 0.9, "T2": 0.8, "T3": 0.7, "E1": 0.2, "E2": 0.1}
    sof = _sector_of(
        {"T1": "Tech", "T2": "Tech", "T3": "Tech", "E1": "Energy", "E2": "Energy"}
    )

    raw_top2 = sorted(scores, key=scores.get, reverse=True)[:2]
    assert raw_top2 == ["T1", "T2"]  # both Tech — the concentration problem

    out = sector_demean_scores(scores, 1.0, sof)
    neutral_top2 = sorted(out, key=out.get, reverse=True)[:2]
    sectors = {sof(t) for t in neutral_top2}
    assert "Energy" in sectors  # selection now spans sectors
