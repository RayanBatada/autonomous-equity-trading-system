"""Dynamic walk-forward CV windows for the autoresearch eval loop.

The eval windows were hardcoded to H2-2025 (`loop.CV_WINDOWS`), so by mid-2026
the self-improvement loop only ever backtested proposals on a ~1-year-old
regime — it never saw current data. These tests pin a builder that anchors the
windows to the latest available data instead, so the eval tracks the present
regime and auto-extends as data accumulates.
"""

from __future__ import annotations

from datetime import date

from sma.autoresearch.loop import _build_cv_windows


def test_returns_requested_number_of_windows():
    wins = _build_cv_windows(date(2026, 6, 4))
    assert len(wins) == 5


def test_last_window_ends_at_anchor_date():
    # The most recent window must end exactly at the latest available data.
    wins = _build_cv_windows(date(2026, 6, 4))
    assert wins[-1][1] == date(2026, 6, 4)


def test_windows_are_oldest_first():
    wins = _build_cv_windows(date(2026, 6, 4))
    starts = [s for s, _ in wins]
    assert starts == sorted(starts)


def test_windows_are_contiguous_and_non_overlapping():
    # Each window starts the day after the previous one ends — a clean tiling.
    wins = _build_cv_windows(date(2026, 6, 4))
    from datetime import timedelta

    for (_prev_s, prev_e), (next_s, _next_e) in zip(wins, wins[1:], strict=False):
        assert next_s == prev_e + timedelta(days=1)


def test_each_window_spans_window_days():
    from datetime import timedelta

    wins = _build_cv_windows(date(2026, 6, 4), window_days=42)
    for s, e in wins:
        assert (e - s) == timedelta(days=41)  # inclusive span of 42 days


def test_not_stale_anchors_to_whatever_date_given():
    # Proves the windows are NOT hardcoded — a 2027 anchor yields 2027 windows.
    wins = _build_cv_windows(date(2027, 1, 15))
    assert wins[-1][1] == date(2027, 1, 15)
    assert all(s.year >= 2026 for s, _ in wins)


def test_respects_custom_counts():
    wins = _build_cv_windows(date(2026, 6, 4), n_windows=3, window_days=30)
    from datetime import timedelta

    assert len(wins) == 3
    assert wins[-1][1] == date(2026, 6, 4)
    for s, e in wins:
        assert (e - s) == timedelta(days=29)


def test_rejects_more_windows_than_schema_supports():
    # experiment_log persists only sharpe_w1..w5; asking for more must fail
    # loudly rather than silently dropping w6+ at insert time.
    import pytest

    with pytest.raises(ValueError, match="w1..w5"):
        _build_cv_windows(date(2026, 6, 4), n_windows=6)
