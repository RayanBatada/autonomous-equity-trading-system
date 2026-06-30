
from sma.backtest.windows import (
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    window_dates,
)


def test_windows_are_non_overlapping_and_chronological():
    assert TRAIN_START < TRAIN_END
    assert TRAIN_END < VAL_START
    assert VAL_START < VAL_END
    assert VAL_END < TEST_START


def test_window_dates_returns_correct_range():
    assert window_dates("train") == (TRAIN_START, TRAIN_END)
    assert window_dates("val") == (VAL_START, VAL_END)
    start, end = window_dates("test")
    assert start == TEST_START
    assert end >= TEST_START  # test ends "now"


def test_window_dates_unknown_raises():
    import pytest
    with pytest.raises(ValueError):
        window_dates("not-a-window")  # type: ignore
