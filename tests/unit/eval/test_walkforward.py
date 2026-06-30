"""Walk-forward eval (2026-06-12): the single-window-start-model eval scores
months-stale predictions late in the window — real evidence needs models
retrained at boundaries through the window, exactly like production's weekly
retrain. Predictor already resolves latest-model<=asof per call, so the
harness just pre-trains boundary models into a cache dir."""

from datetime import date

from sma.eval.walkforward import retrain_boundaries


def test_boundaries_every_n_sessions_starting_at_window_start():
    sessions = [date(2025, 7, d) for d in range(1, 32) if date(2025, 7, d).weekday() < 5]
    b = retrain_boundaries(sessions, every=5)
    assert b[0] == sessions[0]
    assert b == sessions[::5]


def test_boundaries_handle_short_windows():
    sessions = [date(2025, 7, 1), date(2025, 7, 2)]
    assert retrain_boundaries(sessions, every=21) == [date(2025, 7, 1)]
    assert retrain_boundaries([], every=21) == []
