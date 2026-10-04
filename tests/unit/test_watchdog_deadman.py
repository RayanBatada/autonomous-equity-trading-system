"""The watchdog pass has a dead-man limit (flaw hunt 2026-10-01, A2)."""

import pytest


def test_main_arms_a_deadman_alarm_and_clears_it(monkeypatch):
    """Flaw hunt 2026-10-01 A2: a watchdog pass that hangs (it did, from 15:53
    on 10/1, blocked on an Alpaca call) stops every later checkpoint, because
    launchd will not start a second copy. main() arms SIGALRM so a hung pass
    dies and the next checkpoint runs."""
    import signal

    from sma import watchdog as wd

    alarms = []
    monkeypatch.setattr(signal, "alarm", lambda s: alarms.append(s) or 0)
    monkeypatch.setattr(signal, "signal", lambda *a: None)
    monkeypatch.setattr(wd, "check", lambda: 0)
    assert wd.main() == 0
    assert alarms == [wd.PASS_DEADLINE_S, 0]
    assert 60 <= wd.PASS_DEADLINE_S <= 1800


def test_deadman_handler_raises_so_the_pass_exits():
    from sma import watchdog as wd

    with pytest.raises(TimeoutError):
        wd._on_deadman(14, None)
