"""The decide-preflight calendar lookup retries a transient DNS/network blip
instead of fail-closing the whole decide (2026-06-24 audit; same failure mode
as the 2026-06-09 stop-loss DNS kill)."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from sma.live.preflight import _next_session_with_retry


def test_next_session_retries_then_succeeds():
    alpaca = MagicMock()
    target = date(2026, 5, 26)
    alpaca.next_session_date.side_effect = [RuntimeError("dns blip"), target]
    with patch("sma.live.retry._SLEEP", lambda _s: None):
        got = _next_session_with_retry(alpaca, date(2026, 5, 22))
    assert got == target
    assert alpaca.next_session_date.call_count == 2


def test_next_session_propagates_after_exhausting_attempts():
    alpaca = MagicMock()
    alpaca.next_session_date.side_effect = RuntimeError("dns down")
    with (
        patch("sma.live.retry._SLEEP", lambda _s: None),
        pytest.raises(RuntimeError, match="dns down"),
    ):
        _next_session_with_retry(alpaca, date(2026, 5, 22))
    assert alpaca.next_session_date.call_count == 3
