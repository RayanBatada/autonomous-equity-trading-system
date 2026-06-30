"""Tests for sma.monitoring.check_critical_jobs_fired.

Empirical 2026-05-22 failure mode: decide.daily silently didn't fire for a
trading day (preflight bug), no alert surfaced. The monitor closes that
visibility gap by checking sentinels at 22:30 ET.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from sma.monitoring import CRITICAL_LABELS, check_critical_jobs_fired
from sma.sentinels import write_sentinel


def _trading_day_alpaca(asof: date) -> MagicMock:
    alpaca = MagicMock()
    alpaca.sessions_between.return_value = [asof]  # asof itself is a trading day
    return alpaca


def _holiday_alpaca() -> MagicMock:
    alpaca = MagicMock()
    alpaca.sessions_between.return_value = []  # no trading session = holiday
    return alpaca


def test_no_alert_when_all_critical_sentinels_present(monkeypatch, tmp_path):
    """All critical jobs have sentinels for today → no notification."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))

    asof = date(2026, 5, 26)
    for label in CRITICAL_LABELS:
        write_sentinel(
            label=label,
            asof=asof,
            payload={"label": label, "asof": asof.isoformat()},
        )

    notifies: list[tuple[str, str]] = []
    missing = check_critical_jobs_fired(
        asof=asof,
        alpaca=_trading_day_alpaca(asof),
        notify_fn=lambda title, message: notifies.append((title, message)),
    )

    assert missing == []
    assert notifies == []


def test_alert_when_decide_sentinel_missing(monkeypatch, tmp_path):
    """Trading day with no decide sentinel → exactly one notification."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))

    asof = date(2026, 5, 26)
    # Don't write any sentinels.

    notifies: list[tuple[str, str]] = []
    missing = check_critical_jobs_fired(
        asof=asof,
        alpaca=_trading_day_alpaca(asof),
        notify_fn=lambda title, message: notifies.append((title, message)),
    )

    assert missing == list(CRITICAL_LABELS)
    assert len(notifies) == len(CRITICAL_LABELS)
    title, message = notifies[0]
    assert "missed" in title.lower()
    assert "com.sma.live.decide.daily" in message
    assert asof.isoformat() in message


def test_no_alert_on_market_holiday(monkeypatch, tmp_path):
    """Memorial Day, Christmas, etc. — decide is correctly blocked by
    preflight quality.failed (no Mon prices). Don't alert on those."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))

    asof = date(2026, 5, 25)  # Memorial Day

    notifies: list[tuple[str, str]] = []
    missing = check_critical_jobs_fired(
        asof=asof,
        alpaca=_holiday_alpaca(),
        notify_fn=lambda title, message: notifies.append((title, message)),
    )

    assert missing == []
    assert notifies == []


def test_calendar_failure_alerts_and_still_checks(monkeypatch, tmp_path):
    """If the Alpaca calendar lookup raises, monitoring must NOT crash before
    alerting (the 2026-06-05 audit finding). It notifies that market status is
    unknown and still checks the critical sentinels, so a real missed decide on
    a weekday still surfaces."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))

    asof = date(2026, 4, 30)  # Thursday
    notifies: list[tuple[str, str]] = []
    alpaca = MagicMock()
    alpaca.sessions_between.side_effect = RuntimeError("calendar api down")

    # No sentinels written -> decide is missing.
    missing = check_critical_jobs_fired(
        asof=asof,
        alpaca=alpaca,
        notify_fn=lambda title, message: notifies.append((title, message)),
    )

    assert "com.sma.live.decide.daily" in missing  # still checked despite calendar failure
    blob = " ".join(t + " " + m for t, m in notifies).lower()
    assert "market status" in blob or "calendar" in blob  # degraded-calendar alert sent
    assert "decide" in blob  # and the missing-job alert still fired
