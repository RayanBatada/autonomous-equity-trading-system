"""Risk-rail / drift percentage fields are fractions in [0, 1]; out-of-range
values (e.g. stop_loss_pct=8.0 meaning 800%, silently disabling the stop) must
be rejected at config load, not accepted (2026-06-05 audit)."""

import pytest
from pydantic import ValidationError

from sma.config import LiveDrift, LiveRails


def test_live_rails_rejects_out_of_range_pct():
    with pytest.raises(ValidationError):
        LiveRails(stop_loss_pct=8.0)  # 800% — would silently disable the stop
    with pytest.raises(ValidationError):
        LiveRails(max_position_pct=1.5)
    with pytest.raises(ValidationError):
        LiveRails(cash_floor_pct=-0.1)


def test_live_rails_accepts_valid_fractions():
    r = LiveRails(stop_loss_pct=0.08, max_sector_pct=0.35, max_position_pct=0.10)
    assert r.stop_loss_pct == 0.08
    assert r.max_sector_pct == 0.35


def test_live_drift_rejects_out_of_range_pct():
    with pytest.raises(ValidationError):
        LiveDrift(catastrophic_loss_abort_pct=3.0)
    with pytest.raises(ValidationError):
        LiveDrift(buy_miss_alert_threshold_pct=-0.2)
