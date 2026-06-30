"""cash_floor rail tests."""

from sma.risk.cash_floor import check_cash_floor
from sma.risk.rails import RiskRails


def test_cash_floor_rejects_when_post_order_cash_below_floor():
    rails = RiskRails(cash_floor_pct=0.05)
    triggered, reason = check_cash_floor(
        rails=rails, account_value=100_000.0, cash_after_order=4_999.0,
    )
    assert triggered is True
    assert "5%" in reason or "floor" in reason


def test_cash_floor_passes_when_post_order_cash_above_floor():
    rails = RiskRails(cash_floor_pct=0.05)
    triggered, _ = check_cash_floor(
        rails=rails, account_value=100_000.0, cash_after_order=5_001.0,
    )
    assert triggered is False


def test_cash_floor_passes_at_exact_floor():
    rails = RiskRails(cash_floor_pct=0.05)
    # Exactly at floor → not below, so allowed.
    triggered, _ = check_cash_floor(
        rails=rails, account_value=100_000.0, cash_after_order=5_000.0,
    )
    assert triggered is False


def test_cash_floor_disabled_when_pct_is_zero():
    rails = RiskRails(cash_floor_pct=0.0)
    triggered, _ = check_cash_floor(
        rails=rails, account_value=100_000.0, cash_after_order=-5_000.0,
    )
    assert triggered is False


def test_cash_floor_handles_zero_account_value():
    rails = RiskRails(cash_floor_pct=0.05)
    triggered, _ = check_cash_floor(
        rails=rails, account_value=0.0, cash_after_order=0.0,
    )
    assert triggered is False
