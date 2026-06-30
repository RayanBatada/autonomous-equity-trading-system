"""stop_loss rail tests."""

from sma.risk.rails import RiskRails
from sma.risk.stop_loss import check_stop_loss


def test_stop_loss_triggers_at_threshold():
    rails = RiskRails(stop_loss_pct=0.08)
    triggered, reason = check_stop_loss(
        rails=rails, ticker="AAPL", cost_basis=100.0, current_price=92.0,
    )
    assert triggered is True
    assert "AAPL" in reason


def test_stop_loss_does_not_trigger_above_threshold():
    rails = RiskRails(stop_loss_pct=0.08)
    triggered, _ = check_stop_loss(
        rails=rails, ticker="AAPL", cost_basis=100.0, current_price=92.5,
    )
    assert triggered is False


def test_stop_loss_disabled_when_pct_is_zero():
    rails = RiskRails(stop_loss_pct=0.0)
    triggered, _ = check_stop_loss(
        rails=rails, ticker="AAPL", cost_basis=100.0, current_price=50.0,
    )
    assert triggered is False


def test_stop_loss_handles_zero_cost_basis():
    rails = RiskRails(stop_loss_pct=0.08)
    triggered, _ = check_stop_loss(
        rails=rails, ticker="AAPL", cost_basis=0.0, current_price=10.0,
    )
    assert triggered is False


def test_stop_loss_triggers_well_below_threshold():
    rails = RiskRails(stop_loss_pct=0.08)
    triggered, _ = check_stop_loss(
        rails=rails, ticker="AAPL", cost_basis=100.0, current_price=50.0,
    )
    assert triggered is True
