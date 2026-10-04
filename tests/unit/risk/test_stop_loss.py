"""stop_loss rail tests."""

from sma.risk.rails import RiskRails
from sma.risk.stop_loss import check_stop_loss, check_take_profit, check_trailing_stop


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


# --- take-profit + trailing-stop (2026-07-01) --------------------------------
def test_take_profit_triggers_at_threshold():
    rails = RiskRails(take_profit_pct=0.20)
    triggered, reason = check_take_profit(
        rails=rails, ticker="AAPL", cost_basis=100.0, current_price=120.0,
    )
    assert triggered is True
    assert "AAPL" in reason


def test_take_profit_does_not_trigger_below_threshold():
    rails = RiskRails(take_profit_pct=0.20)
    triggered, _ = check_take_profit(
        rails=rails, ticker="AAPL", cost_basis=100.0, current_price=119.0,
    )
    assert triggered is False


def test_take_profit_disabled_when_pct_is_zero():
    rails = RiskRails(take_profit_pct=0.0)
    triggered, _ = check_take_profit(
        rails=rails, ticker="AAPL", cost_basis=100.0, current_price=500.0,
    )
    assert triggered is False


def test_take_profit_handles_zero_cost_basis():
    rails = RiskRails(take_profit_pct=0.20)
    triggered, _ = check_take_profit(
        rails=rails, ticker="AAPL", cost_basis=0.0, current_price=10.0,
    )
    assert triggered is False


def test_trailing_stop_triggers_at_threshold():
    rails = RiskRails(trailing_stop_pct=0.10)
    triggered, reason = check_trailing_stop(
        rails=rails, ticker="AAPL", peak_price=200.0, current_price=180.0,
    )
    assert triggered is True
    assert "AAPL" in reason


def test_trailing_stop_does_not_trigger_above_threshold():
    rails = RiskRails(trailing_stop_pct=0.10)
    triggered, _ = check_trailing_stop(
        rails=rails, ticker="AAPL", peak_price=200.0, current_price=181.0,
    )
    assert triggered is False


def test_trailing_stop_disabled_when_pct_is_zero():
    rails = RiskRails(trailing_stop_pct=0.0)
    triggered, _ = check_trailing_stop(
        rails=rails, ticker="AAPL", peak_price=200.0, current_price=10.0,
    )
    assert triggered is False


def test_trailing_stop_handles_nonpositive_peak():
    rails = RiskRails(trailing_stop_pct=0.10)
    triggered, _ = check_trailing_stop(
        rails=rails, ticker="AAPL", peak_price=0.0, current_price=10.0,
    )
    assert triggered is False
