import pytest

from sma.backtest.risk import RiskRails, check_order


@pytest.fixture
def rails():
    return RiskRails()  # default values


def test_default_rails_match_spec(rails):
    assert rails.max_position_pct == 0.05
    assert rails.max_sector_pct == 0.25
    assert rails.max_drawdown_pct == 0.15
    assert rails.stop_loss_pct == 0.08


def test_check_order_passes_for_compliant_order(rails):
    ok, reason = check_order(
        rails=rails,
        ticker="AAPL",
        order_dollars=4_000,           # 4% of 100k = compliant
        account_value=100_000,
        current_positions={},
        sector="Technology",
        sector_exposure={"Technology": 0.0},
        current_drawdown=0.0,
    )
    assert ok is True
    assert reason == ""


def test_check_order_rejects_position_above_5pct(rails):
    ok, reason = check_order(
        rails=rails,
        ticker="AAPL",
        order_dollars=6_000,           # 6% of 100k
        account_value=100_000,
        current_positions={},
        sector="Technology",
        sector_exposure={"Technology": 0.0},
        current_drawdown=0.0,
    )
    assert ok is False
    assert "position" in reason.lower()


def test_check_order_rejects_when_existing_position_plus_order_exceeds_cap(rails):
    ok, reason = check_order(
        rails=rails,
        ticker="AAPL",
        order_dollars=2_000,           # 2% but already at 4%
        account_value=100_000,
        current_positions={"AAPL": 4_000},
        sector="Technology",
        sector_exposure={"Technology": 0.04},
        current_drawdown=0.0,
    )
    assert ok is False
    assert "position" in reason.lower()


def test_check_order_rejects_when_sector_exposure_exceeds_25pct(rails):
    ok, reason = check_order(
        rails=rails,
        ticker="MSFT",
        order_dollars=3_000,
        account_value=100_000,
        current_positions={},
        sector="Technology",
        sector_exposure={"Technology": 0.24},  # already 24%
        current_drawdown=0.0,
    )
    assert ok is False
    assert "sector" in reason.lower()


def test_check_order_rejects_when_account_in_drawdown_pause(rails):
    ok, reason = check_order(
        rails=rails,
        ticker="AAPL",
        order_dollars=1_000,
        account_value=85_000,          # account down from 100k peak
        current_positions={},
        sector="Technology",
        sector_exposure={"Technology": 0.0},
        current_drawdown=0.16,          # > 15% drawdown
    )
    assert ok is False
    assert "drawdown" in reason.lower()
