"""drawdown rail tests."""

from sma.risk.drawdown import check_drawdown
from sma.risk.rails import RiskRails


def test_drawdown_triggers_above_threshold():
    rails = RiskRails(max_drawdown_pct=0.15)
    triggered, reason = check_drawdown(rails=rails, current_drawdown=0.20)
    assert triggered is True
    assert "20" in reason or "drawdown" in reason


def test_drawdown_passes_at_threshold():
    rails = RiskRails(max_drawdown_pct=0.15)
    # At cap → not strictly exceeded.
    triggered, _ = check_drawdown(rails=rails, current_drawdown=0.15)
    assert triggered is False


def test_drawdown_passes_well_below_threshold():
    rails = RiskRails(max_drawdown_pct=0.15)
    triggered, _ = check_drawdown(rails=rails, current_drawdown=0.05)
    assert triggered is False


def test_drawdown_disabled_when_pct_is_one():
    rails = RiskRails(max_drawdown_pct=1.0)
    triggered, _ = check_drawdown(rails=rails, current_drawdown=0.99)
    assert triggered is False
