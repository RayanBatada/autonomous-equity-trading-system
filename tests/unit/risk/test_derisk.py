"""Drawdown-scaled cash floor (2026-06-15): the IC analysis showed the model
inverts in reversal regimes (momentum crashes), which are persistent. The bot
runs ~91% deployed, so a crash hits at full exposure with only the blunt 30%
catastrophic breaker as backstop. derisk_cash_floor raises the effective cash
floor as drawdown deepens, auto-reducing exposure in crashes — graduated, not
binary. Default-off (slope 0) until backtested."""
from sma.risk.derisk import derisk_cash_floor


def test_no_derisk_below_start_or_when_disabled():
    # disabled (slope 0): always base floor
    assert derisk_cash_floor(0.05, 0.20, start=0.05, slope=0.0) == 0.05
    # below the start threshold: base floor
    assert derisk_cash_floor(0.05, 0.03, start=0.05, slope=3.0) == 0.05
    assert derisk_cash_floor(0.05, 0.05, start=0.05, slope=3.0) == 0.05


def test_floor_rises_linearly_past_start():
    # drawdown 0.10, start 0.05, slope 3.0 -> 0.05 + 3.0*(0.05) = 0.20
    assert derisk_cash_floor(0.05, 0.10, start=0.05, slope=3.0) == 0.20
    # deeper drawdown -> more cash
    assert derisk_cash_floor(0.05, 0.15, start=0.05, slope=3.0) == 0.35


def test_floor_clamped_at_cap():
    # a severe drawdown can't push the floor past the cap (never fully cash)
    assert derisk_cash_floor(0.05, 0.50, start=0.05, slope=3.0, cap=0.60) == 0.60


def test_floor_never_below_base():
    # negative/zero drawdown (at peak) -> base
    assert derisk_cash_floor(0.05, 0.0, start=0.05, slope=3.0) == 0.05
