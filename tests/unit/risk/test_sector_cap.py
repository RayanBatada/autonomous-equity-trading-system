"""sector_cap rail tests."""

from sma.risk.rails import RiskRails
from sma.risk.sector_cap import check_sector_cap


def test_sector_cap_rejects_above_threshold():
    rails = RiskRails(max_sector_pct=0.25)
    triggered, reason = check_sector_cap(
        rails=rails, sector="Tech", sector_exposure_after=0.30,
    )
    assert triggered is True
    assert "Tech" in reason


def test_sector_cap_passes_at_threshold():
    rails = RiskRails(max_sector_pct=0.25)
    # At cap (not strictly above) → allowed.
    triggered, _ = check_sector_cap(
        rails=rails, sector="Tech", sector_exposure_after=0.25,
    )
    assert triggered is False


def test_sector_cap_passes_below_threshold():
    rails = RiskRails(max_sector_pct=0.25)
    triggered, _ = check_sector_cap(
        rails=rails, sector="Tech", sector_exposure_after=0.10,
    )
    assert triggered is False


def test_sector_cap_disabled_when_pct_is_one():
    rails = RiskRails(max_sector_pct=1.0)
    triggered, _ = check_sector_cap(
        rails=rails, sector="Tech", sector_exposure_after=0.99,
    )
    assert triggered is False
