"""Tests for the static GICS sector mapping in sma.sectors."""

from pathlib import Path

import yaml

from sma.sectors import SECTORS, sector_for, sector_map_for


def test_sector_for_known_ticker_returns_expected_sector():
    assert sector_for("AAPL") == "Information Technology"
    assert sector_for("JPM") == "Financials"
    assert sector_for("SPY") == "ETF"
    assert sector_for("LIN") == "Materials"
    assert sector_for("NEE") == "Utilities"
    assert sector_for("DIS") == "Communication Services"


def test_sector_for_unknown_ticker_returns_unknown():
    assert sector_for("ZZZ_NONEXISTENT") == "Unknown"


def test_all_universe_tickers_have_a_sector_mapping():
    """Every ticker in universe.yaml must be in SECTORS (no fallback to Unknown)."""
    universe_path = Path(__file__).parents[2] / "src" / "sma" / "universe.yaml"
    with universe_path.open() as fh:
        data = yaml.safe_load(fh)
    tickers = data["universe"]["tickers"]
    missing = [t for t in tickers if t not in SECTORS]
    assert missing == [], f"Tickers in universe.yaml not in SECTORS: {missing}"


def test_sector_cap_now_meaningful_distinct_sectors():
    """The real sector map must expose multiple distinct sectors (not just 'Unknown')."""
    universe_path = Path(__file__).parents[2] / "src" / "sma" / "universe.yaml"
    with universe_path.open() as fh:
        data = yaml.safe_load(fh)
    tickers = data["universe"]["tickers"]
    sector_values = set(sector_map_for(tickers).values())
    assert len(sector_values) >= 5, (
        f"Expected at least 5 distinct sectors, got {len(sector_values)}: {sector_values}"
    )
    assert "Unknown" not in sector_values, (
        "Universe tickers should not map to 'Unknown' -- stub may still be active"
    )


def test_real_estate_sector_now_populated():
    """2026-05-24 expansion added the Real Estate sector (was absent entirely).
    Verifies the 25%-per-sector cap rail can now bound a real-estate concentration
    instead of silently dumping REITs into 'Unknown'."""
    universe_path = Path(__file__).parents[2] / "src" / "sma" / "universe.yaml"
    with universe_path.open() as fh:
        data = yaml.safe_load(fh)
    tickers = data["universe"]["tickers"]
    sector_map = sector_map_for(tickers)
    re_tickers = {t for t, sec in sector_map.items() if sec == "Real Estate"}
    assert len(re_tickers) >= 3, (
        f"Real Estate sector should have >= 3 names; got {re_tickers}"
    )
    # Data center REITs specifically — the AI-infra angle that motivated the
    # add.
    assert "EQIX" in re_tickers
    assert "DLR" in re_tickers
