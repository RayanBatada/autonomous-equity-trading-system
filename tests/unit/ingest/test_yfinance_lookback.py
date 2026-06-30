"""2026-06-11 KLAC 10:1 split incident: with lookback_days=1, a provider-side
back-adjustment (split) rescales history the nightly fetch never re-syncs —
leaving a permanent 10x discontinuity at the fetch boundary inside the
quality gate's 30-day extreme-moves scan. One corrupt/rescaled stored bar
then blocks EVERY night for a month. The fetch window must cover the scan
window so provider re-adjustments propagate consistently."""

from sma.ingest.sources.yfinance_prices import YFinancePricesSource


def test_default_lookback_covers_quality_scan_window():
    src = YFinancePricesSource()
    assert src.lookback_days >= 40, (
        "lookback must exceed the quality gate's 30-day extreme-moves scan "
        "so split re-adjustments re-sync the whole scanned window"
    )
