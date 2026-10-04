from sma.ingest.sources._finnhub_earnings_symbols import (
    earnings_reverse_map,
    earnings_vendor_aliases,
)


def test_aliases_translate_dash_to_dot():
    """Non-share-class tickers just get the dash-to-dot translation shared
    with alpaca/finnhub_news (a no-op for tickers with no dash)."""
    assert earnings_vendor_aliases("AAPL") == ["AAPL"]
    assert earnings_vendor_aliases("MSFT") == ["MSFT"]


def test_aliases_b_class_ticker_also_gets_a_class_alias():
    """Regression (found live 2026-08-16): Finnhub's earnings endpoints
    (company_earnings AND earnings_calendar) report Berkshire under symbol
    'BRK.A' regardless of query spelling -- it keys company-level earnings
    under a single symbol, not per share class. BRK-B must recognize BOTH
    'BRK.B' (its own dot form, queried with) and 'BRK.A' (what Finnhub
    actually returns) as itself."""
    assert earnings_vendor_aliases("BRK-B") == ["BRK.B", "BRK.A"]


def test_aliases_non_b_class_ticker_unaffected():
    """The A-class fold is scoped to '-B' tickers only -- it must not apply
    to a ticker that merely happens to contain a dash for some other
    reason, or the exemption becomes an overbroad blanket rule."""
    assert earnings_vendor_aliases("TMUS") == ["TMUS"]


def test_reverse_map_is_one_to_many():
    """Same reasoning as the alpaca_news fix (commit 91299f5): a naive 1:1
    {vendor: canonical} dict is non-injective whenever two canonical
    tickers could alias to the same vendor spelling. The reverse map must
    keep a LIST per vendor symbol, not silently keep only the last."""
    reverse = earnings_reverse_map(["AAPL", "BRK-B"])
    assert reverse["AAPL"] == ["AAPL"]
    assert reverse["BRK.B"] == ["BRK-B"]
    assert reverse["BRK.A"] == ["BRK-B"]
    assert "BRK-B" not in reverse  # only vendor-notation keys, never canonical


def test_reverse_map_collision_keeps_both_canonical_tickers():
    """If two canonical tickers in the SAME batch alias to the same vendor
    symbol, the reverse map must return both, not drop one."""
    reverse = earnings_reverse_map(["BRK-B", "BRK.A"])
    assert reverse["BRK.A"] == ["BRK-B", "BRK.A"]
