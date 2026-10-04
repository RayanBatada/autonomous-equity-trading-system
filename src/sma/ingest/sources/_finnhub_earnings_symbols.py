"""Shared Finnhub earnings-endpoint symbol translation.

Finnhub's two earnings endpoints (company_earnings, earnings_calendar) key
company-level earnings under Finnhub's OWN symbol, not necessarily the one
we queried with. Two translations stack:

1. Share-class notation: Finnhub uses DOT notation (BRK.B) where the
   universe is yfinance-canonical DASH notation (BRK-B) -- the same
   translation alpaca_prices, alpaca_news, and finnhub_news already do.
2. Company-level folding: verified live 2026-08-16, Finnhub reports
   Berkshire's earnings under a single symbol for the whole company
   regardless of share class -- BOTH company_earnings and earnings_calendar
   return rows/events under symbol="BRK.A" whether queried as "BRK-B" or
   "BRK.B". Translation (1) alone still left `backfill_earnings` storing 4
   rows under ticker='BRK.A' (a symbol outside the universe -- canonical is
   BRK-B -- so nothing read them), and left earnings_calendar's
   `sym in ticker_set` filter dropping every Berkshire row (BRK.A is never
   in ticker_set).

The reverse map (vendor symbol -> canonical ticker) is ONE-TO-MANY, same
reasoning as the alpaca_news fix (commit 91299f5): a naive 1:1
`{vendor: canonical}` dict is non-injective whenever two canonical tickers
alias to the same vendor spelling, so a caller matching against a BATCH of
vendor rows in one response (earnings_calendar) needs a list per vendor
symbol, not a single value that silently drops or misattributes one of them.

Generic, not Berkshire-hardcoded: any "-B" share-class ticker in the
universe gets its "-A" class registered as an additional alias, on the
same company-level-earnings assumption an A-class report IS the company's
report for the corresponding B-class ticker. Today BRK-B is the only such
ticker in universe.yaml.
"""

from collections import defaultdict


def earnings_vendor_aliases(ticker: str) -> list[str]:
    """Vendor-notation spellings Finnhub's earnings endpoints may use for
    `ticker`, in preference order.

    Element 0 is always the dot form of `ticker` (a no-op for tickers with
    no dash) -- that's what we QUERY with. Any additional entries are
    alternate spellings we must also recognize in a RESPONSE.
    """
    dot = ticker.replace("-", ".")
    aliases = [dot]
    if ticker.endswith("-B"):
        aliases.append(f"{ticker[:-2]}.A")
    return aliases


def earnings_reverse_map(tickers: list[str]) -> dict[str, list[str]]:
    """vendor-symbol -> canonical ticker(s) for a batch, one-to-many.

    Used by callers matching a single vendor response against MANY
    requested tickers at once (earnings_calendar). Keys are vendor-notation
    symbols only; canonical tickers with no vendor-notation change (no dash)
    are still present because `earnings_vendor_aliases` always includes the
    dot form, which is a no-op replace when there's nothing to translate.
    """
    reverse: dict[str, list[str]] = defaultdict(list)
    for t in tickers:
        for alias in earnings_vendor_aliases(t):
            reverse[alias].append(t)
    return dict(reverse)
