"""Load the trading universe from universe.yaml.

`universe.tickers` may be either:
  - a plain list of symbols (legacy), or
  - a map of {symbol: added-date} (current) recording point-in-time membership.

load_universe() returns the deduped/uppercased/sorted symbol list for BOTH forms
(iterating a dict yields its keys), so it is unchanged by the format. The dated
form additionally powers universe_asof() for point-in-time backtests.
"""

from datetime import date
from pathlib import Path

import yaml


def load_universe(path: Path | str) -> list[str]:
    raw = yaml.safe_load(Path(path).read_text())
    tickers = raw["universe"]["tickers"]
    if not tickers:
        raise ValueError("universe.tickers is empty")
    return sorted({t.upper() for t in tickers})


def load_universe_membership(path: Path | str) -> dict[str, date | None]:
    """Return {SYMBOL: added_date}. added_date is None for the legacy list form
    or any entry with no recorded date (treated as "always present")."""
    raw = yaml.safe_load(Path(path).read_text())
    tickers = raw["universe"]["tickers"]
    if not tickers:
        raise ValueError("universe.tickers is empty")
    out: dict[str, date | None] = {}
    if isinstance(tickers, dict):
        for sym, added in tickers.items():
            if added is None or isinstance(added, date):
                out[sym.upper()] = added
            else:
                out[sym.upper()] = date.fromisoformat(str(added))
    else:  # legacy list form — no dates recorded
        for sym in tickers:
            out[sym.upper()] = None
    return out


def universe_asof(path: Path | str, asof_date: date) -> list[str]:
    """Sorted symbols that were in the universe as of asof_date (added on/before
    it). Entries with no recorded added date are treated as always present, so a
    legacy list-form universe returns every symbol at any asof."""
    membership = load_universe_membership(path)
    return sorted(
        sym for sym, added in membership.items() if added is None or added <= asof_date
    )
