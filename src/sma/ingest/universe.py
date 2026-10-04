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


def _as_date(v) -> date | None:
    if v is None or isinstance(v, date):
        return v
    return date.fromisoformat(str(v))


#: Sections of universe_history.yaml that carry membership intervals. Both have
#: the same {SYMBOL: {added, removed}} shape and are merged into one map; the
#: split is documentation, not semantics. Later sections win on a duplicate.
_MEMBERSHIP_SECTIONS = ("members", "former_tickers")


def load_training_membership(
    history_path: Path | str,
) -> dict[str, tuple[date | None, date | None]]:
    """Point-in-time membership intervals, from universe_history.yaml.

    Returns {SYMBOL: (added, removed)}; added None = "since data start",
    removed None = "still a member". Missing or empty file → {} (PIT training
    is additive — a fresh checkout trains exactly as before).

    Two sections, merged:
      `members:`        CURRENT universe names and the date each became a
                        plausible member (point-in-time S&P 500 membership).
                        Their universe.yaml `added` dates are operational file
                        dates from April 2026, so without this section training
                        treats today's universe as having always existed — the
                        survivorship the breadth study measured at +0.76pp of
                        30d top-15 excess overall and +2.73pp on 2023+.
      `former_tickers:` names that LEFT the universe, with a real `removed`
                        date, so delisted/dropped names contribute rows for
                        exactly their membership window and the cross-section
                        keeps the losers today's universe survived.

    Applied by model.loader.build_training_set(membership=...); wired into
    `train` by default, disable with `--no-pit-universe`.
    """
    p = Path(history_path)
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text()) or {}
    out: dict[str, tuple[date | None, date | None]] = {}
    for section in _MEMBERSHIP_SECTIONS:
        for sym, meta in (raw.get(section) or {}).items():
            meta = meta or {}
            out[sym.upper()] = (
                _as_date(meta.get("added")), _as_date(meta.get("removed")),
            )
    return out


def universe_asof(path: Path | str, asof_date: date) -> list[str]:
    """Sorted symbols that were in the universe as of asof_date (added on/before
    it). Entries with no recorded added date are treated as always present, so a
    legacy list-form universe returns every symbol at any asof."""
    membership = load_universe_membership(path)
    return sorted(
        sym for sym, added in membership.items() if added is None or added <= asof_date
    )
