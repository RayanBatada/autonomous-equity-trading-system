"""PIT S&P 500 membership helper from fja05680/sp500 snapshot CSV."""
import csv
from datetime import date
from functools import lru_cache

PIT_CSV = "/Users/youruser/.sma-pit/sp500_pit.csv"


def _norm(t: str) -> str:
    return t.upper().replace("-", ".").strip()


@lru_cache(maxsize=1)
def _snapshots():
    rows = []
    with open(PIT_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            d = date.fromisoformat(row["date"])
            members = frozenset(_norm(t) for t in row["tickers"].split(","))
            rows.append((d, members))
    rows.sort(key=lambda x: x[0])
    return rows


def members_asof(d: date) -> frozenset:
    """S&P 500 members as of date d (latest snapshot on/before d)."""
    snaps = _snapshots()
    out = frozenset()
    for sd, members in snaps:
        if sd <= d:
            out = members
        else:
            break
    return out


def is_member(ticker: str, d: date) -> bool:
    return _norm(ticker) in members_asof(d)
