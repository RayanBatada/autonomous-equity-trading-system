"""Event-trigger detection for Mon-Thu thesis refreshes."""

from dataclasses import dataclass, field
from datetime import date

from sma.ingest.store import Store


@dataclass
class TriggerConfig:
    price_move_pct: float = 0.05
    material_filing_types: list[str] = field(
        default_factory=lambda: ["8-K", "10-K", "10-Q"]
    )


def refresh_order(
    *,
    triggered: list[str],
    held: list[str],
    universe: list[str],
) -> list[str]:
    """Order tonight's thesis refreshes: HELD names first, then event-triggered.

    Why held names must be included at all (2026-07-29): the thesis overlay's
    `strong_bearish` **exit trigger only fires on tickers we currently hold**, but
    `tickers_needing_refresh` triggers solely on today's earnings / filing / >=5%
    price move. Holding a name was never a trigger, so a position could sit in the
    book indefinitely with no thesis and the exit signal could never fire. Measured
    on the live book that day: **7 of 11 held names had stale-or-missing theses**
    (DKNG 42d, MPC 30d, MRNA 27d, HUM never), i.e. the exit rule was dead on the
    majority of the book during the project's worst drawdown.

    Why held names must come FIRST: the run has a deadline budget that stops
    STARTING new tickers once decide needs the writer lock — 19:58 ET on asof
    while decide is still pending, plus a 02:30 ET hard floor the next morning
    (`sma.agents.__main__._deadline_reached`). Anything still in the tail at
    that point is simply skipped, so putting the book last would reintroduce
    the same starvation. Held names are also cheap — a book is ~10-15 names,
    pennies of Haiku.

    Held names outside `universe` are dropped (never send a delisted/removed
    ticker to the LLM). Deterministic: both groups are sorted.
    """
    universe_set = set(universe)
    held_sorted = sorted({t for t in held if t in universe_set})
    held_set = set(held_sorted)
    rest = sorted({t for t in triggered if t in universe_set and t not in held_set})
    return held_sorted + rest


def tickers_needing_refresh(
    store: Store,
    asof_date: date,
    universe: list[str],
    cfg: TriggerConfig,
) -> list[str]:
    """Return universe tickers with at least one active trigger on asof_date.

    Triggers:
      1. Actual earnings released today (eps_actual IS NOT NULL).
      2. Material filing today (filing_type in cfg.material_filing_types).
      3. |today's adj_close vs yesterday's| >= cfg.price_move_pct.

    Returned list is sorted, deduplicated, and intersected with universe.
    """
    universe_set = set(universe)
    triggered: set[str] = set()

    # 1. Earnings actually released today
    for (t,) in store.conn.execute(
        """
        SELECT DISTINCT ticker FROM earnings
        WHERE report_date = ? AND eps_actual IS NOT NULL
        """,
        [asof_date],
    ).fetchall():
        if t in universe_set:
            triggered.add(t)

    # 2. Material filing today
    if cfg.material_filing_types:
        placeholders = ",".join("?" * len(cfg.material_filing_types))
        for (t,) in store.conn.execute(
            f"""
            SELECT DISTINCT ticker FROM filings
            WHERE filed_at::DATE = ? AND filing_type IN ({placeholders})
            """,
            [asof_date, *cfg.material_filing_types],
        ).fetchall():
            if t in universe_set:
                triggered.add(t)

    # 3. Significant price move today
    for (t,) in store.conn.execute(
        """
        WITH canonical AS (
            -- One row per (ticker, date), yfinance preferred over alpaca, so the
            -- LAG compares trading days — not a yfinance row against an alpaca
            -- row of the SAME day (which produced false/missed price-move
            -- triggers). Matches eval/evaluate_strategy's precedence.
            SELECT ticker, date, adj_close,
                   ROW_NUMBER() OVER (
                       PARTITION BY ticker, date
                       ORDER BY CASE source WHEN 'yfinance' THEN 0
                                            WHEN 'alpaca' THEN 1 ELSE 2 END
                   ) AS rn
            FROM prices
            WHERE date >= ? - INTERVAL '5 days' AND adj_close IS NOT NULL
        ),
        p AS (
            SELECT ticker, date, adj_close,
                   LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) AS prev
            FROM canonical WHERE rn = 1
        )
        SELECT ticker FROM p
        WHERE date = ? AND prev IS NOT NULL
          AND ABS(adj_close - prev) / NULLIF(prev, 0) >= ?
        """,
        [asof_date, asof_date, cfg.price_move_pct],
    ).fetchall():
        if t in universe_set:
            triggered.add(t)

    return sorted(triggered)
