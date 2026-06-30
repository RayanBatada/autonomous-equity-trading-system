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
