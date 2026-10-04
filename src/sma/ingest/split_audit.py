"""Cross-source split-consistency audit (2026-10-01).

The bug this exists for: `prices` mixed split-adjusted and unadjusted rows.
MNST split 2:1 on 2026-08-11. The nightly 45-day yfinance window rewrote rows
inside the window on the post-split scale while rows older than the window
kept the pre-split scale, leaving a fake -51% day at 2026-07-17 -> 07-20 in the
series the model reads. The live-attribution study (2026-10-01) traced a month
of MNST sitting in the top 10 to it.

The rule (one function, shared by scripts/audit_split_consistency.py and the
`no_split_inconsistency` ingest quality check):

  * For every ticker, compute each source's own close-to-close return over its
    own consecutive rows.
  * Both sources have a return for the same date over the same interval and
    |r_left - r_right| > `threshold` (default 20%): flag.
  * Only one source has a return for that date (the other is missing the day,
    or covers a different interval) and |r| > `solo_threshold` (default 40%):
    flag.

Left is the yfinance series (or the feature-served series, see
FEATURE_SERIES_SQL); right is alpaca. Alpaca bars are RAW (never
split-adjusted, by design: they are the same scale as our fills), so a real
split shows up as an alpaca-only jump next to a flat yfinance day. Those are
classified `alpaca_raw_split` and are expected, not contamination: nothing
the model reads uses alpaca's scale (alpaca adj_close is NULL, and the feature
query requires adj_close IS NOT NULL). Every other kind means the series the
features read has a discontinuity the other source does not confirm.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

DEFAULT_THRESHOLD = 0.20
DEFAULT_SOLO_THRESHOLD = 0.40

# Kinds that do NOT touch the feature series (alpaca is raw by design).
BENIGN_KINDS = frozenset({"alpaca_raw_split", "alpaca_solo_jump"})

# Clean split ratios (new shares per old share, both directions). A one-day
# price ratio within _SPLIT_RATIO_TOL of 1/k or k for one of these reads as a
# split, not a market move.
_SPLIT_FACTORS = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 25.0, 50.0)
_SPLIT_RATIO_TOL = 0.06

# The series yfinance itself stores: Close with auto_adjust=False, i.e.
# split-adjusted but not dividend-adjusted.
YFINANCE_SERIES_SQL = """
    SELECT ticker, date, close AS px FROM prices
    WHERE source = 'yfinance' AND close IS NOT NULL AND NOT isnan(close) AND close > 0
"""

# Exactly the rows sma.model.predictor / sma.model.__main__ hand to the
# feature builder: one row per (ticker, date), yfinance > alpaca > anything
# else (yfinance_hist), adj_close IS NOT NULL. This is the series that matters.
FEATURE_SERIES_SQL = """
    SELECT ticker, date, adj_close AS px FROM (
        SELECT ticker, date, adj_close,
               ROW_NUMBER() OVER (
                   PARTITION BY ticker, date
                   ORDER BY CASE source WHEN 'yfinance' THEN 0
                                        WHEN 'alpaca' THEN 1 ELSE 2 END
               ) AS rn
        FROM prices WHERE adj_close IS NOT NULL
    ) WHERE rn = 1 AND NOT isnan(adj_close) AND adj_close > 0
"""

ALPACA_SERIES_SQL = """
    SELECT ticker, date, close AS px FROM prices
    WHERE source = 'alpaca' AND close IS NOT NULL AND NOT isnan(close) AND close > 0
"""


@dataclass(frozen=True)
class SplitFlag:
    ticker: str
    date: date
    kind: str  # see _classify
    left_ret: float | None
    right_ret: float | None

    @property
    def feature_affecting(self) -> bool:
        return self.kind not in BENIGN_KINDS

    def describe(self) -> str:
        def f(x):
            return "-" if x is None else f"{x:+.1%}"

        return (
            f"{self.ticker} {self.date} {self.kind} "
            f"left={f(self.left_ret)} alpaca={f(self.right_ret)}"
        )


def _split_like(ratio: float | None) -> bool:
    if ratio is None or ratio <= 0:
        return False
    for k in _SPLIT_FACTORS:
        for target in (k, 1.0 / k):
            if abs(ratio / target - 1.0) <= _SPLIT_RATIO_TOL:
                return True
    return False


def _classify(lr: float, rr: float, threshold: float) -> str:
    """Name a comparable-day divergence. The RELATIVE one-day ratio between
    the sources is what a split moves (MSTR 10:1 on a +9% day: alpaca -89.1%,
    yfinance +9.1%, relative ratio 0.100)."""
    rel = (1.0 + rr) / (1.0 + lr) if (1.0 + lr) > 0 else None
    if _split_like(rel):
        if abs(lr) < threshold:
            return "alpaca_raw_split"  # expected: alpaca is raw by design
        if abs(rr) < threshold:
            return "left_scale_break"  # the bad one: MNST 2026-07-20
    return "divergence"


def find_split_inconsistencies(
    conn,
    *,
    left_sql: str = YFINANCE_SERIES_SQL,
    right_sql: str = ALPACA_SERIES_SQL,
    threshold: float = DEFAULT_THRESHOLD,
    solo_threshold: float = DEFAULT_SOLO_THRESHOLD,
    since: date | None = None,
    until: date | None = None,
    tickers: list[str] | None = None,
) -> list[SplitFlag]:
    """Flag (ticker, date) pairs breaking the rule in the module docstring.

    `since`/`until` bound the FLAGGED dates; returns are still computed off
    each source's previous row even when it predates `since`. Read-only.
    """
    sql = f"""
        WITH l0 AS ({left_sql}), r0 AS ({right_sql}),
        l AS (
            SELECT ticker, date, px,
                   LAG(px)   OVER (PARTITION BY ticker ORDER BY date) AS ppx,
                   LAG(date) OVER (PARTITION BY ticker ORDER BY date) AS pdate
            FROM l0 WHERE ($tickers IS NULL OR ticker = ANY($tickers))
        ),
        r AS (
            SELECT ticker, date, px,
                   LAG(px)   OVER (PARTITION BY ticker ORDER BY date) AS ppx,
                   LAG(date) OVER (PARTITION BY ticker ORDER BY date) AS pdate
            FROM r0 WHERE ($tickers IS NULL OR ticker = ANY($tickers))
        )
        SELECT COALESCE(l.ticker, r.ticker) AS ticker,
               COALESCE(l.date, r.date) AS date,
               l.px / l.ppx - 1 AS lr, l.pdate AS lpd,
               r.px / r.ppx - 1 AS rr, r.pdate AS rpd
        FROM l FULL OUTER JOIN r ON l.ticker = r.ticker AND l.date = r.date
        WHERE ($since IS NULL OR COALESCE(l.date, r.date) >= $since)
          AND ($until IS NULL OR COALESCE(l.date, r.date) <= $until)
          AND (abs(l.px / l.ppx - 1) > $thr / 2 OR abs(r.px / r.ppx - 1) > $thr / 2)
        ORDER BY 2, 1
    """
    rows = conn.execute(
        sql,
        {"tickers": tickers, "since": since, "until": until, "thr": float(threshold)},
    ).fetchall()
    out: list[SplitFlag] = []
    for ticker, d, lr, lpd, rr, rpd in rows:
        comparable = lr is not None and rr is not None and lpd == rpd
        if comparable:
            if abs(lr - rr) <= threshold:
                continue
            kind = _classify(lr, rr, threshold)
            out.append(SplitFlag(ticker, d, kind, lr, rr))
            continue
        # One source missing this day (or the two cover different intervals):
        # each side's own jump stands alone against the solo threshold.
        if lr is not None and abs(lr) > solo_threshold:
            out.append(SplitFlag(ticker, d, "left_solo_jump", lr, rr))
        if rr is not None and abs(rr) > solo_threshold:
            out.append(SplitFlag(ticker, d, "alpaca_solo_jump", lr, rr))
    return out


def trailing_session_start(conn, *, asof: date, sessions: int) -> date | None:
    """The oldest of the last `sessions` distinct price dates <= asof."""
    row = conn.execute(
        "SELECT MIN(date) FROM (SELECT DISTINCT date FROM prices WHERE date <= ? "
        "ORDER BY date DESC LIMIT ?)",
        [asof, int(sessions)],
    ).fetchone()
    return row[0] if row else None
