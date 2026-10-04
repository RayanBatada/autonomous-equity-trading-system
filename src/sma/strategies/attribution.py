"""Per-sleeve attribution: what each sleeve proposed, and what it earned.

Tables (migration 10, sma.ingest.store):
  sleeve_targets(asof_date, session, sleeve, ticker, weight, mode,
                 capital_fraction, run_id, created_at)
  sleeve_daily_returns(asof_date, sleeve, mode, ret, gross, run_id)

Return convention (both modes): a row keyed asof_date=D is the return of the
book decided on the evening of D, over the NEXT session S: from D's close to
S's close. S is the next date with SPY in `prices`, so a book is scored only
once the night's ingest has landed S (decide at 20:00 on S scores the book
from D; reconcile retries anything still pending). `ret` is on the sleeve's
own capital; `gross` is the book's gross weight in sleeve-capital terms.

Shadow sleeves: ret = sum_i w_i * (adj_close_i(S) / adj_close_i(D) - 1).
Frictionless: no 16.6bp/side cost, no open-auction slippage, and the whole
move from D's close counts even though a real order would fill at S's open.
Good enough to rank a shadow sleeve against the incumbent on the same basis
(score the incumbent's own book the same way if you want an apples-to-apples
comparison); not a P&L claim.

Live sleeves: the realized aggregate book return, equity(S) / equity(D) - 1
from account_snapshots, divided by the sum of live capital fractions (the
remainder is uninvested cash earning ~0 on paper). With ONE live sleeve this
is exact: it is the book. With N>1 live sleeves it is an approximation that
assigns every live sleeve the same return on its capital; the right
decomposition (position-level P&L split by each sleeve's share of the
aggregate target weight) needs fills attributed to sleeves and is not built
yet. Deposits/withdrawals would also pollute it (none on the paper account).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date

logger = logging.getLogger(__name__)

REFERENCE_TICKER = "SPY"


def persist_sleeve_targets(conn, *, asof: date, session: str, proposals, run_id: int) -> int:
    """Write every proposal that produced a book (live and shadow). Re-running
    the same (asof, session, sleeve) replaces its rows. Returns rows written."""
    written = 0
    for p in proposals:
        if p.book is None:
            continue
        conn.execute(
            "DELETE FROM sleeve_targets WHERE asof_date = ? AND session = ? AND sleeve = ?",
            [asof, session, p.name],
        )
        rows = [
            [asof, session, p.name, t, float(w), p.mode, float(p.capital_fraction), run_id]
            for t, w in p.book.weights.items()
        ]
        if rows:
            conn.executemany(
                "INSERT INTO sleeve_targets (asof_date, session, sleeve, ticker, weight, "
                "mode, capital_fraction, run_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        written += len(rows)
    return written


def next_session(conn, asof: date) -> date | None:
    row = conn.execute(
        "SELECT MIN(date) FROM prices WHERE ticker = ? AND date > ? AND close IS NOT NULL",
        [REFERENCE_TICKER, asof],
    ).fetchone()
    return row[0] if row and row[0] is not None else None


def _closes(conn, tickers: list[str], day: date) -> dict[str, float]:
    """adj_close per ticker on exactly `day`, same source priority as decide."""
    if not tickers:
        return {}
    rows = conn.execute(
        """
        SELECT ticker, adj_close FROM (
            SELECT ticker, adj_close,
                   ROW_NUMBER() OVER (
                       PARTITION BY ticker
                       ORDER BY CASE source WHEN 'yfinance' THEN 0
                                             WHEN 'alpaca' THEN 1 ELSE 2 END
                   ) AS rn
            FROM prices
            WHERE ticker = ANY($tickers) AND date = $day AND adj_close IS NOT NULL
        ) t WHERE rn = 1
        """,
        {"tickers": list(tickers), "day": day},
    ).fetchall()
    return {t: float(c) for t, c in rows if c is not None and c > 0}


@dataclass
class SleeveScore:
    sleeve: str
    mode: str
    ret: float
    gross: float
    realized_on: date
    missing: list[str]


def score_shadow(conn, asof: date) -> dict[str, SleeveScore]:
    """Hypothetical next-session return of every SHADOW sleeve book persisted
    for `asof`. Empty dict when there is nothing to score yet.

    A ticker with no close on the next session is skipped (not scored) while
    that session is the newest one in `prices` (ingest may still be partial);
    once a later session exists it is genuinely missing (halted, delisted)
    and counts as a 0% return, listed in `missing`.
    """
    s1 = next_session(conn, asof)
    if s1 is None:
        return {}
    later_exists = next_session(conn, s1) is not None
    books: dict[str, dict[str, float]] = {}
    for sleeve, ticker, w in conn.execute(
        "SELECT sleeve, ticker, weight FROM sleeve_targets "
        "WHERE asof_date = ? AND mode = 'shadow' ORDER BY sleeve, ticker",
        [asof],
    ).fetchall():
        books.setdefault(sleeve, {})[ticker] = float(w)
    out: dict[str, SleeveScore] = {}
    for sleeve, weights in books.items():
        tickers = [t for t, w in weights.items() if w != 0]
        c0 = _closes(conn, tickers, asof)
        c1 = _closes(conn, tickers, s1)
        missing = [t for t in tickers if t not in c0 or t not in c1]
        if missing and not later_exists:
            continue
        ret = sum(weights[t] * (c1[t] / c0[t] - 1.0) for t in tickers if t not in missing)
        out[sleeve] = SleeveScore(
            sleeve, "shadow", ret, float(sum(weights.values())), s1, missing,
        )
    return out


# Alpaca's OFFICIAL daily close (see sma.live.reconcile.backfill_official_closes).
OFFICIAL_EQUITY_SOURCES = frozenset({"portfolio_history_daily", "portfolio_history_daily_close"})
# reconcile only heals snapshots this many trading days back
# (reconcile.BACKFILL_LOOKBACK_TRADING_DAYS); older rows are final as they are.
HEAL_WINDOW_SESSIONS = 5


def _equity(conn, day: date) -> float | None:
    """Snapshot equity for `day`, or None while it is not yet trustworthy: a
    proxy/after-hours equity inside reconcile's heal window may still be
    replaced by the official close, and a return written from it would never
    be revisited (score_pending writes once)."""
    row = conn.execute(
        "SELECT equity, equity_source FROM account_snapshots WHERE asof_date = ?", [day]
    ).fetchone()
    if not row or not row[0]:
        return None
    if row[1] not in OFFICIAL_EQUITY_SOURCES:
        newer = conn.execute(
            "SELECT COUNT(*) FROM account_snapshots WHERE asof_date > ?", [day]
        ).fetchone()[0]
        if newer < HEAL_WINDOW_SESSIONS:
            return None
    return float(row[0])


def score_live(conn, asof: date) -> dict[str, SleeveScore]:
    """Pro-rata attribution of the realized book return to live sleeves (exact
    for a single live sleeve; see the module docstring for N>1)."""
    s1 = next_session(conn, asof)
    if s1 is None:
        return {}
    e0, e1 = _equity(conn, asof), _equity(conn, s1)
    if not e0 or not e1:
        return {}
    rows = conn.execute(
        "SELECT sleeve, MAX(capital_fraction), SUM(weight) FROM sleeve_targets "
        "WHERE asof_date = ? AND mode = 'live' GROUP BY sleeve ORDER BY sleeve",
        [asof],
    ).fetchall()
    total_f = sum(float(f or 0.0) for _, f, _ in rows)
    if not rows or total_f <= 0:
        return {}
    agg = e1 / e0 - 1.0
    return {
        sleeve: SleeveScore(sleeve, "live", agg / total_f, float(gross or 0.0), s1, [])
        for sleeve, _f, gross in rows
    }


def score_pending(conn, *, run_id: int, lookback_days: int = 30) -> int:
    """Score every (asof, sleeve) in sleeve_targets from the last
    `lookback_days` that has no sleeve_daily_returns row yet and whose next
    session has landed. Idempotent; a dark night just leaves rows pending
    until the next run. Returns rows written."""
    pending = conn.execute(
        """
        SELECT DISTINCT t.asof_date FROM sleeve_targets t
        LEFT JOIN sleeve_daily_returns r
          ON r.asof_date = t.asof_date AND r.sleeve = t.sleeve
        WHERE r.sleeve IS NULL
          AND t.asof_date >= (SELECT MAX(asof_date) FROM sleeve_targets) - ?::INTEGER
        ORDER BY 1
        """,
        [lookback_days],
    ).fetchall()
    written = 0
    for (asof,) in pending:
        scores = {**score_live(conn, asof), **score_shadow(conn, asof)}
        for s in scores.values():
            if not math.isfinite(s.ret):
                logger.warning("sleeve %s %s: non-finite return, not written", s.sleeve, asof)
                continue
            if s.missing:
                logger.warning(
                    "sleeve %s %s: no price for %s on %s, counted as 0%%",
                    s.sleeve, asof, ",".join(s.missing), s.realized_on,
                )
            conn.execute(
                "INSERT INTO sleeve_daily_returns (asof_date, sleeve, mode, ret, gross, run_id) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                [asof, s.sleeve, s.mode, s.ret, s.gross, run_id],
            )
            written += 1
    return written


@dataclass
class SleeveStatus:
    name: str
    mode: str
    capital_fraction: float
    enabled: bool
    last_asof: date | None
    last_ret: float | None
    last_ret_asof: date | None
    scored_days: int
    cum_ret: float | None


def sleeve_status(conn, sleeves) -> list[SleeveStatus]:
    """Status rows for `python -m sma.live status`. Tolerates the tables not
    existing yet (migration 10 lands on the next writable job run)."""
    out = []
    for s in sleeves:
        last_asof = last_ret = last_ret_asof = cum = None
        n = 0
        try:
            row = conn.execute(
                "SELECT MAX(asof_date) FROM sleeve_targets WHERE sleeve = ?", [s.name]
            ).fetchone()
            last_asof = row[0] if row else None
            rets = conn.execute(
                "SELECT asof_date, ret FROM sleeve_daily_returns WHERE sleeve = ? "
                "ORDER BY asof_date",
                [s.name],
            ).fetchall()
            n = len(rets)
            if rets:
                last_ret_asof, last_ret = rets[-1][0], float(rets[-1][1])
                growth = 1.0
                for _, r in rets:
                    growth *= 1.0 + float(r)
                cum = growth - 1.0
        except Exception:  # noqa: BLE001 - table missing on an unmigrated DB
            pass
        out.append(SleeveStatus(
            s.name, s.mode, float(s.capital_fraction), bool(s.enabled),
            last_asof, last_ret, last_ret_asof, n, cum,
        ))
    return out


def format_sleeve_status(rows: list[SleeveStatus]) -> list[str]:
    lines = ["Sleeves:"]
    for r in rows:
        state = r.mode if r.enabled else f"{r.mode}, disabled"
        last = r.last_asof.isoformat() if r.last_asof else "never"
        line = f"  {r.name:18s} {state:16s} {r.capital_fraction:>5.0%}  last asof {last}"
        if r.last_ret is not None:
            line += (
                f"  last ret {r.last_ret:+.2%} ({r.last_ret_asof.isoformat()})"
                f"  cum {r.cum_ret:+.2%} over {r.scored_days}d"
            )
        lines.append(line)
    return lines
