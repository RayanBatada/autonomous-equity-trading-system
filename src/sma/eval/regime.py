"""Data-driven trend/reversal regime classification.

The model is, by construction, a cross-sectional momentum factor. The regime
that matters for it is whether momentum WORKED or INVERTED over the forward
window — NOT a calendar guess. `per_feature_ic.py` / `feature_subset_experiment.py`
used to hard-split the asof dates at the val-window midpoint and label the
halves "trend" / "reversal" by calendar, which (2026-06-25 diagnosis) mislabeled
strongly-trending months (Jul-Oct 2025, monthly IC +0.12..+0.16) as "reversal"
and gave false confidence to feature A/Bs.

This classifies each asof by the realized forward return of a simple momentum
factor (top-tercile-by-past-return minus bottom-tercile): positive => momentum
worked (trend), negative => momentum inverted (reversal). It's the regime the
model actually lives or dies by.
"""

from __future__ import annotations

import math


def momentum_factor_return(
    past_returns: dict, fwd_returns: dict, *, quantile: int = 3
) -> float | None:
    """Forward return of (top minus bottom) past-return quantile.

    >0  => high-past-return names outperformed (momentum worked = TREND regime)
    <0  => they underperformed (momentum inverted = REVERSAL regime)

    Returns None when there aren't enough names to form both buckets.
    """
    common = [
        t
        for t in past_returns
        if t in fwd_returns
        and past_returns[t] is not None
        and fwd_returns[t] is not None
        and not math.isnan(past_returns[t])
        and not math.isnan(fwd_returns[t])
    ]
    if len(common) < quantile * 2:
        return None
    ranked = sorted(common, key=lambda t: past_returns[t])
    k = len(ranked) // quantile
    if k == 0:
        return None
    bottom, top = ranked[:k], ranked[-k:]
    top_ret = sum(fwd_returns[t] for t in top) / len(top)
    bot_ret = sum(fwd_returns[t] for t in bottom) / len(bottom)
    return top_ret - bot_ret


def label_from_factor(factor_ret: float | None) -> str | None:
    """'trend' if the momentum factor made money, 'reversal' if it lost. None if
    undefined (the caller should drop the date, not guess)."""
    if factor_ret is None or math.isnan(factor_ret):
        return None
    return "trend" if factor_ret >= 0.0 else "reversal"


def _past_return(con, asof, universe: set, sessions: int) -> dict:
    dates = con.execute(
        "SELECT DISTINCT date FROM prices WHERE date <= ? ORDER BY date DESC LIMIT ?",
        [asof, sessions + 1],
    ).fetchall()
    if len(dates) < sessions + 1:
        return {}
    d_now, d_then = dates[0][0], dates[sessions][0]
    rows = con.execute(
        """
        SELECT ticker,
               MAX(CASE WHEN date=? THEN adj_close END),
               MAX(CASE WHEN date=? THEN adj_close END)
        FROM prices WHERE source='yfinance' AND date IN (?, ?)
        GROUP BY ticker
    """,
        [d_now, d_then, d_now, d_then],
    ).fetchall()
    return {
        t: (now / then - 1.0)
        for t, now, then in rows
        if t in universe and now and then and then > 0
    }


def _fwd_return(con, asof, universe: set, sessions: int) -> dict:
    fwd = con.execute(
        "SELECT DISTINCT date FROM prices WHERE date > ? ORDER BY date LIMIT ?",
        [asof, sessions + 1],
    ).fetchall()
    if len(fwd) < sessions + 1:
        return {}
    entry_d, future_d = fwd[0][0], fwd[sessions][0]
    rows = con.execute(
        """
        SELECT ticker,
               MAX(CASE WHEN date=? THEN open*adj_close/NULLIF(close,0) END),
               MAX(CASE WHEN date=? THEN adj_close END)
        FROM prices WHERE source='yfinance' AND date IN (?, ?)
        GROUP BY ticker
    """,
        [entry_d, future_d, entry_d, future_d],
    ).fetchall()
    return {
        t: (f / e - 1.0)
        for t, e, f in rows
        if t in universe and e and f and e > 0
    }


def classify_regimes(
    con, asofs, universe, *, past_sessions: int = 60, fwd_sessions: int = 30, quantile: int = 3
) -> dict:
    """{asof: (label, factor_return)} for each asof where it's defined.

    `con` is a (read-only is fine) DuckDB connection with a `prices` table.
    Drops dates with insufficient history/forward data rather than guessing.
    """
    uni = {t for t in universe if t != "SPY"}
    out = {}
    for asof in asofs:
        past = _past_return(con, asof, uni, past_sessions)
        fwd = _fwd_return(con, asof, uni, fwd_sessions)
        fr = momentum_factor_return(past, fwd, quantile=quantile)
        lab = label_from_factor(fr)
        if lab is not None:
            out[asof] = (lab, fr)
    return out
