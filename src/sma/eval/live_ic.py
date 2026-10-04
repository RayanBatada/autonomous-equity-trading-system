"""Live cross-sectional rank-IC: computation shared by the dashboard's Model
tab (dashboard/tabs/model.py, commit 3bd5f87) and the regime-turn monitoring
check (sma.monitoring.check_regime_turn, added 2026-08-25).

Extracted out of dashboard/tabs/model.py so the monitoring job -- a headless
batch job run from launchd, see src/sma/monitoring/__main__.py -- never needs
to import streamlit (a `dev`-only optional dependency, not something a
production job should require to run `python -m sma.monitoring check`).
Every function below is a verbatim port of the dashboard's original private
helper of the same name (module-private `_` prefix dropped); no logic was
changed, only relocated. dashboard/tabs/model.py now imports these directly
and wraps the DB-backed ones in its own `@st.cache_data` decorators -- the
computation itself lives here exactly once.

Rolling live cross-sectional rank-IC: computed from STORED predictions
joined to realized forward returns from STORED prices -- never re-invoking
the model. This is the honest "is there edge, right now" read (see
CLAUDE.md: judge on IC, not P&L).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from sma.db_connect import read_only_connect

DEFAULT_DB_PATH = Path("data/sma.duckdb")
DEFAULT_UNIVERSE_PATH = Path("src/sma/universe.yaml")

MIN_CROSS_SECTION = 5  # below this a per-date Spearman IC isn't meaningful
IC_HORIZONS_DAYS = (10, 20)
IC_REGIME_WINDOW = 21


def forward_returns(
    prices: pd.DataFrame, asof_dates: list[date], horizon_days: int
) -> pd.DataFrame:
    """Realized forward return for every ticker in `prices`, for each date in
    `asof_dates`, `horizon_days` trading sessions out.

    Mirrors sma.model.loader.build_training_set's label convention exactly:
    entry = the NEXT session's split/dividend-adjusted open (where a DAY_OPG
    buy actually fills, `open * adj_close/close`), exit = adj_close
    horizon_days sessions after asof. Using close[asof] as entry would credit
    the untradable close[asof]->open[asof+1] gap -- "realized forward return"
    means the same thing everywhere in this codebase.

    `prices` must have columns [ticker, date, open, close, adj_close] (one
    row per ticker/date -- callers de-duplicate multi-source rows first, see
    ic_prices_df). An asof date within `horizon_days` sessions of the end of
    `prices`' date range, or not present in `prices` at all, simply produces
    no rows for it (no realized return exists yet) -- never a fabricated or
    zero-filled value. Missing per-ticker prices and non-positive
    open/close/adj_close are skipped the same way.

    Returns columns: asof_date, ticker, fwd_return.
    """
    cols = ["asof_date", "ticker", "fwd_return"]
    if prices.empty or not asof_dates:
        return pd.DataFrame(columns=cols)

    all_dates = sorted(prices["date"].unique())
    idx_of = {d: i for i, d in enumerate(all_dates)}
    n = len(all_dates)

    plan = []
    for asof in asof_dates:
        idx = idx_of.get(asof)
        if idx is None:
            continue
        entry_idx, future_idx = idx + 1, idx + horizon_days
        if future_idx >= n:
            continue
        plan.append((asof, all_dates[entry_idx], all_dates[future_idx]))
    if not plan:
        return pd.DataFrame(columns=cols)
    plan_df = pd.DataFrame(plan, columns=["asof_date", "entry_date", "future_date"])

    entry = prices.merge(
        plan_df[["asof_date", "entry_date"]], left_on="date", right_on="entry_date"
    )
    entry = entry[(entry["open"] > 0) & (entry["close"] > 0)].copy()
    entry["entry_px"] = entry["open"] * entry["adj_close"] / entry["close"]

    future = prices.merge(
        plan_df[["asof_date", "future_date"]], left_on="date", right_on="future_date"
    )
    future = future[future["adj_close"] > 0].rename(columns={"adj_close": "future_px"})

    merged = entry[["asof_date", "ticker", "entry_px"]].merge(
        future[["asof_date", "ticker", "future_px"]], on=["asof_date", "ticker"]
    )
    merged = merged[merged["entry_px"] > 0].copy()
    merged["fwd_return"] = merged["future_px"] / merged["entry_px"] - 1.0
    return merged[cols].reset_index(drop=True)


def rank_ic_series(
    predictions: pd.DataFrame,
    forward_returns_df: pd.DataFrame,
    min_n: int = MIN_CROSS_SECTION,
) -> pd.DataFrame:
    """Per-decide-date cross-sectional rank-IC: Spearman correlation between
    that date's prediction ranks and realized forward returns.

    Inner-joins predictions to forward_returns_df on (asof_date, ticker) --
    tickers with no forward return yet (insufficient price history) or not
    present that date simply drop out of the cross-section rather than
    erroring; this is how non-universe/missing-price names get excluded.
    Dates whose surviving cross-section has fewer than `min_n` names are
    excluded too (an IC on a handful of names isn't a meaningful read), as
    are dates with zero-variance predicted_value or fwd_return (Spearman is
    undefined/NaN there -- skip rather than emit a NaN IC that would poison
    the mean/t-stat downstream).

    Returns columns: asof_date, ic, n. Empty/no-surviving-date input returns
    an empty frame with these columns.
    """
    from scipy.stats import spearmanr

    cols = ["asof_date", "ic", "n"]
    if predictions.empty or forward_returns_df.empty:
        return pd.DataFrame(columns=cols)

    merged = predictions.merge(
        forward_returns_df, on=["asof_date", "ticker"], how="inner"
    )
    if merged.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for asof, grp in merged.groupby("asof_date"):
        grp = grp.dropna(subset=["predicted_value", "fwd_return"])
        if len(grp) < min_n:
            continue
        if grp["predicted_value"].nunique() < 2 or grp["fwd_return"].nunique() < 2:
            continue
        ic = spearmanr(grp["predicted_value"], grp["fwd_return"]).correlation
        if ic != ic:  # NaN guard (defensive; nunique>=2 above should prevent this)
            continue
        rows.append({"asof_date": asof, "ic": float(ic), "n": int(len(grp))})
    return pd.DataFrame(rows, columns=cols)


def smoothed_ic(
    ic_series: pd.Series, window: int = IC_REGIME_WINDOW, min_periods: int = 5
) -> pd.Series:
    """Light rolling-mean overlay. `min_periods` (well under `window`) lets
    the smoothed line start early in a short live history instead of
    rendering as all-NaN until `window` dates accumulate."""
    return ic_series.rolling(window=window, min_periods=min_periods).mean()


def trailing_ic_regime(ic_series: pd.Series, window: int = IC_REGIME_WINDOW) -> dict:
    """Mean IC + a simple t-stat over the trailing `window` decide dates --
    the compact "regime read".

    CAVEAT (surface this in the UI, don't just bury it here): consecutive
    decide dates' realized forward returns overlap by (horizon_days - 1)
    sessions, so these `window` IC points are far from independent draws --
    the textbook sqrt(n) t-stat overstates significance. Treat it as a rough
    signpost, not a rigorous hypothesis test.

    Uses the repo's existing |t| > 2 significance bar (see
    sma.eval.performance.regression_alpha's docstring: "require |t| > 2
    before claiming ANY skill"). Returns {mean, t_stat, n, level} where level
    is one of "positive", "negative", "neutral", or "insufficient" (fewer
    than 2 valid points, or zero variance -- can't compute a t-stat).
    """
    tail = ic_series.dropna().tail(window)
    n = len(tail)
    if n < 2:
        return {
            "mean": float(tail.iloc[-1]) if n == 1 else None,
            "t_stat": None,
            "n": n,
            "level": "insufficient",
        }
    mean = float(tail.mean())
    std = float(tail.std(ddof=1))
    if std == 0:
        return {"mean": mean, "t_stat": None, "n": n, "level": "insufficient"}
    t_stat = mean / std * (n**0.5)
    if mean > 0 and t_stat > 2:
        level = "positive"
    elif mean < 0 and t_stat < -2:
        level = "negative"
    else:
        level = "neutral"
    return {"mean": mean, "t_stat": t_stat, "n": n, "level": level}


def decide_dates(db_path: Path = DEFAULT_DB_PATH) -> list[date]:
    """Real live decide dates -- from intended_orders (decide.py's own
    output table), NOT every predictions.asof_date. `predictions` also holds
    `sma model backfill-predictions` walk-forward evaluation rows (2025-07
    onward, multiple model_ids per day) that were never actually used to
    trade; real live decide dates start 2026-04-29."""
    if not Path(db_path).exists():
        return []
    con = read_only_connect(db_path)
    try:
        rows = con.execute(
            "SELECT DISTINCT asof_date FROM intended_orders ORDER BY 1"
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def live_predictions_df(
    db_path: Path = DEFAULT_DB_PATH, universe_path: Path = DEFAULT_UNIVERSE_PATH
) -> pd.DataFrame:
    """One row per (decide date, ticker): predicted_value from the latest
    model_id when a date has more than one (matches the convention already
    used by sma.agents.__main__'s thesis-context builder: `ORDER BY model_id
    DESC LIMIT 1`). Restricted to real decide dates (decide_dates) and the
    current universe, excluding SPY -- a benchmark, not a candidate, same
    exclusion sma.digest.py makes.
    """
    cols = ["asof_date", "ticker", "predicted_value"]
    if not Path(db_path).exists() or not Path(universe_path).exists():
        return pd.DataFrame(columns=cols)
    from sma.ingest.universe import load_universe

    universe = [t for t in load_universe(universe_path) if t != "SPY"]
    dates = decide_dates(db_path)
    if not dates:
        return pd.DataFrame(columns=cols)
    con = read_only_connect(db_path)
    try:
        df = con.execute(
            """
            SELECT asof_date, ticker, predicted_value
            FROM (
                SELECT asof_date, ticker, predicted_value,
                       ROW_NUMBER() OVER (
                           PARTITION BY asof_date, ticker ORDER BY model_id DESC
                       ) AS rn
                FROM predictions
                WHERE target = 'ret_30d_forward'
                  AND asof_date = ANY($decide_dates)
                  AND ticker = ANY($universe)
            ) t
            WHERE rn = 1
            """,
            {"decide_dates": dates, "universe": universe},
        ).fetchdf()
    finally:
        con.close()
    if not df.empty:
        df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.date
    return df


def ic_prices_df(
    db_path: Path = DEFAULT_DB_PATH, universe_path: Path = DEFAULT_UNIVERSE_PATH
) -> pd.DataFrame:
    """De-duplicated (ticker, date) price rows for the current universe,
    preferring the 'yfinance' source over others. Feeds forward_returns for
    the Model Edge IC section / regime-turn monitoring check."""
    cols = ["ticker", "date", "open", "close", "adj_close"]
    if not Path(db_path).exists() or not Path(universe_path).exists():
        return pd.DataFrame(columns=cols)
    from sma.ingest.universe import load_universe

    universe = [t for t in load_universe(universe_path) if t != "SPY"]
    con = read_only_connect(db_path)
    try:
        df = con.execute(
            """
            SELECT ticker, date, open, close, adj_close
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY ticker, date
                           ORDER BY CASE source WHEN 'yfinance' THEN 0 ELSE 1 END
                       ) AS rn
                FROM prices
                WHERE ticker = ANY($universe)
                  AND open IS NOT NULL AND close IS NOT NULL AND adj_close IS NOT NULL
            ) t
            WHERE rn = 1
            ORDER BY ticker, date
            """,
            {"universe": universe},
        ).fetchdf()
    finally:
        con.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def model_edge_ic_df(
    horizon_days: int,
    db_path: Path = DEFAULT_DB_PATH,
    universe_path: Path = DEFAULT_UNIVERSE_PATH,
) -> pd.DataFrame:
    """Live rolling IC series at `horizon_days`, wired end-to-end from the DB:
    predictions + prices -> forward_returns -> rank_ic_series."""
    preds = live_predictions_df(db_path, universe_path)
    prices = ic_prices_df(db_path, universe_path)
    if preds.empty or prices.empty:
        return pd.DataFrame(columns=["asof_date", "ic", "n"])
    asof_dates = sorted(preds["asof_date"].unique())
    fwd = forward_returns(prices, asof_dates, horizon_days)
    return rank_ic_series(preds, fwd)
