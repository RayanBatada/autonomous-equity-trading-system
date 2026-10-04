"""Cross-sectional momentum-dispersion de-risk detector.

Diagnosis 2026-06-25 #2: the model is a momentum factor that inverts in
reversal regimes, and those factor unwinds happen ABOVE the index MA200 (the
refuted SPY<MA200 gate misses them). The candidate regime signal is
cross-sectional momentum DISPERSION: when the spread of trailing returns across
the universe blows out vs its own history, factor unwinds and momentum crashes
cluster. This module turns prices into a per-date extra cash floor:

    dispersion(d)  = cross-sectional std of `lookback`-day returns at d
    z(d)           = (dispersion(d) − trailing mean) / trailing std,
                     computed over dates STRICTLY BEFORE d (point-in-time)
    floor(d)       = base + slope · max(0, z(d) − z_start), capped

Everything here is pure and look-ahead-free; the simulator consumes the floor
series via `extra_cash_floor_by_date` (validation only — live wiring happens
ONLY if the multi-window gate passes: beat OFF on Sharpe AND maxDD in every
reversal window, no bull-window loss, plateau across floor caps).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

__all__ = [
    "breadth_series",
    "dispersion_floor_series",
    "momentum_dispersion_series",
    "rolling_z",
]


def momentum_dispersion_series(
    prices: pd.DataFrame,
    *,
    lookback: int = 60,
    min_names: int = 20,
) -> dict[date, float]:
    """Cross-sectional std of `lookback`-session returns per date.

    `prices` needs columns ticker/date/adj_close. For each date with at least
    `min_names` tickers having a price both at d and `lookback` sessions
    earlier (per-ticker session grid), returns std over the cross-section of
    (adj_close[d] / adj_close[d - lookback sessions] − 1). Dates with fewer
    resolvable names are omitted (thin cross-sections are noise).
    """
    out: dict[date, float] = {}
    piv = prices.pivot_table(index="date", columns="ticker", values="adj_close")
    piv = piv.sort_index()
    if len(piv) <= lookback:
        return out
    rets = piv / piv.shift(lookback) - 1.0
    for d, row in rets.iterrows():
        vals = row.dropna()
        if len(vals) < min_names:
            continue
        out[d] = float(vals.std(ddof=1))
    return out


def breadth_series(
    prices: pd.DataFrame,
    *,
    ma_window: int = 50,
    min_names: int = 20,
) -> dict[date, float]:
    """Fraction of tickers trading ABOVE their own `ma_window`-session moving
    average per date (classic breadth). A breadth COLLAPSE (low values vs
    history) is the de-risk trigger — feed −breadth (or a low-percentile
    transform) into rolling_z + dispersion_floor_series. PIT: the MA at d uses
    sessions ≤ d only. Dates with fewer than `min_names` resolvable names are
    omitted."""
    out: dict[date, float] = {}
    piv = prices.pivot_table(index="date", columns="ticker", values="adj_close")
    piv = piv.sort_index()
    if len(piv) < ma_window:
        return out
    ma = piv.rolling(ma_window, min_periods=ma_window).mean()
    above = piv > ma
    valid = piv.notna() & ma.notna()
    for d in piv.index:
        n = int(valid.loc[d].sum())
        if n < min_names:
            continue
        out[d] = float(above.loc[d][valid.loc[d]].mean())
    return out


def rolling_z(
    series: dict[date, float],
    *,
    window: int = 252,
    min_window: int = 60,
) -> dict[date, float]:
    """Z-score each value against the trailing `window` values STRICTLY before
    its date (point-in-time: a date's own value never enters its own baseline).
    Dates with fewer than `min_window` prior values are omitted. A degenerate
    flat history (std ~ 0) uses a tiny epsilon so a genuine shock still
    registers rather than dividing by zero."""
    out: dict[date, float] = {}
    dates = sorted(series)
    vals = [series[d] for d in dates]
    for i, d in enumerate(dates):
        hist = vals[max(0, i - window):i]  # strictly before d
        if len(hist) < min_window:
            continue
        mean = sum(hist) / len(hist)
        var = sum((v - mean) ** 2 for v in hist) / (len(hist) - 1)
        std = var ** 0.5
        out[d] = (vals[i] - mean) / max(std, 1e-12)
    return out


def dispersion_floor_series(
    z_by_date: dict[date, float],
    *,
    base_floor: float,
    z_start: float = 1.0,
    slope: float = 0.15,
    cap: float = 0.5,
) -> dict[date, float]:
    """Map dispersion z-scores to an effective cash floor per date (same shape
    as risk.derisk.derisk_cash_floor: inert below z_start or at slope<=0; rises
    by `slope` per z past it; clamped to `cap` so the book is never forced
    fully to cash)."""
    out: dict[date, float] = {}
    for d, z in z_by_date.items():
        if slope <= 0.0 or z <= z_start:
            out[d] = base_floor
        else:
            out[d] = min(cap, base_floor + slope * (z - z_start))
    return out
