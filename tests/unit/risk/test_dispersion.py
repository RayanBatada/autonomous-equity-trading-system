"""Cross-sectional momentum-dispersion de-risk detector (diagnosis 2026-06-25
#2: a regime signal that fires during above-MA200 factor unwinds — dispersion,
NOT the refuted SPY<MA200 gate). Pure, point-in-time, prices-only."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def _prices(tickers_rets: dict[str, float], n_days: int = 80,
            start: date = date(2024, 1, 1)) -> pd.DataFrame:
    """Each ticker follows a constant daily return — so 60d returns differ
    across tickers by construction, giving a known cross-sectional spread."""
    rows = []
    for t, r in tickers_rets.items():
        px = 100.0
        d = start
        for _ in range(n_days):
            rows.append({"ticker": t, "date": d, "adj_close": px})
            px *= 1.0 + r
            d += timedelta(days=1)
    return pd.DataFrame(rows)


def test_momentum_dispersion_series_measures_cross_sectional_spread():
    from sma.risk.dispersion import momentum_dispersion_series

    tight = momentum_dispersion_series(
        _prices({"A": 0.001, "B": 0.0011, "C": 0.0009}), lookback=60, min_names=3,
    )
    wide = momentum_dispersion_series(
        _prices({"A": 0.004, "B": -0.003, "C": 0.0}), lookback=60, min_names=3,
    )
    # Only dates with >= lookback history resolve.
    assert tight and wide
    last_tight = tight[max(tight)]
    last_wide = wide[max(wide)]
    assert last_wide > last_tight > 0


def test_momentum_dispersion_requires_min_names():
    from sma.risk.dispersion import momentum_dispersion_series

    out = momentum_dispersion_series(
        _prices({"A": 0.001, "B": 0.002}), lookback=60, min_names=3,
    )
    assert out == {}


def test_rolling_z_is_point_in_time():
    """z at date d uses ONLY dates strictly before d — the value at d must not
    influence its own z (look-ahead guard)."""
    from sma.risk.dispersion import rolling_z

    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(40)]
    series = {d: 1.0 for d in dates}
    series[dates[-1]] = 100.0  # a shock on the LAST day
    z = rolling_z(series, window=20, min_window=10)
    # The shock day's z is huge (measured against the flat history)...
    assert z[dates[-1]] > 5
    # ...and no EARLIER date's z was affected (all ~0 vs flat history).
    assert all(abs(z[d]) < 1e-9 for d in dates[11:-1] if d in z)


def test_rolling_z_needs_min_window():
    from sma.risk.dispersion import rolling_z

    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(8)]
    z = rolling_z({d: float(i) for i, d in enumerate(dates)}, window=20, min_window=10)
    assert z == {}  # never enough history


def test_dispersion_floor_series_maps_z_to_floor():
    from sma.risk.dispersion import dispersion_floor_series

    z = {date(2024, 1, 1): 0.0, date(2024, 1, 2): 1.0,
         date(2024, 1, 3): 2.0, date(2024, 1, 4): 5.0}
    f = dispersion_floor_series(z, base_floor=0.0, z_start=1.0, slope=0.15, cap=0.5)
    assert f[date(2024, 1, 1)] == 0.0          # below z_start → base
    assert f[date(2024, 1, 2)] == 0.0          # at z_start → base
    assert abs(f[date(2024, 1, 3)] - 0.15) < 1e-12   # 1 z past start
    assert f[date(2024, 1, 4)] == 0.5          # capped


def test_dispersion_floor_series_zero_slope_is_inert():
    from sma.risk.dispersion import dispersion_floor_series

    z = {date(2024, 1, 1): 9.0}
    f = dispersion_floor_series(z, base_floor=0.1, z_start=1.0, slope=0.0, cap=0.5)
    assert f[date(2024, 1, 1)] == 0.1


def test_breadth_series_measures_fraction_above_ma():
    from sma.risk.dispersion import breadth_series

    # A: rising the whole time (above its MA); B: falling (below its MA).
    up = _prices({"A": 0.002, "B": 0.002, "C": 0.002}, n_days=80)
    down = _prices({"A": -0.002, "B": -0.002, "C": -0.002}, n_days=80)
    b_up = breadth_series(up, ma_window=50, min_names=3)
    b_down = breadth_series(down, ma_window=50, min_names=3)
    assert b_up and b_down
    assert b_up[max(b_up)] == 1.0     # all rising names above their MA
    assert b_down[max(b_down)] == 0.0  # all falling names below


def test_breadth_series_min_names_gate():
    from sma.risk.dispersion import breadth_series

    out = breadth_series(_prices({"A": 0.001}, n_days=80), ma_window=50, min_names=3)
    assert out == {}
