import math
from datetime import date, timedelta

import pandas as pd
import pytest

from sma.features.technical import (
    dist_from_52w_high,
    dollar_volume_20d,
    gap_open,
    rel_strength_spy_60d,
    ret_1d,
    ret_5d,
    ret_20d,
    ret_60d,
    rsi_14,
    vol_20d,
    vol_60d,
    volume_z_20d,
)


def _series(closes: list[float], start: date = date(2025, 1, 6)) -> pd.DataFrame:
    """Build a single-ticker DataFrame from a list of adj_close values.

    Uses business-day-style dates (skip weekends would complicate; use plain
    consecutive days since the feature functions only count rows, not calendar days).
    """
    rows = []
    d = start
    for c in closes:
        rows.append({
            "ticker": "AAA", "date": d,
            "open": c, "high": c, "low": c, "close": c,
            "adj_close": c, "volume": 1_000_000,
        })
        d = d + timedelta(days=1)
    return pd.DataFrame(rows)


def _series_with_oc(
    opens: list[float],
    closes: list[float],
    volumes: list[int] | None = None,
    ticker: str = "AAA",
    start: date = date(2025, 1, 6),
) -> pd.DataFrame:
    """Build a single-ticker DataFrame with distinct open and close columns."""
    assert len(opens) == len(closes)
    if volumes is None:
        volumes = [1_000_000] * len(closes)
    rows = []
    d = start
    for o, c, v in zip(opens, closes, volumes, strict=True):
        rows.append({
            "ticker": ticker, "date": d,
            "open": o, "high": max(o, c), "low": min(o, c), "close": c,
            "adj_close": c, "volume": v,
        })
        d = d + timedelta(days=1)
    return pd.DataFrame(rows)


# ---- ret_1d ----

def test_ret_1d_basic():
    prices = _series([100.0, 110.0])
    asof = prices["date"].max()
    assert ret_1d(prices, asof) == pytest.approx(0.10)

def test_ret_1d_returns_none_with_one_row():
    prices = _series([100.0])
    asof = prices["date"].max()
    assert ret_1d(prices, asof) is None

def test_ret_1d_lookahead_clean():
    base = _series([100.0, 110.0])
    asof = base["date"].max()
    expected = ret_1d(base, asof)
    # Add a future row that should be invisible
    extended = _series([100.0, 110.0, 999.0])
    assert ret_1d(extended, asof) == expected

# ---- ret_5d ----

def test_ret_5d_basic():
    # 6 rows: index 0..5. ret_5d compares index 5 (asof) vs index 0.
    prices = _series([100.0, 101.0, 102.0, 103.0, 104.0, 110.0])
    asof = prices["date"].max()
    assert ret_5d(prices, asof) == pytest.approx(0.10)

def test_ret_5d_returns_none_with_5_rows():
    prices = _series([100.0, 101.0, 102.0, 103.0, 104.0])  # only 5 rows
    asof = prices["date"].max()
    assert ret_5d(prices, asof) is None

def test_ret_5d_lookahead_clean():
    base = _series([100.0, 101.0, 102.0, 103.0, 104.0, 110.0])
    asof = base["date"].max()
    expected = ret_5d(base, asof)
    extended = _series([100.0, 101.0, 102.0, 103.0, 104.0, 110.0, 999.0, 999.0])
    assert ret_5d(extended, asof) == expected

# ---- ret_20d ----

def test_ret_20d_basic():
    closes = [100.0] * 20 + [120.0]  # 21 rows; asof is at +20% vs 20 days ago
    prices = _series(closes)
    asof = prices["date"].max()
    assert ret_20d(prices, asof) == pytest.approx(0.20)

def test_ret_20d_returns_none_with_20_rows():
    prices = _series([100.0] * 20)
    asof = prices["date"].max()
    assert ret_20d(prices, asof) is None

def test_ret_20d_lookahead_clean():
    closes = [100.0] * 20 + [120.0]
    base = _series(closes)
    asof = base["date"].max()
    expected = ret_20d(base, asof)
    extended = _series(closes + [9.99, 9.99])
    assert ret_20d(extended, asof) == expected

# ---- ret_60d ----

def test_ret_60d_basic():
    closes = [100.0] * 60 + [150.0]  # +50% vs 60 days ago
    prices = _series(closes)
    asof = prices["date"].max()
    assert ret_60d(prices, asof) == pytest.approx(0.50)

def test_ret_60d_returns_none_with_60_rows():
    prices = _series([100.0] * 60)
    asof = prices["date"].max()
    assert ret_60d(prices, asof) is None

def test_ret_60d_lookahead_clean():
    closes = [100.0] * 60 + [150.0]
    base = _series(closes)
    asof = base["date"].max()
    expected = ret_60d(base, asof)
    extended = _series(closes + [1.0])
    assert ret_60d(extended, asof) == expected


# ---- vol_20d ----

def test_vol_20d_basic():
    # 21 rows of [100, 101, 100, 101, ...] produce predictable std.
    # Returns: alternating +1%, -0.99%, ... Just use a flat series with known std.
    # 21 closes: first 20 are 100.0, last is 110.0. Returns: 19 zeros then +10%.
    # std([0]*19 + [0.10]) with ddof=1: mean ~ 0.00476, but easier to compute directly.
    closes = [100.0] * 20 + [110.0]
    prices = _series(closes)
    asof = prices["date"].max()
    result = vol_20d(prices, asof)
    # Manually: daily returns from adj_close of 21 rows = pct_change of closes[1:] vs closes[0:]
    # returns = [0.0]*19 + [0.10]; std(ddof=1) = sqrt(sum((r - mean)^2) / 19)
    rets = [0.0] * 19 + [0.10]
    mean = sum(rets) / 20
    expected_std = math.sqrt(sum((r - mean) ** 2 for r in rets) / 19)
    assert result == pytest.approx(expected_std)

def test_vol_20d_returns_none_with_20_rows():
    prices = _series([100.0] * 20)
    asof = prices["date"].max()
    assert vol_20d(prices, asof) is None

def test_vol_20d_lookahead_clean():
    closes = [100.0] * 20 + [110.0]
    base = _series(closes)
    asof = base["date"].max()
    expected = vol_20d(base, asof)
    extended = _series(closes + [999.0])
    assert vol_20d(extended, asof) == expected


# ---- vol_60d ----

def test_vol_60d_basic():
    closes = [100.0] * 60 + [110.0]
    prices = _series(closes)
    asof = prices["date"].max()
    result = vol_60d(prices, asof)
    rets = [0.0] * 59 + [0.10]
    mean = sum(rets) / 60
    expected_std = math.sqrt(sum((r - mean) ** 2 for r in rets) / 59)
    assert result == pytest.approx(expected_std)

def test_vol_60d_returns_none_with_60_rows():
    prices = _series([100.0] * 60)
    asof = prices["date"].max()
    assert vol_60d(prices, asof) is None

def test_vol_60d_lookahead_clean():
    closes = [100.0] * 60 + [110.0]
    base = _series(closes)
    asof = base["date"].max()
    expected = vol_60d(base, asof)
    extended = _series(closes + [999.0])
    assert vol_60d(extended, asof) == expected


# ---- rsi_14 ----

def test_rsi_14_uptrend():
    # Strict monotonic uptrend: every day closes 1 higher. All gains, no losses.
    closes = [float(100 + i) for i in range(16)]  # 16 rows -> 15 changes
    prices = _series(closes)
    asof = prices["date"].max()
    result = rsi_14(prices, asof)
    assert result is not None
    assert result > 95.0

def test_rsi_14_downtrend():
    # Strict monotonic downtrend: every day closes 1 lower. All losses, no gains.
    closes = [float(115 - i) for i in range(16)]
    prices = _series(closes)
    asof = prices["date"].max()
    result = rsi_14(prices, asof)
    assert result is not None
    assert result < 5.0

def test_rsi_14_mixed_hand_computed():
    # 15 rows => 14 changes: gains=[1,0,1,0,1,0,1,0,1,0,1,0,1,0], losses=[0,1,0,1,...]
    # avg_gain = 7/14 = 0.5, avg_loss = 7/14 = 0.5
    # RS = 1.0, RSI = 100 - 100/(1+1) = 50.0
    closes = [float(100 + (i % 2 == 0)) for i in range(15)]
    # Actually: 100, 101, 100, 101, 100, 101, 100, 101, 100, 101, 100, 101, 100, 101, 100
    # changes: +1,-1,+1,-1,+1,-1,+1,-1,+1,-1,+1,-1,+1,-1 (14 alternating)
    # gains=[1,0,1,0,1,0,1,0,1,0,1,0,1,0], losses=[0,1,0,1,0,1,0,1,0,1,0,1,0,1]
    # avg_gain=0.5, avg_loss=0.5, RS=1, RSI=50
    prices = _series(closes)
    asof = prices["date"].max()
    result = rsi_14(prices, asof)
    assert result == pytest.approx(50.0)

def test_rsi_14_returns_none_with_14_rows():
    prices = _series([100.0] * 14)
    asof = prices["date"].max()
    assert rsi_14(prices, asof) is None

def test_rsi_14_lookahead_clean():
    closes = [float(100 + i) for i in range(16)]
    base = _series(closes)
    asof = base["date"].max()
    expected = rsi_14(base, asof)
    extended = _series(closes + [999.0])
    assert rsi_14(extended, asof) == expected


# ---- volume_z_20d ----

def test_volume_z_20d_basic():
    # prev 20 volumes: 10 rows of 900 and 10 rows of 1100 -> mean=1000
    # std(ddof=1) of that series is ~102.6 (not 100 because ddof=1 on balanced split)
    # asof volume: 1200. z = (1200 - 1000) / std
    import statistics
    volumes = [900] * 10 + [1100] * 10 + [1200]
    expected_z = (1200 - 1000) / statistics.stdev([900] * 10 + [1100] * 10)
    closes = [100.0] * 21
    prices = _series_with_oc([100.0] * 21, closes, volumes)
    asof = prices["date"].max()
    result = volume_z_20d(prices, asof)
    assert result == pytest.approx(expected_z)

def test_volume_z_20d_returns_none_with_20_rows():
    prices = _series([100.0] * 20)
    asof = prices["date"].max()
    assert volume_z_20d(prices, asof) is None

def test_volume_z_20d_lookahead_clean():
    volumes = [900] * 10 + [1100] * 10 + [1200]
    closes = [100.0] * 21
    base = _series_with_oc([100.0] * 21, closes, volumes)
    asof = base["date"].max()
    expected = volume_z_20d(base, asof)
    extended_closes = closes + [100.0]
    extended_volumes = volumes + [9_000_000]
    extended = _series_with_oc([100.0] * 22, extended_closes, extended_volumes)
    assert volume_z_20d(extended, asof) == expected


# ---- dollar_volume_20d ----

def test_dollar_volume_20d_basic():
    # 20 rows: price=100, volume=500 -> dollar_vol per day = 50000
    # mean over 20 days = 50000
    closes = [100.0] * 20
    volumes = [500] * 20
    prices = _series_with_oc([100.0] * 20, closes, volumes)
    asof = prices["date"].max()
    assert dollar_volume_20d(prices, asof) == pytest.approx(50_000.0)

def test_dollar_volume_20d_returns_none_with_19_rows():
    prices = _series([100.0] * 19)
    asof = prices["date"].max()
    assert dollar_volume_20d(prices, asof) is None

def test_dollar_volume_20d_lookahead_clean():
    closes = [100.0] * 20
    volumes = [500] * 20
    base = _series_with_oc([100.0] * 20, closes, volumes)
    asof = base["date"].max()
    expected = dollar_volume_20d(base, asof)
    extended = _series_with_oc([100.0] * 21, closes + [999.0], volumes + [9_000_000])
    assert dollar_volume_20d(extended, asof) == expected


# ---- gap_open ----

def test_gap_open_basic():
    # prior close = 100, asof open = 105 -> gap = (105 - 100) / 100 = 0.05
    opens = [100.0, 105.0]
    closes = [100.0, 108.0]
    prices = _series_with_oc(opens, closes)
    asof = prices["date"].max()
    assert gap_open(prices, asof) == pytest.approx(0.05)

def test_gap_open_returns_none_with_one_row():
    prices = _series_with_oc([100.0], [100.0])
    asof = prices["date"].max()
    assert gap_open(prices, asof) is None

def test_gap_open_lookahead_clean():
    opens = [100.0, 105.0]
    closes = [100.0, 108.0]
    base = _series_with_oc(opens, closes)
    asof = base["date"].max()
    expected = gap_open(base, asof)
    extended = _series_with_oc(opens + [999.0], closes + [999.0])
    assert gap_open(extended, asof) == expected


# ---- dist_from_52w_high ----

def test_dist_from_52w_high_basic():
    # 252 rows: first 251 at 150.0, last (asof) at 100.0.
    # max over 252 rows = 150.0. dist = (100/150) - 1 = -1/3
    closes = [150.0] * 251 + [100.0]
    prices = _series(closes)
    asof = prices["date"].max()
    result = dist_from_52w_high(prices, asof)
    assert result == pytest.approx((100.0 / 150.0) - 1.0)

def test_dist_from_52w_high_returns_none_with_251_rows():
    prices = _series([100.0] * 251)
    asof = prices["date"].max()
    assert dist_from_52w_high(prices, asof) is None

def test_dist_from_52w_high_lookahead_clean():
    closes = [150.0] * 251 + [100.0]
    base = _series(closes)
    asof = base["date"].max()
    expected = dist_from_52w_high(base, asof)
    extended = _series(closes + [999.0])
    assert dist_from_52w_high(extended, asof) == expected


# ---- rel_strength_spy_60d ----

def _spy_series(closes: list[float], start: date = date(2025, 1, 6)) -> pd.DataFrame:
    """Build a SPY ticker DataFrame for rel_strength_spy_60d tests."""
    rows = []
    d = start
    for c in closes:
        rows.append({
            "ticker": "SPY", "date": d,
            "open": c, "high": c, "low": c, "close": c,
            "adj_close": c, "volume": 10_000_000,
        })
        d = d + timedelta(days=1)
    return pd.DataFrame(rows)

def test_rel_strength_spy_60d_basic():
    # Target up 10% vs 60d ago; SPY up 5%. Rel strength = 0.10 - 0.05 = 0.05
    target_closes = [100.0] * 60 + [110.0]  # +10%
    spy_closes = [200.0] * 60 + [210.0]      # +5%
    target_prices = _series(target_closes)
    spy_prices = _spy_series(spy_closes)
    asof = target_prices["date"].max()
    result = rel_strength_spy_60d(target_prices, spy_prices, asof)
    assert result == pytest.approx(0.05)

def test_rel_strength_spy_60d_returns_none_if_spy_insufficient():
    target_closes = [100.0] * 61
    spy_closes = [200.0] * 60  # only 60 rows -> ret_60d returns None
    target_prices = _series(target_closes)
    spy_prices = _spy_series(spy_closes)
    asof = target_prices["date"].max()
    assert rel_strength_spy_60d(target_prices, spy_prices, asof) is None

def test_rel_strength_spy_60d_lookahead_clean():
    target_closes = [100.0] * 60 + [110.0]
    spy_closes = [200.0] * 60 + [210.0]
    base_target = _series(target_closes)
    base_spy = _spy_series(spy_closes)
    asof = base_target["date"].max()
    expected = rel_strength_spy_60d(base_target, base_spy, asof)
    extended_target = _series(target_closes + [999.0])
    extended_spy = _spy_series(spy_closes + [999.0])
    assert rel_strength_spy_60d(extended_target, extended_spy, asof) == expected


def _trend_prices(n=120, drift=0.01, start=100.0):
    """Synthetic single-ticker frame with constant daily drift."""
    from datetime import date, timedelta

    import pandas as pd
    rows = []
    px = start
    d = date(2025, 1, 1)
    for i in range(n):
        px *= (1 + drift) * (1.002 if i % 2 else 0.998)  # mild noise so vol > 0
        rows.append({"ticker": "AAA", "date": d, "open": px, "high": px,
                     "low": px, "close": px, "adj_close": px, "volume": 1e6})
        d += timedelta(days=1)
    return pd.DataFrame(rows)


def test_vol_adj_mom_60d_scales_relative_return_by_vol():
    """2026-06-12 feature wave: risk-adjusted relative momentum — the classic
    robust form of the momentum factor (raw winners are often just high-vol
    names; scaling by realized vol ranks QUALITY of trend)."""
    from sma.features.technical import vol_adj_mom_60d

    target = _trend_prices(drift=0.01)
    spy = _trend_prices(drift=0.002)
    v = vol_adj_mom_60d(target, spy, target["date"].iloc[-1])
    assert v is not None and v > 0
    # steadier outperformance (same excess, lower vol) must score HIGHER
    import pandas as pd
    noisy = target.copy()
    noise = [1.03 if i % 2 else 0.985 for i in range(len(noisy))]
    noisy["adj_close"] = noisy["adj_close"] * pd.Series(noise)
    v_noisy = vol_adj_mom_60d(noisy, spy, target["date"].iloc[-1])
    assert v_noisy is not None and v > v_noisy


def test_downside_vol_ratio_60d_flags_asymmetric_crashers():
    from sma.features.technical import downside_vol_ratio_60d

    smooth = _trend_prices(drift=0.005)
    r_smooth = downside_vol_ratio_60d(smooth, smooth["date"].iloc[-1])
    # all-up drift has ~no downside vol
    assert r_smooth is not None and r_smooth < 0.2

    crashy = _trend_prices(drift=0.005)
    px = crashy["adj_close"].to_list()
    for i in range(70, 115, 9):  # periodic sharp down days
        for j in range(i, len(px)):
            px[j] *= 0.94
    crashy["adj_close"] = px
    r_crashy = downside_vol_ratio_60d(crashy, crashy["date"].iloc[-1])
    assert r_crashy is not None and r_crashy > r_smooth


def test_reversal_5d_z_is_vol_scaled_short_horizon_return():
    from sma.features.technical import reversal_5d_z

    p = _trend_prices(drift=0.0)
    px = p["adj_close"].to_list()
    for j in range(len(px) - 5, len(px)):  # 5-day pop on a flat series
        px[j] = px[j] * 1.10
    # add mild noise so vol_20d > 0
    px = [v * (1.001 if i % 2 else 0.999) for i, v in enumerate(px)]
    p["adj_close"] = px
    z = reversal_5d_z(p, p["date"].iloc[-1])
    assert z is not None and z > 1.0  # large pop relative to tiny vol
