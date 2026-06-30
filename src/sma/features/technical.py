"""Technical-indicator feature functions for the XGBoost quant model.

All functions take a per-ticker price DataFrame (sorted ascending by date)
and an asof_date, and return a single float (the feature value at that date)
or None if insufficient history.

Lookahead contract: for any (ticker_prices, asof_date), the function's output
must depend ONLY on rows where date <= asof_date. Adding rows with date >
asof_date to the input must not change the output. Tested adversarially.
"""

from datetime import date

import pandas as pd


def _rows_up_to(prices: pd.DataFrame, asof_date: date) -> pd.DataFrame:
    """Return the prices subset with date <= asof_date, sorted ascending."""
    return prices[prices["date"] <= asof_date].sort_values("date")


def _adj_close_n_back(prices: pd.DataFrame, asof_date: date, n: int) -> float | None:
    """Return adj_close from n trading days before asof_date.

    Returns None if there are fewer than n+1 rows at or before asof_date.
    """
    visible = _rows_up_to(prices, asof_date)
    if len(visible) < n + 1:
        return None
    # asof row is the last; n trading days back is the (n+1)-th from the end.
    return float(visible.iloc[-(n + 1)]["adj_close"])


def ret_n_days(prices: pd.DataFrame, asof_date: date, n: int) -> float | None:
    """N-trading-day total return: (adj_close[asof] / adj_close[asof - n]) - 1.

    Returns None if insufficient history or if either price is non-positive.
    """
    visible = _rows_up_to(prices, asof_date)
    if len(visible) < n + 1:
        return None
    asof_px = float(visible.iloc[-1]["adj_close"])
    prev_px = float(visible.iloc[-(n + 1)]["adj_close"])
    if prev_px <= 0 or asof_px <= 0:
        return None
    return (asof_px / prev_px) - 1.0


def ret_1d(prices: pd.DataFrame, asof_date: date) -> float | None:
    return ret_n_days(prices, asof_date, 1)


def ret_5d(prices: pd.DataFrame, asof_date: date) -> float | None:
    return ret_n_days(prices, asof_date, 5)


def ret_20d(prices: pd.DataFrame, asof_date: date) -> float | None:
    return ret_n_days(prices, asof_date, 20)


def ret_60d(prices: pd.DataFrame, asof_date: date) -> float | None:
    return ret_n_days(prices, asof_date, 60)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _daily_returns_from_adjclose(visible: pd.DataFrame) -> pd.Series:
    return visible["adj_close"].pct_change().dropna()


# ---------------------------------------------------------------------------
# T3: Volatility
# ---------------------------------------------------------------------------

def vol_20d(prices: pd.DataFrame, asof_date: date) -> float | None:
    """Standard deviation of last 20 daily returns. None if fewer than 21 rows."""
    visible = _rows_up_to(prices, asof_date)
    if len(visible) < 21:
        return None
    rets = _daily_returns_from_adjclose(visible)
    return float(rets.iloc[-20:].std(ddof=1))


def vol_60d(prices: pd.DataFrame, asof_date: date) -> float | None:
    """Standard deviation of last 60 daily returns. None if fewer than 61 rows."""
    visible = _rows_up_to(prices, asof_date)
    if len(visible) < 61:
        return None
    rets = _daily_returns_from_adjclose(visible)
    return float(rets.iloc[-60:].std(ddof=1))


# ---------------------------------------------------------------------------
# T4: RSI
# ---------------------------------------------------------------------------

def rsi_14(prices: pd.DataFrame, asof_date: date) -> float | None:
    """14-day RSI using Wilder smoothing on adj_close. None if fewer than 15 rows."""
    visible = _rows_up_to(prices, asof_date)
    if len(visible) < 15:
        return None

    closes = visible["adj_close"].values
    # Walk ALL visible rows (NOT just the last 15): seed avg gain/loss from the
    # first 14 deltas, then Wilder-smooth (EMA, alpha=1/14) over the rest. The EMA
    # forgets its seed after ~100 steps, so the value is stable regardless of how
    # much history is passed — train (full history) and serve (400-day window)
    # agree to within rounding. Do NOT "optimize" this to a fixed 15-row slice:
    # that changes the feature distribution and would require retraining every model.
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]

    # Seed with the first 14 changes.
    avg_gain = sum(gains[:14]) / 14.0
    avg_loss = sum(losses[:14]) / 14.0

    # Wilder smooth over any remaining deltas.
    for gain, loss in zip(gains[14:], losses[14:], strict=True):
        avg_gain = (avg_gain * 13.0 + gain) / 14.0
        avg_loss = (avg_loss * 13.0 + loss) / 14.0

    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


# ---------------------------------------------------------------------------
# T5: Volume
# ---------------------------------------------------------------------------

def volume_z_20d(prices: pd.DataFrame, asof_date: date) -> float | None:
    """Z-score of asof volume vs the trailing 20 days (excluding asof).

    None if fewer than 21 rows or if the trailing std is zero.
    """
    visible = _rows_up_to(prices, asof_date)
    if len(visible) < 21:
        return None
    prev_20_vol = visible["volume"].iloc[-21:-1]
    today_vol = float(visible["volume"].iloc[-1])
    mean = float(prev_20_vol.mean())
    std = float(prev_20_vol.std(ddof=1))
    if std == 0.0:
        return None
    return (today_vol - mean) / std


def dollar_volume_20d(prices: pd.DataFrame, asof_date: date) -> float | None:
    """Mean of (close * volume) over the last 20 trading days including asof.

    None if fewer than 20 rows.
    """
    visible = _rows_up_to(prices, asof_date)
    if len(visible) < 20:
        return None
    last_20 = visible.iloc[-20:]
    dv = last_20["close"] * last_20["volume"]
    return float(dv.mean())


# ---------------------------------------------------------------------------
# T6: Cross-sectional + price
# ---------------------------------------------------------------------------

def gap_open(prices: pd.DataFrame, asof_date: date) -> float | None:
    """(open[asof] - close[asof - 1 trading day]) / close[asof - 1 trading day].

    None if fewer than 2 rows or prior close <= 0.
    """
    visible = _rows_up_to(prices, asof_date)
    if len(visible) < 2:
        return None
    prior_close = float(visible.iloc[-2]["close"])
    if prior_close <= 0:
        return None
    asof_open = float(visible.iloc[-1]["open"])
    return (asof_open - prior_close) / prior_close


def dist_from_52w_high(prices: pd.DataFrame, asof_date: date) -> float | None:
    """(adj_close[asof] / max(adj_close over last 252 days including asof)) - 1.

    Returns 0.0 if asof is at the high; negative otherwise. None if <252 rows.
    """
    visible = _rows_up_to(prices, asof_date)
    if len(visible) < 252:
        return None
    last_252 = visible.iloc[-252:]
    high = float(last_252["adj_close"].max())
    asof_close = float(visible.iloc[-1]["adj_close"])
    return (asof_close / high) - 1.0


def rel_strength_spy_60d(
    target_prices: pd.DataFrame,
    spy_prices: pd.DataFrame,
    asof_date: date,
) -> float | None:
    """ret_60d(target) - ret_60d(SPY). None if either return is None.

    Caller passes separate single-ticker DataFrames for the target and SPY.
    """
    target_ret = ret_60d(target_prices, asof_date)
    spy_ret = ret_60d(spy_prices, asof_date)
    if target_ret is None or spy_ret is None:
        return None
    return target_ret - spy_ret


def rel_strength_sector_etf_30d(
    target_prices: pd.DataFrame,
    sector_etf_prices: pd.DataFrame,
    asof_date: date,
) -> float | None:
    """ret_30d(target) - ret_30d(sector_etf).

    Uses the SPDR sector ETF as the benchmark (XLK for IT, XLF for
    Financials, etc.). Cleaner than the per-name peer-mean version
    because the benchmark doesn't drift with universe composition — a
    small sector with 2 universe peers gives a noisier mean than one
    with 20, but the ETF is always the same index.

    Caller passes a single-ticker DataFrame for the appropriate sector
    ETF (look up via sma.sectors.sector_etf_for(ticker)). None when
    either return is None (insufficient history).
    """
    target_ret = ret_n_days(target_prices, asof_date, 30)
    if target_ret is None:
        return None
    etf_ret = ret_n_days(sector_etf_prices, asof_date, 30)
    if etf_ret is None:
        return None
    return target_ret - etf_ret


def rel_strength_sector_30d(
    target_prices: pd.DataFrame,
    sector_peer_prices: list[pd.DataFrame],
    asof_date: date,
) -> float | None:
    """ret_30d(target) - mean(ret_30d(p)) over OTHER tickers in same sector.

    Captures within-sector leadership: a name beating its sector is a
    different signal than a name riding sector momentum. SPY-relative
    strength conflates the two; this isolates the alpha component.

    Caller passes a list of single-ticker price DataFrames for the
    target's sector peers EXCLUDING the target itself. None when:
      - target has no ret_30d (insufficient history), OR
      - fewer than 2 peers with valid ret_30d (sector mean isn't
        meaningful with 0-1 peer)

    Excluding self prevents the ticker from being part of its own
    benchmark — a single-ticker sector would otherwise always score 0.
    """
    target_ret = ret_n_days(target_prices, asof_date, 30)
    if target_ret is None:
        return None
    peer_rets = [ret_n_days(p, asof_date, 30) for p in sector_peer_prices]
    peer_rets = [r for r in peer_rets if r is not None]
    if len(peer_rets) < 2:
        return None
    sector_mean = sum(peer_rets) / len(peer_rets)
    return target_ret - sector_mean


# ---------------------------------------------------------------------------
# 2026-06-12 feature wave (strategy review): risk-adjusted and asymmetry
# transforms of price data already on hand. Raw momentum partially ranks
# high-vol names; these isolate trend QUALITY and crash asymmetry.
# ---------------------------------------------------------------------------

def vol_adj_mom_60d(
    target_prices: pd.DataFrame,
    spy_prices: pd.DataFrame,
    asof_date: date,
) -> float | None:
    """SPY-relative 60d return divided by 60d realized vol (risk-adjusted
    relative momentum). None when either leg is unavailable or vol ~ 0."""
    rel = rel_strength_spy_60d(target_prices, spy_prices, asof_date)
    vol = vol_60d(target_prices, asof_date)
    if rel is None or vol is None:
        return None
    if vol < 1e-9:
        return 0.0  # flat series: no trend quality signal, stay neutral
    return float(rel / vol)


def downside_vol_ratio_60d(prices: pd.DataFrame, asof_date: date) -> float | None:
    """std(negative daily returns) / std(all daily returns) over the last 60
    sessions — crash-asymmetry. ~0 for smooth uptrends; →1 when volatility is
    dominated by down moves. None with <61 rows or ~zero total vol."""
    visible = _rows_up_to(prices, asof_date)
    if len(visible) < 61:
        return None
    rets = _daily_returns_from_adjclose(visible).iloc[-60:]
    total = float(rets.std(ddof=1))
    if total < 1e-12:
        return 0.0  # no vol at all: no asymmetry signal
    downs = rets[rets < 0]
    if len(downs) < 2:
        return 0.0
    return float(downs.std(ddof=1) * (len(downs) / len(rets)) ** 0.5 / total)


def reversal_5d_z(prices: pd.DataFrame, asof_date: date) -> float | None:
    """5d return scaled by 20d daily vol — the short-term reversal input in
    comparable units across names. None when either leg is unavailable."""
    r5 = ret_n_days(prices, asof_date, 5)
    v20 = vol_20d(prices, asof_date)
    if r5 is None or v20 is None:
        return None
    if v20 < 1e-9:
        return 0.0  # flat series: neutral
    return float(r5 / v20)
