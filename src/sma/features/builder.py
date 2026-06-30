"""Batch builder: compute features for every ticker on a given asof_date.

Feature list grew to 13 on 2026-05-10 with the addition of
`politician_flow_30d` (net dollar flow from House Periodic Transaction
Reports over the prior 30 days). When the caller doesn't supply flow
data — or for historical asof_dates predating PTR backfill (2024-01-01) —
the feature defaults to 0.0 for every ticker.

Models trained before 2026-05-10 only know about the original 12 features;
predictor/__main__ uses `model.feature_names_in_` (exposed by xgboost's
sklearn API) to pass exactly the columns the model was trained on, so
older models keep predicting correctly until the next weekly retrain
incorporates the new feature.
"""

from datetime import date

import pandas as pd

from sma.features import technical

FEATURE_NAMES: list[str] = [
    "ret_1d", "ret_5d", "ret_20d", "ret_60d",
    "vol_20d", "vol_60d",
    "rsi_14",
    "volume_z_20d", "dollar_volume_20d",
    "rel_strength_spy_60d",
    "gap_open",
    "dist_from_52w_high",
    # Phase 5.5 follow-up (2026-05-10): politician trade signal.
    "politician_flow_30d",
    # 2026-05-24: within-sector leadership. SPY-relative conflates "name
    # has alpha" with "sector is hot"; this isolates the alpha component
    # by subtracting the mean 30d return of OTHER tickers in the same
    # GICS sector. Falls back to 0.0 when sector has <2 peers with valid
    # ret_30d (the SECTORS dict is shipped per-release so coverage is
    # complete by construction, but this guards against future single-
    # ticker sectors and the bootstrap window).
    "rel_strength_sector_30d",
    # 2026-05-24: days to next earnings report, capped at 60. The model
    # target is 30-day forward return — if earnings fall inside that
    # window, the target distribution is meaningfully different (binary
    # event risk dominates baseline drift). Capped because beyond 60d
    # earnings shouldn't drive 30d-forward signal, and a uniform "60"
    # is more learnable than a sparse heavy tail.
    "days_to_next_earnings",
    # 2026-05-26: news coverage volume over the prior 7 days, as an
    # attention/momentum proxy. Aggregates across all news sources
    # (Finnhub + Alpaca/Benzinga + NewsAPI). Heavy-tailed distribution
    # in practice (median ~48, max ~744 for GOOGL on 5/22) so applied as
    # log1p to compress for the tree splitter. Defaults to 0 when no
    # news_counts dict supplied or ticker has no rows in the window.
    "news_count_7d_log",
    # 2026-05-26: SPDR sector ETF relative strength. ret_30d(ticker) -
    # ret_30d(sector_etf) where sector_etf comes from SECTOR_ETF_FOR_GICS
    # (XLK for IT, XLF for Financials, etc.). Cleaner than the per-name
    # rel_strength_sector_30d because the benchmark (an ETF) doesn't drift
    # with universe composition. Both features ship together — the model
    # learns which one is more predictive per sector. Defaults to 0 when
    # the ticker's sector has no ETF mapping (currently ETF + Sector ETF
    # buckets) or when the ETF's prices are missing.
    "rel_strength_sector_etf_30d",
    # 2026-06-12 feature wave (risk-adjusted momentum / asymmetry / reversal)
    "vol_adj_mom_60d",
    "downside_vol_ratio_60d",
    "reversal_5d_z",
    # 2026-06-12: first fundamentals feature — latest reported EPS surprise
    # ((actual−est)/|est|, clamped ±1; 0.0 = no report / neutral), unblocked
    # by the yfinance_hist earnings backfill. Caller resolves per-asof
    # (loader._compute_latest_surprises / predictor._fetch_latest_surprises).
    "earnings_surprise_last",
]


def build_features(
    prices: pd.DataFrame,
    universe: list[str],
    asof_date: date,
    politician_flows: dict[str, float] | None = None,
    earnings_calendar: dict[str, date] | None = None,
    earnings_surprises: dict[str, float] | None = None,
    news_counts_7d: dict[str, int] | None = None,
) -> pd.DataFrame:
    """For each ticker in universe, compute the 16 features as of asof_date.

    Args:
        prices: multi-ticker price DataFrame with columns
            ticker, date, open, high, low, close, adj_close, volume.
            Must include SPY rows so rel_strength_spy_60d can resolve.
        universe: tickers to score. SPY itself is included if it's in the list.
        asof_date: date at which to evaluate the features.
        politician_flows: optional ticker → net 30d politician dollar flow.
        earnings_calendar: optional ticker → next earnings report_date AFTER
            asof_date. When None or ticker absent, days_to_next_earnings
            defaults to 60 (the cap — i.e. "no earnings soon").
        news_counts_7d: optional ticker → news article count over [asof-7d, asof].
            When None or ticker absent, news_count_7d_log defaults to 0
            (log1p(0) = 0 — no news attention signal).

    Returns:
        DataFrame indexed by ticker with FEATURE_NAMES as columns. Tickers
        for which any feature is None (insufficient history) are DROPPED, not
        emitted with NaN. The training loader handles missing-row exclusion.
    """
    # Local import to avoid circular dep when sma.sectors is loaded first.
    from sma.sectors import SECTOR_ETF_FOR_GICS, SECTORS

    spy_prices = prices[prices["ticker"] == "SPY"]
    # Pre-extract sector-ETF price slices once so the per-ticker loop
    # doesn't re-filter the full prices DataFrame N times.
    sector_etf_prices: dict[str, pd.DataFrame] = {}
    for etf in set(SECTOR_ETF_FOR_GICS.values()):
        slice_ = prices[prices["ticker"] == etf]
        if not slice_.empty:
            sector_etf_prices[etf] = slice_

    # Pre-group prices by ticker once so the per-ticker loop doesn't
    # re-scan the full DataFrame N times. Also pre-group by sector so the
    # sector-relative-strength lookup is O(1) instead of O(universe).
    prices_by_ticker = {t: prices[prices["ticker"] == t] for t in universe}
    sector_members: dict[str, list[str]] = {}
    for t in universe:
        sec = SECTORS.get(t)
        # ETFs are benchmarks, not within-sector members: SPY/NANC ("ETF") and
        # the SPDR sector ETFs ("Sector ETF", e.g. XLK/XLE) have no meaningful
        # GICS-sector peer group, so they neither form nor join peer buckets.
        if sec is None or sec in ("ETF", "Sector ETF"):
            continue
        sector_members.setdefault(sec, []).append(t)

    rows: list[dict] = []
    for ticker in universe:
        target_prices = prices_by_ticker.get(ticker)
        if target_prices is None or target_prices.empty:
            continue

        # Sector peers (excluding self). Default to 0.0 when peers are
        # absent or insufficient — letting the model learn that "no
        # within-sector signal" maps to neutral.
        sec = SECTORS.get(ticker)
        if sec is None or sec in ("ETF", "Sector ETF"):
            sector_rs: float | None = 0.0
        else:
            peers = [t for t in sector_members.get(sec, []) if t != ticker]
            peer_prices = [
                prices_by_ticker[t] for t in peers
                if prices_by_ticker.get(t) is not None
                and not prices_by_ticker[t].empty
            ]
            sector_rs = technical.rel_strength_sector_30d(
                target_prices, peer_prices, asof_date,
            )
            if sector_rs is None:
                sector_rs = 0.0  # insufficient peers — neutral

        # Sector ETF relative strength. Independent of peer membership —
        # benchmarks against an actual tradable index (XLK, XLF, etc.).
        sector_etf_ticker = (
            SECTOR_ETF_FOR_GICS.get(sec) if sec else None
        )
        if sector_etf_ticker and sector_etf_ticker in sector_etf_prices:
            etf_rs = technical.rel_strength_sector_etf_30d(
                target_prices, sector_etf_prices[sector_etf_ticker], asof_date,
            )
            if etf_rs is None:
                etf_rs = 0.0
        else:
            etf_rs = 0.0  # ETF itself or missing benchmark prices

        feats = {
            "ret_1d": technical.ret_1d(target_prices, asof_date),
            "ret_5d": technical.ret_5d(target_prices, asof_date),
            "ret_20d": technical.ret_20d(target_prices, asof_date),
            "ret_60d": technical.ret_60d(target_prices, asof_date),
            "vol_20d": technical.vol_20d(target_prices, asof_date),
            "vol_60d": technical.vol_60d(target_prices, asof_date),
            "rsi_14": technical.rsi_14(target_prices, asof_date),
            "volume_z_20d": technical.volume_z_20d(target_prices, asof_date),
            "dollar_volume_20d": technical.dollar_volume_20d(target_prices, asof_date),
            "rel_strength_spy_60d": technical.rel_strength_spy_60d(
                target_prices, spy_prices, asof_date
            ),
            "gap_open": technical.gap_open(target_prices, asof_date),
            "dist_from_52w_high": technical.dist_from_52w_high(target_prices, asof_date),
            # 2026-06-12 feature wave: risk-adjusted relative momentum,
            # crash-asymmetry, and vol-scaled short-term reversal — trend
            # QUALITY transforms of price data already on hand.
            "vol_adj_mom_60d": technical.vol_adj_mom_60d(
                target_prices, spy_prices, asof_date
            ),
            "downside_vol_ratio_60d": technical.downside_vol_ratio_60d(
                target_prices, asof_date
            ),
            "reversal_5d_z": technical.reversal_5d_z(target_prices, asof_date),
            "earnings_surprise_last": float(
                (earnings_surprises or {}).get(ticker, 0.0)
            ),
            # 2026-05-10: net House-PTR dollar flow over the prior 30
            # days. 0.0 when the caller didn't supply flow data or no
            # politician trades touched this ticker in the window.
            "politician_flow_30d": (
                float(politician_flows.get(ticker, 0.0))
                if politician_flows is not None else 0.0
            ),
            "rel_strength_sector_30d": sector_rs,
            "days_to_next_earnings": _days_to_next_earnings(
                ticker, asof_date, earnings_calendar,
            ),
            "news_count_7d_log": _news_count_7d_log(ticker, news_counts_7d),
            "rel_strength_sector_etf_30d": etf_rs,
        }
        # Drop tickers that have any None feature (insufficient history).
        if any(v is None for v in feats.values()):
            continue
        feats["ticker"] = ticker
        rows.append(feats)

    if not rows:
        return pd.DataFrame(columns=["ticker"] + FEATURE_NAMES).set_index("ticker")

    df = pd.DataFrame(rows).set_index("ticker")
    return df[FEATURE_NAMES]  # column order = FEATURE_NAMES


# Cap chosen so the feature has a finite range the model can ingest. Beyond
# 60 days, earnings shouldn't drive a 30-day-forward return signal — name
# is effectively "mid-quarter, no upcoming event."
_EARNINGS_PROXIMITY_CAP_DAYS = 60


def _news_count_7d_log(
    ticker: str, news_counts_7d: dict[str, int] | None,
) -> float:
    """log1p of news article count over the prior 7 days.

    log1p compresses the heavy right tail (top names get 500-700+ articles
    while the median is ~50). Tree splits work fine on raw counts too,
    but the log transform reduces the dynamic range without losing the
    monotonic relationship — useful since the same absolute uptick at
    the high end (700 → 720) and low end (5 → 25) probably mean different
    things, and log evens them out.

    Returns 0.0 (log1p(0)) when:
      - news_counts_7d is None (no news data loaded)
      - ticker absent from the dict (no coverage in window)
    """
    import math
    if news_counts_7d is None:
        return 0.0
    count = news_counts_7d.get(ticker, 0)
    return float(math.log1p(max(int(count), 0)))


def _days_to_next_earnings(
    ticker: str,
    asof_date: date,
    earnings_calendar: dict[str, date] | None,
) -> float:
    """Calendar days from asof_date to ticker's next earnings report.

    Returned as float for consistency with the rest of FEATURE_NAMES
    (XGBoost handles int fine but the training-set + builder tests assert
    float dtype across all columns; cheaper to cast here than diverge).

    Capped at _EARNINGS_PROXIMITY_CAP_DAYS. Defaults to the cap when:
      - earnings_calendar is None (no calendar loaded)
      - ticker absent from calendar (no scheduled report)
      - next report is more than cap days out
      - next report somehow falls on/before asof (shouldn't happen if the
        loader filters report_date > asof, but guarded anyway)
    """
    if earnings_calendar is None:
        return float(_EARNINGS_PROXIMITY_CAP_DAYS)
    next_date = earnings_calendar.get(ticker)
    if next_date is None:
        return float(_EARNINGS_PROXIMITY_CAP_DAYS)
    days = (next_date - asof_date).days
    if days <= 0:
        return float(_EARNINGS_PROXIMITY_CAP_DAYS)
    return float(min(days, _EARNINGS_PROXIMITY_CAP_DAYS))
