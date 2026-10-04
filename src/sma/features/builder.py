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
from typing import NamedTuple

import pandas as pd

from sma.features import technical
from sma.features.parallel import contiguous_chunks, map_ordered, resolve_workers
from sma.features.window import PriceWindow, SortedPrices

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


# More chunks than workers so the tail stays balanced: tickers differ in how
# much price history they carry (a name delisted mid-window has a third of the
# rows of a full-history name), so equal-sized chunks are NOT equal-cost.
# Handing out 4 small chunks per worker instead of 1 big one lets a worker that
# drew cheap names pick up more.
_TICKER_CHUNKS_PER_WORKER = 4


class _AsofContext(NamedTuple):
    """Everything a feature row needs beyond its own ticker symbol.

    Assembled ONCE per asof_date by build_features and thereafter read-only.
    Bundling it is what lets the parallel path ship the heavy inputs to each
    worker a single time (via the pool initializer) instead of attaching them
    to every ticker chunk.

    `sectors` / `sector_etf_for_gics` are passed in rather than re-imported in
    the worker deliberately: a spawned child would import sma.sectors fresh and
    miss any patch the caller applied, so passing the mappings keeps the
    parallel path faithful to whatever the parent actually read.

    The price inputs are PriceWindows, not frames: already truncated to
    date <= asof_date and sorted, once each, before anything reads them. That
    is what stops the ~22 feature functions from each redoing the mask and the
    sort. It also SHRINKS what the parallel path ships — a worker gets each
    ticker's visible history, not its whole history.
    """

    asof_date: date
    windows_by_ticker: dict[str, PriceWindow]
    spy_window: PriceWindow
    sector_etf_windows: dict[str, PriceWindow]
    ret_30d_by_ticker: dict[str, float | None]
    sector_members: dict[str, list[str]]
    sectors: dict[str, str]
    sector_etf_for_gics: dict[str, str]
    politician_flows: dict[str, float] | None
    earnings_calendar: dict[str, date] | None
    earnings_surprises: dict[str, float] | None
    news_counts_7d: dict[str, int] | None


def _feature_rows_for_tickers(
    tickers: list[str], ctx: _AsofContext,
) -> list[dict]:
    """Compute one feature dict per ticker, in `tickers` order.

    This is the whole per-ticker body of build_features, lifted out verbatim so
    that the serial path and each parallel worker run the SAME code over
    different slices of the universe. Tickers with no price rows, or with any
    feature that comes back None, are dropped — the caller's row list is not
    parallel to `tickers`.

    Must stay a module-level function: `spawn` pickles the task by qualified
    name, so a closure or a local def would not survive the trip to a worker.
    """
    asof_date = ctx.asof_date
    rows: list[dict] = []
    for ticker in tickers:
        # One window per ticker, built once by build_features and read ~22
        # times below. Absent = the ticker has no price rows at all; empty =
        # it has rows but none at or before asof_date, which every feature
        # would have resolved to None anyway (and the None sweep at the bottom
        # of the loop would then have dropped the row).
        target_prices = ctx.windows_by_ticker.get(ticker)
        if target_prices is None or target_prices.rows.empty:
            continue

        # Sector peers (excluding self). Default to 0.0 when peers are
        # absent or insufficient — letting the model learn that "no
        # within-sector signal" maps to neutral.
        sec = ctx.sectors.get(ticker)
        if sec is None or sec in ("ETF", "Sector ETF"):
            sector_rs: float | None = 0.0
        else:
            # Peers keep universe order — see the bit-identical note on
            # rel_strength_sector_30d_from_returns. A peer with no price rows
            # is absent from the map and yields None, exactly as the old path
            # dropped it from the frame list before computing.
            peers = [t for t in ctx.sector_members.get(sec, []) if t != ticker]
            sector_rs = technical.rel_strength_sector_30d_from_returns(
                ctx.ret_30d_by_ticker.get(ticker),
                [ctx.ret_30d_by_ticker.get(t) for t in peers],
            )
            if sector_rs is None:
                sector_rs = 0.0  # insufficient peers — neutral

        # Sector ETF relative strength. Independent of peer membership —
        # benchmarks against an actual tradable index (XLK, XLF, etc.).
        sector_etf_ticker = (
            ctx.sector_etf_for_gics.get(sec) if sec else None
        )
        if sector_etf_ticker and sector_etf_ticker in ctx.sector_etf_windows:
            etf_rs = technical.rel_strength_sector_etf_30d(
                target_prices, ctx.sector_etf_windows[sector_etf_ticker], asof_date,
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
                target_prices, ctx.spy_window, asof_date
            ),
            "gap_open": technical.gap_open(target_prices, asof_date),
            "dist_from_52w_high": technical.dist_from_52w_high(target_prices, asof_date),
            # 2026-06-12 feature wave: risk-adjusted relative momentum,
            # crash-asymmetry, and vol-scaled short-term reversal — trend
            # QUALITY transforms of price data already on hand.
            "vol_adj_mom_60d": technical.vol_adj_mom_60d(
                target_prices, ctx.spy_window, asof_date
            ),
            "downside_vol_ratio_60d": technical.downside_vol_ratio_60d(
                target_prices, asof_date
            ),
            "reversal_5d_z": technical.reversal_5d_z(target_prices, asof_date),
            "earnings_surprise_last": float(
                (ctx.earnings_surprises or {}).get(ticker, 0.0)
            ),
            # 2026-05-10: net House-PTR dollar flow over the prior 30
            # days. 0.0 when the caller didn't supply flow data or no
            # politician trades touched this ticker in the window.
            "politician_flow_30d": (
                float(ctx.politician_flows.get(ticker, 0.0))
                if ctx.politician_flows is not None else 0.0
            ),
            "rel_strength_sector_30d": sector_rs,
            "days_to_next_earnings": _days_to_next_earnings(
                ticker, asof_date, ctx.earnings_calendar,
            ),
            "news_count_7d_log": _news_count_7d_log(ticker, ctx.news_counts_7d),
            "rel_strength_sector_etf_30d": etf_rs,
        }
        # Drop tickers that have any None feature (insufficient history).
        if any(v is None for v in feats.values()):
            continue
        feats["ticker"] = ticker
        rows.append(feats)
    return rows


def build_features(
    prices: pd.DataFrame,
    universe: list[str],
    asof_date: date,
    politician_flows: dict[str, float] | None = None,
    earnings_calendar: dict[str, date] | None = None,
    earnings_surprises: dict[str, float] | None = None,
    news_counts_7d: dict[str, int] | None = None,
    workers: int = 1,
    sorted_prices: SortedPrices | None = None,
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
        workers: processes to spread the per-ticker feature computation over.
            DEFAULTS TO 1 (serial, no pool) on purpose: `spawn` costs ~1s of
            interpreter startup per worker, which is a net LOSS on a single
            cross-section — and MORE of a loss since 2026-08-27 shrank that
            cross-section 4x while leaving spawn where it was. Measured on one
            real 266-name asof at production scale: 0.568s serial, 1.661s at 2
            workers, 2.845s at 4. The live predict path must not pay it.
            Callers that build many asofs should parallelise at the asof level
            instead — see build_training_set(feature_workers=...), which is
            what the retrain uses and which calls this with workers=1.
            Output is bit-identical at any worker count.
        sorted_prices: optional prebuilt SortedPrices index over `prices`. Only
            for callers that build MANY asof_dates off ONE price frame: the
            index carries the per-ticker groupby and sort, so handing the same
            one to every asof pays them once instead of once per date. It MUST
            be built from this exact `prices` frame — nothing checks that, and
            a stale index would quietly serve stale history. None (the default)
            builds one here, which is what every single-asof caller wants.

    Returns:
        DataFrame indexed by ticker with FEATURE_NAMES as columns. Tickers
        for which any feature is None (insufficient history) are DROPPED, not
        emitted with NaN. The training loader handles missing-row exclusion.
    """
    # Local import to avoid circular dep when sma.sectors is loaded first.
    from sma.sectors import SECTOR_ETF_FOR_GICS, SECTORS

    # ONE pass over `prices` to split it by ticker and sort each group by date.
    # Splitting used to be a dict comprehension,
    # {t: prices[prices["ticker"] == t] for t in universe}, which ran a
    # full-frame boolean scan PER TICKER — O(universe x rows), so the cost grew
    # superlinearly in universe size. The breadth study (2026-08-17) measured
    # 1.82 s/asof at 264 names vs 5.08 s/asof at 494: 2.8x the cost for 1.87x
    # the names, extrapolating to ~64 min of feature build per retrain at 500
    # names. groupby makes it linear (1ee39b1).
    #
    # SortedPrices additionally sorts each group ONCE so the per-(ticker, asof)
    # truncation below is a searchsorted prefix slice. Groups are built for
    # EVERY ticker present, not just the universe, which is what makes the SPY
    # and sector-ETF benchmark windows below free.
    index = sorted_prices if sorted_prices is not None else SortedPrices(prices)

    # One window per ticker per asof_date — the whole point. Each is built once
    # here and then read by all ~22 feature functions; before this, every one of
    # them re-masked and re-sorted the ticker's full history for itself (67.0s
    # of a 101.4s build, 226,352 calls on a 266-name x 2y slice).
    windows_by_ticker: dict[str, PriceWindow] = {}
    for t in universe:
        w = index.window(t, asof_date)
        # A universe ticker with no price rows is simply absent from the index;
        # the pre-1ee39b1 comprehension gave it an empty frame instead. Both are
        # handled identically by the guards in _feature_rows_for_tickers.
        if w is not None:
            windows_by_ticker[t] = w

    # `index.empty_window` reproduces the old `prices.iloc[:0]` result (same
    # columns and dtypes) for a frame that carries no SPY rows.
    spy_window = index.window("SPY", asof_date)
    if spy_window is None:
        spy_window = index.empty_window(asof_date)
    sector_etf_windows: dict[str, PriceWindow] = {}
    for etf in set(SECTOR_ETF_FOR_GICS.values()):
        w = index.window(etf, asof_date)
        if w is not None:
            sector_etf_windows[etf] = w
    # Pre-group by sector so the sector-relative-strength lookup is O(1)
    # instead of O(universe).
    sector_members: dict[str, list[str]] = {}
    for t in universe:
        sec = SECTORS.get(t)
        # ETFs are benchmarks, not within-sector members: SPY/NANC ("ETF") and
        # the SPDR sector ETFs ("Sector ETF", e.g. XLK/XLE) have no meaningful
        # GICS-sector peer group, so they neither form nor join peer buckets.
        if sec is None or sec in ("ETF", "Sector ETF"):
            continue
        sector_members.setdefault(sec, []).append(t)

    # Each ticker's 30d return, computed ONCE. The sector-leadership feature
    # needs every peer's 30d return for every target, so computing it inside
    # the per-ticker loop meant a sector of S names did S*(S-1) computations
    # for S distinct values — the real quadratic in this function (53% of a
    # 266-name asof under cProfile, 3.27s of 6.19s). Order of the peer list
    # below is preserved so the mean's float sum is bit-identical.
    #
    # This stays SERIAL even at workers > 1, and deliberately: every chunk
    # needs every OTHER chunk's 30d returns to form its sector peer mean, so
    # fanning it out would need a gather barrier and a second pool dispatch per
    # asof. Off a prebuilt window it is now nearly free; what dominates the
    # serial preamble instead is SortedPrices, 0.067s of a 0.559s 266-name
    # cross-section at production scale — an Amdahl floor of ~2.8x at 4
    # workers, down from ~3.6x only because the parallel half got 4x cheaper.
    # Callers that build many asofs pass `sorted_prices` and skip even that.
    ret_30d_by_ticker: dict[str, float | None] = {
        t: technical.ret_n_days(w, asof_date, 30)
        for t, w in windows_by_ticker.items()
    }

    ctx = _AsofContext(
        asof_date=asof_date,
        windows_by_ticker=windows_by_ticker,
        spy_window=spy_window,
        sector_etf_windows=sector_etf_windows,
        ret_30d_by_ticker=ret_30d_by_ticker,
        sector_members=sector_members,
        sectors=SECTORS,
        sector_etf_for_gics=SECTOR_ETF_FOR_GICS,
        politician_flows=politician_flows,
        earnings_calendar=earnings_calendar,
        earnings_surprises=earnings_surprises,
        news_counts_7d=news_counts_7d,
    )

    n_workers = resolve_workers(workers, len(universe))
    if n_workers == 1:
        rows = _feature_rows_for_tickers(list(universe), ctx)
    else:
        # Chunks are CONTIGUOUS slices of `universe` and their results are
        # concatenated in chunk order, so `rows` lands in exactly the order the
        # serial loop produces it — which is what makes the output frame
        # bit-identical rather than merely equal as a set.
        chunks = contiguous_chunks(
            list(universe), n_workers * _TICKER_CHUNKS_PER_WORKER,
        )
        rows = [
            row
            for chunk_rows in map_ordered(
                _feature_rows_for_tickers, chunks, ctx, workers=n_workers,
            )
            for row in chunk_rows
        ]

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
