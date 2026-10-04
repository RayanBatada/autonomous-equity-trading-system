"""Dispersion de-risk overlay A/B — diagnosis 2026-06-25 #2, run 2026-07-20.

Question: does a cross-sectional momentum-dispersion cash-floor overlay
(risk/dispersion.py) improve the strategy in reversal regimes without hurting
bulls? Gate (from the diagnosis): ON must beat OFF on Sharpe AND maxDD in
EVERY reversal window and ≥2 bull windows, with a plateau across floor caps —
look-ahead-free.

Design notes (honesty):
- The predictor model is the DEPLOYED 7/20 artifact pinned to an early date, so
  every historical asof is IN-SAMPLE for it. Absolute numbers are optimistic;
  the A/B is still informative because BOTH arms share identical predictions —
  the comparison isolates the overlay. Same standard as the ab_wave campaigns.
- The detector is strictly PIT: dispersion at d uses returns ending at d; its
  z-score baseline uses dispersion values strictly BEFORE d.
- Runs against a SCRATCH COPY of the DB + a scratch models dir — production
  files untouched, no lock contention with the evening pipeline.

Usage: .venv/bin/python scripts/dispersion_derisk_ab.py [--probe]
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sma.backtest.earnings_blackout import load_upcoming_earnings
from sma.backtest.simulator import simulate
from sma.backtest.slippage import SlippageModel
from sma.backtest.strategies.xgb_top_k import XGBoostTopKStrategy
from sma.ingest.universe import load_universe
from sma.model.predictor import Predictor
from sma.risk.dispersion import (
    dispersion_floor_series,
    momentum_dispersion_series,
    rolling_z,
)
from sma.risk.rails import RiskRails
from sma.sectors import sector_map_for

SCRATCH = Path(
    "/private/tmp/claude-501/-Users-youruser-SecondBrain/"
    "9c290692-7caf-4a33-9129-9d332a7cd4cb/scratchpad/dispersion-ab"
)
DB = SCRATCH / "sma-research.duckdb"
MODELS = SCRATCH / "models"

SIM_START = date(2023, 1, 1)
WARMUP_START = date(2021, 6, 1)  # 60d lookback + 252d z baseline before SIM_START


def load_prices(con, universe, start, end) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT ticker, date, open, close, adj_close, volume
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY ticker, date
                ORDER BY CASE source WHEN 'yfinance' THEN 0
                                     WHEN 'yfinance_hist' THEN 1
                                     WHEN 'alpaca' THEN 2 ELSE 3 END
            ) AS rn
            FROM prices
            WHERE ticker = ANY($t) AND date BETWEEN $s AND $e
              AND adj_close IS NOT NULL
        ) q WHERE rn = 1 ORDER BY ticker, date
        """,
        {"t": list(universe), "s": start, "e": end},
    ).df()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def momentum_factor_label(prices: pd.DataFrame, w_start: date, w_end: date) -> str:
    """TREND/REVERSAL label for a window: monthly winners-minus-losers of
    trailing-60-session return, averaged over the window. Negative = REVERSAL."""
    piv = prices.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
    dates = [d for d in piv.index]
    month_starts = []
    cur = None
    for d in dates:
        key = (d.year, d.month)
        if key != cur and w_start <= d <= w_end:
            month_starts.append(d)
            cur = key
        elif key != cur:
            cur = key
    factor_rets = []
    for ms in month_starts:
        i = dates.index(ms)
        if i < 60 or i + 21 >= len(dates):
            continue
        past = piv.iloc[i] / piv.iloc[i - 60] - 1.0
        fwd = piv.iloc[i + 21] / piv.iloc[i] - 1.0
        both = pd.concat([past.rename("p"), fwd.rename("f")], axis=1).dropna()
        if len(both) < 40:
            continue
        q = len(both) // 5
        srt = both.sort_values("p")
        factor_rets.append(float(srt.tail(q)["f"].mean() - srt.head(q)["f"].mean()))
    if not factor_rets:
        return "UNKNOWN"
    avg = sum(factor_rets) / len(factor_rets)
    return f"{'REVERSAL' if avg < 0 else 'TREND'} ({avg:+.3f})"


def window_stats(daily: list[tuple[date, float]], w_start: date, w_end: date):
    rs = [r for d, r in daily if w_start <= d <= w_end]
    if len(rs) < 10:
        return None
    mean = sum(rs) / len(rs)
    var = sum((r - mean) ** 2 for r in rs) / (len(rs) - 1)
    sharpe = (mean / (var ** 0.5) * 252 ** 0.5) if var > 0 else 0.0
    eq = 1.0
    peak = 1.0
    maxdd = 0.0
    for r in rs:
        eq *= 1 + r
        peak = max(peak, eq)
        maxdd = max(maxdd, (peak - eq) / peak)
    total = eq - 1.0
    return {"sharpe": sharpe, "maxdd": maxdd, "ret": total, "n": len(rs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="3-month timing probe")
    ap.add_argument("--slope", type=float, default=0.25)
    ap.add_argument("--z-start", type=float, default=1.0)
    ap.add_argument("--detector", choices=("dispersion", "breadth", "both"),
                    default="both")
    ap.add_argument("--skip-off", action="store_true",
                    help="skip the OFF baseline (reuse a prior run's)")
    args = ap.parse_args()

    sim_end = date(2026, 7, 17)
    if args.probe:
        sim_start = date(2025, 4, 1)
        probe_end = date(2025, 6, 30)
    else:
        sim_start = SIM_START
        probe_end = sim_end

    universe = load_universe("src/sma/universe.yaml")
    con = duckdb.connect(str(DB), read_only=True)
    t0 = time.time()
    prices_all = load_prices(con, universe, WARMUP_START, sim_end)
    earnings = load_upcoming_earnings(DB, sim_start, probe_end)
    print(f"loaded {len(prices_all)} price rows in {time.time()-t0:.1f}s", flush=True)

    # ---- detector (PIT, prices-only), from the warmup-extended span ----------
    equities = [t for t in universe if t not in {
        "SPY", "NANC", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP",
        "XLRE", "XLU", "XLV", "XLY",
    }]
    from sma.risk.dispersion import breadth_series
    eq_px = prices_all[prices_all["ticker"].isin(equities)]
    zs: dict[str, dict] = {}
    if args.detector in ("dispersion", "both"):
        sig = momentum_dispersion_series(eq_px, lookback=60, min_names=60)
        zs["dispersion"] = rolling_z(sig, window=252, min_window=120)
    if args.detector in ("breadth", "both"):
        # Breadth COLLAPSE de-risks: z of NEGATIVE breadth → low breadth vs
        # its own trailing history = high z = floor rises.
        sigb = {d: -b for d, b in breadth_series(
            eq_px, ma_window=50, min_names=60).items()}
        zs["breadth"] = rolling_z(sigb, window=252, min_window=120)
    for k, zk in zs.items():
        print(f"{k} z series: {len(zk)} dates", flush=True)

    arms: dict[str, dict[date, float] | None] = {}
    if not args.skip_off:
        arms["OFF"] = None
    for tag, zk in zs.items():
        arms[f"{tag} cap=0.3"] = dispersion_floor_series(
            zk, base_floor=0.0, z_start=args.z_start, slope=args.slope, cap=0.3)
        arms[f"{tag} cap=0.5"] = dispersion_floor_series(
            zk, base_floor=0.0, z_start=args.z_start, slope=args.slope, cap=0.5)
    for name, floors in arms.items():
        if floors:
            hot = sum(1 for d, f in floors.items()
                      if sim_start <= d <= probe_end and f > 0)
            span = sum(1 for d in floors if sim_start <= d <= probe_end)
            print(f"{name}: floor>0 on {hot}/{span} sim dates", flush=True)

    prices_sim = prices_all[prices_all["date"] >= WARMUP_START]
    sector_map = sector_map_for(universe)

    class MemoPredictor:
        """Memoize predict() by asof: all arms share one pinned model + the
        same dates, so predictions are identical across arms — arm 1 pays the
        XGB/feature cost, later arms replay from the dict (pure function of
        (model, asof, universe); the strategy's hysteresis state stays in the
        per-arm strategy object, NOT here)."""

        def __init__(self, inner):
            self._inner = inner
            self._memo = {}

        def predict_for(self, asof_date, universe, *a, **kw):
            key = asof_date
            if key not in self._memo:
                self._memo[key] = self._inner.predict_for(
                    asof_date, universe, *a, **kw)
            return self._memo[key]

        def __getattr__(self, name):
            return getattr(self._inner, name)

    shared_predictor = MemoPredictor(
        Predictor(models_dir=MODELS, db_path=DB, conn=con))
    results = {}
    for name, floors in arms.items():
        strat = XGBoostTopKStrategy(
            predictor=shared_predictor,
            universe=universe, k=15, hold_rank=30, sector_neutralize=1.0,
        )
        rails = RiskRails(stop_loss_pct=0.0, max_position_pct=strat.target_weight,
                          min_hold_days=7)
        t0 = time.time()
        r = simulate(
            strategy=strat, universe=universe, prices=prices_sim,
            sector_map=sector_map, earnings_blackouts=earnings,
            window_name="train", start_date=sim_start, end_date=probe_end,
            initial_cash=100_000.0, slippage_model=SlippageModel(),
            rails=rails, extra_cash_floor_by_date=floors,
        )
        # BacktestResult carries daily_returns but not dates; the sim iterates
        # the sorted in-window trading dates, so reconstruct and zip STRICT
        # (a length mismatch means the assumption broke — fail loud).
        trading_dates = sorted({
            d for d in prices_sim["date"].unique()
            if sim_start <= d <= probe_end
        })
        daily = list(zip(trading_dates, r.daily_returns, strict=True))
        results[name] = (r, daily)
        print(f"{name}: sharpe={r.sharpe:.3f} ret={r.total_return:+.2%} "
              f"maxDD={r.max_drawdown:.2%} trades={r.num_trades} "
              f"({time.time()-t0:.0f}s)", flush=True)
    con.close()

    if args.probe:
        return

    # ---- per-window verdict --------------------------------------------------
    windows = [
        ("2023H1", date(2023, 1, 1), date(2023, 6, 30)),
        ("2023H2", date(2023, 7, 1), date(2023, 12, 31)),
        ("2024H1", date(2024, 1, 1), date(2024, 6, 30)),
        ("2024H2", date(2024, 7, 1), date(2024, 12, 31)),
        ("2025H1", date(2025, 1, 1), date(2025, 6, 30)),
        ("2025H2", date(2025, 7, 1), date(2025, 12, 31)),
        ("2026YTD", date(2026, 1, 1), sim_end),
    ]
    print("\n=== per-window (label = monthly 60d winners-minus-losers avg) ===")
    hdr = f"{'window':9} {'label':118}"
    print(f"{'window':9} {'regime':18} " + "  ".join(f"{k:>24}" for k in arms))
    for wname, ws, we in windows:
        label = momentum_factor_label(
            prices_all[prices_all["date"] <= we], ws, we)
        cells = []
        for name in arms:
            st = window_stats(results[name][1], ws, we)
            cells.append(
                f"shp {st['sharpe']:+.2f} dd {st['maxdd']:.1%}" if st else "n/a")
        print(f"{wname:9} {label:18} " + "  ".join(f"{c:>24}" for c in cells))
    _ = hdr


if __name__ == "__main__":
    main()
