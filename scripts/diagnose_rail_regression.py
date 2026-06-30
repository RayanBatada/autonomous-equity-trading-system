"""Phase 3 rail-regression diagnostic: which rail dropped val Sharpe?

Phase 2 (no rails, no theses):    val Sharpe +0.083
Phase 4 (all rails, no theses):   val Sharpe -0.272
Delta: -0.355 across the four Phase 3 rails.

Procedure: run xgb_top_k on the val window 5 times — once with all rails on
(baseline), then once with each rail individually disabled. Whichever rail's
removal recovers the most Sharpe is the primary suspect.

`use_theses=False` throughout so we match the "no theses" Phase 2 vs Phase 4
comparison, isolating the rails as the only moving part.

Run:
    cd ~/code/Stock-Market-Predictor-Agents
    .venv/bin/python scripts/diagnose_rail_regression.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sma.backtest.__main__ import _build_strategy
from sma.backtest.risk import RiskRails
from sma.backtest.simulator import simulate
from sma.backtest.windows import window_dates
from sma.eval.evaluate_strategy import _load_prices_for_window, load_upcoming_earnings
from sma.ingest.universe import load_universe
from sma.sectors import sector_map_for

DB_PATH = Path("data/sma.duckdb")
UNIVERSE_PATH = Path("src/sma/universe.yaml")


def run_one(label: str, *, rails: RiskRails, earnings_blackouts: dict | None,
            strategy, prices, universe: list[str]) -> dict:
    """Run one configuration and return a metrics dict."""
    start_date, end_date = window_dates("val")
    result = simulate(
        strategy=strategy,
        universe=universe,
        prices=prices,
        sector_map=sector_map_for(universe),
        earnings_blackouts=earnings_blackouts,
        window_name="val",
        start_date=start_date,
        end_date=end_date,
        initial_cash=100_000.0,
        rails=rails,
    )
    return {
        "label": label,
        "sharpe": result.sharpe,
        "sortino": result.sortino,
        "total_return": result.total_return,
        "max_drawdown": result.max_drawdown,
        "hit_rate": result.hit_rate,
        "num_trades": result.num_trades,
    }


def main() -> None:
    universe = load_universe(UNIVERSE_PATH)
    print(f"Loading val-window prices for {len(universe)} tickers...")
    start_date, end_date = window_dates("val")
    prices = _load_prices_for_window(DB_PATH, universe, start_date, end_date)
    earnings = load_upcoming_earnings(DB_PATH, start_date, end_date)
    print(f"Prices: {len(prices)} rows, earnings: {sum(len(v) for v in earnings.values())} dates "
          f"across {len(earnings)} tickers")

    # Build the strategy ONCE; it caches the model + predictor internally.
    print("Building xgb_top_k strategy (use_theses=False)...")
    strategy = _build_strategy("xgb_top_k", seed=None, universe=universe, use_theses=False)

    configs = [
        ("baseline (all rails ON)", RiskRails(), earnings),
        ("stop_loss disabled",
            RiskRails(stop_loss_pct=999.0), earnings),
        ("cash_floor disabled",
            RiskRails(cash_floor_pct=0.0), earnings),
        ("sector_cap disabled",
            RiskRails(max_sector_pct=1.0), earnings),
        ("earnings_blackout disabled", RiskRails(), {}),
    ]

    results: list[dict] = []
    for label, rails, blackouts in configs:
        print(f"\n=== Running: {label} ===")
        r = run_one(label, rails=rails, earnings_blackouts=blackouts,
                    strategy=strategy, prices=prices, universe=universe)
        results.append(r)
        print(f"  Sharpe:        {r['sharpe']:+.4f}")
        print(f"  Sortino:       {r['sortino']:+.4f}")
        print(f"  Total return:  {r['total_return']:+.2%}")
        print(f"  Max drawdown:  {r['max_drawdown']:.2%}")
        print(f"  Hit rate:      {r['hit_rate']:.2%}")
        print(f"  Trades:        {r['num_trades']}")

    print("\n" + "=" * 80)
    print("SUMMARY (sorted by Sharpe recovery vs baseline)")
    print("=" * 80)
    baseline_sharpe = results[0]["sharpe"]
    print(f"\nBaseline Sharpe (all rails on):   {baseline_sharpe:+.4f}")
    print("Phase 2 reference (no rails):    +0.0830")
    print(f"Total regression to recover:    {0.0830 - baseline_sharpe:+.4f}\n")

    header = (
        f"{'Configuration':<32} {'Sharpe':>10} {'Δ baseline':>14}  "
        f"{'% recovered':>14}"
    )
    print(header)
    print("-" * len(header))
    denom = 0.0830 - baseline_sharpe
    for r in sorted(results, key=lambda x: x["sharpe"], reverse=True):
        delta = r["sharpe"] - baseline_sharpe
        pct = (delta / denom) * 100 if denom > 0 else 0.0
        print(
            f"{r['label']:<32} {r['sharpe']:>+10.4f} {delta:>+14.4f}  "
            f"{pct:>12.1f}%"
        )


if __name__ == "__main__":
    main()
