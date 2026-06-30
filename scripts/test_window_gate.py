"""Test-window gate: with-vs-without theses, with shipping rail config (stop_loss off).

This is the gate that determines `live.use_theses` for Phase 5 paper trading.
Decision rule: ship with theses if with-theses Sharpe ≥ without-theses Sharpe + 0.05.

Run:
    cd ~/code/Stock-Market-Predictor-Agents
    .venv/bin/python scripts/test_window_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sma.backtest.__main__ import _build_strategy
from sma.backtest.simulator import simulate
from sma.backtest.windows import window_dates
from sma.eval.evaluate_strategy import _load_prices_for_window
from sma.ingest.universe import load_universe
from sma.risk.earnings_blackout import load_upcoming_earnings
from sma.risk.rails import RiskRails
from sma.sectors import sector_map_for

DB_PATH = Path("data/sma.duckdb")
UNIVERSE_PATH = Path("src/sma/universe.yaml")
DECISION_DELTA = 0.05   # ship-with-theses threshold

# Phase 5 ships with stop_loss disabled per the 2026-04-28 rail diagnostic.
SHIPPING_RAILS = RiskRails(stop_loss_pct=0.0)


def run_one(label: str, *, use_theses: bool, strategy, prices, universe,
            earnings) -> dict:
    start_date, end_date = window_dates("test")
    result = simulate(
        strategy=strategy,
        universe=universe,
        prices=prices,
        sector_map=sector_map_for(universe),
        earnings_blackouts=earnings,
        window_name="test",
        start_date=start_date,
        end_date=end_date,
        initial_cash=100_000.0,
        rails=SHIPPING_RAILS,
    )
    return {
        "label": label,
        "use_theses": use_theses,
        "sharpe": result.sharpe,
        "sortino": result.sortino,
        "total_return": result.total_return,
        "max_drawdown": result.max_drawdown,
        "hit_rate": result.hit_rate,
        "num_trades": result.num_trades,
    }


def main() -> None:
    universe = load_universe(UNIVERSE_PATH)
    print(f"Loading test-window prices for {len(universe)} tickers...")
    start_date, end_date = window_dates("test")
    prices = _load_prices_for_window(DB_PATH, universe, start_date, end_date)
    earnings = load_upcoming_earnings(DB_PATH, start_date, end_date)
    print(f"Prices: {len(prices)} rows, "
          f"earnings: {sum(len(v) for v in earnings.values())} dates "
          f"across {len(earnings)} tickers")
    print(f"Test window: {start_date} → {end_date}")
    print(f"Rails (shipping): {SHIPPING_RAILS}")
    print()

    print("=== Run 1: WITHOUT theses ===")
    strategy_baseline = _build_strategy(
        "xgb_top_k", seed=None, universe=universe, use_theses=False,
    )
    baseline = run_one(
        "without theses (baseline)",
        use_theses=False, strategy=strategy_baseline,
        prices=prices, universe=universe, earnings=earnings,
    )
    for k in ("sharpe", "sortino", "total_return", "max_drawdown", "hit_rate", "num_trades"):
        v = baseline[k]
        if k in ("total_return", "max_drawdown", "hit_rate"):
            print(f"  {k:18s}: {v:+.2%}")
        elif k == "num_trades":
            print(f"  {k:18s}: {v}")
        else:
            print(f"  {k:18s}: {v:+.4f}")
    print()

    print("=== Run 2: WITH theses ===")
    strategy_theses = _build_strategy(
        "xgb_top_k", seed=None, universe=universe, use_theses=True,
    )
    with_theses = run_one(
        "with theses",
        use_theses=True, strategy=strategy_theses,
        prices=prices, universe=universe, earnings=earnings,
    )
    for k in ("sharpe", "sortino", "total_return", "max_drawdown", "hit_rate", "num_trades"):
        v = with_theses[k]
        if k in ("total_return", "max_drawdown", "hit_rate"):
            print(f"  {k:18s}: {v:+.2%}")
        elif k == "num_trades":
            print(f"  {k:18s}: {v}")
        else:
            print(f"  {k:18s}: {v:+.4f}")
    print()

    print("=" * 76)
    print("GATE DECISION")
    print("=" * 76)
    delta = with_theses["sharpe"] - baseline["sharpe"]
    decision = "use_theses=TRUE" if delta >= DECISION_DELTA else "use_theses=FALSE"
    print(f"Δ Sharpe:                {delta:+.4f}")
    print(f"Decision threshold:      ≥ +{DECISION_DELTA:.2f}")
    print(f"Recommended Phase 5 cfg: {decision}")
    print()
    if delta >= DECISION_DELTA:
        print("Theses provide meaningful edge on truly held-out data. Ship Phase 5")
        print("with use_theses=True. Update config.yaml live.use_theses if needed.")
    else:
        print("Theses do NOT meet the +0.05 Sharpe bar on truly held-out data.")
        print("Ship Phase 5 with use_theses=False. Theses remain a dashboard-only")
        print("feature; live decide skips the LLM thesis fetch entirely.")


if __name__ == "__main__":
    main()
