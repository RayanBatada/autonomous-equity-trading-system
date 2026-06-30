"""B-group backtest comparison on the val window (2025-07-01 .. 2025-12-31).

Runs the xgb_top_k strategy under several configs and prints a metrics table so
we ship only configs that improve risk-adjusted return. Backtestable levers:
stop_loss_pct (B4), concentration k / weight_tilt (B1-ish). Cash-drag dead_zone
and min_hold are live-decide-only (not modeled here).
"""
from __future__ import annotations

from sma.backtest.__main__ import _DEFAULT_DB_PATH, _DEFAULT_MODELS_DIR, _get_universe
from sma.backtest.strategies.xgb_top_k import XGBoostTopKStrategy
from sma.eval.evaluate_strategy import evaluate_strategy
from sma.model.predictor import Predictor
from sma.risk.rails import RiskRails

universe = _get_universe()
print(f"universe: {len(universe)} tickers; window=val (2025-07-01..2025-12-31)\n")


def run(name, *, k=15, target_weight=0.10, weight_tilt=True, stop_loss=0.0):
    predictor = Predictor(models_dir=_DEFAULT_MODELS_DIR, db_path=_DEFAULT_DB_PATH)
    strat = XGBoostTopKStrategy(
        predictor=predictor, universe=universe, k=k,
        target_weight_per_position=target_weight, weight_tilt=weight_tilt,
    )
    rails = RiskRails(
        stop_loss_pct=stop_loss, max_sector_pct=0.35, max_position_pct=0.10,
        cash_floor_pct=0.05, max_drawdown_pct=0.15,
    )
    r = evaluate_strategy(
        strategy=strat, window="val", universe=universe, rails=rails,
        membership="current",
    )
    print(f"{name:<22} sharpe={r.sharpe:+.3f}  ret={r.total_return:+.2%}  "
          f"maxDD={r.max_drawdown:+.2%}  calmar={r.calmar:+.2f}  "
          f"hold={r.avg_holding_days:.1f}d  trades={r.num_trades}  hit={r.hit_rate:.0%}")
    return r


# Baseline = current live config (k=15, weight_tilt on, stop_loss disabled).
print(f"{'CONFIG':<22} metrics")
print("-" * 100)
base = run("baseline(live)")
# B4: re-evaluate the stop-loss rail (currently disabled).
run("B4 stop_loss=0.08", stop_loss=0.08)
run("B4 stop_loss=0.05", stop_loss=0.05)
run("B4 stop_loss=0.12", stop_loss=0.12)
# B1-ish: is the built-in rank conviction-tilt helping? more/less concentration?
run("weight_tilt=OFF", weight_tilt=False)
run("k=10 (concentrated)", k=10)
run("k=20 (diversified)", k=20)
print("\nbaseline sharpe =", round(base.sharpe, 3))
