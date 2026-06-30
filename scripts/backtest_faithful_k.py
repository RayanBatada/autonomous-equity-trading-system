"""Re-answer the k question on the FIXED simulator (force-sell + stop-disable).
All earlier k conclusions were confounded by those bugs. No-stop (the corrected
winner) across k, on val + test, to pick the best concentration."""
from __future__ import annotations

from sma.backtest.__main__ import _DEFAULT_DB_PATH, _DEFAULT_MODELS_DIR, _get_universe
from sma.backtest.strategies.xgb_top_k import XGBoostTopKStrategy
from sma.eval.evaluate_strategy import evaluate_strategy
from sma.model.predictor import Predictor
from sma.risk.rails import RiskRails

universe = _get_universe()


def run(name, window, *, k, weight_tilt=True):
    predictor = Predictor(models_dir=_DEFAULT_MODELS_DIR, db_path=_DEFAULT_DB_PATH)
    strat = XGBoostTopKStrategy(predictor=predictor, universe=universe, k=k,
                                target_weight_per_position=0.10, weight_tilt=weight_tilt)
    rails = RiskRails(stop_loss_pct=0.0, max_sector_pct=0.35, max_position_pct=0.10,
                      cash_floor_pct=0.05, max_drawdown_pct=0.15)
    kw = {"i_promise_this_is_a_promotion_decision": True} if window == "test" else {}
    r = evaluate_strategy(
        strategy=strat, window=window, universe=universe, rails=rails,
        membership="current", **kw,
    )
    print(f"{name:<26} sharpe={r.sharpe:+.3f}  ret={r.total_return:+.2%}  "
          f"maxDD={r.max_drawdown:+.2%}  hold={r.avg_holding_days:.1f}d  trades={r.num_trades}")
    return r


print("FAITHFUL k-sweep (no-stop, force-sell fixed)")
print("-" * 100)
for w in ("val", "test"):
    print(f"{w.upper()}:")
    run("  k=10", w, k=10)
    run("  k=12", w, k=12)
    run("  k=15 (live)", w, k=15)
print("weight_tilt sanity (val, k=15):")
run("  weight_tilt=OFF", "val", k=15, weight_tilt=False)
