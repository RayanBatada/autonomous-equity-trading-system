"""Test the horizon-mismatch fix: the model targets 30 trading days but the bot
churns ~13 trades/day (holds ~5d). Lengthen holds / cut churn via min_hold_days
(hold dropped names longer) + rebalance_dead_zone_pct (skip small rebalances),
now that the simulator models both. If higher mh/dz cuts turnover AND improves
risk-adjusted return, the horizon mismatch is real and this is the fix.
k=15, no stop (the validated config). VAL + TEST.
"""
from __future__ import annotations

from sma.backtest.__main__ import _DEFAULT_DB_PATH, _DEFAULT_MODELS_DIR, _get_universe
from sma.backtest.strategies.xgb_top_k import XGBoostTopKStrategy
from sma.eval.evaluate_strategy import evaluate_strategy
from sma.model.predictor import Predictor
from sma.risk.rails import RiskRails

universe = _get_universe()


def run(name, window, *, min_hold, dead_zone):
    predictor = Predictor(models_dir=_DEFAULT_MODELS_DIR, db_path=_DEFAULT_DB_PATH)
    strat = XGBoostTopKStrategy(predictor=predictor, universe=universe, k=15,
                                target_weight_per_position=0.10, weight_tilt=True)
    rails = RiskRails(stop_loss_pct=0.0, max_sector_pct=0.35, max_position_pct=0.10,
                      cash_floor_pct=0.05, max_drawdown_pct=0.15,
                      min_hold_days=min_hold, rebalance_dead_zone_pct=dead_zone)
    kw = {"i_promise_this_is_a_promotion_decision": True} if window == "test" else {}
    r = evaluate_strategy(
        strategy=strat, window=window, universe=universe, rails=rails,
        membership="current", **kw,
    )
    print(f"{name:<30} sharpe={r.sharpe:+.3f}  ret={r.total_return:+.2%}  "
          f"maxDD={r.max_drawdown:+.2%}  hold={r.avg_holding_days:.1f}d  trades={r.num_trades}")
    return r


CONFIGS = [
    ("baseline (mh=1, dz=0.10)", 1, 0.10),
    ("dz=0.25", 1, 0.25),
    ("mh=10", 10, 0.10),
    ("mh=10 + dz=0.25", 10, 0.25),
    ("mh=20 + dz=0.40", 20, 0.40),
]
for w in ("val", "test"):
    print(f"{w.upper()} (horizon: model=30d, lengthen holds / cut churn):")
    print("-" * 100)
    for name, mh, dz in CONFIGS:
        run(f"  {name}", w, min_hold=mh, dead_zone=dz)
