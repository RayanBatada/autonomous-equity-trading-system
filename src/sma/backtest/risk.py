"""Per-order risk checks. Pure functions called by the simulator before every fill.

Phase 5 extracted both RiskRails and check_order to sma.risk; this module is
now a back-compat shim. New code should import directly from sma.risk.
"""

from sma.risk.pipeline import check_order
from sma.risk.rails import RiskRails

__all__ = ["RiskRails", "check_order"]


def eval_rails_for(strategy) -> "RiskRails":
    """Rails consistent with the STRATEGY's contract for research evals.

    The default RiskRails position cap (5%) silently REJECTED every decision
    of a k=10 x 10% strategy, so historical backtests measured the <=5% TAIL
    of the top-K (2026-06-10 inverse-selection finding). Every research eval
    path (evaluate, detect-overfit, autoresearch) must build rails from the
    strategy target instead of defaults.
    """
    return RiskRails(
        stop_loss_pct=0.0,
        max_position_pct=getattr(strategy, "target_weight", 0.05),
    )
