"""Phase 5: Alpaca paper trading live module.

Three launchd-fired daily jobs:
  - decide (18:35 ET, Mon-Fri): strategy → risk → orders → submit
  - stop_loss_sweep (09:25 ET, Mon-Fri): force-sell positions hitting stop-loss
  - reconcile (16:30 ET, Mon-Fri): pull fills + snapshot account + drift detection

The risk pipeline (sma.risk) is shared with the simulator so paper-vs-sim P&L
stays comparable.
"""
