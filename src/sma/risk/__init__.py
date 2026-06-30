"""Shared risk module for simulator and live trading.

Phase 5 extracted the risk rails out of `sma.backtest.risk` so that both the
backtest simulator and the live paper-trading engine consume the same code
path. Single source of truth = no drift between sim and paper.
"""

from sma.risk.rails import RiskRails

__all__ = ["RiskRails"]
