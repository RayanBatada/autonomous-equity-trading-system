"""Codex post-audit review (2026-06-09): the 2e2c085 bearish-veto fix keys on
`current_holdings`, and the SIMULATOR passes it — but live decide_once called
strategy.decide(asof_date, prices) only. Live therefore saw held=∅, treated
every held merely-bearish name as a new-buy candidate, vetoed it, and
translate() force-sold it — the exact regression 2e2c085 was meant to stop,
still live (and --use-theses IS on in the decide plist)."""

from datetime import date
from unittest.mock import MagicMock

from sma.backtest.strategies.base import StrategyDecision
from sma.ingest.store import Store
from sma.live.decide import decide_once
from sma.risk.rails import RiskRails


def _alpaca_with_positions(positions: dict):
    alpaca = MagicMock()
    alpaca.get_account.return_value = {
        "equity": 100_000.0,
        "cash": 50_000.0,
        "trading_blocked": False,
        "account_blocked": False,
    }
    alpaca.get_positions.return_value = positions
    return alpaca


class _SpyHoldingsStrategy:
    """Accepts current_holdings — must receive the live held set."""

    def __init__(self):
        self.received = "NOT-CALLED"

    def decide(self, *, asof_date, prices, current_holdings=None):
        self.received = current_holdings
        # keep the held names so the paranoia rail (0 decisions with held
        # positions) doesn't abort the dry-run
        return [
            StrategyDecision(asof_date=asof_date, ticker=t, target_weight=0.05)
            for t in sorted(current_holdings or ["HOOD"])
        ]


class _LegacyStrategy:
    """No current_holdings kwarg — must still be callable (no TypeError)."""

    def __init__(self):
        self.called = False

    def decide(self, *, asof_date, prices):
        self.called = True
        return [StrategyDecision(asof_date=asof_date, ticker="HOOD", target_weight=0.05)]


def _run(tmp_path, strategy, positions):
    store = Store(path=tmp_path / "t.duckdb").connect()
    try:
        return decide_once(
            asof=date(2026, 4, 30),
            store=store,
            alpaca=_alpaca_with_positions(positions),
            universe=["AAPL", "HOOD"],
            strategy=strategy,
            sector_for=lambda t: "tech",
            rails=RiskRails(stop_loss_pct=0.0),
            dry_run=True,
        )
    finally:
        store.conn.close()


def test_decide_passes_live_holdings_to_strategy(tmp_path):
    spy = _SpyHoldingsStrategy()
    _run(
        tmp_path, spy,
        {"HOOD": {"shares": 12, "cost_basis": 80.0}, "AVGO": {"shares": 14, "cost_basis": 300.0}},
    )
    assert spy.received == {"HOOD", "AVGO"}, (
        "live decide must pass current holdings — without them the bearish "
        "veto force-sells held names (2e2c085 regression)"
    )


def test_decide_supports_legacy_strategy_without_holdings_kwarg(tmp_path):
    legacy = _LegacyStrategy()
    _run(tmp_path, legacy, {"HOOD": {"shares": 12, "cost_basis": 80.0}})
    assert legacy.called
