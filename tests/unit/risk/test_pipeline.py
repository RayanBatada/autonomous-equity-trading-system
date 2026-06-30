"""sma.risk.pipeline.apply tests."""

from datetime import date

from sma.backtest.strategies.base import StrategyDecision
from sma.risk.pipeline import RiskContext, apply
from sma.risk.rails import RiskRails


def _decision(ticker: str, weight: float = 0.05) -> StrategyDecision:
    return StrategyDecision(
        asof_date=date(2026, 1, 5), ticker=ticker, target_weight=weight,
    )


def _ctx(**overrides) -> RiskContext:
    base = dict(
        rails=RiskRails(stop_loss_pct=0.0),
        account_value=100_000.0,
        cash=50_000.0,
        current_positions_dollars={},
        sector_exposure_pct={},
        sector_for=lambda t: "Tech",
        current_drawdown=0.0,
        upcoming_earnings={},
        asof_date=date(2026, 1, 5),
    )
    base.update(overrides)
    return RiskContext(**base)


def test_pipeline_passes_decision_through_when_all_rails_clear():
    ctx = _ctx()
    out = apply([_decision("AAPL")], ctx)
    assert len(out) == 1
    assert out[0].ticker == "AAPL"


def test_pipeline_drops_decision_when_target_sector_exceeds_cap():
    """The sector cap bounds the TARGET portfolio (sum of decision target_weights),
    NOT current holdings. Two 6% Tech targets under a 10% cap: first accepted,
    second rejected (0.06 + 0.06 = 0.12 > 0.10)."""
    ctx = _ctx(rails=RiskRails(
        stop_loss_pct=0.0, max_sector_pct=0.10, max_position_pct=0.10,
    ))
    out = apply([_decision("AAPL", weight=0.06), _decision("MSFT", weight=0.06)], ctx)
    assert [d.ticker for d in out] == ["AAPL"]


def test_pipeline_sector_cap_counts_retained_holdings():
    """Codex audit / regression: a NEW same-sector buy must be rejected when
    retained holdings already fill the sector. translate() keeps model-approved
    held names even if apply() trims the batch, so apply() MUST count current
    sector exposure — accumulating only the batch's decisions from 0 let new
    buys breach the cap on top of retained holds (the real bug). Here a NEW name
    (not currently held) is bought into a sector already at 8% under a 10% cap:
    0.08 + 0.05 = 0.13 > 0.10 → rejected."""
    ctx = _ctx(
        rails=RiskRails(stop_loss_pct=0.0, max_sector_pct=0.10),
        sector_exposure_pct={"Tech": 0.08},   # retained Tech holdings
        current_positions_dollars={},          # AAPL is a NEW name, not held
    )
    out = apply([_decision("AAPL", weight=0.05)], ctx)
    assert out == []


def test_pipeline_sector_cap_rebalanced_name_replaces_own_weight():
    """No double-count: a HELD name rebalancing up counts its TARGET, not
    current+target. AAPL held at 8% (the only Tech holding) rebalances to 9%
    under a 10% cap → accepted (0.08 - 0.08 + 0.09 = 0.09 <= 0.10), not the
    old double-counted 0.08 + 0.09 = 0.17."""
    ctx = _ctx(
        rails=RiskRails(stop_loss_pct=0.0, max_sector_pct=0.10, max_position_pct=0.10),
        sector_exposure_pct={"Tech": 0.08},
        current_positions_dollars={"AAPL": 8_000.0},  # 8% of 100k
    )
    out = apply([_decision("AAPL", weight=0.09)], ctx)
    assert [d.ticker for d in out] == ["AAPL"]


def test_pipeline_drops_decision_within_earnings_blackout():
    ctx = _ctx(upcoming_earnings={"AAPL": [date(2026, 1, 7)]})  # 2 days from asof
    out = apply([_decision("AAPL")], ctx)
    assert out == []


def test_pipeline_drops_decision_exceeding_position_cap():
    ctx = _ctx(rails=RiskRails(stop_loss_pct=0.0, max_position_pct=0.05))
    out = apply([_decision("AAPL", weight=0.10)], ctx)  # exceeds 5% cap
    assert out == []


def test_pipeline_blocks_all_buys_when_drawdown_exceeded():
    ctx = _ctx(
        rails=RiskRails(stop_loss_pct=0.0, max_drawdown_pct=0.15),
        current_drawdown=0.20,  # 20% > 15% cap
    )
    out = apply([_decision("AAPL"), _decision("MSFT")], ctx)
    assert out == []


def test_pipeline_accumulates_sector_exposure_across_batch():
    """Decision sector exposure accumulates across the batch from 0. Three 5%
    Tech targets under a 10% cap: first two accepted (0.10 == cap), third
    rejected (0.15 > 0.10)."""
    ctx = _ctx(rails=RiskRails(stop_loss_pct=0.0, max_sector_pct=0.10))
    out = apply(
        [_decision("A", 0.05), _decision("B", 0.05), _decision("C", 0.05)], ctx
    )
    assert [d.ticker for d in out] == ["A", "B"]


def test_pipeline_accepts_when_sector_below_cap_after_decision():
    ctx = _ctx(rails=RiskRails(stop_loss_pct=0.0, max_sector_pct=0.25))
    out = apply([_decision("AAPL", weight=0.05)], ctx)  # 5% < 25% cap
    assert len(out) == 1


# --- 2026-06-05 audit: rails must gate exposure-INCREASING decisions only ---


def test_drawdown_allows_reductions_but_blocks_new_buys():
    """A drawdown must still let the book DERISK. A held name being trimmed
    (target < current) passes even in drawdown; a new buy is blocked."""
    ctx = _ctx(
        rails=RiskRails(stop_loss_pct=0.0, max_drawdown_pct=0.15),
        current_drawdown=0.20,
        current_positions_dollars={"AAPL": 10_000.0},  # 10%
        sector_exposure_pct={"Tech": 0.10},
    )
    out = apply([_decision("AAPL", 0.05), _decision("MSFT", 0.05)], ctx)
    assert [d.ticker for d in out] == ["AAPL"]  # trim passes; new buy blocked


def test_earnings_blackout_allows_reductions():
    """Earnings blackout blocks new buys/adds, not a reduction of a held name."""
    ctx = _ctx(
        current_positions_dollars={"AAPL": 10_000.0},
        sector_exposure_pct={"Tech": 0.10},
        upcoming_earnings={"AAPL": [date(2026, 1, 7)]},
    )
    out = apply([_decision("AAPL", 0.05)], ctx)  # trim of a held name in blackout
    assert [d.ticker for d in out] == ["AAPL"]


def test_sector_cap_dropped_holding_frees_room():
    """A held name NOT in the decision set is force-sold by translate(), so it
    must NOT count against the sector cap (a real cause of the idle-cash drag)."""
    ctx = _ctx(
        rails=RiskRails(stop_loss_pct=0.0, max_sector_pct=0.10),
        sector_exposure_pct={"Tech": 0.08},
        current_positions_dollars={"OLD": 8_000.0},  # dropped (not in decisions)
    )
    out = apply([_decision("AAPL", 0.05)], ctx)  # new Tech buy
    # OLD will be force-sold → frees the 8% → AAPL's 5% fits under the 10% cap.
    assert [d.ticker for d in out] == ["AAPL"]


def test_sector_cap_processes_reductions_before_buys():
    """A same-sector trim frees cap room BEFORE same-batch buys are evaluated."""
    ctx = _ctx(
        rails=RiskRails(stop_loss_pct=0.0, max_sector_pct=0.10),
        sector_exposure_pct={"Tech": 0.08},
        current_positions_dollars={"AAA": 8_000.0},  # held, kept (in decisions)
    )
    # AAA trims 8%→2% (frees 6%); BBB is a new 5% buy. Both fit once AAA trims.
    out = apply([_decision("BBB", 0.05), _decision("AAA", 0.02)], ctx)
    assert set(d.ticker for d in out) == {"AAA", "BBB"}


def test_position_cap_coalesces_duplicate_tickers():
    """Duplicate decisions for one ticker must be coalesced so the per-ticker
    position cap can't be bypassed by two sub-cap decisions."""
    ctx = _ctx(rails=RiskRails(stop_loss_pct=0.0, max_position_pct=0.05))
    out = apply([_decision("AAPL", 0.04), _decision("AAPL", 0.04)], ctx)
    assert [d.ticker for d in out] == ["AAPL"]  # one AAPL, not two


def test_sector_cap_does_not_block_a_reduction_above_cap():
    """A trim that LOWERS sector exposure but stays above the cap must still be
    accepted — rejecting it would freeze the oversized position (Codex review of
    the 2026-06-05 refactor). The caps gate increases, never reductions."""
    ctx = _ctx(
        rails=RiskRails(stop_loss_pct=0.0, max_sector_pct=0.30, max_position_pct=1.0),
        sector_exposure_pct={"Tech": 0.40},
        current_positions_dollars={"AAA": 40_000.0},  # 40%, above the 30% cap
    )
    out = apply([_decision("AAA", 0.35)], ctx)  # trim 40%->35%, still > 30%
    assert [d.ticker for d in out] == ["AAA"]


def test_position_cap_does_not_block_a_reduction():
    """Trimming an oversized holding toward (but not under) the position cap must
    be allowed — it derisks."""
    ctx = _ctx(
        rails=RiskRails(stop_loss_pct=0.0, max_position_pct=0.10, max_sector_pct=1.0),
        sector_exposure_pct={"Tech": 0.20},
        current_positions_dollars={"AAA": 20_000.0},  # 20%, above the 10% cap
    )
    out = apply([_decision("AAA", 0.15)], ctx)  # trim 20%->15%, still > 10%
    assert [d.ticker for d in out] == ["AAA"]


def test_sector_room_not_freed_for_min_hold_protected_positions():
    """Codex module review (2026-06-11 HIGH): apply() freed sector room for
    every held-but-dropped name assuming translate() would force-sell it —
    but translate's min_hold rail can VETO that exit. The freed room was
    fiction: a same-sector buy then pushed REAL exposure past the cap.
    Min-hold-protected holdings must keep their sector room reserved."""
    from datetime import date as _date

    from sma.backtest.strategies.base import StrategyDecision
    from sma.risk.pipeline import RiskContext, apply
    from sma.risk.rails import RiskRails

    rails = RiskRails(stop_loss_pct=0.0, max_sector_pct=0.30, max_position_pct=0.20)
    ctx_kwargs = dict(
        rails=rails,
        account_value=100_000.0,
        cash=50_000.0,
        # AAPL: 25% of account, tech. Fresh (inside min_hold) and DROPPED by
        # the model this cycle.
        current_positions_dollars={"AAPL": 25_000.0},
        sector_exposure_pct={"tech": 0.25},
        sector_for=lambda t: "tech",
        current_drawdown=0.0,
        upcoming_earnings={},
        asof_date=_date(2026, 4, 30),
    )
    # NVDA buy at 15%: only fits under the 30% tech cap if AAPL's 25% is
    # (wrongly) credited as leaving.
    decisions = [
        StrategyDecision(asof_date=_date(2026, 4, 30), ticker="NVDA", target_weight=0.15)
    ]

    # WITHOUT protection (legacy): room freed → buy passes (documents the hole)
    accepted = apply(decisions, RiskContext(**ctx_kwargs))
    assert [d.ticker for d in accepted] == ["NVDA"]

    # WITH protection: AAPL's exit is min-hold-vetoed → room stays reserved →
    # 25% + 15% > 30% cap → buy rejected
    accepted = apply(
        decisions,
        RiskContext(**ctx_kwargs, min_hold_protected=frozenset({"AAPL"})),
    )
    assert accepted == []
