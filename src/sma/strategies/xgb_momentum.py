"""The incumbent as a sleeve: XGBoost 30-day cross-sectional momentum, top-K.

No decision logic lives here. `build_incumbent_strategy` is the one builder
for XGBoostTopKStrategy on the live path (sma.live.__main__._build_strategy
delegates to it), and `propose` runs it through
sma.strategies.base.call_strategy_decide, the same call decide_once makes. At
capital_fraction 1.0 the sleeve path therefore hands the rails the exact same
decisions, in the same order, as the old direct path (golden test:
tests/integration/live/test_sleeve_golden.py).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from sma.strategies.base import (
    SleeveContext,
    Strategy,
    TargetBook,
    book_from_decisions,
    call_strategy_decide,
)
from sma.strategies.registry import SleeveBuild, register


def build_incumbent_strategy(
    universe: list[str], *, use_theses: bool, db: str, store, settings=None, predictor=None,
):
    from sma.backtest.strategies.xgb_top_k import XGBoostTopKStrategy
    from sma.model.predictor import Predictor

    # 2026-09-01 (sma.live.replay): callers that already know the scores for
    # asof_date (replay reads them from the `predictions` table instead of
    # running the model) may pass a `predictor` duck-typing
    # Predictor.predict_for(asof_date, universe) -> dict[ticker, score]. This
    # is the ONLY thing replay swaps out -- k/hold_rank/sector_neutralize/
    # min_score/use_theses below stay identical to the real decide job, so
    # the two callers can never drift on strategy configuration.
    if predictor is None:
        # Pass the live store's connection to the predictor so it doesn't open a
        # second (conflicting) connection on the same DB file.
        predictor = Predictor(
            models_dir=Path("models_artifacts"),
            db_path=Path(db),
            conn=store.conn,
        )
    # Theses lookups also reuse the live store conn (same single-conn rule).
    # 2026-05-29: k=15 (was strategy default k=10). After bumping sector cap
    # 0.25 -> 0.35 on 5/26, IT exposure stabilized at 29.4% (~5.6% cap
    # headroom) but the top-10 had 8 IT names -- every available IT pick
    # (PANW/ARM/ORCL/AVGO/DDOG) was sector-cap-blocked. Non-IT picks that
    # would diversify cleanly (BBY/FDX/DKNG, all positive predictions) were
    # at ranks 11-13, outside the K=10 ceiling. Bumping to K=15 brings
    # them into the buy pool without changing per-name risk (weight_tilt
    # still active; top names get target_weight=10%, bottom of K=15 gets
    # floor=4%). Expected effect: deployment 43% -> ~75%.
    # Strategy params from config (2026-06-12 ship: sector_neutralize=1.0 +
    # hold_rank=30 + rails.min_hold_days=7 -- corrected-eval val sharpe +1.60
    # vs -1.58 baseline; plateau-checked in scripts/ab_wave4_robustness.py).
    strat_cfg = getattr(getattr(settings, "live", None), "strategy", None)
    k = getattr(strat_cfg, "k", 15) if strat_cfg else 15
    return XGBoostTopKStrategy(
        predictor=predictor,
        universe=universe,
        k=k,
        hold_rank=getattr(strat_cfg, "hold_rank", None) if strat_cfg else None,
        sector_neutralize=(
            getattr(strat_cfg, "sector_neutralize", 0.0) if strat_cfg else 0.0
        ),
        # Entry conviction floor (default None = off). New buys must clear the bar.
        min_score=getattr(strat_cfg, "min_score", None) if strat_cfg else None,
        use_theses=use_theses,
        store=store if use_theses else None,
    )


@register
class XgbMomentumSleeve(Strategy):
    name = "xgb_momentum"
    session = "open"
    # The model's target is the 30-day forward return; the book is rebalanced
    # nightly with rank hysteresis and a min-hold rail.
    horizon_days = 30
    # Raw incumbent book grosses ~1.24 (15 names, 10% top weight tapering to
    # ~4%); the rails clip it. See Strategy.max_gross.
    max_gross = None

    def __init__(self, incumbent):
        self.incumbent = incumbent

    @classmethod
    def from_build(cls, build: SleeveBuild) -> XgbMomentumSleeve:
        factory = build.incumbent_factory or build_incumbent_strategy
        return cls(
            factory(
                build.universe,
                use_theses=build.use_theses,
                db=build.db,
                store=build.store,
                settings=build.settings,
                predictor=build.predictor,
            )
        )

    def propose(self, asof: date, ctx: SleeveContext) -> TargetBook:
        decisions = call_strategy_decide(
            self.incumbent,
            asof=asof,
            prices=ctx.prices,
            current_holdings=set(ctx.current_holdings),
        )
        return book_from_decisions(decisions)
