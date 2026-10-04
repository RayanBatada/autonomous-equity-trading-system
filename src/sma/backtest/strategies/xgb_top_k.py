"""XGBoostTopK strategy: long the top K tickers by predicted forward return.

Phase 4 Task 11: optional ``use_theses=True`` layers three asymmetric C-rules
on top of the Phase 2/3 quant baseline using LLM-generated theses:

1. **Bearish veto.** Drop a buy candidate when the most recent thesis
   conviction is ``bearish`` or ``strong_bearish`` (``_apply_thesis_buy_filter``).
2. **Bullish tilt.** Multiply the quant score by 1.05 when conviction is
   ``bullish`` or ``strong_bullish`` (``_tilt_score``).
3. **Strong-bearish exit trigger.** Force-sell a currently-held ticker when
   conviction is ``strong_bearish`` (``_held_exits_on_thesis``).

An outside-top-30 catalyst override was specced as a fourth rule but never
wired into ``decide``; its helper and the ``OVERRIDE_CATALYST_FLAGS`` set were
removed in 2026-08 along with ``_thesis_exit_decisions``, a duplicate of
``_held_exits_on_thesis`` that only tests ever called.

A thesis is considered fresh when its ``asof_date`` falls within the last
``THESIS_STALE_DAYS`` (7) calendar days. Stale or missing theses are
treated as neutral and cannot fire any of the rules.

When ``use_theses=False`` the helpers are identity-no-ops and the strategy
preserves the Phase 3 baseline contract exactly.
"""

import json
import math
from collections import defaultdict
from collections.abc import Callable
from datetime import date

import pandas as pd
from loguru import logger

from sma.backtest.strategies.base import StrategyDecision
from sma.ingest.store import Store
from sma.model.predictor import Predictor
from sma.sectors import sector_for


def sector_demean_scores(
    scores: dict[str, float],
    lam: float,
    sector_of: Callable[[str], str],
) -> dict[str, float]:
    """Demean each score by its sector mean: ``score - lam * mean(sector)``.

    Makes the top-K rank on within-sector relative strength so selection spans
    sectors instead of concentrating in whatever sector the model is broadly
    bullish on. ``lam`` (0..1) is the neutralization strength: 0 is an exact
    identity, 1 fully removes each sector's average level.

    Sectors with fewer than two scored members are left UNCHANGED — demeaning a
    lone name gives ``(1-lam)*score`` and would collapse a strong solo name
    toward 0. ``sector_of`` maps ticker -> sector label (e.g. sma.sectors.sector_for).
    """
    if lam <= 0.0:
        return dict(scores)
    by_sector: dict[str, list[str]] = defaultdict(list)
    for ticker in scores:
        by_sector[sector_of(ticker)].append(ticker)
    out = dict(scores)
    for members in by_sector.values():
        # Demean over FINITE members only: a NaN/inf score must not contaminate
        # the whole sector's mean. Non-finite scores pass through unchanged (they
        # sort the same as they would with neutralization off).
        finite = [t for t in members if math.isfinite(scores[t])]
        if len(finite) < 2:
            continue
        mean = sum(scores[t] for t in finite) / len(finite)
        for t in finite:
            out[t] = scores[t] - lam * mean
    return out


class XGBoostTopKStrategy:
    """Pick top K tickers by predicted return; equal-weight at target_weight each.

    With the default 5% per position and K=20, the strategy targets 100% of
    the account deployed (subject to the simulator's 5% position cap rail).

    Tickers without a prediction (insufficient feature history) are skipped.
    If fewer than K tickers have predictions, the strategy returns one
    decision per available ticker.
    """

    name = "xgb_top_k"

    # --- Phase 4 thesis-integration constants ----------------------------
    THESIS_STALE_DAYS = 7
    BULLISH_TILT_MULT = 1.05
    BULLISH_CONVICTIONS = {"bullish", "strong_bullish"}
    BEARISH_CONVICTIONS = {"bearish", "strong_bearish"}

    def __init__(
        self,
        predictor: Predictor,
        universe: list[str],
        k: int = 10,
        hold_rank: int | None = None,
        target_weight_per_position: float = 0.10,
        weight_tilt: bool = True,
        weight_tilt_floor_ratio: float = 0.4,
        sector_neutralize: float = 0.0,
        min_score: float | None = None,
        use_theses: bool = False,
        use_politician_flow: bool = True,
        politician_flow_lookback_days: int = 30,
        politician_flow_max_tilt: float = 0.20,
        politician_flow_normalizer: float = 50_000.0,
        store: Store | None = None,
        tilt_strict: bool = False,
    ):
        """Top-K XGBoost strategy.

        v1 manual tilt (2026-05-09): k defaults to 10 (was 20) and
        target_weight_per_position to 0.10 (was 0.05) — same total deployable
        capital but in fewer, higher-conviction names. weight_tilt=True scales
        each pick's target_weight from `weight_tilt_floor_ratio * max` (lowest
        of top-K) up to `max` (top of top-K), based on score rank within the
        selected slice. Risk pipeline (sector_cap, position_cap) bounds the
        worst case downstream.
        """
        if k <= 0:
            raise ValueError(f"k must be positive; got {k}")
        if not (0.0 < target_weight_per_position <= 1.0):
            raise ValueError(
                f"target_weight_per_position must be in (0, 1]; got {target_weight_per_position}"
            )
        if not (0.0 < weight_tilt_floor_ratio <= 1.0):
            raise ValueError(
                f"weight_tilt_floor_ratio must be in (0, 1]; got {weight_tilt_floor_ratio}"
            )
        if not (0.0 <= sector_neutralize <= 1.0):
            raise ValueError(
                f"sector_neutralize must be in [0, 1]; got {sector_neutralize}"
            )
        self.predictor = predictor
        self.universe = list(universe)
        self.k = k
        # Rank hysteresis buffer. Default None => k (DISABLED / plain top-K). Set
        # hold_rank > k to keep a held name while ranked in the [k, hold_rank)
        # band instead of churning it. OPT-IN: a 19-day backtest was inconclusive
        # (turnover -17% but return worse in-sample; the cost benefit is a
        # long-horizon effect a short window can't show), so it's off by default
        # until a longer honest window can validate it. Only active when the
        # caller passes current_holdings.
        self.hold_rank = hold_rank if hold_rank is not None else k
        self.target_weight = target_weight_per_position
        self.weight_tilt = weight_tilt
        self.weight_tilt_floor_ratio = weight_tilt_floor_ratio
        # Sector-relative scoring strength (0 = off / plain top-K). OPT-IN /
        # default-OFF until a backtest on an honest window clears the ship bar.
        self.sector_neutralize = sector_neutralize
        # Entry conviction floor on the RAW model score (predicted 30d return,
        # before tilts/demean). None = off. A candidate is only eligible as a
        # NEW buy when its raw score >= min_score, so weak-signal days hold cash
        # instead of filling K slots with mediocre names. Held names keep their
        # rank-hysteresis eligibility regardless (this is an ENTRY gate, not an
        # exit). OPT-IN / default-OFF until a backtest clears the ship bar.
        self.min_score = min_score
        self.use_theses = use_theses
        self.use_politician_flow = use_politician_flow
        self.politician_flow_lookback_days = politician_flow_lookback_days
        self.politician_flow_max_tilt = politician_flow_max_tilt
        self.politician_flow_normalizer = politician_flow_normalizer
        self.store = store
        # Live default (False): a buggy tilt() must not crash live decisions —
        # swallow and fall back to untilted. Autoresearch eval sets this True so
        # a crashing proposal surfaces as eval_error instead of masquerading as
        # the untilted baseline score.
        self.tilt_strict = tilt_strict

    # ------------------------------------------------------------------
    # Phase 4 thesis helpers (early-return when use_theses=False)
    # ------------------------------------------------------------------
    def _recent_thesis(self, ticker: str, asof_date: date) -> dict | None:
        """Fetch the most recent thesis row for ``ticker`` within the
        ``THESIS_STALE_DAYS`` window of ``asof_date``.

        Returns a dict with keys ``conviction``, ``score``, ``flags``,
        ``action_hint``, ``reasoning`` -- or None when use_theses is off,
        no store is wired, or no fresh thesis exists.
        """
        if not self.use_theses or self.store is None or self.store.conn is None:
            return None
        # DuckDB does not accept positional params inside INTERVAL, so
        # the staleness window literal is interpolated from the class
        # constant. The value is hard-coded class state, not user input.
        sql = f"""
            SELECT conviction, score, flags, action_hint, reasoning
            FROM theses
            WHERE ticker = ?
              AND asof_date <= ?
              AND asof_date >= ? - INTERVAL '{self.THESIS_STALE_DAYS} days'
            ORDER BY asof_date DESC, run_id DESC
            LIMIT 1
        """
        row = self.store.conn.execute(
            sql, [ticker, asof_date, asof_date]
        ).fetchone()
        if row is None:
            return None
        flags = row[2]
        if isinstance(flags, str):
            try:
                flags = json.loads(flags)
            except (json.JSONDecodeError, TypeError):
                flags = []
        elif flags is None:
            flags = []
        return {
            "conviction": row[0],
            "score": row[1] if row[1] is not None else 0.0,
            "flags": flags,
            "action_hint": row[3],
            "reasoning": row[4],
        }

    def _apply_thesis_buy_filter(
        self, candidates: list[str], asof_date: date
    ) -> list[str]:
        """Drop bearish-thesis candidates from the buy pool. Identity when use_theses is off."""
        if not self.use_theses:
            return list(candidates)
        out: list[str] = []
        for t in candidates:
            thesis = self._recent_thesis(t, asof_date)
            if thesis is None or thesis["conviction"] not in self.BEARISH_CONVICTIONS:
                out.append(t)
            else:
                # 2026-08 study: thesis vetoes were unlogged, costing hours to
                # reconstruct efficacy after the fact. The vetoed slot is NOT
                # backfilled from rank K+1 (verified 2026-08-06: `top` is
                # fixed by top-K selection above before this filter runs), so
                # the trade count for the day simply shrinks by one.
                logger.info(
                    "thesis_veto: {} vetoed on {} thesis, buy slot left unfilled (no backfill)",
                    t,
                    thesis["conviction"],
                )
        return out

    def _held_exits_on_thesis(self, ticker: str, asof_date: date) -> bool:
        """True if a HELD ticker should EXIT on its thesis. Only ``strong_bearish``
        triggers an exit — a merely ``bearish`` held name is retained, not churned
        out (the buy veto is for new buys only)."""
        if not self.use_theses:
            return False
        thesis = self._recent_thesis(ticker, asof_date)
        exits = thesis is not None and thesis["conviction"] == "strong_bearish"
        if exits:
            # The live-fired exit path, called inline from decide(). The 2026-08
            # study found strategy-level thesis interventions never appeared in
            # live.decide logs because they were logged on the unused duplicate
            # (_thesis_exit_decisions, since deleted) instead of this one.
            logger.info(
                "thesis_exit: {} exited on {} thesis", ticker, thesis["conviction"]
            )
        return exits

    def _tilt_score(
        self, ticker: str, base_score: float, asof_date: date
    ) -> float:
        """Multiply ``base_score`` by ``BULLISH_TILT_MULT`` when conviction is (strong_)bullish."""
        if not self.use_theses:
            return base_score
        thesis = self._recent_thesis(ticker, asof_date)
        if thesis is None:
            return base_score
        if thesis["conviction"] in self.BULLISH_CONVICTIONS:
            # Sign-safe: a bullish thesis must move the score UP. Plain
            # multiplication made NEGATIVE scores MORE negative (demoting
            # bullish names) — audit xgb_top_k:242.
            return base_score + (self.BULLISH_TILT_MULT - 1.0) * abs(base_score)
        return base_score

    # ------------------------------------------------------------------
    # Politician-flow overlay (2026-05-10)
    # ------------------------------------------------------------------
    def _fetch_politician_flow(self, asof_date: date) -> dict[str, float]:
        """Net politician dollar flow per ticker over the lookback window.

        Uses Politician trade disclosures from `politician_trades` (House
        Clerk PTR feed). Returns ticker → net dollar value, where buys add
        (amount_min + amount_max) / 2 and sells subtract. Empty dict when
        the table doesn't exist (older DBs without the v5 migration) or
        the source is intentionally disabled.
        """
        if self.store is None or self.store.conn is None:
            return {}
        from datetime import timedelta
        start = asof_date - timedelta(days=self.politician_flow_lookback_days)
        try:
            rows = self.store.conn.execute(
                """
                SELECT ticker, SUM(
                    CASE WHEN transaction_type = 'P' THEN (amount_min + amount_max) / 2.0
                         WHEN transaction_type LIKE 'S%' THEN -1 * (amount_min + amount_max) / 2.0
                         ELSE 0 END
                )
                FROM politician_trades
                WHERE ticker IS NOT NULL
                  -- Window on filing_date (public DISCLOSURE date), not
                  -- transaction_date: trades disclose ~58d after execution,
                  -- so a transaction_date window leaks non-public info into
                  -- backtests. Exclude options/derivatives. MUST match
                  -- Predictor._fetch_politician_flows exactly (Codex
                  -- module review 2026-06-11 HIGH).
                  AND COALESCE(asset_type, '') NOT IN ('Stock Option', 'OP')
                  AND filing_date >= ?
                  AND filing_date <= ?
                GROUP BY ticker
                """,
                [start, asof_date],
            ).fetchall()
        except Exception as e:
            logger.warning("politician flow lookup failed (db missing v5 migration?): {}", e)
            return {}
        return {ticker: float(net or 0.0) for ticker, net in rows}

    # ------------------------------------------------------------------
    # Decide
    # ------------------------------------------------------------------
    def decide(
        self,
        asof_date: date,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        current_holdings: set[str] | None = None,
    ) -> list[StrategyDecision]:
        try:
            scores = self.predictor.predict_for(asof_date, self.universe)
        except FileNotFoundError:
            # No model available for this date yet (e.g., backtest before
            # any model was trained). Return no decisions; the simulator
            # holds cash that day.
            return []

        if not scores:
            return []

        # Phase 4 (rule 2): apply bullish tilt to the quant scores before
        # ranking. When use_theses is False this is identity per ticker.
        tilted_scores = {
            ticker: self._tilt_score(ticker, score, asof_date)
            for ticker, score in scores.items()
        }

        # 2026-05-10: politician-flow overlay. Multiply each score by
        # (1 + clamped_flow_signal) where signal is normalized net dollar
        # flow over `politician_flow_lookback_days`, clamped to
        # ±politician_flow_max_tilt. Direction is the only meaningful signal
        # — politicians distributing a name (sell flow > buy flow) → score
        # down, accumulating → score up. The magnitude of the tilt is small
        # by design (default ±20% on the score) so it complements rather
        # than overrides the quant model.
        if self.use_politician_flow and self.store is not None:
            flows = self._fetch_politician_flow(asof_date)
            for ticker in tilted_scores:
                flow = flows.get(ticker, 0.0)
                if flow == 0.0:
                    continue
                # Normalize and clamp.
                signal = flow / self.politician_flow_normalizer
                signal = max(-self.politician_flow_max_tilt,
                             min(self.politician_flow_max_tilt, signal))
                # Sign-safe (same bug class as the bullish tilt): buy-flow must
                # move the score UP even when the score is negative.
                tilted_scores[ticker] += signal * abs(tilted_scores[ticker])

        # Sector-relative scoring (opt-in): demean each score by its sector mean
        # so the top-K ranks on within-sector relative strength and spans sectors
        # rather than concentrating in one theme. Applied after the thesis +
        # politician overlays, before ranking; selection is purely rank-based so
        # the resulting (possibly negative) scores are safe. lam=0 is a no-op.
        if self.sector_neutralize > 0.0:
            tilted_scores = sector_demean_scores(
                tilted_scores, self.sector_neutralize, sector_for
            )

        # Sort tickers by (possibly tilted) predicted return descending.
        ranked = sorted(tilted_scores.items(), key=lambda kv: kv[1], reverse=True)

        # Rank hysteresis (turnover control): KEEP a currently-held name while it
        # stays ranked above hold_rank (a buffer beyond k) rather than churning it
        # the moment it drops below k, then fill the remaining k slots with the
        # best NEW names. A daily re-rank on a 30-day signal otherwise bleeds
        # round-trip costs. With current_holdings=None (e.g. the autoresearch eval
        # path), this reduces to plain top-k — no behavior change.
        rank_of = {t: i for i, (t, _) in enumerate(ranked)}
        held = current_holdings or set()
        keep = [t for t, _ in ranked if t in held and rank_of[t] < self.hold_rank][: self.k]
        n_fill = self.k - len(keep)
        new_buy_pool = [t for t, _ in ranked if t not in held and rank_of[t] < self.k]
        # Entry conviction floor (opt-in): drop new-buy candidates whose RAW
        # model score is below the bar so a weak-signal day fills fewer slots
        # (holds cash) rather than buying mediocre names. Uses the untilted
        # `scores` so the threshold is a plain predicted-return level regardless
        # of sector-demean scaling. None = no-op. Held names are unaffected.
        if self.min_score is not None:
            # Require a FINITE score at/above the floor. A bare `>= min_score`
            # rejects NaN by accident (NaN comparisons are always False) but lets
            # a +inf score through — a degenerate model output that must never be
            # bought. isfinite() rejects both NaN and ±inf explicitly.
            new_buy_pool = [
                t for t in new_buy_pool
                if math.isfinite(scores.get(t, float("-inf")))
                and scores.get(t, float("-inf")) >= self.min_score
            ]
        new_buys = new_buy_pool[:n_fill]
        selected = set(keep) | set(new_buys)
        top = [(t, s) for t, s in ranked if t in selected]

        # Phase 4 (rule 1): bearish-thesis handling differs for NEW BUYS vs HOLDS.
        # A new buy is vetoed on (strong_)bearish — don't buy weakness. A HELD name
        # is only dropped (force-sold) on STRONG_bearish; a merely-bearish held name
        # is RETAINED, not churned out (2026-06-05 audit). When use_theses is False
        # both helpers are identity, so this is plain top-K.
        kept_buys = set(
            self._apply_thesis_buy_filter([t for t, _ in top if t not in held], asof_date)
        )
        kept_top = [
            (t, s)
            for t, s in top
            if (t in held and not self._held_exits_on_thesis(t, asof_date))
            or (t not in held and t in kept_buys)
        ]
        if not kept_top:
            return []

        # v1 manual tilt: weight by rank within the selected top-K. Top pick
        # gets `target_weight`; bottom of the top-K gets `target_weight *
        # weight_tilt_floor_ratio`. Linear interpolation in between. When
        # weight_tilt is False, every pick gets `target_weight` (legacy).
        n = len(kept_top)
        floor = self.target_weight * self.weight_tilt_floor_ratio

        def _weight_for_rank(rank: int) -> float:
            if not self.weight_tilt or n <= 1:
                return self.target_weight
            # rank=0 → target_weight (top pick); rank=n-1 → floor.
            t = rank / (n - 1)
            return self.target_weight - t * (self.target_weight - floor)

        raw_weights = [_weight_for_rank(rank) for rank in range(len(kept_top))]
        # Gross preservation (audit xgb_top_k:415): the linear tilt averages
        # ~(1+floor)/2 of target, deploying only ~70% gross on a full top-K —
        # a live cash-drag. Renormalize so sum == n*target while keeping the
        # rank shape; a single name may exceed target_weight, which the risk
        # rails clip at max_position_pct (sim check_order enforces the same).
        raw_sum = sum(raw_weights)
        if self.weight_tilt and raw_sum > 0:
            scale = (n * self.target_weight) / raw_sum
            # Clamp at target_weight: the risk pipeline REJECTS (not clips)
            # any decision above max_position_pct, and target_weight is the
            # strategy's contract with that cap. Single-pass renorm+clamp
            # keeps the rank shape and recovers ~91% gross (vs ~70% before)
            # without ever producing a rails-rejected weight.
            raw_weights = [min(w * scale, self.target_weight) for w in raw_weights]
        decisions = [
            StrategyDecision(
                asof_date=asof_date,
                ticker=ticker,
                target_weight=raw_weights[rank],
            )
            for rank, (ticker, _score) in enumerate(kept_top)
        ]

        # Phase 6: post-process via the agent-editable tilt() function.
        # The autoresearch loop rewrites the body of `sma.strategy.active.tilt`;
        # default body is identity, so this is a no-op until the loop produces
        # an experiment we want to test. TiltContext is built from scores +
        # whatever ancillary state we already have. The risk pipeline runs
        # AFTER this, so a misbehaving tilt body is bounded by sector_cap,
        # position_cap, and cash_floor.
        try:
            from sma.strategy.active import TiltContext, tilt
            ctx = TiltContext(
                quant_scores={t: float(s) for t, s in scores.items()},
                theses=None,            # populated by future autoresearch experiments
                portfolio_dollars={},   # decide_once owns positions; not threaded here yet
                sector_exposure={},
                # Real sector mapping (was a lambda returning 'Unknown',
                # blinding every autoresearch tilt() proposal to sectors —
                # strategy review 2026-06-11).
                sector_for=sector_for,
                account_equity=0.0,
                cash=0.0,
            )
            decisions = tilt(asof_date=asof_date, decisions=decisions, ctx=ctx)
        except Exception as e:
            # Autoresearch eval (tilt_strict=True): surface the failure so the
            # experiment is recorded as eval_error rather than silently scored
            # as the untilted baseline.
            if self.tilt_strict:
                raise
            # Live default: a buggy tilt body must NOT take down the decision
            # loop. Log and fall through with the un-tilted decisions.
            logger.warning("tilt() raised; using untilted decisions: {}", e)

        return decisions
