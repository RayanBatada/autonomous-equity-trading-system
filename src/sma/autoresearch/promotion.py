"""Pure deploy-gate decision for an autoresearch-produced model.

Mirrors the retrain's gate sequence (src/sma/model/__main__.py) so a model from
config search is held to the SAME bar as a weekly retrain:
  1. IC floor  — held-out rank-IC below the noise band fails CLOSED.
  2. label-type transition — raw vs demean: RMSE incomparable, promote on IC.
  3. train-window transition — different training distribution: RMSE
     incomparable, promote on the IC floor (already passed at step 1).
  4. RMSE ratio — otherwise, promote unless materially worse OOS.

Kept pure (no I/O) so it is trivially testable and so the autoresearch path
cannot drift from the retrain's decision. A follow-up will refactor the retrain
to call this same function.
"""
from __future__ import annotations

from dataclasses import dataclass

from sma.model.persistence import passes_ic_floor, should_promote

# Minimum CV-IC lift a config-search winner must show over the incumbent to
# auto-promote. ~1 SE of a walk-forward CV-IC estimate (see the noise-band
# reasoning at persistence.GATE_MIN_CV_IC), so a candidate must rank MEASURABLY
# better OOS, not just numerically higher by noise — otherwise we'd churn the
# live model every run on coin-flips.
IC_PROMOTE_MARGIN = 0.005


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str


def evaluate_promotion(
    *,
    new_cv_ic: float | None,
    new_cv_rmse: float | None,
    incumbent_cv_rmse: float | None,
    inc_label_type: str | None,
    new_label_type: str,
    inc_train_start: str | None,
    new_train_start: str,
    cv_ran: bool = True,
) -> PromotionDecision:
    """Decide whether the new model replaces the incumbent. Same order as the
    retrain gate so the two paths cannot diverge."""
    if not passes_ic_floor(new_cv_ic, cv_ran=cv_ran):
        return PromotionDecision(
            False, "CV rank-IC below floor (ranks worse than chance OOS)"
        )
    if inc_label_type is not None and inc_label_type != new_label_type:
        return PromotionDecision(
            True, f"label_type transition {inc_label_type} -> {new_label_type}"
        )
    if incumbent_cv_rmse is not None and inc_train_start != new_train_start:
        return PromotionDecision(
            True,
            f"train_start change {inc_train_start} -> {new_train_start} "
            "(promote on IC floor; RMSE incomparable across training windows)",
        )
    if should_promote(new_cv_rmse, incumbent_cv_rmse):
        return PromotionDecision(True, "CV RMSE within ratio of incumbent")
    return PromotionDecision(False, "CV RMSE materially worse than incumbent")


def evaluate_search_promotion(
    *,
    new_cv_ic: float | None,
    incumbent_cv_ic: float | None,
    inc_label_type: str | None,
    new_label_type: str,
    inc_train_start: str | None,
    new_train_start: str,
    cv_ran: bool = True,
    margin: float = IC_PROMOTE_MARGIN,
) -> PromotionDecision:
    """Promotion gate for a config-search winner. Unlike the retrain (which
    compares RMSE in the no-transition case), this compares CV-IC directly,
    because the search SELECTS by IC — gating on RMSE would re-introduce the
    ranking-blindness the IC work removed.

    Sequence:
      1. IC floor — below the noise band fails CLOSED (never ship below-chance).
      2. No incumbent CV-IC to compare — passing the floor is enough.
      3. Label/window mismatch — the search trains apples-to-apples with the
         incumbent; if those differ, IC is NOT comparable across distributions,
         so DEFER (don't auto-deploy), even at a higher raw IC. The search does
         not vary label/window in Phase 1, so this is a conservative guard, not
         the deliberate-transition promote the retrain does.
      4. Same surface — require a real IC lift (>= margin) over the incumbent.
    """
    if not passes_ic_floor(new_cv_ic, cv_ran=cv_ran):
        return PromotionDecision(
            False, "CV rank-IC below floor (ranks worse than chance OOS)"
        )
    # `inc_label_type` is None ONLY when no model is deployed — incumbent_label_type
    # reads legacy artifacts as "raw", never None — so it, NOT incumbent_cv_ic
    # (which is also None for a legacy/NaN-IC incumbent), is the "is there an
    # incumbent" signal. Keying existence off incumbent_cv_ic let a
    # mismatched-surface incumbent be bypassed (Codex review 2026-06-17).
    if inc_label_type is None:
        return PromotionDecision(True, "no incumbent model; passes floor")
    if inc_label_type != new_label_type or inc_train_start != new_train_start:
        return PromotionDecision(
            False,
            f"label/window differs from incumbent "
            f"({inc_label_type}/{inc_train_start} vs {new_label_type}/{new_train_start}); "
            "IC not comparable, deferring",
        )
    if incumbent_cv_ic is None:
        # Incumbent on the SAME surface but no comparable CV-IC (legacy/NaN):
        # can't prove an improvement, so defer rather than deploy.
        return PromotionDecision(
            False, "incumbent has no comparable CV-IC; deferring (can't prove improvement)"
        )
    # new_cv_ic is finite here (passed the floor) and incumbent_cv_ic is not None.
    if new_cv_ic >= incumbent_cv_ic + margin:
        return PromotionDecision(
            True,
            f"CV-IC {new_cv_ic:+.4f} beats incumbent {incumbent_cv_ic:+.4f} "
            f"by >= {margin}",
        )
    return PromotionDecision(
        False,
        f"CV-IC {new_cv_ic:+.4f} does not beat incumbent {incumbent_cv_ic:+.4f} "
        f"by {margin}",
    )
