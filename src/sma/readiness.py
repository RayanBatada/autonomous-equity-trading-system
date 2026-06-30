"""Live-readiness contract for cross-job dependency checks.

Given a sentinel and a consuming job's waivers, decide whether the upstream
dependency is READY (proceed), WAIT (sentinel not yet written), or FAIL
(blocking failure that no waiver covers).

Used by `sma.live.preflight.run_preflight` for the decide job's dependency
chain (ingest, predict, agents). Replaces the previous diverging
`EXPECTED_SOURCES` / `CRITICAL_INGEST_SOURCES` set comparisons with one
explicit contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ReadinessState(StrEnum):
    READY = "READY"
    WAIT = "WAIT"
    FAIL = "FAIL"


# Jobs that have NO quality gate: they write a success-only sentinel (no
# `quality` block) and only after their work has persisted. A quality-less
# sentinel is treated as a success ONLY for these labels (and only with
# completion evidence). Every other label — including the quality-GATED jobs
# (ingest, agents), which always emit a quality block — must NOT be allowed to
# look ready without one; an absent block there means malformed/crashed and
# must fail closed.
_NON_GATED_LABELS = frozenset(
    {
        "com.sma.model.predict.daily",
        "com.sma.model.retrain.weekly",
        # NOTE: backup is NOT here — its sentinel carries a quality block keyed on
        # verification (runner.py), so it must be gated, not auto-passed.
    }
)


@dataclass(frozen=True)
class ReadinessResult:
    state: ReadinessState
    explanation: str

    def is_ready(self) -> bool:
        return self.state == ReadinessState.READY


def sentinel_lineage_stale(*, consumer: dict | None, upstream: dict | None) -> bool:
    """True iff `consumer`'s sentinel was built from a DIFFERENT upstream run
    than the current upstream sentinel records.

    2026-06-09: ingest failed (DNS), predict silently produced predictions off
    stale features and wrote a healthy sentinel. When ingest later healed
    (new run_id), both sentinels read READY and decide would have traded on
    the stale predictions. The consumer records `ingest_run_id` (the upstream
    run it consumed); a mismatch with the upstream's current `run_id` means
    the consumer must re-run first.

    Tolerant by design: missing sentinels or missing fields (old-format
    consumers, pre-lineage) are NOT stale — the transition must not brick the
    pipeline.
    """
    if not consumer or not upstream:
        return False
    consumed = consumer.get("ingest_run_id")
    current = upstream.get("run_id")
    if consumed is None or current is None:
        return False
    return consumed != current


def live_readiness(
    *,
    label: str,
    sentinel: dict | None,
    waivers: frozenset[str],
) -> ReadinessResult:
    """Decide if `label`'s sentinel signals readiness for downstream consumers.

    - sentinel is None: WAIT (job has not yet written its sentinel for this asof)
    - sentinel has no `quality` block (non-gated job, e.g. predict): READY
    - sentinel.quality.passed and no blocking failures: READY
    - sentinel.quality has blocking failures, all waived: READY (with waived note)
    - sentinel has unwaived blocking failures: FAIL
    """
    if sentinel is None:
        return ReadinessResult(
            state=ReadinessState.WAIT,
            explanation=f"sentinel for {label} not yet written",
        )

    # A sentinel with NO `quality` block comes from a non-gated job (e.g.
    # predict), which writes its sentinel only on success. Treat it as READY
    # ONLY for a recognized non-gated label AND only with completion evidence
    # (`completed_at`). Anything else — a gated job missing its block (always a
    # bug/crash), an unknown label, or an empty/truncated payload — falls
    # through to fail-closed. Without this scoping a malformed hard-dep sentinel
    # could green-light trading; with it, the original freeze (a quality-less
    # but valid predict sentinel, 2026-06-04) still resolves to READY.
    if "quality" not in sentinel:
        if label in _NON_GATED_LABELS and sentinel.get("completed_at"):
            return ReadinessResult(
                state=ReadinessState.READY,
                explanation=f"{label} ready (no quality gate)",
            )
        return ReadinessResult(
            state=ReadinessState.FAIL,
            explanation=f"{label} sentinel has no quality block and is not a "
            f"recognized non-gated job with completion evidence",
        )

    quality = sentinel.get("quality")
    if not isinstance(quality, dict):
        # Present-but-malformed quality (null/list/scalar): fail closed rather
        # than raising AttributeError on the .get() calls below.
        return ReadinessResult(
            state=ReadinessState.FAIL,
            explanation=f"{label} quality block is malformed (not an object)",
        )
    blocking = list(quality.get("blocking_failures", []))

    if quality.get("passed", False) and not blocking:
        return ReadinessResult(
            state=ReadinessState.READY,
            explanation=f"{label} quality passed",
        )

    unwaived_blocking = [b for b in blocking if b not in waivers]
    if unwaived_blocking:
        return ReadinessResult(
            state=ReadinessState.FAIL,
            explanation=f"{label} has unwaived blocking failures: "
            f"{','.join(unwaived_blocking)}",
        )

    # No unwaived blocking failures remain. But a sentinel that explicitly says
    # passed=False with NO blocking reason is malformed/failed — fail closed
    # rather than green-lighting on the (empty) all-waived branch.
    if not quality.get("passed", False) and not blocking:
        return ReadinessResult(
            state=ReadinessState.FAIL,
            explanation=f"{label} quality.passed is False with no blocking reason",
        )

    waived_str = ",".join(sorted(b for b in blocking if b in waivers))
    return ReadinessResult(
        state=ReadinessState.READY,
        explanation=f"{label} ready (waived blocking: {waived_str})"
        if waived_str
        else f"{label} ready",
    )
