"""Review past autoresearch config-search runs from their sentinels.

Each `search` run writes a com.sma.autoresearch.nightly-<date>.json sentinel
holding EVERY config it evaluated (rank, CV-IC, params) plus the promote/hold
decision. This module reads them back so you can watch what the search would
have done — even on a run where it HELD and deployed nothing — and see where to
improve (near-miss holds, regions of the space that keep scoring well, etc.).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

# Must match promotion.IC_PROMOTE_MARGIN; imported lazily to avoid a cycle.
_PARAM_ORDER = (
    "max_depth", "learning_rate", "n_estimators",
    "subsample", "colsample_bytree", "min_child_weight",
)


def load_search_runs(sentinel_dir: Path, limit: int = 20) -> list[dict]:
    """Recent config-search runs, newest first, parsed from the
    com.sma.autoresearch.nightly-*.json sentinels. Skips non-search or
    unreadable sentinels."""
    runs: list[dict] = []
    paths = sorted(
        sentinel_dir.glob("com.sma.autoresearch.nightly-*.json"), reverse=True
    )
    for p in paths:
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict) or d.get("kind") != "config_search":
            continue
        runs.append(d)
        if len(runs) >= limit:
            break
    return runs


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def _gap_to_promotion(run: dict, margin: float) -> float | None:
    """best_cv_ic - (incumbent_cv_ic + margin). >= 0 means it cleared the bar.
    None when there's no comparable incumbent."""
    best, inc = run.get("best_cv_ic"), run.get("incumbent_cv_ic")
    if not _is_num(best) or not _is_num(inc):
        return None
    return best - (inc + margin)


def _ic(x) -> str:
    return f"{x:+.4f}" if _is_num(x) else "nan"


def format_history(runs: list[dict], margin: float = 0.005) -> str:
    """Human-readable review: one block per run (decision + every config tried),
    then a summary including the closest near-miss hold."""
    if not runs:
        return "no autoresearch search runs recorded yet (first run Mon 07:00 ET)."

    lines: list[str] = []
    promoted = held = 0
    closest_miss: tuple[float, str] | None = None  # (gap<0 nearest to 0, date)

    for r in runs:
        asof = r.get("asof", "?")
        dry, prom = r.get("dry_run"), r.get("promoted")
        verb = "DRY-RUN" if dry else ("PROMOTED" if prom else "HELD")
        if not dry:
            if prom:
                promoted += 1
            else:
                held += 1
        gap = _gap_to_promotion(r, margin)
        if (
            not prom and not dry and gap is not None and gap < 0
            and (closest_miss is None or gap > closest_miss[0])
        ):
            closest_miss = (gap, asof)
        gap_s = f"{gap:+.4f}" if gap is not None else "n/a"
        lines.append(
            f"{asof}  {verb:<8} best {_ic(r.get('best_cv_ic'))}  "
            f"incumbent {_ic(r.get('incumbent_cv_ic'))}  "
            f"gap-to-promote {gap_s}  ({r.get('n_configs', '?')} configs)"
        )
        for c in (r.get("configs") or [])[:10]:
            params = c.get("params") or {}
            ps = " ".join(f"{k}={params[k]}" for k in _PARAM_ORDER if k in params)
            lines.append(f"      #{c.get('rank', '?')}  CV-IC {_ic(c.get('cv_ic'))}  {ps}")

    summary = f"\n{len(runs)} runs: {promoted} promoted, {held} held."
    if closest_miss is not None:
        summary += (
            f" Closest hold: {closest_miss[0]:+.4f} from promoting "
            f"(on {closest_miss[1]}) — widen the search or lower the margin if "
            "these keep just missing."
        )
    return "\n".join(lines) + summary
