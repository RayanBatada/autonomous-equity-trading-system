"""Deterministic process-pool fan-out for the feature build.

The feature build is the cost centre of the whole project: 65-80 min of every
weekly retrain and the dominant cost of every walk-forward study (the
label-horizon study ran 339 boundaries). 1ee39b1 made it linear in universe
size; this module makes it use more than one core.

It is embarrassingly parallel by construction — an asof_date's cross-section is
computed from read-only inputs and never reads another asof's result, and a
ticker's feature row never reads another ticker's row (the one cross-ticker
input, the sector peer mean, is precomputed by the caller before the fan-out).
This module is the single place that owns the fan-out so both entry points
(`build_features` over ticker chunks, `build_training_set` over asof dates) get
the same determinism guarantees:

  * **"spawn", always.** It is the macOS default and the only start method that
    is safe in a process that may already hold threads; forcing it explicitly
    means Linux CI exercises exactly what production runs, rather than fork.
  * **Heavy read-only inputs ship ONCE per worker**, through the pool
    initializer rather than per task. A 2170-asof retrain pays 4 pickles of the
    price frame, not 2170.
  * **Results come back in SUBMISSION order** (`Executor.map`), never in
    completion order, and the caller concatenates them in that order. No part
    of the reduction depends on which worker finished first — which is what
    makes the output bit-identical to the serial path rather than merely close.
  * **workers=1 builds no pool at all.** It runs the same task function in this
    process, in order: the exact serial path, not a one-worker imitation of it.

TWO TRAPS, both a consequence of "spawn", both benign at the shipped defaults:

1. Do not fan out work whose inputs are patched at module scope. A spawned
   child imports `sma.sectors` fresh and would NOT see a
   `patch.dict("sma.sectors.SECTORS", ...)` applied in the parent. The feature
   builder sidesteps this by passing the sector mappings through its context
   object rather than re-importing them in the worker, but a future fan-out
   should not assume module state travels.

2. A spawned child re-imports the PARENT'S MAIN MODULE. `python -m sma.model`
   and `python -m sma.autoresearch` both guard their entry point behind
   `if __name__ == "__main__"`, so the child re-import is inert — but a
   one-off `scripts/foo.py` that calls into a fan-out from unguarded top-level
   code hits multiprocessing's "attempt to start a new process before the
   current process has finished its bootstrapping phase". Python raises that in
   the CHILD, which stops the fork bomb — but observed cost is a HUNG PARENT,
   not a clean traceback, so it is not something to shrug at. It cannot happen
   by accident: every `workers` default in this codebase is 1, so a script
   either opts in explicitly (and puts its body behind a main guard, as both
   `python -m` entry points already do) or stays serial.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

# Cap the auto-default at 4 rather than "all cores". The box this runs on is a
# 10-core / 16GB Mac that is also serving the dashboard and may be running the
# evening trading jobs; each worker holds its own copy of the price and news
# frames, so more workers costs RAM linearly for a speedup that flattens out
# (the parent's own serial assembly is the remaining floor).
MAX_AUTO_WORKERS = 4


def default_feature_workers() -> int:
    """The auto-default: `min(4, cpu_count - 1)`, floored at 1.

    cpu_count - 1 deliberately leaves a core for the parent process and for
    whatever else is on the box; the retrain is a background job, not something
    anyone is waiting at a terminal for.
    """
    cpus = os.cpu_count() or 1
    return max(1, min(MAX_AUTO_WORKERS, cpus - 1))


def resolve_workers(workers: int | None, n_units: int) -> int:
    """Normalise a requested worker count against the work actually available.

    `None` means "auto" (`default_feature_workers()`). Never returns more
    workers than there are units of work — spawning a process that would get
    nothing to do costs ~1s of interpreter startup for zero benefit.
    """
    if workers is None:
        workers = default_feature_workers()
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise TypeError(f"workers must be an int or None; got {workers!r}")
    if workers < 1:
        raise ValueError(f"workers must be >= 1; got {workers}")
    if n_units <= 0:
        return 1
    return max(1, min(workers, n_units))


# Worker-process globals. Set once per worker by the pool initializer, then read
# by every task that worker runs — this is what keeps the heavy shared inputs
# out of the per-task payload.
_TASK: Callable[[Any, Any], Any] | None = None
_SHARED: Any = None


def _init_worker(task: Callable[[Any, Any], Any], shared: Any) -> None:
    global _TASK, _SHARED
    _TASK, _SHARED = task, shared


def _run_unit(unit: Any) -> Any:
    return _TASK(unit, _SHARED)  # type: ignore[misc]


def map_ordered(
    task: Callable[[Any, Any], Any],
    units: Iterable[Any],
    shared: Any,
    *,
    workers: int | None,
    chunksize: int = 1,
) -> list[Any]:
    """Run `task(unit, shared)` for every unit; return results in UNIT ORDER.

    `task` must be a module-level function (spawn pickles it by qualified name)
    and `shared` must be picklable. Results are returned in the order of
    `units`, whatever order the workers actually finished in — callers rely on
    that for bit-identical output.

    chunksize stays at 1 by default: per-asof tasks are ~2.5s each, so the
    dispatch round-trip is noise next to them, and unit-at-a-time hand-out is
    what keeps the tail balanced when some asofs are dearer than others.
    """
    units = list(units)
    n_workers = resolve_workers(workers, len(units))
    if n_workers == 1:
        return [task(u, shared) for u in units]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(task, shared),
    ) as pool:
        return list(pool.map(_run_unit, units, chunksize=chunksize))


def contiguous_chunks(items: Sequence[Any], n_chunks: int) -> list[list[Any]]:
    """Split `items` into at most `n_chunks` CONTIGUOUS, non-empty chunks.

    Contiguous (not round-robin) so that concatenating the chunks' results in
    chunk order reproduces the original item order exactly, with no re-sort to
    get wrong.
    """
    n_items = len(items)
    if n_items == 0:
        return []
    n_chunks = max(1, min(n_chunks, n_items))
    base, extra = divmod(n_items, n_chunks)
    out: list[list[Any]] = []
    start = 0
    for i in range(n_chunks):
        stop = start + base + (1 if i < extra else 0)
        out.append(list(items[start:stop]))
        start = stop
    return out
