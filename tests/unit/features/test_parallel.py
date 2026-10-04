"""Tests for the deterministic process-pool fan-out (sma.features.parallel).

These pin the two properties the bit-identity of the feature build rests on:
results come back in SUBMISSION order (never completion order), and workers=1
runs in-process with no pool at all.
"""

import os
import time

import pytest

from sma.features.parallel import (
    MAX_AUTO_WORKERS,
    contiguous_chunks,
    default_feature_workers,
    map_ordered,
    resolve_workers,
)

# ---------------------------------------------------------------------------
# Module-level task functions. `spawn` pickles the task by qualified name, so a
# closure or a local def would not survive the trip to a worker — that is a
# real constraint on every caller, and these test it as such.
# ---------------------------------------------------------------------------

def _square(unit, shared):
    return unit * unit + shared


def _pid_of(unit, shared):
    return os.getpid()


def _pid_after_rendezvous(unit, shared):
    """Announce this PID, then wait for `n_workers` distinct PIDs to announce.

    Without the rendezvous, "two workers each did some work" is a race the test
    usually loses: `_pid_of` returns instantly, so the first worker to finish
    spawning can drain all eight units before the second one has finished its
    ~1s interpreter startup, and the assertion sees a single PID. The barrier
    makes the property the test is actually claiming — the pool really does run
    work on more than one process — deterministic. The deadline means a worker
    that never arrives FAILS the test instead of hanging it.
    """
    import pathlib

    announce_dir, n_workers = shared
    d = pathlib.Path(announce_dir)
    (d / str(os.getpid())).touch()
    deadline = time.monotonic() + 60.0
    while len(list(d.iterdir())) < n_workers and time.monotonic() < deadline:
        time.sleep(0.01)
    return os.getpid()


def _sleep_then_echo(unit, shared):
    """Sleeps LONGER for EARLIER units, so completion order is the reverse of
    submission order. If map_ordered ever returned results as they finished,
    this test would catch it."""
    time.sleep((shared - unit) * 0.05)
    return unit


# ---------------------------------------------------------------------------
# default_feature_workers / resolve_workers
# ---------------------------------------------------------------------------

def test_default_is_min_four_and_cpus_minus_one():
    """The auto-default leaves a core for the parent and for whatever else is
    on the box, and caps at 4 because each worker holds its own copy of the
    price frame. Computed, not hardcoded — a 2-core CI runner gets 1."""
    assert default_feature_workers() == max(
        1, min(MAX_AUTO_WORKERS, (os.cpu_count() or 1) - 1)
    )
    assert default_feature_workers() >= 1


def test_resolve_workers_none_means_auto():
    assert resolve_workers(None, 1000) == default_feature_workers()


def test_resolve_workers_never_exceeds_the_work_available():
    """Spawning a process that would get nothing to do costs ~1s of
    interpreter startup for zero benefit."""
    assert resolve_workers(8, 3) == 3
    assert resolve_workers(8, 0) == 1
    assert resolve_workers(4, 100) == 4


def test_resolve_workers_rejects_nonsense():
    with pytest.raises(ValueError):
        resolve_workers(0, 10)
    with pytest.raises(ValueError):
        resolve_workers(-2, 10)
    with pytest.raises(TypeError):
        resolve_workers(2.5, 10)
    with pytest.raises(TypeError):
        resolve_workers(True, 10)  # bool is an int subclass; not a worker count


# ---------------------------------------------------------------------------
# contiguous_chunks
# ---------------------------------------------------------------------------

def test_contiguous_chunks_partition_in_order():
    items = list(range(10))
    chunks = contiguous_chunks(items, 3)
    assert [x for c in chunks for x in c] == items, "concatenation must restore order"
    assert [len(c) for c in chunks] == [4, 3, 3]
    assert all(chunks), "no empty chunks"


def test_contiguous_chunks_handles_more_chunks_than_items():
    assert contiguous_chunks([1, 2], 9) == [[1], [2]]
    assert contiguous_chunks([], 4) == []


# ---------------------------------------------------------------------------
# map_ordered
# ---------------------------------------------------------------------------

def test_map_ordered_serial_runs_in_this_process():
    """workers=1 is the exact serial path: no pool, same PID."""
    out = map_ordered(_pid_of, list(range(5)), None, workers=1)
    assert out == [os.getpid()] * 5


def test_map_ordered_parallel_actually_uses_other_processes(tmp_path):
    out = map_ordered(
        _pid_after_rendezvous, list(range(8)), (str(tmp_path), 2), workers=2,
    )
    assert os.getpid() not in out, "work must land in worker processes"
    assert len(set(out)) == 2, "both workers must have taken units"


def test_map_ordered_returns_submission_order_not_completion_order():
    """THE determinism guarantee. Unit 0 sleeps longest and finishes last; the
    result list must still start with 0."""
    units = [0, 1, 2, 3]
    assert map_ordered(_sleep_then_echo, units, 4, workers=2) == units


def test_map_ordered_parallel_matches_serial():
    units = list(range(20))
    assert (
        map_ordered(_square, units, 7, workers=3)
        == map_ordered(_square, units, 7, workers=1)
    )


def test_map_ordered_on_empty_units_builds_no_pool():
    assert map_ordered(_square, [], 0, workers=4) == []
