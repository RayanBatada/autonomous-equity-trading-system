"""read_only_connect retries a conflicting writer lock instead of crashing."""

import pytest

from sma.db_connect import read_only_connect


class _LockError(Exception):
    pass


def test_retries_then_succeeds_on_lock_conflict():
    """A transient conflicting lock (writer overlap) is waited out, not raised."""
    calls = {"n": 0}

    def fake_connect(path, read_only):
        calls["n"] += 1
        if calls["n"] < 3:  # fail twice, succeed on the 3rd
            raise _LockError("Could not set lock on file: Conflicting lock is held")
        return "CONN"

    out = read_only_connect(
        "x.duckdb", base_delay=0.0, max_delay=0.0, _connect=fake_connect
    )
    assert out == "CONN"
    assert calls["n"] == 3


def test_non_lock_error_is_raised_immediately():
    """A non-lock error (real corruption, missing file) must NOT be retried."""
    calls = {"n": 0}

    def fake_connect(path, read_only):
        calls["n"] += 1
        raise ValueError("database is corrupt")

    with pytest.raises(ValueError, match="corrupt"):
        read_only_connect("x.duckdb", base_delay=0.0, _connect=fake_connect)
    assert calls["n"] == 1  # no retry


def test_exhausts_retries_then_reraises_lock_error():
    """If the writer never releases, give up after `retries` and raise the lock error."""
    def fake_connect(path, read_only):
        raise _LockError("Conflicting lock is held")

    with pytest.raises(_LockError, match="Conflicting lock"):
        read_only_connect(
            "x.duckdb", retries=3, base_delay=0.0, max_delay=0.0, _connect=fake_connect
        )
