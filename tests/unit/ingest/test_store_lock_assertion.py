import pytest

from sma.ingest.store import Store, WriterLockNotHeld
from sma.locks import writer_lock


def test_writable_connect_raises_when_lock_not_held(tmp_path, monkeypatch):
    lock_path = tmp_path / ".sma-writer.lock"
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", lock_path)
    db_path = tmp_path / "test.duckdb"
    store = Store(path=db_path)
    with pytest.raises(WriterLockNotHeld):
        store.connect(read_only=False)


def test_writable_connect_succeeds_when_lock_held(tmp_path, monkeypatch):
    lock_path = tmp_path / ".sma-writer.lock"
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", lock_path)
    db_path = tmp_path / "test.duckdb"
    with writer_lock(lock_path=lock_path, label="test"):
        store = Store(path=db_path)
        store.connect(read_only=False)
        store.conn.close()


def test_read_only_connect_does_not_require_lock(tmp_path, monkeypatch):
    lock_path = tmp_path / ".sma-writer.lock"
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", lock_path)
    db_path = tmp_path / "test.duckdb"
    # First create the DB by writing once (with lock)
    with writer_lock(lock_path=lock_path, label="setup"):
        Store(path=db_path).connect(read_only=False).conn.close()
    # Now read_only without lock should work
    store = Store(path=db_path)
    store.connect(read_only=True)
    store.conn.close()


def test_writer_lock_held_by_other_pid_raises(tmp_path, monkeypatch):
    """Simulate the case where the PID file exists but contains a different PID."""
    lock_path = tmp_path / ".sma-writer.lock"
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", lock_path)
    pid_path = lock_path.with_suffix(".pid")
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    # Write a fake PID (we'll never have PID 1 as our process)
    pid_path.write_text("1 fake-holder\n")

    db_path = tmp_path / "test.duckdb"
    store = Store(path=db_path)
    with pytest.raises(WriterLockNotHeld, match="held by PID"):
        store.connect(read_only=False)
