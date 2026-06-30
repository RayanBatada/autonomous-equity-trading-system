"""Test configuration and shared fixtures.

This conftest provides an autouse fixture that ensures every test runs with
the writer_lock held and DEFAULT_LOCK_PATH pointed at a per-test temp location.
This satisfies the Store.connect(read_only=False) assertion added in Task 4
without requiring every test file to be updated individually.

Tests in test_store_lock_assertion.py that explicitly test the "lock not held"
scenario use their own monkeypatch.setattr calls on DEFAULT_LOCK_PATH, which
override this fixture for the duration of those tests.
"""

import pytest


@pytest.fixture(autouse=True)
def _hold_writer_lock_for_test(tmp_path_factory, monkeypatch):
    """Acquire the writer_lock for each test and point DEFAULT_LOCK_PATH at a
    per-test temp location so tests are isolated and the Store.connect()
    assertion is satisfied automatically.

    Uses tmp_path_factory (not tmp_path) so the conftest-owned lock directory
    is NOT visible inside the test's own tmp_path. Tests that assert on
    tmp_path contents (e.g. test_atomic_write_does_not_leak_tempfile) are
    therefore unaffected.

    Tests that need to exercise the "lock not held" path (e.g.
    test_store_lock_assertion.py) override DEFAULT_LOCK_PATH with their own
    monkeypatch.setattr call pointing to a different path, so they remain
    unaffected by the lock held here.
    """
    import sma.locks as _locks
    from sma.locks import writer_lock

    lock_dir = tmp_path_factory.mktemp("_writer_lock")
    lock_path = lock_dir / ".sma-writer.lock"
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", lock_path)
    with writer_lock(lock_path=lock_path, label="test"):
        yield


@pytest.fixture(autouse=True)
def _isolate_sentinel_dir(tmp_path_factory, monkeypatch):
    """Point SMA_SENTINEL_DIR at a per-test temp dir for EVERY test.

    Without this, any test that exercises a sentinel-writing path (directly or
    via a CLI subprocess) resolves the CWD-relative default and writes REAL
    files into data/sentinels/ — on 2026-06-09 the integration suite polluted
    production with fixture-dated sentinels (and a higher-run_id fake could
    even suppress a real job's write via the monotonicity guard). Env vars
    inherit into subprocess-spawned CLIs, so this isolates those too. Uses
    tmp_path_factory so tests asserting on their own tmp_path contents are
    unaffected; tests can still monkeypatch.setenv to a specific dir.
    """
    monkeypatch.setenv(
        "SMA_SENTINEL_DIR", str(tmp_path_factory.mktemp("_sentinels"))
    )
