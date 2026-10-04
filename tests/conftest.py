"""Test configuration and shared fixtures.

This conftest provides an autouse fixture that ensures every test runs with
the writer_lock held and DEFAULT_LOCK_PATH pointed at a per-test temp location.
This satisfies the Store.connect(read_only=False) assertion added in Task 4
without requiring every test file to be updated individually.

Tests in test_store_lock_assertion.py that explicitly test the "lock not held"
scenario use their own monkeypatch.setattr calls on DEFAULT_LOCK_PATH, which
override this fixture for the duration of those tests.
"""

import traceback
from pathlib import Path

import pytest

# The one real production DuckDB file. Resolved once at import time relative
# to this conftest's own location (repo_root/tests/conftest.py), NOT to cwd,
# so the guard below still works no matter where pytest is invoked from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROD_DB = (_REPO_ROOT / "data" / "sma.duckdb").resolve()
_PROD_DB_WAL = Path(str(_PROD_DB) + ".wal")


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
    # Isolate the heavy-job lock too (repo-root-absolute in production): a
    # train-CLI test must never queue behind a LIVE retrain/autoresearch run —
    # on 2026-07-02 the suite sat blocked 20+ min on exactly that.
    monkeypatch.setattr(
        _locks, "HEAVY_JOB_LOCK_PATH", lock_dir / ".sma-heavy.lock"
    )
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


@pytest.fixture(autouse=True)
def _isolate_dead_ticker_state(tmp_path_factory, monkeypatch):
    """Same rationale as _isolate_sentinel_dir, for
    sma.ingest.dead_ticker_state's no_dead_or_frozen_tickers dedup file:
    without this, any test that reaches
    sma.ingest.quality.notify_new_dead_or_frozen_tickers (directly, or via the
    ingest CLI's `run` command) resolves the CWD-relative default and writes
    into the real data/state/dead_frozen_tickers.json."""
    monkeypatch.setenv(
        "SMA_DEAD_TICKER_STATE_PATH",
        str(tmp_path_factory.mktemp("_dead_ticker_state") / "dead_frozen_tickers.json"),
    )


@pytest.fixture(autouse=True)
def _no_real_ntfy_pushes(monkeypatch):
    """2026-08-31: this repo's real .env carries a real SMA_NTFY_TOPIC (the
    nightly trade-push feature reads it via sma.ingest.notify.send_ntfy,
    called directly from sma.live.__main__ rather than only through the
    already-mocked notify_failure). Without this, any test that exercises a
    real (non-dry-run) decide/stop-loss-sweep path without explicitly
    mocking send_ntfy would fire a REAL push to Rayan's phone during
    `pytest`. Unset it for every test; a test that specifically exercises
    send_ntfy/notify_failure's ntfy behavior sets it back with its own
    monkeypatch.setenv call, which — since it runs inside the test body,
    after fixture setup — always wins."""
    monkeypatch.delenv("SMA_NTFY_TOPIC", raising=False)


@pytest.fixture(autouse=True)
def _no_real_agents_pages(monkeypatch):
    """2026-10-04: the agents job now pages when most thesis calls fail. Any
    test that drives a failing agents run (several do, on purpose) would
    otherwise raise a real macOS notification, or a real Telegram message
    when the host's env carries a token. Tests that check the page patch it
    themselves, inside the test body, which wins over this."""
    monkeypatch.setattr("sma.agents.__main__.notify_failure", lambda **kw: None)


def _resolve_db_path(path) -> Path | None:
    """Best-effort resolve of a duckdb target path. Returns None for the
    in-memory sentinel (":memory:") or anything else that isn't a real path."""
    if path is None:
        return None
    text = str(path)
    if text == ":memory:" or text.startswith("md:") or text.startswith("motherduck:"):
        return None
    try:
        return Path(text).resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def _is_prod_db_path(path) -> bool:
    resolved = _resolve_db_path(path)
    return resolved in (_PROD_DB, _PROD_DB_WAL)


def _refuse_writable_prod_open(label: str, path) -> None:
    stack = "".join(traceback.format_stack()[:-1])
    raise RuntimeError(
        f"BLOCKED: {label} tried to open the PRODUCTION DuckDB at "
        f"{_resolve_db_path(path)} WRITABLE from inside the test suite.\n\n"
        "Tests must never open data/sma.duckdb (or its .wal) read-write --\n"
        "the prod DB may only be opened writable from a launchd job. This\n"
        "guard exists because a test doing exactly this applied migrations\n"
        "10-12 (one still UNCOMMITTED) to the real prod DB on 2026-09-27 by\n"
        "running under `pytest -n auto -m \"not serial\"`.\n\n"
        "Fix the test: point it at tmp_path / an explicit fixture DB instead\n"
        "of relying on a CWD-relative default like \"data/sma.duckdb\". Do NOT\n"
        "weaken or remove this fixture -- it is the regression test for that\n"
        "incident. Read-only opens of the prod DB remain allowed.\n\n"
        f"Call stack (most recent call last):\n{stack}"
    )


@pytest.fixture(autouse=True)
def _guard_prod_duckdb_never_writable(monkeypatch):
    """Refuse ANY writable open of the real repo data/sma.duckdb during tests.

    2026-09-27 incident: `_schema_version` in the real data/sma.duckdb shows
    migrations 10 and 11 applied at 22:57:05 ET and migration 12 (still
    UNCOMMITTED at the time) applied at 23:18:50 ET -- both timestamps to the
    second matching a full `pytest -n auto -m "not serial"` run starting from
    the repo root. Root cause: `Store.connect(read_only=False)` only checks
    that the writer_lock is held (see `WriterLockNotHeld` above), and
    `_hold_writer_lock_for_test` above HOLDS that lock for every single test
    regardless of which DB path is being opened. So any test (or module-level
    default) that constructs `Store(path="data/sma.duckdb")` -- or calls
    `duckdb.connect("data/sma.duckdb")` directly -- without read_only=True
    silently passes the writer-lock check and runs pending migrations
    against PRODUCTION.

    This fixture makes that a loud, immediate failure instead of a silent
    write, and IS the regression test for the incident: keep it enabled
    permanently, do not skip/xfail it, and do not narrow the path match.

    Read-only opens of the real prod DB are explicitly allowed (the
    dashboard, `replay`, `read_only_connect`, and the opt-in
    SMA_GOLDEN_PROD=1 test in test_sleeve_golden.py all need this).

    Imports `duckdb` / `sma.ingest.store` lazily (inside this function, not
    at module scope) for the same reason every `sma.*` import in the fixtures
    above is lazy: conftest.py itself must stay importable by whatever
    interpreter loads it even when that interpreter lacks this project's
    deps -- e.g. a subprocess-spawned `python -m pytest` targeting a single
    unrelated test node (see test_smoke_paper.py's env-flag test), where
    `python` need not resolve to this repo's venv.
    """
    import duckdb

    import sma.ingest.store as _store_module

    real_duckdb_connect = duckdb.connect

    def _guarded_duckdb_connect(*args, **kwargs):
        path = kwargs.get("database", args[0] if args else None)
        read_only = kwargs.get("read_only", args[1] if len(args) > 1 else False)
        if not read_only and _is_prod_db_path(path):
            _refuse_writable_prod_open("duckdb.connect", path)
        return real_duckdb_connect(*args, **kwargs)

    real_store_connect = _store_module.Store.connect

    def _guarded_store_connect(self, *, read_only: bool = False):
        if not read_only and _is_prod_db_path(self.path):
            _refuse_writable_prod_open("Store.connect", self.path)
        return real_store_connect(self, read_only=read_only)

    monkeypatch.setattr(duckdb, "connect", _guarded_duckdb_connect)
    monkeypatch.setattr(_store_module.Store, "connect", _guarded_store_connect)
