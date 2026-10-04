"""senate/house weekly ingest CLIs must write their launchd sentinels.

Review 2026-07-20 (HIGH): neither `_main` wrote a sentinel, so the watchdog could
NEVER see the Sunday jobs as done — with the new daytime checkpoints + widened
late-kick windows it would re-kick both on every healthy Sunday (duplicate
scrapes + a 30s writer-lock loser crash per checkpoint)."""

from __future__ import annotations

import sys
from datetime import date
from unittest.mock import patch

from sma.sentinels import read_sentinel


def test_senate_main_writes_sentinel(tmp_path, monkeypatch):
    # NOTE: do NOT re-patch DEFAULT_LOCK_PATH — the autouse conftest fixture
    # already patches AND holds it (Store.connect asserts against that path);
    # the CLI's own writer_lock briefly takes the CWD-relative real lock, which
    # is the documented-safe pattern for in-process CLI tests.
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))

    from sma.ingest.sources import senate_trades as st

    with (
        patch.object(sys, "argv", ["senate", "--db", str(tmp_path / "t.duckdb")]),
        patch.object(st, "run_senate_ingest", return_value={"inserted": 0}),
    ):
        st._main()

    assert read_sentinel(
        label="com.sma.senate-ingest.weekly", asof=date.today()
    ) is not None


def test_house_main_writes_sentinel(tmp_path, monkeypatch):
    # NOTE: do NOT re-patch DEFAULT_LOCK_PATH — the autouse conftest fixture
    # already patches AND holds it (Store.connect asserts against that path);
    # the CLI's own writer_lock briefly takes the CWD-relative real lock, which
    # is the documented-safe pattern for in-process CLI tests.
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))

    from sma.ingest.sources import politician_trades as pt

    with (
        patch.object(sys, "argv", ["house", "--db", str(tmp_path / "t.duckdb")]),
        patch("sma.ingest.sources.politician_trades.run_ingest", return_value={"inserted": 0}),
    ):
        pt._main()

    assert read_sentinel(
        label="com.sma.house-ingest.weekly", asof=date.today()
    ) is not None
