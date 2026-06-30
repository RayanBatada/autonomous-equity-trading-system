from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sma.live.preflight import (
    IngestNotCompleteError,
    NextSessionNotTomorrow,
    UpstreamMissedDeadline,
    UpstreamReadinessFailed,
    run_preflight,
)
from sma.sentinels import write_sentinel


def _mock_alpaca(next_session_date: date):
    alpaca = MagicMock()
    alpaca.next_session_date.return_value = next_session_date
    return alpaca


def test_preflight_passes_when_all_upstream_sentinels_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    asof = date(2026, 4, 30)  # Thursday
    for label in (
        "com.sma.ingest.daily",
        "com.sma.model.predict.daily",
        "com.sma.agents.daily",
    ):
        write_sentinel(
            label=label,
            asof=asof,
            payload={
                "label": label,
                "asof": asof.isoformat(),
                "run_id": 1,
                "quality": {"passed": True, "blocking_failures": []},
            },
        )
    alpaca = _mock_alpaca(next_session_date=date(2026, 5, 1))
    # Should not raise; max_wait_s=0 means no real waiting
    run_preflight(asof=asof, db_path=Path("does/not/matter"), alpaca=alpaca, max_wait_s=0)


def test_preflight_raises_when_ingest_sentinel_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    asof = date(2026, 4, 30)
    alpaca = _mock_alpaca(next_session_date=date(2026, 5, 1))
    with pytest.raises(IngestNotCompleteError):
        run_preflight(asof=asof, db_path=Path("x"), alpaca=alpaca, max_wait_s=0)


def test_preflight_raises_when_predict_sentinel_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    asof = date(2026, 4, 30)
    write_sentinel(
        label="com.sma.ingest.daily",
        asof=asof,
        payload={"quality": {"passed": True, "blocking_failures": []}, "run_id": 1},
    )
    # predict + agents sentinels missing
    alpaca = _mock_alpaca(next_session_date=date(2026, 5, 1))
    with pytest.raises(UpstreamMissedDeadline):
        run_preflight(asof=asof, db_path=Path("x"), alpaca=alpaca, max_wait_s=0)


def test_preflight_raises_when_upstream_quality_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    asof = date(2026, 4, 30)
    write_sentinel(
        label="com.sma.ingest.daily",
        asof=asof,
        payload={
            "label": "com.sma.ingest.daily",
            "asof": asof.isoformat(),
            "run_id": 1,
            "quality": {"passed": False, "blocking_failures": ["all_tickers_have_price"]},
        },
    )
    alpaca = _mock_alpaca(next_session_date=date(2026, 5, 1))
    with pytest.raises(UpstreamReadinessFailed, match="all_tickers_have_price"):
        run_preflight(asof=asof, db_path=Path("x"), alpaca=alpaca, max_wait_s=0)


def _seed_upstream_sentinels(tmp_path, asof: date) -> None:
    """Helper: stamp the three preflight-blocking sentinels READY for asof."""
    for label in (
        "com.sma.ingest.daily",
        "com.sma.model.predict.daily",
        "com.sma.agents.daily",
    ):
        write_sentinel(
            label=label,
            asof=asof,
            payload={"quality": {"passed": True, "blocking_failures": []}, "run_id": 1},
        )


def _seed_hard_deps_ready(asof: date) -> None:
    """Seed only the HARD (blocking) decide deps — ingest + predict — as READY."""
    for label in ("com.sma.ingest.daily", "com.sma.model.predict.daily"):
        write_sentinel(
            label=label,
            asof=asof,
            payload={"quality": {"passed": True, "blocking_failures": []}, "run_id": 1},
        )


def test_preflight_proceeds_when_agents_overlay_failed(tmp_path, monkeypatch):
    """Agents is an ADVISORY dependency: a failed agents run (e.g. a network blip
    → all_tickers_failed) must NOT hard-block trading. With ingest+predict READY,
    preflight proceeds even though the agents quality verdict failed."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    asof = date(2026, 4, 30)  # Thursday
    _seed_hard_deps_ready(asof)
    write_sentinel(
        label="com.sma.agents.daily",
        asof=asof,
        payload={
            "quality": {"passed": False, "blocking_failures": ["all_tickers_failed"]},
            "run_id": 1,
        },
    )
    alpaca = _mock_alpaca(next_session_date=date(2026, 5, 1))
    # Must NOT raise — advisory deps never block decide.
    run_preflight(asof=asof, db_path=Path("x"), alpaca=alpaca, max_wait_s=0)


def test_preflight_proceeds_when_agents_sentinel_missing(tmp_path, monkeypatch):
    """Agents advisory: even a totally missing agents sentinel (the agents job
    never ran) does not block decide, as long as the hard deps are READY."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    asof = date(2026, 4, 30)
    _seed_hard_deps_ready(asof)
    alpaca = _mock_alpaca(next_session_date=date(2026, 5, 1))
    # Must NOT raise — a missing advisory sentinel is logged, not fatal.
    run_preflight(asof=asof, db_path=Path("x"), alpaca=alpaca, max_wait_s=0)


def test_preflight_proceeds_when_advisory_sentinel_corrupt(tmp_path, monkeypatch):
    """A corrupt advisory sentinel (unparseable JSON) must NOT abort preflight.
    read_sentinel raises on bad JSON; the advisory loop has to swallow that and
    proceed, or a single garbled agents file would re-freeze trading."""
    sentinel_dir = tmp_path / "sentinels"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))
    asof = date(2026, 4, 30)
    _seed_hard_deps_ready(asof)
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    (sentinel_dir / f"com.sma.agents.daily-{asof.isoformat()}.json").write_text("{not valid json")
    alpaca = _mock_alpaca(next_session_date=date(2026, 5, 1))
    # Must NOT raise.
    run_preflight(asof=asof, db_path=Path("x"), alpaca=alpaca, max_wait_s=0)


def test_preflight_raises_when_next_session_too_far_out(tmp_path, monkeypatch):
    """Calendar sanity bound is 14 days; beyond that is treated as bad data."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    asof = date(2026, 4, 30)
    _seed_upstream_sentinels(tmp_path, asof)
    alpaca = _mock_alpaca(next_session_date=date(2026, 6, 15))  # 46 days out
    with pytest.raises(NextSessionNotTomorrow):
        run_preflight(asof=asof, db_path=Path("x"), alpaca=alpaca, max_wait_s=0)


def test_preflight_raises_when_next_session_not_in_future(tmp_path, monkeypatch):
    """next_session must be strictly after asof — a same-day or past date is
    nonsense (Alpaca SDK returning stale or cached value)."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    asof = date(2026, 4, 30)
    _seed_upstream_sentinels(tmp_path, asof)
    alpaca = _mock_alpaca(next_session_date=date(2026, 4, 30))
    with pytest.raises(NextSessionNotTomorrow):
        run_preflight(asof=asof, db_path=Path("x"), alpaca=alpaca, max_wait_s=0)


def test_preflight_accepts_post_holiday_next_session(tmp_path, monkeypatch):
    """Empirical 2026-05-22: Friday before Memorial Day. Alpaca correctly
    reported next session = Tue 5/26 (skipping Mon 5/25 holiday). Pre-fix
    decide refused to fire because the check insisted on Mon 5/25. The fix
    treats any next session within 14 days as valid; OPG orders queue at
    Alpaca and fill at the next genuine open auction."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    friday_before_memorial_day = date(2026, 5, 22)
    _seed_upstream_sentinels(tmp_path, friday_before_memorial_day)
    alpaca = _mock_alpaca(next_session_date=date(2026, 5, 26))  # skip Mon holiday
    # Should not raise.
    run_preflight(
        asof=friday_before_memorial_day,
        db_path=Path("x"),
        alpaca=alpaca,
        max_wait_s=0,
    )


def test_preflight_accepts_normal_next_day_session(tmp_path, monkeypatch):
    """Vanilla case: Thursday → next session is Friday. Still accepted."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    asof = date(2026, 4, 30)  # Thursday
    _seed_upstream_sentinels(tmp_path, asof)
    alpaca = _mock_alpaca(next_session_date=date(2026, 5, 1))  # Friday
    run_preflight(asof=asof, db_path=Path("x"), alpaca=alpaca, max_wait_s=0)
