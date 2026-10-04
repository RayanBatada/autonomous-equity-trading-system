"""Split-consistency audit rule + the no_split_inconsistency quality check
(2026-10-01, MNST 2:1 on 2026-08-11)."""

from datetime import date, timedelta

import pytest

from sma.ingest.quality import (
    _check_no_split_inconsistency,
    notify_degraded_quality_checks,
    run_quality_checks,
)
from sma.ingest.split_audit import (
    FEATURE_SERIES_SQL,
    find_split_inconsistencies,
)
from sma.ingest.store import Store


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    yield s
    s.close()


D0 = date(2026, 7, 13)


def _days(n):
    out, d = [], D0
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _put(store, ticker, source, closes, adj=True):
    for d, c in closes:
        store.conn.execute(
            "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, 1)",
            [ticker, d, c, c, c, c, (c if adj else None), source],
        )


def _mnst(store):
    """yfinance: pre-split scale through day 4, adjusted after (the bug).
    alpaca: raw, continuous until the real split on day 10."""
    days = _days(14)
    yf = [(d, 98.0 if i < 5 else 48.0) for i, d in enumerate(days)]
    al = [(d, 96.0 if i < 10 else 48.0) for i, d in enumerate(days)]
    _put(store, "MNST", "yfinance", yf)
    _put(store, "MNST", "alpaca", al, adj=False)
    return days


def test_rule_flags_yfinance_break_and_classifies_raw_alpaca_split(store):
    days = _mnst(store)
    flags = find_split_inconsistencies(store.conn)
    kinds = {(f.date, f.kind) for f in flags}
    assert (days[5], "left_scale_break") in kinds
    assert (days[10], "alpaca_raw_split") in kinds
    bad = [f for f in flags if f.feature_affecting]
    assert [(f.ticker, f.date) for f in bad] == [("MNST", days[5])]


def test_clean_series_and_confirmed_moves_are_not_flagged(store):
    days = _days(10)
    # a real -30% day both sources agree on is a market move, not a split
    px = [(d, 100.0 if i < 5 else 70.0) for i, d in enumerate(days)]
    _put(store, "AAPL", "yfinance", px)
    _put(store, "AAPL", "alpaca", px, adj=False)
    assert find_split_inconsistencies(store.conn) == []


def test_solo_jump_with_other_source_missing(store):
    days = _days(6)
    _put(store, "OXY", "yfinance", [(d, 50.0 if i < 3 else 24.0) for i, d in enumerate(days)])
    flags = find_split_inconsistencies(store.conn)
    assert [(f.date, f.kind) for f in flags] == [(days[3], "left_solo_jump")]
    # 35% alone (under 40%) with the other source missing is not flagged
    store.conn.execute("DELETE FROM prices")
    _put(store, "OXY", "yfinance", [(d, 50.0 if i < 3 else 32.5) for i, d in enumerate(days)])
    assert find_split_inconsistencies(store.conn) == []


def test_relative_ratio_catches_split_on_a_moving_day(store):
    # MSTR 10:1 on a +9% day: alpaca -89.1%, yfinance +9.1%
    days = _days(4)
    _put(store, "MSTR", "yfinance", [(days[0], 100.0), (days[1], 100.0), (days[2], 109.1)])
    _put(
        store, "MSTR", "alpaca", [(days[0], 1000.0), (days[1], 1000.0), (days[2], 109.1)], adj=False
    )
    flags = find_split_inconsistencies(store.conn)
    assert [f.kind for f in flags] == ["alpaca_raw_split"]


def test_feature_series_reads_what_the_model_reads(store):
    days = _mnst(store)
    flags = find_split_inconsistencies(store.conn, left_sql=FEATURE_SERIES_SQL)
    assert [(f.date, f.kind) for f in flags if f.feature_affecting] == [
        (days[5], "left_scale_break")
    ]


def test_quality_check_degraded_names_tickers_and_pages(store):
    days = _mnst(store)
    chk = _check_no_split_inconsistency(store, days[-1])
    assert chk.passed and not chk.blocking and chk.degraded
    assert "MNST" in chk.detail and "repair-splits" in chk.detail
    report = run_quality_checks(store, asof_date=days[-1], universe=["MNST"], run_id=1)
    names = [c.name for c in report.checks]
    assert "no_split_inconsistency" in names
    sent = []
    notified = notify_degraded_quality_checks(
        report, asof=days[-1], notify_fn=lambda **kw: sent.append(kw)
    )
    assert "no_split_inconsistency" in notified
    assert any("MNST" in m["message"] for m in sent)


def test_quality_check_clean_when_only_raw_alpaca_split(store):
    days = _days(14)
    _put(store, "MNST", "yfinance", [(d, 48.0) for d in days])
    _put(
        store,
        "MNST",
        "alpaca",
        [(d, 96.0 if i < 10 else 48.0) for i, d in enumerate(days)],
        adj=False,
    )
    chk = _check_no_split_inconsistency(store, days[-1])
    assert chk.passed and not chk.degraded


def test_quality_check_threshold_knob_and_lookback(store):
    days = _days(10)
    # a 25% disagreement: flagged at 20%, not at 30%
    _put(store, "X", "yfinance", [(d, 100.0 if i < 5 else 75.0) for i, d in enumerate(days)])
    _put(store, "X", "alpaca", [(d, 100.0) for d in days], adj=False)
    assert _check_no_split_inconsistency(store, days[-1], threshold=0.20).degraded
    assert not _check_no_split_inconsistency(store, days[-1], threshold=0.30).degraded
    # outside the trailing window -> not flagged
    assert not _check_no_split_inconsistency(store, days[-1], sessions=3).degraded
    report = run_quality_checks(
        store, asof_date=days[-1], universe=["X"], run_id=1, split_threshold=0.30
    )
    chk = next(c for c in report.checks if c.name == "no_split_inconsistency")
    assert not chk.degraded


def test_config_knob_default():
    from sma.config import IngestConfig

    cfg = IngestConfig(
        default_lookback_days=1,
        rate_limits={
            "finnhub": {"requests_per_minute": 1},
            "newsapi": {"requests_per_day": 1},
            "edgar": {"requests_per_second": 1},
        },
        retries={"max": 1, "base_delay": 1.0, "jitter": 0.0},
        circuit_breaker={"failures_to_open": 1, "cooldown_minutes": 1},
    )
    assert cfg.split_inconsistency_threshold == 0.20
