"""python -m sma.ingest repair-splits (2026-10-01, MNST 2:1 on 2026-08-11)."""

import json
from datetime import date, timedelta

import pandas as pd
import pytest
from click.testing import CliRunner

import sma.ingest.repair_splits as rs
from sma.ingest.store import Store

DAYS = [d for d in (date(2026, 7, 13) + timedelta(days=i) for i in range(20)) if d.weekday() < 5]


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    yield s
    s.close()


def _seed(store):
    """yfinance: pre-split scale for the first 5 days (the bug); alpaca raw with
    the real split on day 10; legacy alpaca rows carry adj_close = raw close."""
    for i, d in enumerate(DAYS):
        y = 98.0 if i < 5 else 48.0
        a = 96.0 if i < 10 else 48.0
        store.conn.execute(
            "INSERT INTO prices VALUES ('MNST', ?, ?, ?, ?, ?, ?, 1, 'yfinance', 1)",
            [d, y, y, y, y, y],
        )
        store.conn.execute(
            "INSERT INTO prices VALUES ('MNST', ?, ?, ?, ?, ?, ?, 1, 'alpaca', 1)",
            [d, a, a, a, a, (a if i < 3 else None)],
        )


def _clean_df(n=None):
    days = DAYS if n is None else DAYS[:n]
    idx = pd.to_datetime(days)
    v = [48.0] * len(days)
    return pd.DataFrame(
        {"Open": v, "High": v, "Low": v, "Close": v, "Adj Close": v, "Volume": [1] * len(days)},
        index=idx,
    )


def _yf(store):
    return dict(
        store.conn.execute(
            "SELECT date, close FROM prices WHERE ticker='MNST' AND source='yfinance'"
        ).fetchall()
    )


def test_repair_replaces_history_and_clears_flag(store):
    _seed(store)
    rep = rs.repair_ticker(store, "MNST", run_id=7, dry_run=False, fetch_fn=lambda t: _clean_df())
    assert rep.status == "replaced"
    assert any("left_scale_break" in f for f in rep.flags_before)
    assert not any("left_scale_break" in f for f in rep.flags_after)
    assert any("alpaca_raw_split" in f for f in rep.flags_after)  # raw alpaca stays raw
    assert set(_yf(store).values()) == {48.0}
    assert rep.samples[DAYS[4]][:2] == (98.0, 48.0)
    run_ids = {
        r[0]
        for r in store.conn.execute("SELECT run_id FROM prices WHERE source='yfinance'").fetchall()
    }
    assert run_ids == {7}


def test_dry_run_writes_nothing(store):
    _seed(store)
    before = _yf(store)
    rep = rs.repair_ticker(store, "MNST", run_id=7, dry_run=True, fetch_fn=lambda t: _clean_df())
    assert rep.status.startswith("dry-run")
    assert not any("left_scale_break" in f for f in rep.flags_after)
    assert _yf(store) == before


def test_truncated_or_failed_refetch_is_not_written(store):
    _seed(store)
    before = _yf(store)
    rep = rs.repair_ticker(
        store, "MNST", run_id=7, dry_run=False, fetch_fn=lambda t: _clean_df(n=3)
    )
    assert rep.status.startswith("skipped") and _yf(store) == before

    def boom(t):
        raise OSError("dns")

    rep = rs.repair_ticker(store, "MNST", run_id=7, dry_run=False, fetch_fn=boom)
    assert rep.status.startswith("skipped: fetch failed") and _yf(store) == before


def test_worse_refetch_rolls_back(store):
    _seed(store)
    before = _yf(store)
    bad = _clean_df()
    bad.loc[bad.index[2:], ["Open", "High", "Low", "Close", "Adj Close"]] = 10.0
    bad.loc[bad.index[8:], ["Open", "High", "Low", "Close", "Adj Close"]] = 48.0
    rep = rs.repair_ticker(store, "MNST", run_id=7, dry_run=False, fetch_fn=lambda t: bad)
    assert rep.status.startswith("rolled back")
    assert _yf(store) == before


def test_null_legacy_alpaca_adj(store):
    _seed(store)
    assert rs.null_legacy_alpaca_adj(store, dry_run=True) == 3
    assert rs.null_legacy_alpaca_adj(store, dry_run=False) == 3
    assert (
        store.conn.execute(
            "SELECT COUNT(*) FROM prices WHERE source='alpaca' AND adj_close IS NOT NULL"
        ).fetchone()[0]
        == 0
    )


def test_fetch_full_history_is_explicitly_unadjusted_close(monkeypatch):
    seen = {}

    def fake_download(t, **kw):
        seen.update(kw)
        return _clean_df()

    import yfinance as yf

    monkeypatch.setattr(yf, "download", fake_download)
    rs.fetch_full_history("MNST", end=date(2026, 9, 30))
    assert seen["auto_adjust"] is False
    assert seen["start"] == "2016-01-01" and seen["end"] == "2026-10-01"


def test_cli_job_writes_log_and_sentinel(tmp_path, monkeypatch):
    from sma.ingest.__main__ import cli

    db = tmp_path / "t.duckdb"
    s = Store(db).connect()
    _seed(s)
    s.close()
    sent = tmp_path / "sentinels"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sent))
    monkeypatch.setattr(rs, "fetch_full_history", lambda t, end: _clean_df())

    r = CliRunner().invoke(cli, ["repair-splits", "--db", str(db), "--all-flagged", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "tickers=['MNST']" in r.output and "98.000 -> 48.000" in r.output
    assert not sent.exists()

    r = CliRunner().invoke(cli, ["repair-splits", "--db", str(db), "--tickers", "mnst"])
    assert r.exit_code == 0, r.output
    s = Store(db).connect()
    try:
        assert set(_yf(s).values()) == {48.0}
        src, status, rows = s.conn.execute(
            "SELECT source, status, rows_inserted FROM ingest_log "
            "WHERE source = 'yfinance_repair_splits'"
        ).fetchone()
        assert (status, rows) == ("ok", len(DAYS))
        assert (
            s.conn.execute(
                "SELECT COUNT(*) FROM prices WHERE source='alpaca' AND adj_close IS NOT NULL"
            ).fetchone()[0]
            == 0
        )
    finally:
        s.close()
    files = list(sent.glob("com.sma.ingest.repair_splits*"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["tickers"] == {"MNST": "replaced"}


def test_cli_requires_exactly_one_selector(tmp_path):
    from sma.ingest.__main__ import cli

    r = CliRunner().invoke(cli, ["repair-splits", "--db", str(tmp_path / "x.duckdb")])
    assert r.exit_code != 0
