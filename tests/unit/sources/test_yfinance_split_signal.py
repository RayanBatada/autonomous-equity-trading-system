"""2026-10-01: any window row whose close moved >20% vs the stored close for the
same date is a split signal -> full-history refetch inside the nightly ingest
(MNST 2:1 on 2026-08-11 slipped past the oldest-row adj_close check)."""

from datetime import date

import pandas as pd
import pytest

import sma.ingest.sources.yfinance_prices as mod
from sma.ingest.store import Store


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    yield s
    s.close()


def _df(closes: dict[str, float]) -> pd.DataFrame:
    idx = pd.to_datetime(list(closes))
    v = list(closes.values())
    return pd.DataFrame(
        {"Open": v, "High": v, "Low": v, "Close": v, "Adj Close": v, "Volume": [1] * len(v)},
        index=idx,
    )


def _stored(store, run_id, closes: dict[str, float]):
    for d, c in closes.items():
        store.conn.execute(
            "INSERT INTO prices VALUES ('MNST', ?, ?, ?, ?, ?, ?, 1, 'yfinance', ?)",
            [d, c, c, c, c, c, run_id],
        )


def _run(store, monkeypatch, window, full):
    calls = []

    def fake_download(tickers, **kw):
        calls.append(kw.get("start"))
        return window if len(calls) == 1 else full

    monkeypatch.setattr(mod.yf, "download", fake_download)
    src = mod.YFinancePricesSource(lookback_days=5)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")
    res = src.fetch(["MNST"], date(2026, 7, 23), store, run_id)
    return res, calls


def test_mid_window_restatement_triggers_full_refetch(store, monkeypatch):
    # oldest overlap row (7/17) agrees, so the old adj_close check is silent;
    # 7/20 and later come back on the post-split scale vs stored pre-split.
    _stored(
        store, 1, {"2026-07-14": 98.0, "2026-07-17": 97.5, "2026-07-20": 95.4, "2026-07-21": 94.5}
    )
    window = _df({"2026-07-17": 97.5, "2026-07-20": 47.7, "2026-07-21": 47.2, "2026-07-22": 47.8})
    full = _df(
        {
            "2026-07-14": 49.0,
            "2026-07-17": 48.75,
            "2026-07-20": 47.7,
            "2026-07-21": 47.2,
            "2026-07-22": 47.8,
        }
    )
    res, calls = _run(store, monkeypatch, window, full)
    assert res.status == "ok"
    assert calls == [calls[0], "2016-01-01"], "split signal must refetch full history"
    got = dict(
        store.conn.execute(
            "SELECT CAST(date AS VARCHAR), close FROM prices WHERE ticker='MNST' "
            "AND source='yfinance'"
        ).fetchall()
    )
    assert got["2026-07-14"] == pytest.approx(49.0)
    assert got["2026-07-17"] == pytest.approx(48.75)


def test_small_revision_is_not_a_split_signal(store, monkeypatch):
    _stored(store, 1, {"2026-07-17": 97.5, "2026-07-20": 95.4})
    window = _df({"2026-07-17": 97.5, "2026-07-20": 90.0, "2026-07-21": 94.0})
    res, calls = _run(store, monkeypatch, window, window)
    assert res.status == "ok"
    assert len(calls) == 1
    got = store.conn.execute(
        "SELECT close FROM prices WHERE ticker='MNST' AND date='2026-07-20'"
    ).fetchone()[0]
    assert got == pytest.approx(90.0)
