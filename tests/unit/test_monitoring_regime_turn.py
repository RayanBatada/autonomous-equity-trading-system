"""Tests for sma.monitoring.check_regime_turn -- the trailing-21-decide-date
IC regime-turn crossing detector added 2026-08-25.

Reuses sma.eval.live_ic.trailing_ic_regime (the SAME function
dashboard/tabs/model.py's Model Edge section uses, commit 3bd5f87) -- these
tests mostly drive it with synthetic IC series (order/values only matter,
not a real DB) to prove the CROSSING state machine: fires once on a
crossing, silent in steady state, fires again on a recross. The last test
wires a small real DuckDB through the production default path
(ic_series=None) to prove the DB plumbing (sma.eval.live_ic.model_edge_ic_df)
is actually reached, not just the injected-series unit tests above it.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from loguru import logger as _loguru_logger

from sma.monitoring import check_regime_turn
from sma.monitoring.regime_state import read_regime_state

ASOF = date(2026, 8, 25)


def _positive_series(n_tail: int = 21) -> pd.Series:
    # trailing_ic_regime's default window is 21; matches the pattern used in
    # tests/unit/dashboard/test_model_edge_ic.py's own regime tests.
    return pd.Series([0.05] * (n_tail - 1) + [0.049])


def _negative_series(n_tail: int = 21) -> pd.Series:
    return pd.Series([-0.05] * (n_tail - 1) + [-0.049])


def _neutral_series(n_tail: int = 21) -> pd.Series:
    # Alternating sign -> mean ~0, well inside the |t|<2 neutral band.
    vals = ([0.05, -0.05] * ((n_tail // 2) + 1))[:n_tail]
    return pd.Series(vals)


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(
        "SMA_REGIME_STATE_PATH", str(tmp_path / "state" / "regime_ic_10d.json")
    )


def test_crossing_to_positive_fires_once():
    notifies = []
    fired = check_regime_turn(
        asof=ASOF,
        ic_series=_positive_series(),
        notify_fn=lambda title, message: notifies.append((title, message)),
    )

    assert fired == "positive"
    assert len(notifies) == 1
    title, message = notifies[0]
    assert "regime" in title.lower()
    assert "turned positive" in message
    assert "IC=" in message
    assert "t=" in message

    state = read_regime_state()
    assert state["level"] == "positive"
    assert state["asof"] == ASOF.isoformat()


def test_steady_state_is_silent():
    notifies = []
    notify_fn = lambda title, message: notifies.append((title, message))  # noqa: E731

    fired1 = check_regime_turn(asof=ASOF, ic_series=_positive_series(), notify_fn=notify_fn)
    fired2 = check_regime_turn(asof=ASOF, ic_series=_positive_series(), notify_fn=notify_fn)

    assert fired1 == "positive"
    assert fired2 is None
    assert len(notifies) == 1  # only the first call fired


def test_recross_to_negative_fires_again():
    notifies = []
    notify_fn = lambda title, message: notifies.append((title, message))  # noqa: E731

    check_regime_turn(asof=ASOF, ic_series=_positive_series(), notify_fn=notify_fn)
    fired = check_regime_turn(asof=ASOF, ic_series=_negative_series(), notify_fn=notify_fn)

    assert fired == "negative"
    assert len(notifies) == 2
    assert "turned negative" in notifies[1][1]


def test_neutral_wobble_does_not_fire_but_rearms_the_next_positive_crossing():
    notifies = []
    notify_fn = lambda title, message: notifies.append((title, message))  # noqa: E731

    check_regime_turn(asof=ASOF, ic_series=_positive_series(), notify_fn=notify_fn)  # fires
    fired_neutral = check_regime_turn(
        asof=ASOF, ic_series=_neutral_series(), notify_fn=notify_fn
    )
    assert fired_neutral is None
    assert len(notifies) == 1
    assert read_regime_state()["level"] == "neutral"

    # Re-entering "positive" from "neutral" is a real crossing again.
    fired_again = check_regime_turn(
        asof=ASOF, ic_series=_positive_series(), notify_fn=notify_fn
    )
    assert fired_again == "positive"
    assert len(notifies) == 2


def test_no_signal_yet_returns_none_and_does_not_notify():
    notifies = []
    fired = check_regime_turn(
        asof=ASOF,
        ic_series=pd.Series(dtype=float),
        notify_fn=lambda title, message: notifies.append((title, message)),
    )

    assert fired is None
    assert notifies == []


def test_never_raises_on_a_broken_series():
    # Not a pandas Series at all -- .dropna()/.tail() would blow up;
    # check_regime_turn must swallow it and return None, not crash the rest
    # of the evening monitoring run.
    fired = check_regime_turn(asof=ASOF, ic_series="not-a-series", notify_fn=lambda t, m: None)
    assert fired is None


# ---------------------------------------------------------------------------
# Real DB wiring: ic_series=None (the production default) reaches
# sma.eval.live_ic.model_edge_ic_df, the SAME function the dashboard uses --
# proving check_regime_turn REUSES that computation rather than reimplementing
# its own DB queries.
# ---------------------------------------------------------------------------


def _make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "t.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE predictions (
            asof_date DATE, ticker VARCHAR, target VARCHAR,
            predicted_value DOUBLE, model_id VARCHAR,
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE prices (
            ticker VARCHAR, date DATE, open DOUBLE, high DOUBLE, low DOUBLE,
            close DOUBLE, adj_close DOUBLE, volume BIGINT, source VARCHAR,
            run_id BIGINT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE intended_orders (
            intended_order_id VARCHAR, asof_date DATE, ticker VARCHAR,
            side VARCHAR, target_shares INTEGER, source VARCHAR,
            status VARCHAR, run_id BIGINT
        )
        """
    )
    con.close()
    return db_path


def _make_universe_yaml(tmp_path: Path, tickers: list[str]) -> Path:
    path = tmp_path / "universe.yaml"
    body = "\n".join(f"    - {t}" for t in tickers)
    path.write_text(f"universe:\n  tickers:\n{body}\n")
    return path


def _seed_positive_regime_db(db_path: Path, tickers: list[str]) -> list[date]:
    """Populate `db_path` with synthetic predictions/prices that produce a
    real 'positive' trailing-IC regime read through the production DB-wired
    path (model_edge_ic_df) -- shared by the wiring test and the lock-retry
    test below, both of which need a real query to succeed against a real
    DuckDB file (not an injected ic_series).

    predicted_value is the ticker's fixed rank (0..5); prices follow that
    same rank as a drift PLUS i.i.d. noise, so realized forward returns
    correlate with the ranking on average but not perfectly on every single
    date (a deterministic exact-copy design gives Spearman IC=1.0 on every
    date -- zero variance in the IC series itself, which trailing_ic_regime
    correctly reports as "insufficient" since a t-stat needs std>0; real
    live IC is never that clean). A fixed seed keeps this reproducible.
    Returns the decide dates inserted (needed as the `asof` for the check).
    """
    import numpy as np

    rng = np.random.default_rng(42)
    n_days = 60
    horizon = 10
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(n_days)]

    con = duckdb.connect(str(db_path))

    # Per-ticker daily drift monotonic in rank i, plus noise on the level
    # each day -- breaks perfect rank-monotonicity so realized Spearman IC
    # varies day to day instead of being a constant 1.0.
    price = {tkr: 100.0 + i * 5 for i, tkr in enumerate(tickers)}
    for d in dates:
        for i, tkr in enumerate(tickers):
            drift = (i + 1) * 0.15
            noise = rng.normal(0, 1.0)
            price[tkr] = max(1.0, price[tkr] + drift + noise)
            px = price[tkr]
            con.execute(
                "INSERT INTO prices (ticker, date, open, high, low, close, adj_close, "
                " volume, source, run_id) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 'yfinance', 1)",
                [tkr, d, px, px, px, px, px],
            )

    # Decide dates: every date that leaves room for a horizon-10 forward
    # return within the price history above.
    decide_dates = [d for i, d in enumerate(dates) if i + 1 + horizon < n_days]
    assert len(decide_dates) >= 21  # need a full trailing window

    for d in decide_dates:
        for i, tkr in enumerate(tickers):
            con.execute(
                "INSERT INTO predictions (asof_date, ticker, target, predicted_value, "
                " model_id) VALUES (?, ?, 'ret_30d_forward', ?, 'm1')",
                [d, tkr, float(i)],
            )
    for d in decide_dates:
        con.execute(
            "INSERT INTO intended_orders (intended_order_id, asof_date, ticker, side, "
            " target_shares, source, status, run_id) "
            "VALUES (gen_random_uuid(), ?, ?, 'buy', 1, 'test', 'filled', 1)",
            [d, tickers[0]],
        )
    con.close()
    return decide_dates


def test_check_regime_turn_wires_into_live_db_and_fires_on_real_signal(tmp_path: Path):
    """Proves the ic_series=None default path actually reaches the DB via
    sma.eval.live_ic.model_edge_ic_df (not a reimplementation)."""
    tickers = ["T0", "T1", "T2", "T3", "T4", "T5"]
    db_path = _make_db(tmp_path)
    universe_path = _make_universe_yaml(tmp_path, tickers)
    decide_dates = _seed_positive_regime_db(db_path, tickers)

    notifies = []
    fired = check_regime_turn(
        asof=decide_dates[-1],
        db_path=db_path,
        universe_path=universe_path,
        notify_fn=lambda title, message: notifies.append((title, message)),
    )

    assert fired == "positive"
    assert len(notifies) == 1


# ---------------------------------------------------------------------------
# Lock-conflict retry (2026-09-09 03:15 and 2026-09-10 23:01 monitoring.err.log
# incidents): "monitoring: regime-turn check errored: IOException('IO Error:
# Could not set lock on file ... Conflicting lock is held in
# /opt/anaconda3/bin/python3.12 ...')". Audit (`grep -rn "duckdb.connect"
# src/sma/`) found every read-only open on this path (sma.eval.live_ic's
# decide_dates/live_predictions_df/ic_prices_df) already routes through
# db_connect.read_only_connect -- both incidents were the retry budget
# (~90s) being exhausted by that external anaconda process's writable hold,
# not a bypass of the helper. This test locks the routing in as a
# regression guard: a lock conflict on only the FIRST open attempt (a truly
# transient overlap) must be retried and absorbed, not surfaced as
# "regime-turn check errored".
# ---------------------------------------------------------------------------


def test_regime_check_retries_a_transient_lock_and_does_not_error(tmp_path, monkeypatch):
    """Spies on the real duckdb.connect (not a stub) to prove two things about
    the production DB-wired path: every open request is read_only=True (i.e.
    routed through read_only_connect, not a raw/writable open), and a lock
    conflict on the very first attempt is retried and succeeds -- the regime
    check completes normally (state file written, correct crossing fires)
    instead of logging 'regime-turn check errored' and silently skipping the
    night's read."""
    tickers = ["T0", "T1", "T2", "T3", "T4", "T5"]
    db_path = _make_db(tmp_path)
    universe_path = _make_universe_yaml(tmp_path, tickers)
    decide_dates = _seed_positive_regime_db(db_path, tickers)

    real_connect = duckdb.connect
    read_only_flags: list[bool] = []

    def flaky_connect(path, read_only=False):
        read_only_flags.append(read_only)
        if len(read_only_flags) == 1:  # transient: only the first attempt is locked
            raise duckdb.IOException(
                f'IO Error: Could not set lock on file "{path}": Conflicting '
                "lock is held in /opt/anaconda3/bin/python3.12 (PID 1674) by "
                "user youruser. See also "
                "https://duckdb.org/docs/stable/connect/concurrency"
            )
        return real_connect(path, read_only=read_only)

    monkeypatch.setattr("sma.db_connect.time.sleep", lambda s: None)
    monkeypatch.setattr("sma.db_connect.duckdb.connect", flaky_connect)

    warnings: list[str] = []
    sink_id = _loguru_logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")
    try:
        notifies = []
        fired = check_regime_turn(
            asof=decide_dates[-1],
            db_path=db_path,
            universe_path=universe_path,
            notify_fn=lambda title, message: notifies.append((title, message)),
        )
    finally:
        _loguru_logger.remove(sink_id)

    assert len(read_only_flags) >= 2  # it actually retried, not a lucky first pass
    assert all(read_only_flags)  # every open on this path is read-only
    assert fired == "positive"
    assert len(notifies) == 1
    state = read_regime_state()
    assert state is not None  # the state WAS written -- the read succeeded
    assert state["level"] == "positive"
    assert not any("errored" in w for w in warnings)
