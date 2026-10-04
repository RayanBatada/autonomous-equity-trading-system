"""Model Edge (live IC) section (dashboard/tabs/model.py), added 2026-08-14
per Rayan's ask: a rolling live cross-sectional rank-IC read on the Model
tab, computed from STORED predictions joined to realized forward returns
from STORED prices (never re-invoking the model) -- so the dashboard shows
exactly what the live pipeline actually produced.

Covers:
  - _forward_returns: realized N-session forward return per (asof, ticker),
    using the SAME entry/exit convention as sma.model.loader.
    build_training_set's label (next session's adjusted open -> adj_close
    N sessions later) -- "realized forward return" means the same thing
    everywhere in this codebase.
  - _rank_ic_series: per-decide-date cross-sectional Spearman IC between
    prediction ranks and realized forward returns.
  - _smoothed_ic / _trailing_ic_regime: the light smoothing overlay and the
    trailing-21-session regime read (mean + overlap-caveat t-stat).
  - _decide_dates / _live_predictions_df / _ic_prices_df / _model_edge_ic_df:
    the DB-backed wiring -- real decide dates come from intended_orders
    (decide.py's own output table), NOT every predictions.asof_date, which
    also holds `sma model backfill-predictions` walk-forward evaluation rows
    never actually used to trade.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from dashboard.tabs import model


@pytest.fixture(autouse=True)
def _clear_model_edge_ic_caches():
    # st.cache_data caches by (function identity, args); these take no args
    # (or just an int horizon), so a stale result from a prior test's
    # monkeypatched DB_PATH/_UNIVERSE_PATH would otherwise leak across tests.
    for fn in (
        model._decide_dates,
        model._live_predictions_df,
        model._ic_prices_df,
        model._model_edge_ic_df,
    ):
        fn.clear()
    yield
    for fn in (
        model._decide_dates,
        model._live_predictions_df,
        model._ic_prices_df,
        model._model_edge_ic_df,
    ):
        fn.clear()


def _dates(n: int, start: date = date(2026, 5, 4)) -> list[date]:
    """n consecutive calendar dates (the pure functions only care about
    ordering, not a real trading calendar)."""
    return [start + timedelta(days=i) for i in range(n)]


# ── _forward_returns (pure) ───────────────────────────────────────────────


def test_forward_returns_basic_entry_and_exit_math():
    d = _dates(6)
    # Full calendar coverage (baseline) so trading-day *position* lines up
    # with the intended offsets -- _forward_returns builds its calendar from
    # the dates actually present in `prices`, not a fabricated date range.
    prices = pd.DataFrame(
        [("A", dd, 100.0, 100.0, 100.0) for dd in d],
        columns=["ticker", "date", "open", "close", "adj_close"],
    )
    prices.loc[prices["date"] == d[3], "adj_close"] = 110.0  # exit, 3 sessions after asof=d[0]

    out = model._forward_returns(prices, [d[0]], horizon_days=3)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["ticker"] == "A"
    assert row["asof_date"] == d[0]
    assert row["fwd_return"] == pytest.approx(0.10)


def test_forward_returns_uses_split_adjusted_open_not_raw_open():
    """entry_px = open * adj_close/close on the entry day -- a raw open would
    be wrong across a split/dividend adjustment factor."""
    d = _dates(5)
    prices = pd.DataFrame(
        [("A", dd, 100.0, 100.0, 100.0) for dd in d],
        columns=["ticker", "date", "open", "close", "adj_close"],
    )
    # entry day (d[1], asof=d[0]+1): raw open 100, but adj_close/close factor
    # is 0.5 (a 2-for-1 split priced in) -> adjusted entry = 50.
    prices.loc[prices["date"] == d[1], ["open", "close", "adj_close"]] = [100.0, 100.0, 50.0]
    prices.loc[prices["date"] == d[3], "adj_close"] = 55.0  # exit, 3 sessions after asof=d[0]

    out = model._forward_returns(prices, [d[0]], horizon_days=3)

    assert out.iloc[0]["fwd_return"] == pytest.approx(55.0 / 50.0 - 1.0)


def test_forward_returns_excludes_asof_too_close_to_end_of_history():
    """A decide date within `horizon_days` sessions of the end of the price
    history has no realized return YET -- it must be absent, never a
    zero-filled/fabricated point."""
    d = _dates(5)
    prices = pd.DataFrame(
        [(t, dd, 100.0, 100.0, 100.0) for t in ["A"] for dd in d],
        columns=["ticker", "date", "open", "close", "adj_close"],
    )

    # asof = d[3] (idx 3): entry idx 4, future idx would be 3+10=13, way past
    # the 5-date history -> no row.
    out = model._forward_returns(prices, [d[3]], horizon_days=10)

    assert out.empty
    assert list(out.columns) == ["asof_date", "ticker", "fwd_return"]


def test_forward_returns_skips_ticker_missing_entry_or_future_price():
    """A ticker with a prediction that lacks a price row on the entry/future
    date (gap, late listing, etc.) is excluded from that date's output --
    never an error, never a fabricated value."""
    d = _dates(4)
    prices = pd.DataFrame(
        [
            ("A", d[0], 100.0, 100.0, 100.0),
            ("A", d[1], 100.0, 100.0, 100.0),
            ("A", d[2], 100.0, 100.0, 100.0),
            ("A", d[3], 100.0, 100.0, 105.0),
            ("B", d[1], 50.0, 50.0, 50.0),
            # B has no row on d[3] (the future date) -- e.g. delisted/gap.
        ],
        columns=["ticker", "date", "open", "close", "adj_close"],
    )

    out = model._forward_returns(prices, [d[0]], horizon_days=3)

    assert set(out["ticker"]) == {"A"}


def test_forward_returns_skips_non_positive_entry_price():
    d = _dates(4)
    prices = pd.DataFrame(
        [
            ("A", d[0], 100.0, 100.0, 100.0),
            ("A", d[1], 0.0, 0.0, 0.0),  # non-positive entry -- must skip
            ("A", d[2], 100.0, 100.0, 100.0),
            ("A", d[3], 100.0, 100.0, 105.0),
        ],
        columns=["ticker", "date", "open", "close", "adj_close"],
    )

    out = model._forward_returns(prices, [d[0]], horizon_days=3)

    assert out.empty


def test_forward_returns_empty_prices_returns_empty_frame_with_columns():
    out = model._forward_returns(pd.DataFrame(), [date(2026, 5, 4)], horizon_days=10)

    assert out.empty
    assert list(out.columns) == ["asof_date", "ticker", "fwd_return"]


def test_forward_returns_unknown_asof_date_is_simply_absent():
    d = _dates(6)
    prices = pd.DataFrame(
        [("A", dd, 100.0, 100.0, 100.0) for dd in d],
        columns=["ticker", "date", "open", "close", "adj_close"],
    )
    not_a_trading_date = date(2099, 1, 1)

    out = model._forward_returns(prices, [not_a_trading_date], horizon_days=2)

    assert out.empty


# ── _rank_ic_series (pure) ────────────────────────────────────────────────


def _perfect_rank_frames(asof: date, n: int = 6):
    tickers = [f"T{i}" for i in range(n)]
    preds = pd.DataFrame(
        {"asof_date": [asof] * n, "ticker": tickers, "predicted_value": list(range(n))}
    )
    fwd = pd.DataFrame(
        {"asof_date": [asof] * n, "ticker": tickers, "fwd_return": [float(i) for i in range(n)]}
    )
    return preds, fwd


def test_rank_ic_series_perfect_monotonic_relationship_gives_ic_near_one():
    asof = date(2026, 5, 4)
    preds, fwd = _perfect_rank_frames(asof, n=6)

    out = model._rank_ic_series(preds, fwd)

    assert len(out) == 1
    assert out.iloc[0]["asof_date"] == asof
    assert out.iloc[0]["ic"] == pytest.approx(1.0)
    assert out.iloc[0]["n"] == 6


def test_rank_ic_series_excludes_dates_below_min_cross_section():
    asof = date(2026, 5, 4)
    preds, fwd = _perfect_rank_frames(asof, n=3)  # below the min_n=5 floor

    out = model._rank_ic_series(preds, fwd, min_n=5)

    assert out.empty


def test_rank_ic_series_drops_tickers_missing_a_realized_return():
    """A prediction with no matching forward-return row (inner join) just
    drops out of that date's cross-section -- non-universe/missing-price
    names are excluded, never an error."""
    asof = date(2026, 5, 4)
    preds, fwd = _perfect_rank_frames(asof, n=6)
    fwd = fwd[fwd["ticker"] != "T0"].reset_index(drop=True)  # T0 has no price

    out = model._rank_ic_series(preds, fwd, min_n=5)

    assert len(out) == 1
    assert out.iloc[0]["n"] == 5


def test_rank_ic_series_constant_predictions_skip_the_date():
    """Zero-variance predictions make Spearman undefined (NaN) -- skip that
    date's point rather than emit a NaN that would poison the mean/t-stat."""
    asof = date(2026, 5, 4)
    n = 6
    tickers = [f"T{i}" for i in range(n)]
    preds = pd.DataFrame(
        {"asof_date": [asof] * n, "ticker": tickers, "predicted_value": [0.01] * n}
    )
    fwd = pd.DataFrame(
        {"asof_date": [asof] * n, "ticker": tickers, "fwd_return": [float(i) for i in range(n)]}
    )

    out = model._rank_ic_series(preds, fwd)

    assert out.empty


def test_rank_ic_series_computes_independently_per_date():
    asof1, asof2 = date(2026, 5, 4), date(2026, 5, 5)
    preds1, fwd1 = _perfect_rank_frames(asof1, n=6)
    preds2, fwd2 = _perfect_rank_frames(asof2, n=6)
    # Invert the relationship on the second date.
    fwd2 = fwd2.assign(fwd_return=fwd2["fwd_return"].to_numpy()[::-1])
    preds = pd.concat([preds1, preds2], ignore_index=True)
    fwd = pd.concat([fwd1, fwd2], ignore_index=True)

    out = model._rank_ic_series(preds, fwd).set_index("asof_date")

    assert out.loc[asof1, "ic"] == pytest.approx(1.0)
    assert out.loc[asof2, "ic"] == pytest.approx(-1.0)


def test_rank_ic_series_empty_input_returns_empty_frame_with_columns():
    out = model._rank_ic_series(pd.DataFrame(), pd.DataFrame())

    assert out.empty
    assert list(out.columns) == ["asof_date", "ic", "n"]


# ── _smoothed_ic (pure) ────────────────────────────────────────────────────


def test_smoothed_ic_matches_plain_rolling_mean():
    s = pd.Series([0.01, 0.02, 0.03, -0.01, 0.04, 0.05])

    out = model._smoothed_ic(s, window=3, min_periods=1)

    expected = s.rolling(window=3, min_periods=1).mean()
    pd.testing.assert_series_equal(out, expected)


# ── _trailing_ic_regime (pure) ─────────────────────────────────────────────


def test_trailing_ic_regime_positive_when_mean_positive_and_tstat_above_2():
    ic = pd.Series([0.05] * 20 + [0.049])

    out = model._trailing_ic_regime(ic, window=21)

    assert out["level"] == "positive"
    assert out["mean"] > 0
    assert out["t_stat"] > 2
    assert out["n"] == 21


def test_trailing_ic_regime_negative_when_mean_negative_and_tstat_below_neg2():
    ic = pd.Series([-0.05] * 20 + [-0.049])

    out = model._trailing_ic_regime(ic, window=21)

    assert out["level"] == "negative"
    assert out["mean"] < 0
    assert out["t_stat"] < -2


def test_trailing_ic_regime_neutral_when_tstat_inconclusive():
    ic = pd.Series(([0.05, -0.05] * 10) + [0.0])  # mean == 0 exactly

    out = model._trailing_ic_regime(ic, window=21)

    assert out["level"] == "neutral"
    assert out["mean"] == pytest.approx(0.0)


def test_trailing_ic_regime_insufficient_with_fewer_than_two_points():
    out = model._trailing_ic_regime(pd.Series([0.05]), window=21)

    assert out["level"] == "insufficient"
    assert out["t_stat"] is None


def test_trailing_ic_regime_insufficient_with_empty_series():
    out = model._trailing_ic_regime(pd.Series(dtype=float), window=21)

    assert out["level"] == "insufficient"
    assert out["n"] == 0


def test_trailing_ic_regime_only_uses_the_trailing_window():
    # 21 strongly negative points, then 21 strongly positive points -- the
    # regime read must reflect only the trailing window, not full history.
    ic = pd.Series([-0.10] * 21 + [0.10] * 21)

    out = model._trailing_ic_regime(ic, window=21)

    assert out["level"] == "positive"
    assert out["n"] == 21


# ── DB-backed: _decide_dates / _live_predictions_df / _ic_prices_df ────────


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


def _seed_prediction(con, asof, ticker, target, value, model_id) -> None:
    con.execute(
        "INSERT INTO predictions (asof_date, ticker, target, predicted_value, model_id) "
        "VALUES (?, ?, ?, ?, ?)",
        [asof, ticker, target, value, model_id],
    )


def _seed_price(con, ticker, d, o, c, ac, source="yfinance") -> None:
    con.execute(
        "INSERT INTO prices (ticker, date, open, high, low, close, adj_close, "
        " volume, source, run_id) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, 1)",
        [ticker, d, o, o, o, c, ac, source],
    )


def _seed_intended_order(con, asof, ticker="AAPL") -> None:
    con.execute(
        "INSERT INTO intended_orders (intended_order_id, asof_date, ticker, side, "
        " target_shares, source, status, run_id) "
        "VALUES (gen_random_uuid(), ?, ?, 'buy', 1, 'test', 'filled', 1)",
        [asof, ticker],
    )


def test_decide_dates_reads_from_intended_orders_not_every_prediction_date(monkeypatch, tmp_path):
    """`predictions` also holds `sma model backfill-predictions` walk-forward
    evaluation rows never actually used to trade -- decide dates must come
    from intended_orders (decide.py's own output), not blindly from every
    predictions.asof_date."""
    db_path = _make_db(tmp_path)
    monkeypatch.setattr(model, "_DB_PATH", db_path)
    con = duckdb.connect(str(db_path))
    _seed_prediction(con, date(2025, 7, 1), "AAPL", "ret_30d_forward", 0.01, "m0")  # backfill-era
    _seed_prediction(con, date(2026, 5, 4), "AAPL", "ret_30d_forward", 0.02, "m1")  # real decide
    _seed_intended_order(con, date(2026, 5, 4))
    con.close()

    out = model._decide_dates()

    assert out == [date(2026, 5, 4)]


def test_live_predictions_df_excludes_spy_and_uses_latest_model_id(monkeypatch, tmp_path):
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "SPY"]
    db_path = _make_db(tmp_path)
    universe_path = _make_universe_yaml(tmp_path, tickers)
    monkeypatch.setattr(model, "_DB_PATH", db_path)
    monkeypatch.setattr(model, "_UNIVERSE_PATH", universe_path)
    con = duckdb.connect(str(db_path))
    asof = date(2026, 5, 4)
    _seed_intended_order(con, asof)
    for t in ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]:
        _seed_prediction(con, asof, t, "ret_30d_forward", 0.01, "m1")
    # AAPL has a second, later-trained model's prediction -- the live read
    # must use the latest model_id, matching the convention already used in
    # sma.agents.__main__ (ORDER BY model_id DESC LIMIT 1).
    _seed_prediction(con, asof, "AAPL", "ret_30d_forward", 0.99, "m2")
    _seed_prediction(con, asof, "SPY", "ret_30d_forward", 0.5, "m1")
    con.close()

    out = model._live_predictions_df()

    assert "SPY" not in set(out["ticker"])
    aapl_row = out[out["ticker"] == "AAPL"].iloc[0]
    assert aapl_row["predicted_value"] == pytest.approx(0.99)
    assert set(out["ticker"]) == {"AAPL", "MSFT", "GOOG", "AMZN", "TSLA"}


def test_live_predictions_df_restricted_to_decide_dates(monkeypatch, tmp_path):
    tickers = ["AAPL", "MSFT"]
    db_path = _make_db(tmp_path)
    universe_path = _make_universe_yaml(tmp_path, tickers)
    monkeypatch.setattr(model, "_DB_PATH", db_path)
    monkeypatch.setattr(model, "_UNIVERSE_PATH", universe_path)
    con = duckdb.connect(str(db_path))
    _seed_intended_order(con, date(2026, 5, 4))
    _seed_prediction(con, date(2026, 5, 4), "AAPL", "ret_30d_forward", 0.01, "m1")
    # Backfill-era date, no matching intended_orders row -- must be excluded.
    _seed_prediction(con, date(2025, 7, 1), "AAPL", "ret_30d_forward", 0.02, "m0")
    con.close()

    out = model._live_predictions_df()

    assert set(out["asof_date"]) == {date(2026, 5, 4)}


def test_ic_prices_df_prefers_yfinance_over_other_sources(monkeypatch, tmp_path):
    tickers = ["AAPL"]
    db_path = _make_db(tmp_path)
    universe_path = _make_universe_yaml(tmp_path, tickers)
    monkeypatch.setattr(model, "_DB_PATH", db_path)
    monkeypatch.setattr(model, "_UNIVERSE_PATH", universe_path)
    con = duckdb.connect(str(db_path))
    d = date(2026, 5, 4)
    _seed_price(con, "AAPL", d, 100.0, 100.0, 100.0, source="alpaca")
    _seed_price(con, "AAPL", d, 200.0, 200.0, 200.0, source="yfinance")
    con.close()

    out = model._ic_prices_df()

    assert len(out) == 1
    assert out.iloc[0]["open"] == pytest.approx(200.0)


# ── DB-backed: _model_edge_ic_df end to end ────────────────────────────────


def test_model_edge_ic_df_end_to_end_wires_predictions_to_realized_returns(monkeypatch, tmp_path):
    """Full pipeline: seeded predictions perfectly rank-correlate with the
    realized forward returns implied by seeded prices -- IC should come back
    ~1.0 for the one decide date old enough to have a realized return."""
    n = 6
    tickers = [f"T{i}" for i in range(n)]
    db_path = _make_db(tmp_path)
    universe_path = _make_universe_yaml(tmp_path, tickers)
    monkeypatch.setattr(model, "_DB_PATH", db_path)
    monkeypatch.setattr(model, "_UNIVERSE_PATH", universe_path)
    con = duckdb.connect(str(db_path))

    trading_dates = _dates(8, start=date(2026, 5, 4))
    asof = trading_dates[0]
    future_date = trading_dates[5]  # horizon_days=5 -> asof_idx(0) + 5 = idx 5

    _seed_intended_order(con, asof, ticker=tickers[0])
    for i, t in enumerate(tickers):
        _seed_prediction(con, asof, t, "ret_30d_forward", float(i), "m1")
        # Full calendar coverage (baseline) so trading-day *position* lines
        # up with the intended offsets, then override the future date's
        # price so higher predicted rank -> higher realized forward return.
        for dd in trading_dates:
            price = 100.0 + float(i) if dd == future_date else 100.0
            _seed_price(con, t, dd, 100.0, 100.0, price)
    con.close()

    out = model._model_edge_ic_df(horizon_days=5)

    assert len(out) == 1
    assert out.iloc[0]["asof_date"] == asof
    assert out.iloc[0]["ic"] == pytest.approx(1.0)


def test_model_edge_ic_df_no_decide_dates_returns_empty_frame_not_crash(monkeypatch, tmp_path):
    db_path = _make_db(tmp_path)
    universe_path = _make_universe_yaml(tmp_path, ["AAPL"])
    monkeypatch.setattr(model, "_DB_PATH", db_path)
    monkeypatch.setattr(model, "_UNIVERSE_PATH", universe_path)

    out = model._model_edge_ic_df(horizon_days=10)

    assert out.empty


def test_model_edge_ic_df_missing_db_returns_empty_frame_not_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(model, "_DB_PATH", tmp_path / "does-not-exist.duckdb")
    monkeypatch.setattr(model, "_UNIVERSE_PATH", tmp_path / "does-not-exist.yaml")

    out = model._model_edge_ic_df(horizon_days=10)

    assert out.empty
