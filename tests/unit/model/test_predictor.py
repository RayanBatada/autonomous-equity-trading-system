"""Tests for sma.model.predictor.Predictor.

TDD: these tests were written before the implementation.

DB setup pattern: create via Store, then INSERT directly.
Price history pattern: generate 300 rows of synthetic daily prices so that
all 12 features (including dist_from_52w_high which needs 252 rows) resolve.
"""

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from sma.features.builder import FEATURE_NAMES
from sma.ingest.store import Store
from sma.model.persistence import save_model
from sma.model.predictor import Predictor  # noqa: E402 (tested module)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_PRICE_ROWS = 300  # enough for dist_from_52w_high (252) + buffer


def _fresh_db(tmp_path: Path) -> Path:
    """Create a schema-initialized DuckDB; return its path."""
    db_path = tmp_path / "test.duckdb"
    store = Store(path=str(db_path))
    store.connect()
    store.close()
    return db_path


def _synthetic_prices(
    ticker: str,
    asof_date: date,
    n_rows: int = N_PRICE_ROWS,
    seed: int = 0,
) -> list[dict]:
    """Generate n_rows rows of fake daily OHLCV data ending on asof_date.

    Returns a list of dicts ready for INSERT INTO prices.
    """
    rng = np.random.default_rng(seed)
    dates = [asof_date - timedelta(days=(n_rows - 1 - i)) for i in range(n_rows)]
    # Simple random-walk price around 100.
    log_rets = rng.normal(0.0005, 0.01, n_rows)
    adj_closes = 100.0 * np.exp(np.cumsum(log_rets))
    rows = []
    for i, d in enumerate(dates):
        ac = float(adj_closes[i])
        rows.append({
            "ticker": ticker,
            "date": d,
            "open": round(ac * rng.uniform(0.99, 1.01), 4),
            "high": round(ac * rng.uniform(1.00, 1.02), 4),
            "low": round(ac * rng.uniform(0.98, 1.00), 4),
            "close": round(ac * rng.uniform(0.99, 1.01), 4),
            "adj_close": round(ac, 4),
            "volume": int(rng.integers(100_000, 1_000_000)),
            "source": "yfinance",
            "run_id": 1,
        })
    return rows


def _insert_prices(db_path: Path, rows: list[dict]) -> None:
    con = duckdb.connect(str(db_path), read_only=False)
    try:
        sql = (
            "INSERT INTO prices "
            "(ticker, date, open, high, low, close, adj_close, volume, source, run_id) "
            "VALUES ($ticker, $date, $open, $high, $low, $close, "
            "$adj_close, $volume, $source, $run_id)"
        )
        con.executemany(sql, rows)
    finally:
        con.close()


def _tiny_model(feature_names: list[str]) -> xgb.XGBRegressor:
    """Fit a trivially small XGBRegressor on random data for the given features."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((40, len(feature_names))), columns=feature_names)
    y = pd.Series(rng.standard_normal(40))
    model = xgb.XGBRegressor(n_estimators=5, max_depth=2, random_state=0)
    model.fit(X, y)
    return model


def _tiny_ranker_model(feature_names: list[str]) -> xgb.XGBRanker:
    """Fit a trivially small XGBRanker on grouped synthetic data."""
    rng = np.random.default_rng(43)
    X = pd.DataFrame(rng.standard_normal((40, len(feature_names))), columns=feature_names)
    y = pd.Series(np.tile([0.0, 1.0, 2.0, 3.0], 10))
    model = xgb.XGBRanker(
        objective="rank:pairwise",
        n_estimators=5,
        max_depth=2,
        random_state=0,
    )
    model.fit(X, y, group=[4] * 10)
    return model


def _save_tiny_model(tmp_path: Path, train_end: date) -> Path:
    """Save a tiny model trained before train_end; return pkl_path."""
    model = _tiny_model(FEATURE_NAMES)
    pkl_path, _ = save_model(
        model=model,
        hyperparams={"n_estimators": 5, "max_depth": 2},
        feature_names=FEATURE_NAMES,
        train_end_date=train_end,
        train_rows=40,
        train_rmse=0.5,
        code_commit="abcdef12",
        training_duration_seconds=0.1,
        output_dir=tmp_path / "models",
    )
    return pkl_path


def _save_tiny_ranker_model(tmp_path: Path, train_end: date) -> Path:
    """Save a tiny ranker model trained before train_end; return pkl_path."""
    model = _tiny_ranker_model(FEATURE_NAMES)
    pkl_path, _ = save_model(
        model=model,
        hyperparams={"objective": "rank:pairwise", "n_estimators": 5, "max_depth": 2},
        feature_names=FEATURE_NAMES,
        train_end_date=train_end,
        train_rows=40,
        train_rmse=0.5,
        code_commit="rankabcd",
        training_duration_seconds=0.1,
        output_dir=tmp_path / "models",
        objective="rank",
    )
    return pkl_path


ASOF = date(2025, 7, 1)
UNIVERSE = ["AAA", "BBB"]


# ---------------------------------------------------------------------------
# Test 1: full round-trip
# ---------------------------------------------------------------------------

def test_predict_for_returns_dict_of_floats(tmp_path):
    db_path = _fresh_db(tmp_path)
    for ticker in UNIVERSE + ["SPY"]:
        _insert_prices(db_path, _synthetic_prices(ticker, ASOF, seed=hash(ticker) % 999))

    pkl_path = _save_tiny_model(tmp_path, train_end=date(2025, 6, 1))

    predictor = Predictor(
        models_dir=pkl_path.parent,
        db_path=db_path,
    )
    result = predictor.predict_for(ASOF, UNIVERSE)

    assert isinstance(result, dict)
    assert set(result.keys()) == set(UNIVERSE)
    for ticker, val in result.items():
        assert isinstance(val, float), f"{ticker} prediction is not float: {type(val)}"


def test_predict_for_serves_saved_ranker_model(tmp_path):
    db_path = _fresh_db(tmp_path)
    for ticker in UNIVERSE + ["SPY"]:
        _insert_prices(db_path, _synthetic_prices(ticker, ASOF, seed=hash(ticker) % 999))

    pkl_path = _save_tiny_ranker_model(tmp_path, train_end=date(2025, 6, 1))

    predictor = Predictor(models_dir=pkl_path.parent, db_path=db_path)
    result = predictor.predict_for(ASOF, UNIVERSE)

    assert set(result.keys()) == set(UNIVERSE)
    assert all(isinstance(val, float) for val in result.values())


# ---------------------------------------------------------------------------
# Test 2: predict_for_with_model_id returns correct model_id
# ---------------------------------------------------------------------------

def test_predict_for_with_model_id_returns_correct_model_id(tmp_path):
    db_path = _fresh_db(tmp_path)
    for ticker in UNIVERSE + ["SPY"]:
        _insert_prices(db_path, _synthetic_prices(ticker, ASOF, seed=hash(ticker) % 999))

    pkl_path = _save_tiny_model(tmp_path, train_end=date(2025, 6, 1))

    predictor = Predictor(models_dir=pkl_path.parent, db_path=db_path)
    predictions, model_id = predictor.predict_for_with_model_id(ASOF, UNIVERSE)

    assert isinstance(predictions, dict)
    assert model_id == pkl_path.stem


# ---------------------------------------------------------------------------
# Test 3: raises FileNotFoundError when no eligible model
# ---------------------------------------------------------------------------

def test_predict_for_raises_when_no_eligible_model(tmp_path):
    db_path = _fresh_db(tmp_path)
    empty_models_dir = tmp_path / "empty_models"
    empty_models_dir.mkdir()

    predictor = Predictor(models_dir=empty_models_dir, db_path=db_path)

    with pytest.raises(FileNotFoundError):
        predictor.predict_for(ASOF, UNIVERSE)


# ---------------------------------------------------------------------------
# Test 4: returns empty dict when no price data
# ---------------------------------------------------------------------------

def test_predict_for_returns_empty_when_no_price_data(tmp_path):
    db_path = _fresh_db(tmp_path)
    # DB is empty - no prices inserted.

    pkl_path = _save_tiny_model(tmp_path, train_end=date(2025, 6, 1))

    predictor = Predictor(models_dir=pkl_path.parent, db_path=db_path)
    result = predictor.predict_for(ASOF, UNIVERSE)

    assert result == {}


# ---------------------------------------------------------------------------
# Test 5: model is cached (load_model called only once)
# ---------------------------------------------------------------------------

def test_predict_for_caches_model(tmp_path):
    db_path = _fresh_db(tmp_path)
    for ticker in UNIVERSE + ["SPY"]:
        _insert_prices(db_path, _synthetic_prices(ticker, ASOF, seed=hash(ticker) % 999))

    pkl_path = _save_tiny_model(tmp_path, train_end=date(2025, 6, 1))

    call_count = {"n": 0}
    original_load = __import__("sma.model.persistence", fromlist=["load_model"]).load_model

    def counting_load(path):
        call_count["n"] += 1
        return original_load(path)

    predictor = Predictor(models_dir=pkl_path.parent, db_path=db_path)

    with patch("sma.model.predictor.load_model", side_effect=counting_load):
        predictor.predict_for(ASOF, UNIVERSE)
        predictor.predict_for(ASOF, UNIVERSE)

    assert call_count["n"] == 1, (
        f"load_model called {call_count['n']} times; expected 1 (cache should absorb second call)"
    )


# ---------------------------------------------------------------------------
# Regression: predict_for_with_model_id must use model.feature_names_in_
#
# Bug seen 2026-05-18: the 5/14 model was trained on 12 features; the codebase
# then added politician_flow_30d (13 features). predict_for_with_model_id passed
# 13 features to a 12-feature model → XGBoost feature_names mismatch crash,
# blocking that day's predict.daily fire and leaving decide on a stale model.
#
# predict_for already handled this via model.feature_names_in_; this regression
# pins the with_model_id path to the same behavior.
# ---------------------------------------------------------------------------

def test_predict_for_with_model_id_handles_model_with_fewer_features(tmp_path):
    """A model trained on FEATURE_NAMES[:-1] (no politician_flow_30d) must
    still be usable by the CLI path: predict_for_with_model_id should subset
    to the model's known features instead of insisting on the full current
    FEATURE_NAMES list. Pre-fix this raised
        ValueError: feature_names mismatch: [...12...] [...13...]
    crashing the 19:30 ET predict.daily fire.
    """
    db_path = _fresh_db(tmp_path)
    for ticker in UNIVERSE + ["SPY"]:
        _insert_prices(db_path, _synthetic_prices(ticker, ASOF, seed=hash(ticker) % 999))

    # Train a model on the SUBSET of FEATURE_NAMES that excludes
    # politician_flow_30d (the feature added in 5/10 commit 606b644).
    subset = [f for f in FEATURE_NAMES if f != "politician_flow_30d"]
    assert subset != FEATURE_NAMES, (
        "test premise: politician_flow_30d must be in current FEATURE_NAMES"
    )
    model = _tiny_model(subset)
    from sma.model.persistence import save_model
    pkl_path, _ = save_model(
        model=model,
        hyperparams={"n_estimators": 5, "max_depth": 2},
        feature_names=subset,
        train_end_date=date(2025, 6, 1),
        train_rows=40,
        train_rmse=0.5,
        code_commit="oldfeats",
        training_duration_seconds=0.1,
        output_dir=tmp_path / "models",
    )

    predictor = Predictor(models_dir=pkl_path.parent, db_path=db_path)
    # This call CRASHED pre-fix with feature_names mismatch.
    predictions, model_id = predictor.predict_for_with_model_id(ASOF, UNIVERSE)

    assert isinstance(predictions, dict)
    assert len(predictions) > 0, "synthetic prices should resolve at least one ticker"
    assert model_id == pkl_path.stem


# ---------------------------------------------------------------------------
# Test 6: SPY is loaded automatically even when not in universe
# ---------------------------------------------------------------------------

def test_predict_for_loads_spy_automatically(tmp_path):
    """Universe is ['AAA', 'BBB'] (no SPY). rel_strength_spy_60d should resolve
    because _load_prices adds SPY to the query."""
    db_path = _fresh_db(tmp_path)
    # Insert AAA, BBB, and SPY - SPY not in universe but must be fetched.
    for ticker in ["AAA", "BBB", "SPY"]:
        _insert_prices(db_path, _synthetic_prices(ticker, ASOF, seed=hash(ticker) % 999))

    pkl_path = _save_tiny_model(tmp_path, train_end=date(2025, 6, 1))

    predictor = Predictor(models_dir=pkl_path.parent, db_path=db_path)
    result = predictor.predict_for(ASOF, ["AAA", "BBB"])

    # Both tickers should have predictions (rel_strength resolved via SPY).
    assert "AAA" in result
    assert "BBB" in result
    # SPY itself should not appear in results since it was not in the universe.
    assert "SPY" not in result


def test_news_count_includes_asof_day_matching_training(tmp_path):
    """Train/predict PARITY for news_count_7d: training counts news with
    CAST(published_at AS DATE) <= asof (asof-day INCLUSIVE). Live predict used a
    raw `published_at <= asof_date` timestamp compare, which binds asof to
    midnight and silently DROPS the entire asof day — shifting the feature below
    what the model learned. Live must count the asof day too."""
    from datetime import datetime
    db_path = _fresh_db(tmp_path)
    store = Store(path=str(db_path))
    store.connect()
    asof = date(2026, 1, 15)
    rows = [
        ("AAA", datetime(2026, 1, 15, 14, 0), "finnhub", "h1", "u1", "x", "hash1", 1),  # asof PM
        ("AAA", datetime(2026, 1, 12, 9, 0), "finnhub", "h2", "u2", "x", "hash2", 1),  # in 7d
        ("AAA", datetime(2026, 1, 4, 9, 0), "finnhub", "h3", "u3", "x", "hash3", 1),  # >7d
    ]
    store.conn.executemany(
        "INSERT INTO news (ticker, published_at, source, headline, url, "
        "body_excerpt, hash, run_id) VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    store.close()
    pred = Predictor(models_dir=tmp_path, db_path=db_path)
    counts = pred._fetch_news_counts_7d(asof, ["AAA"])
    # asof-day + 3d-ago are in [asof-7d, asof]; 11d-ago is out → expect 2.
    assert counts.get("AAA") == 2


def test_politician_flow_train_serve_parity(tmp_path):
    """PARITY: loader._compute_politician_flows (training, DataFrame) and
    predictor._fetch_politician_flows (serve, SQL) must return identical net
    flows for the same data. Auto-guards the P/S/'E' sign-convention skew class."""
    from sma.model.__main__ import _load_politician_trades
    from sma.model.loader import _compute_politician_flows

    db_path = _fresh_db(tmp_path)
    store = Store(path=str(db_path))
    store.connect()
    asof = date(2026, 1, 15)
    # Window is on filing_date (disclosure), so each row carries a filing_date.
    rows = [
        # doc_id, chamber, last, first, txn, filing, ticker, asset_desc, asset_type, txn_type, min, max  # noqa: E501
        ("d1", "H", "X", "Y", date(2026, 1, 5), date(2026, 1, 10), "AAA", "d", "ST", "P", 1000.0, 3000.0),  # noqa: E501
        ("d2", "H", "X", "Y", date(2026, 1, 5), date(2026, 1, 10), "BBB", "d", "ST", "S", 1000.0, 3000.0),  # noqa: E501
        ("d3", "H", "X", "Y", date(2026, 1, 5), date(2026, 1, 10), "CCC", "d", "ST", "S (partial)", 1000.0, 3000.0),  # noqa: E501
        ("d4", "H", "X", "Y", date(2026, 1, 5), date(2026, 1, 10), "DDD", "d", "ST", "E", 1000.0, 3000.0),  # noqa: E501
        # disclosed >30d before asof -> excluded (even though traded recently):
        ("d5", "H", "X", "Y", date(2026, 1, 5), date(2025, 12, 1), "AAA", "d", "ST", "P", 5000.0, 7000.0),  # noqa: E501
        ("d6", "H", "X", "Y", date(2026, 1, 5), date(2026, 1, 10), "EEE", "d", "Stock Option", "P", 1000.0, 3000.0),  # noqa: E501
    ]
    store.conn.executemany(
        "INSERT INTO politician_trades (doc_id, chamber, last_name, first_name, "
        "transaction_date, filing_date, ticker, asset_description, asset_type, "
        "transaction_type, amount_min, amount_max, run_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1)",
        rows,
    )
    store.close()
    df = _load_politician_trades(db_path)
    train = _compute_politician_flows(df, asof, lookback_days=30)
    pred = Predictor(models_dir=tmp_path, db_path=db_path)
    serve = pred._fetch_politician_flows(asof, lookback_days=30)
    assert set(train) == set(serve), f"ticker sets differ: {set(train)} vs {set(serve)}"
    # EEE is an option — both paths must EXCLUDE it from the stock-flow feature.
    assert "EEE" not in train and "EEE" not in serve
    for k in train:
        assert train[k] == pytest.approx(serve[k]), f"{k}: train {train[k]} != serve {serve[k]}"


def test_next_earnings_train_serve_parity(tmp_path):
    """PARITY: loader._compute_next_earnings (training) and
    predictor._fetch_next_earnings_dates (serve) must agree — earliest
    report_date strictly after asof, per ticker."""
    from sma.model.__main__ import _load_earnings_calendar
    from sma.model.loader import _compute_next_earnings

    db_path = _fresh_db(tmp_path)
    store = Store(path=str(db_path))
    store.connect()
    asof = date(2026, 1, 15)
    rows = [
        ("AAA", date(2026, 1, 20), "fmp", 1),   # next after asof
        ("AAA", date(2026, 4, 20), "fmp", 1),   # later — not the min
        ("AAA", date(2026, 1, 10), "fmp", 1),   # before asof — excluded
        ("BBB", date(2026, 2, 1), "fmp", 1),
        ("CCC", date(2026, 1, 15), "fmp", 1),   # == asof, NOT > asof — excluded
    ]
    store.conn.executemany(
        "INSERT INTO earnings (ticker, report_date, source, run_id) VALUES (?,?,?,?)",
        rows,
    )
    store.close()
    df = _load_earnings_calendar(db_path)
    train = _compute_next_earnings(df, asof)
    pred = Predictor(models_dir=tmp_path, db_path=db_path)
    serve = pred._fetch_next_earnings_dates(asof, ["AAA", "BBB", "CCC"])
    assert train == serve


def test_earnings_surprise_train_serve_parity(tmp_path):
    """predictor._fetch_latest_surprises must produce the SAME map as
    loader._compute_latest_surprises for identical rows (train/serve parity;
    the feature shipped 2026-06-12)."""
    import duckdb
    import pandas as pd

    from sma.model.loader import _compute_latest_surprises
    from sma.model.predictor import Predictor

    rows = [
        ("AAA", date(2025, 1, 10), 1.0, 1.2),
        ("AAA", date(2025, 4, 10), 1.0, 0.5),
        ("AAA", date(2025, 7, 10), 1.0, 9.0),   # future vs asof
        ("BBB", date(2025, 3, 1), 2.0, None),    # no actual
        ("CCC", date(2025, 2, 1), 0.10, 0.50),   # clamps to +1
    ]
    db = tmp_path / "t.duckdb"
    con = duckdb.connect(str(db))
    con.execute("""CREATE TABLE earnings (
        ticker TEXT, report_date DATE, eps_estimate DOUBLE, eps_actual DOUBLE,
        revenue_estimate DOUBLE, revenue_actual DOUBLE, source TEXT, run_id BIGINT)""")
    con.executemany(
        "INSERT INTO earnings VALUES (?, ?, ?, ?, NULL, NULL, 'test', 1)", rows
    )
    con.close()

    asof = date(2025, 6, 1)
    p = Predictor(models_dir=tmp_path, db_path=db)
    served = p._fetch_latest_surprises(asof, ["AAA", "BBB", "CCC"])

    earnings_df = pd.DataFrame(
        [{"ticker": t, "report_date": d, "eps_estimate": e, "eps_actual": a}
         for t, d, e, a in rows]
    )
    trained = _compute_latest_surprises(earnings_df, asof)
    assert served == pytest.approx(trained)
    assert served["AAA"] == pytest.approx(-0.5)
