"""Predictor: load XGBoost model + compute features + predict for a given date."""

from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import xgboost as xgb

from sma.db_connect import read_only_connect
from sma.features.builder import FEATURE_NAMES, build_features
from sma.model.ensemble import EnsembleModel
from sma.model.persistence import latest_model_for_date, load_model

DEFAULT_MODELS_DIR = Path("models_artifacts")
DEFAULT_DB_PATH = Path("data/sma.duckdb")
DEFAULT_TARGET = "ret_30d_forward"
DEFAULT_FEATURE_LOOKBACK_DAYS = 400  # need ~252 trading days for dist_from_52w_high


class Predictor:
    """Load the right model for an asof_date, build features, return predictions.

    Caches loaded models in-process so repeated predict_for() calls on the
    same model don't re-deserialize.

    Optionally accepts an existing DuckDB connection (`conn`) to avoid opening
    a new one — required for live mode where the DB is already open in write
    mode (DuckDB rejects same-process connections with different configs).
    """

    def __init__(
        self,
        models_dir: Path = DEFAULT_MODELS_DIR,
        db_path: Path = DEFAULT_DB_PATH,
        target: str = DEFAULT_TARGET,
        conn: duckdb.DuckDBPyConnection | None = None,
    ):
        self.models_dir = models_dir
        self.db_path = db_path
        self.target = target
        self._conn = conn
        self._model_cache: dict[
            Path, xgb.XGBRegressor | xgb.XGBRanker | EnsembleModel
        ] = {}

    def _fetch_news_counts_7d(
        self, asof_date: date, universe: list[str],
    ) -> dict[str, int]:
        """Count news rows per ticker over [asof_date - 7d, asof_date].

        Aggregates across all sources (Finnhub + Alpaca/Benzinga + NewsAPI).
        Empty dict on table-missing or connection error — builder defaults
        the feature to log1p(0)=0 in that case.
        """
        from datetime import timedelta
        start = asof_date - timedelta(days=7)
        # Window must match training (loader counts published_at_date in
        # [start, asof], asof-day INCLUSIVE). A raw `published_at <= asof_date`
        # timestamp compare binds asof to midnight and drops the whole asof day
        # → train/predict skew. Use a half-open timestamp range [start, asof+1d)
        # which is asof-day-inclusive AND sargable (keeps the published_at index,
        # unlike CAST(published_at AS DATE)).
        sql = """
            SELECT ticker, COUNT(*) AS n
            FROM news
            WHERE ticker = ANY($tickers)
              AND published_at >= $start
              AND published_at < $end_excl
            GROUP BY ticker
        """
        params = {
            "tickers": list(universe),
            "start": start,
            "end_excl": asof_date + timedelta(days=1),
        }
        try:
            if self._conn is not None:
                rows = self._conn.execute(sql, params).fetchall()
            else:
                if not self.db_path.exists():
                    return {}
                con = read_only_connect(self.db_path)
                try:
                    rows = con.execute(sql, params).fetchall()
                finally:
                    con.close()
        except Exception:
            return {}
        return {ticker: int(n) for ticker, n in rows}

    def _fetch_next_earnings_dates(
        self, asof_date: date, universe: list[str],
    ) -> dict[str, date]:
        """For each ticker, the earliest report_date strictly after asof_date.

        Empty dict when the `earnings` table is missing or the connection
        fails — builder defaults to the cap value (60d), which is the
        same as "no earnings soon," so an empty calendar is gracefully a
        neutral signal rather than a None-feature drop.
        """
        sql = """
            SELECT ticker, MIN(report_date) AS next_date
            FROM earnings
            WHERE ticker = ANY($tickers)
              AND report_date > $asof
            GROUP BY ticker
        """
        params = {"tickers": list(universe), "asof": asof_date}
        try:
            if self._conn is not None:
                rows = self._conn.execute(sql, params).fetchall()
            else:
                if not self.db_path.exists():
                    return {}
                con = read_only_connect(self.db_path)
                try:
                    rows = con.execute(sql, params).fetchall()
                finally:
                    con.close()
        except Exception:
            return {}
        return {ticker: report_date for ticker, report_date in rows}

    def _fetch_latest_surprises(
        self, asof_date: date, universe: list[str],
    ) -> dict[str, float]:
        """ticker → latest EPS surprise ((actual−est)/|est|, clamped ±1) for
        the most recent report at or before asof_date with BOTH legs present.
        MUST match loader._compute_latest_surprises exactly (train/serve
        parity). Empty dict on any failure — builder reads 0.0 (neutral)."""
        sql = """
            SELECT ticker, eps_estimate, eps_actual
            FROM (
                SELECT ticker, eps_estimate, eps_actual,
                       ROW_NUMBER() OVER (
                           PARTITION BY ticker ORDER BY report_date DESC
                       ) AS rn
                FROM earnings
                WHERE ticker = ANY($tickers)
                  AND report_date <= $asof
                  AND eps_actual IS NOT NULL
                  AND eps_estimate IS NOT NULL
                  AND ABS(eps_estimate) > 1e-6
            ) t
            WHERE rn = 1
        """
        params = {"tickers": list(universe), "asof": asof_date}
        try:
            if self._conn is not None:
                rows = self._conn.execute(sql, params).fetchall()
            else:
                if not self.db_path.exists():
                    return {}
                con = read_only_connect(self.db_path)
                try:
                    rows = con.execute(sql, params).fetchall()
                finally:
                    con.close()
        except Exception:
            return {}
        return {
            t: max(-1.0, min(1.0, (float(a) - float(e)) / abs(float(e))))
            for t, e, a in rows
        }

    def _fetch_politician_flows(
        self, asof_date: date, *, lookback_days: int = 30,
    ) -> dict[str, float]:
        """Net politician dollar flow per ticker over the lookback window,
        read directly from `politician_trades`. Returns empty dict if the
        table doesn't exist (older DBs without the v5 migration) or if no
        rows match — predictor's caller treats that as "feature = 0.0".
        """
        from datetime import timedelta
        start = asof_date - timedelta(days=lookback_days)
        sql = """
            SELECT ticker, SUM(
                CASE WHEN transaction_type = 'P' THEN (amount_min + amount_max) / 2.0
                     WHEN transaction_type LIKE 'S%' THEN -1 * (amount_min + amount_max) / 2.0
                     ELSE 0 END
            )
            FROM politician_trades
            WHERE ticker IS NOT NULL
              -- Exclude options/derivatives so they don't pollute the stock-flow
              -- feature. MUST match _load_politician_trades (training) exactly.
              AND COALESCE(asset_type, '') NOT IN ('Stock Option', 'OP')
              -- Window on filing_date (public DISCLOSURE date), not
              -- transaction_date: trades are disclosed ~58d (avg) after the
              -- trade, so a transaction_date window leaks non-public info.
              -- MUST match loader._compute_politician_flows.
              AND filing_date >= ?
              AND filing_date <= ?
            GROUP BY ticker
        """
        params = [start, asof_date]
        try:
            if self._conn is not None:
                rows = self._conn.execute(sql, params).fetchall()
            else:
                if not self.db_path.exists():
                    return {}
                con = read_only_connect(self.db_path)
                try:
                    rows = con.execute(sql, params).fetchall()
                finally:
                    con.close()
        except Exception:
            return {}
        return {ticker: float(net or 0.0) for ticker, net in rows}

    def predict_for(
        self,
        asof_date: date,
        universe: list[str],
        feature_lookback_days: int = DEFAULT_FEATURE_LOOKBACK_DAYS,
    ) -> dict[str, float]:
        """Find the right model for asof_date, build features for the universe,
        return dict[ticker -> predicted_value].

        Tickers with insufficient history (any feature is None) are absent
        from the result.

        Raises FileNotFoundError if no model is eligible for asof_date.
        """
        pkl_path = latest_model_for_date(self.models_dir, asof_date, self.target)
        predictions, _ = self._predict_with_path(
            asof_date, universe, pkl_path, feature_lookback_days,
        )
        return predictions

    def predict_for_with_model_id(
        self,
        asof_date: date,
        universe: list[str],
        feature_lookback_days: int = DEFAULT_FEATURE_LOOKBACK_DAYS,
    ) -> tuple[dict[str, float], str]:
        """Same as predict_for but also returns the model_id used.

        Resolves the model path ONCE (single `latest_model_for_date` call)
        before delegating to the shared inner helper. A naive delegation that
        re-resolved the path inside predict_for would race a model rotation
        between the two lookups, causing model_id (from the outer lookup) to
        diverge from the model actually used (the inner lookup). The
        predictions table would then carry the wrong model_id, silently
        breaking the per-model attribution + retrain audit trail.
        """
        pkl_path = latest_model_for_date(self.models_dir, asof_date, self.target)
        predictions, model_id = self._predict_with_path(
            asof_date, universe, pkl_path, feature_lookback_days,
        )
        return predictions, model_id

    def _predict_with_path(
        self,
        asof_date: date,
        universe: list[str],
        pkl_path: Path,
        feature_lookback_days: int,
    ) -> tuple[dict[str, float], str]:
        """Shared predict path. Takes an already-resolved model path so both
        public methods agree on which artifact was used."""
        model = self._load_or_get_cached(pkl_path)
        model_id = pkl_path.stem

        prices = self._load_prices(asof_date, universe, feature_lookback_days)
        if prices.empty:
            return {}, model_id

        flows = self._fetch_politician_flows(asof_date, lookback_days=30)
        earnings_cal = self._fetch_next_earnings_dates(asof_date, universe)
        earnings_surprises = self._fetch_latest_surprises(asof_date, universe)
        news_counts = self._fetch_news_counts_7d(asof_date, universe)
        feats_df = build_features(
            prices, universe, asof_date,
            politician_flows=flows,
            earnings_calendar=earnings_cal,
            earnings_surprises=earnings_surprises,
            news_counts_7d=news_counts,
        )
        if feats_df.empty:
            return {}, model_id

        # Use the feature set the MODEL was trained on (older models may know
        # fewer columns than the current FEATURE_NAMES list); XGBoost's
        # sklearn API exposes feature_names_in_ for that purpose. Pre-fix,
        # predict_for_with_model_id bypassed this and crashed on the
        # 2026-05-18 12-feature 5/14 model.
        expected = list(getattr(model, "feature_names_in_", FEATURE_NAMES))
        x = feats_df[expected]
        # One call for both artifact shapes. A single-booster artifact scores
        # exactly as it always has; an EnsembleModel returns the MEAN of its N
        # boosters' predictions (sma.model.ensemble). The averaging lives in
        # the model object rather than here on purpose: the walk-forward CV
        # that gates a retrain calls the same predict(), so the number the
        # gate measures is the number that trades.
        preds = model.predict(x)

        predictions = {
            ticker: float(p) for ticker, p in zip(feats_df.index, preds, strict=True)
        }
        return predictions, model_id

    def _load_or_get_cached(
        self, pkl_path: Path,
    ) -> xgb.XGBRegressor | xgb.XGBRanker | EnsembleModel:
        if pkl_path not in self._model_cache:
            self._model_cache[pkl_path] = load_model(pkl_path)
        return self._model_cache[pkl_path]

    def _load_prices(
        self, asof_date: date, universe: list[str], lookback_days: int
    ) -> pd.DataFrame:
        """Read prices from DuckDB for [asof_date - lookback, asof_date], plus SPY.

        Universe + SPY both fetched. Lookback chosen so all 12 features can
        resolve (dist_from_52w_high needs 252 trading days).

        Uses an injected `self._conn` if available (live mode); otherwise
        opens a fresh read-only connection and closes it after the query.
        """
        # Always include SPY for rel_strength_spy_60d.
        tickers = list(set(list(universe) + ["SPY"]))
        start_date = asof_date - timedelta(days=lookback_days)
        sql = """
            SELECT ticker, date, open, high, low, close, adj_close, volume
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY ticker, date
                           ORDER BY CASE source WHEN 'yfinance' THEN 0
                                                 WHEN 'alpaca' THEN 1
                                                 ELSE 2 END
                       ) AS rn
                FROM prices
                WHERE ticker = ANY($tickers)
                  AND date BETWEEN $start_date AND $end_date
                  AND adj_close IS NOT NULL
            ) t
            WHERE rn = 1
            ORDER BY ticker, date
        """
        params = {"tickers": tickers, "start_date": start_date, "end_date": asof_date}

        if self._conn is not None:
            df = self._conn.execute(sql, params).df()
        else:
            if not self.db_path.exists():
                raise FileNotFoundError(
                    f"DuckDB not found at {self.db_path}. "
                    "Run `python -m sma.ingest run` first."
                )
            con = read_only_connect(self.db_path)
            try:
                df = con.execute(sql, params).df()
            finally:
                con.close()

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
