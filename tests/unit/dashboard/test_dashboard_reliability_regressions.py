from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from dashboard.tabs import autoresearch, pipeline
from sma.sentinels import write_sentinel

ET = ZoneInfo("America/New_York")


def _patch_db(monkeypatch, tmp_path):
    db_path = tmp_path / "dashboard.duckdb"
    monkeypatch.setattr(pipeline, "DB_PATH", db_path)
    return db_path


def test_predict_status_marks_db_rows_stale_when_lineage_mismatches(monkeypatch, tmp_path):
    db_path = _patch_db(monkeypatch, tmp_path)
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE predictions (
            asof_date DATE,
            ticker VARCHAR,
            target VARCHAR,
            predicted_value DOUBLE,
            model_id VARCHAR,
            computed_at TIMESTAMP
        )
        """
    )
    con.execute(
        """
        INSERT INTO predictions VALUES
        ('2026-06-10', 'AAPL', 'next_5d', 0.01, 'sma_xgb_v1_2026-06-10', ?)
        """,
        [datetime(2026, 6, 10, 23, 30)],
    )
    con.close()

    write_sentinel(
        label=pipeline.INGEST_LABEL,
        asof=date(2026, 6, 10),
        payload={"run_id": 200, "completed_at": "2026-06-10T22:30:00Z"},
    )
    write_sentinel(
        label=pipeline.PREDICT_LABEL,
        asof=date(2026, 6, 10),
        payload={
            "run_id": 10,
            "ingest_run_id": 100,
            "completed_at": "2026-06-10T23:30:00Z",
        },
    )

    status, _, output = pipeline._predict_status(
        "2026-06-10", datetime(2026, 6, 10, 21, tzinfo=ET)
    )

    assert status == "STALE (repredict pending)"
    assert "1 predictions" in output


def test_reconciled_fill_asof_uses_run_sentinel_before_latest(monkeypatch, tmp_path):
    db_path = _patch_db(monkeypatch, tmp_path)
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE paper_fills (asof_date DATE)")
    con.execute("INSERT INTO paper_fills VALUES ('2026-06-09'), ('2026-06-10')")
    con.close()

    write_sentinel(
        label=pipeline.RECONCILE_RAN_LABEL,
        asof=date(2026, 6, 11),
        payload={
            "completed_at": "2026-06-11T20:30:00Z",
            "reconciled_asof": "2026-06-09",
        },
    )

    assert pipeline._reconciled_fill_asof("2026-06-11") == "2026-06-09"


def test_reconciled_fill_asof_falls_back_to_latest_batch(monkeypatch, tmp_path):
    db_path = _patch_db(monkeypatch, tmp_path)
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE paper_fills (asof_date DATE)")
    con.execute("INSERT INTO paper_fills VALUES ('2026-06-08'), ('2026-06-10')")
    con.close()

    assert pipeline._reconciled_fill_asof("2026-06-11") == "2026-06-10"


def test_promotion_mask_compares_each_row_to_own_baseline():
    df = pd.DataFrame(
        {
            "monotonicity_score": [3, 3],
            "sharpe_overall": [0.31, 0.31],
            "baseline_overall": [0.20, 0.40],
        }
    )

    assert autoresearch._promotion_mask(df).tolist() == [True, False]
