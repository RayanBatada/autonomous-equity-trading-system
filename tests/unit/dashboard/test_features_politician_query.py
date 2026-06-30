"""Regression: the Features tab's politician_trades query must supply every
column the shared `_compute_politician_flows` reads.

Commit 077e1e9 switched `_compute_politician_flows` to window on `filing_date`
(disclosure date) instead of `transaction_date`. The model loader's query was
updated; the dashboard's copy was not, so the Features tab crashed with
`KeyError: 'filing_date'`. This test runs the dashboard query against a real
DuckDB table and feeds the result to the compute function — exactly the path
that broke.
"""

from datetime import date

import duckdb

from dashboard.tabs.features import _POLITICIAN_TRADES_SQL
from sma.model.loader import _compute_politician_flows


def test_dashboard_politician_query_feeds_compute_flows(tmp_path):
    db = tmp_path / "t.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        "CREATE TABLE politician_trades ("
        "ticker VARCHAR, transaction_date DATE, filing_date DATE, "
        "transaction_type VARCHAR, amount_min DOUBLE, amount_max DOUBLE)"
    )
    # A buy disclosed 2026-06-01 (within the 30d window ending at the asof below).
    con.execute(
        "INSERT INTO politician_trades VALUES "
        "('NVDA', DATE '2026-05-01', DATE '2026-06-01', 'P', 1000, 5000)"
    )
    pol_df = con.execute(_POLITICIAN_TRADES_SQL).df()
    con.close()

    # Must not raise KeyError('filing_date'); the disclosed buy nets a positive flow.
    flows = _compute_politician_flows(pol_df, date(2026, 6, 5))
    assert flows.get("NVDA", 0.0) > 0
