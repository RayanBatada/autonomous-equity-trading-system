from pathlib import Path

import duckdb
import pytest

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sma-fixture.duckdb"


@pytest.fixture
def conn():
    if not FIXTURE.exists():
        pytest.skip("fixture not generated; run scripts/refresh_fixture.py")
    c = duckdb.connect(str(FIXTURE), read_only=True)
    yield c
    c.close()


def test_fixture_has_expected_tickers(conn):
    rows = conn.execute(
        "SELECT DISTINCT ticker FROM prices ORDER BY ticker"
    ).fetchall()
    tickers = [r[0] for r in rows]
    assert tickers == ["AAPL", "GOOGL", "MSFT"]


def test_fixture_spans_at_least_three_months(conn):
    row = conn.execute(
        "SELECT MIN(date), MAX(date) FROM prices"
    ).fetchone()
    assert (row[1] - row[0]).days >= 90


def test_fixture_has_no_obvious_gaps(conn):
    rows = conn.execute("""
        WITH r AS (
            SELECT ticker, MIN(date) AS d0, MAX(date) AS d1, COUNT(*) AS n
            FROM prices GROUP BY ticker
        )
        SELECT ticker, n,
               (DATE_DIFF('day', d0, d1) * 5 / 7) AS expected_business_days
        FROM r
    """).fetchall()
    for ticker, n, expected in rows:
        assert n >= expected * 0.8, f"{ticker}: only {n} of ~{expected} business days"
