from datetime import date
from unittest.mock import patch

import httpx
import pytest

from sma.ingest.sources.edgar_filings import EdgarFilingsSource
from sma.ingest.store import Store


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    yield s
    s.close()


_TICKERS_LOOKUP = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
}


_SUBMISSIONS = {
    "filings": {
        "recent": {
            "accessionNumber": ["0000320193-26-000001", "0000320193-26-000002"],
            "filingDate":      ["2026-04-25",          "2026-03-15"],
            "form":            ["8-K",                 "10-Q"],
            "primaryDocument": ["aapl-20260425.htm",   "aapl-20260315.htm"],
        }
    }
}


def _resp(json_data, status=200):
    request = httpx.Request("GET", "https://www.sec.gov")
    return httpx.Response(status, json=json_data, request=request)


def test_edgar_pulls_recent_filings(store):
    src = EdgarFilingsSource(user_agent="Test (t@example.com)")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="edgar")

    def fake_get(url, **kw):
        if "company_tickers" in url:
            return _resp(_TICKERS_LOOKUP)
        if "submissions" in url:
            return _resp(_SUBMISSIONS)
        return _resp({}, status=404)

    with patch.object(src._client, "get", side_effect=fake_get):
        result = src.fetch(["AAPL"], date(2026, 4, 26), store, run_id)

    assert result.status == "ok"
    rows = store.conn.execute(
        "SELECT ticker, filing_type, accession_no FROM filings ORDER BY filed_at DESC"
    ).fetchall()
    assert ("AAPL", "8-K", "0000320193-26-000001") in rows


def _prospectus_heavy_submissions():
    """20 most-recent filings are all untracked (424B2 debt prospectuses),
    like Morgan Stanley's actual recent filing history. Tracked forms
    (10-K/10-Q/8-K) only start appearing after the 20th entry.
    """
    n_prospectus = 20
    accession_nos = [f"0000000000-26-{i:06d}" for i in range(n_prospectus)]
    filing_dates = [f"2026-04-{(30 - i) if (30 - i) >= 1 else 1:02d}" for i in range(n_prospectus)]
    forms = ["424B2"] * n_prospectus
    primary_docs = [f"prospectus-{i}.htm" for i in range(n_prospectus)]

    # Older tracked filings, still within EDGAR's single "recent" response.
    accession_nos += ["0000000000-26-100001", "0000000000-26-100002"]
    filing_dates += ["2026-02-10", "2026-01-15"]
    forms += ["10-Q", "8-K"]
    primary_docs += ["ms-20260210.htm", "ms-20260115.htm"]

    return {
        "filings": {
            "recent": {
                "accessionNumber": accession_nos,
                "filingDate": filing_dates,
                "form": forms,
                "primaryDocument": primary_docs,
            }
        }
    }


def test_edgar_filters_by_form_before_slicing_to_max_filings(store):
    """Regression test for the MS bug (2026-08-03 data audit): the code used
    to slice to max_filings_per_ticker most-recent filings BEFORE filtering
    by form type. For a prospectus-heavy issuer whose 20 most-recent filings
    are all untracked forms, that meant zero tracked filings were ever found,
    even though older 10-K/10-Q/8-K filings exist in the same response.
    """
    ms_tickers_lookup = {
        "0": {"cik_str": 895421, "ticker": "MS", "title": "Morgan Stanley"},
    }
    submissions = _prospectus_heavy_submissions()

    src = EdgarFilingsSource(user_agent="Test (t@example.com)")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="edgar")

    def fake_get(url, **kw):
        if "company_tickers" in url:
            return _resp(ms_tickers_lookup)
        if "submissions" in url:
            return _resp(submissions)
        return _resp({}, status=404)

    with patch.object(src._client, "get", side_effect=fake_get):
        result = src.fetch(["MS"], date(2026, 4, 30), store, run_id)

    assert result.status == "ok"
    rows = store.conn.execute(
        "SELECT ticker, filing_type, accession_no FROM filings ORDER BY filed_at DESC"
    ).fetchall()
    assert ("MS", "10-Q", "0000000000-26-100001") in rows
    assert ("MS", "8-K", "0000000000-26-100002") in rows
    assert result.rows_inserted == 2


_FPI_TICKERS_LOOKUP = {
    "0": {"cik_str": 1973239, "ticker": "ARM", "title": "Arm Holdings plc"},
}


def _fpi_submissions():
    """Foreign-private-issuer filing mix, matching ARM's live EDGAR history
    (checked 2026-08-05): no 10-K/10-Q/8-K at all, only 20-F (annual report)
    and 6-K (periodic furnished report), plus an untracked Section 16 form.
    """
    return {
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0001973239-26-000001",
                    "0001973239-26-000002",
                    "0001973239-26-000003",
                ],
                "filingDate": ["2026-06-15", "2026-05-20", "2026-04-01"],
                "form": ["20-F", "6-K", "4"],
                "primaryDocument": [
                    "arm-20260615.htm", "arm-20260520.htm", "arm-form4.htm",
                ],
            }
        }
    }


def test_edgar_captures_20f_and_6k_for_foreign_private_issuers(store):
    """ARM/ASML/SPOT/TSM file 20-F/6-K, not 10-K/10-Q/8-K, and previously
    had ZERO filings rows as a result -- invisible to the filings features
    and thesis pipeline. Regression for the 2026-08-05 fix.
    """
    src = EdgarFilingsSource(user_agent="Test (t@example.com)")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="edgar")

    def fake_get(url, **kw):
        if "company_tickers" in url:
            return _resp(_FPI_TICKERS_LOOKUP)
        if "submissions" in url:
            return _resp(_fpi_submissions())
        return _resp({}, status=404)

    with patch.object(src._client, "get", side_effect=fake_get):
        result = src.fetch(["ARM"], date(2026, 6, 16), store, run_id)

    assert result.status == "ok"
    rows = store.conn.execute(
        "SELECT ticker, filing_type, accession_no FROM filings ORDER BY filed_at DESC"
    ).fetchall()
    assert ("ARM", "20-F", "0001973239-26-000001") in rows
    assert ("ARM", "6-K", "0001973239-26-000002") in rows
    # The untracked Section 16 form ('4') must not be captured.
    assert result.rows_inserted == 2


def test_edgar_skips_unknown_ticker(store):
    src = EdgarFilingsSource(user_agent="Test (t@example.com)")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="edgar")

    def fake_get(url, **kw):
        if "company_tickers" in url:
            return _resp(_TICKERS_LOOKUP)
        return _resp({}, status=404)

    with patch.object(src._client, "get", side_effect=fake_get):
        result = src.fetch(["NOPE"], date(2026, 4, 26), store, run_id)

    assert result.status == "ok"
    assert result.rows_inserted == 0
