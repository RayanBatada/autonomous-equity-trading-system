"""Unit tests for the Senate PTR scraper + parser.

Live HTTP is mocked — we test the parsing surface (CSRF extraction, search-
row decode, transactions HTML parser, amount-range parser) against the
HTML/JSON shapes documented in the May 2026 EFD research brief.
"""

from __future__ import annotations

from datetime import date

from sma.ingest.sources.senate_trades import (
    SenateFiling,
    _extract_csrf_from_html,
    _normalize_senate_order_type,
    _office_to_state,
    _parse_amount_range,
    _parse_search_row,
    parse_senate_ptr_html,
)

# ---- CSRF extraction -------------------------------------------------------


def test_extract_csrf_from_typical_django_form():
    html = '''
    <form method="post" action="/search/home/">
      <input type="hidden" name="csrfmiddlewaretoken" value="ABC123def456" />
      <input type="checkbox" name="prohibition_agreement" />
    </form>
    '''
    assert _extract_csrf_from_html(html) == "ABC123def456"


def test_extract_csrf_handles_single_quotes_and_attribute_order():
    html = "<input value='token-xyz' name='csrfmiddlewaretoken' />"
    # Our regex requires name first then value; this attribute-order case
    # should fall through to raise (so we know to update the regex if EFD
    # ever swaps attribute ordering).
    try:
        _extract_csrf_from_html(html)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError on swapped attribute order")


def test_extract_csrf_raises_on_missing_token():
    try:
        _extract_csrf_from_html("<form></form>")
    except RuntimeError as e:
        assert "csrfmiddlewaretoken" in str(e)
        return
    raise AssertionError("expected RuntimeError")


# ---- Search-row decode -----------------------------------------------------


def test_parse_search_row_html_ptr():
    row = [
        "Jane",
        "Doe",
        "Hawaii, Junior Senator",
        '<a href="/search/view/ptr/abc-def-123/">Periodic Transaction Report</a>',
        "05/19/2026 14:32:11",
    ]
    parsed = _parse_search_row(row)
    assert parsed is not None
    assert parsed.first_name == "Jane"
    assert parsed.last_name == "Doe"
    assert parsed.office == "Hawaii, Junior Senator"
    assert parsed.doc_id == "abc-def-123"
    assert parsed.is_paper is False
    assert parsed.filing_date == date(2026, 5, 19)


def test_parse_search_row_paper_filed_is_flagged():
    row = [
        "John", "Smith", "WV - Senator",
        '<a href="/search/view/paper/uuid-zzz/">PTR</a>',
        "01/02/2025",
    ]
    parsed = _parse_search_row(row)
    assert parsed is not None
    assert parsed.is_paper is True
    assert parsed.doc_id == "uuid-zzz"


def test_parse_search_row_missing_href_returns_none():
    assert _parse_search_row(["A", "B", "Office", "no link here", "01/01/2026"]) is None


def test_parse_search_row_bad_date_yields_none_date():
    row = ["A", "B", "Off", '<a href="/search/view/ptr/x/">PTR</a>', "TBD"]
    parsed = _parse_search_row(row)
    assert parsed is not None
    assert parsed.filing_date is None


def test_parse_search_row_too_short_returns_none():
    assert _parse_search_row(["A", "B"]) is None


# ---- Amount range parser ---------------------------------------------------


def test_parse_amount_range_standard():
    lo, hi = _parse_amount_range("$1,001 - $15,000")
    assert lo == 1001.0
    assert hi == 15000.0


def test_parse_amount_range_open_ended_upper():
    lo, hi = _parse_amount_range("$1,000,000 +")
    assert lo == 1_000_000.0
    assert hi is None


def test_parse_amount_range_em_dash():
    lo, hi = _parse_amount_range("$15,001 – $50,000")
    assert lo == 15001.0
    assert hi == 50000.0


def test_parse_amount_range_unparseable():
    assert _parse_amount_range("under $1,000") == (None, None)


# ---- Order-type normalization ---------------------------------------------


def test_normalize_order_type_purchase():  # noqa: N802
    assert _normalize_senate_order_type("Purchase") == "P"


def test_normalize_order_type_partial_sale():
    assert _normalize_senate_order_type("Sale (Partial)") == "S (partial)"


def test_normalize_order_type_full_sale():
    assert _normalize_senate_order_type("Sale (Full)") == "S"


def test_normalize_order_type_exchange():
    assert _normalize_senate_order_type("Exchange") == "E"


def test_normalize_order_type_unknown_preserved_truncated():
    out = _normalize_senate_order_type("Reinvestment of Some Sort")
    # Falls through to raw passthrough, capped to 32 chars.
    assert len(out) <= 32


# ---- Office → state extractor ---------------------------------------------


def test_office_to_state_comma_format():
    assert _office_to_state("Hawaii, Junior Senator") == "Hawaii"


def test_office_to_state_dash_format():
    assert _office_to_state("WV - Senator") == "WV"


def test_office_to_state_empty():
    assert _office_to_state("") is None


# ---- HTML transactions table parser ---------------------------------------


_FIXTURE_HTML = """
<html><body>
<div class="table-responsive">
<table class="table">
  <thead><tr>
    <th>#</th><th>Transaction Date</th><th>Owner</th><th>Ticker</th>
    <th>Asset Name</th><th>Asset Type</th><th>Order Type</th>
    <th>Amount</th><th>Comment</th>
  </tr></thead>
  <tbody>
    <tr>
      <td>1</td><td>05/01/2026</td><td>Self</td><td>NVDA</td>
      <td>NVIDIA Corp</td><td>Stock</td><td>Purchase</td>
      <td>$15,001 - $50,000</td><td>--</td>
    </tr>
    <tr>
      <td>2</td><td>05/02/2026</td><td>Spouse</td><td>MU</td>
      <td>Micron Technology Inc</td><td>Stock</td><td>Sale (Partial)</td>
      <td>$1,001 - $15,000</td><td></td>
    </tr>
    <tr>
      <td>3</td><td>04/29/2026</td><td>Self</td><td>--</td>
      <td>Putnam Sustainable Future Fund</td><td>Mutual Fund</td><td>Purchase</td>
      <td>$50,001 - $100,000</td><td></td>
    </tr>
    <tr>
      <td>4</td><td>04/30/2026</td><td>Self</td><td>BLK</td>
      <td>BlackRock</td><td>Stock</td><td>Purchase</td>
      <td>$1,000,000 +</td><td></td>
    </tr>
  </tbody>
</table>
</div>
</body></html>
"""


def test_parse_senate_html_extracts_stock_rows_skips_mutual_funds():
    trades = parse_senate_ptr_html(_FIXTURE_HTML)
    tickers = [t.ticker for t in trades]
    # NVDA + MU + BLK; mutual fund row dropped.
    assert tickers == ["NVDA", "MU", "BLK"]


def test_parse_senate_html_decodes_transaction_types_and_dates():
    trades = parse_senate_ptr_html(_FIXTURE_HTML)
    by_ticker = {t.ticker: t for t in trades}
    assert by_ticker["NVDA"].transaction_type == "P"
    assert by_ticker["NVDA"].transaction_date == date(2026, 5, 1)
    assert by_ticker["MU"].transaction_type == "S (partial)"
    assert by_ticker["MU"].transaction_date == date(2026, 5, 2)


def test_parse_senate_html_handles_open_ended_high_band():
    trades = parse_senate_ptr_html(_FIXTURE_HTML)
    blk = next(t for t in trades if t.ticker == "BLK")
    assert blk.amount_min == 1_000_000.0
    assert blk.amount_max is None


def test_parse_senate_html_returns_empty_when_no_table():
    trades = parse_senate_ptr_html("<html><body>No transactions disclosed.</body></html>")
    assert trades == []


def test_parse_senate_html_returns_empty_when_table_has_no_rows():
    html = (
        "<html><body><table><thead><tr><th>x</th></tr></thead>"
        "<tbody></tbody></table></body></html>"
    )
    assert parse_senate_ptr_html(html) == []


# ---- store_senate_trades smoke (requires schema) -------------------------


def test_store_senate_trades_writes_with_chamber_senate(tmp_path):
    """End-to-end: parse fixture → insert into a real DuckDB, query back."""
    from sma.ingest.sources.senate_trades import store_senate_trades
    from sma.ingest.store import Store

    db_path = tmp_path / "t.duckdb"
    store = Store(path=str(db_path)).connect()
    try:
        rid = store.allocate_run_id()
        filing = SenateFiling(
            doc_id="test-doc-1",
            first_name="Jane",
            last_name="Doe",
            office="Hawaii, Junior Senator",
            filing_date=date(2026, 5, 19),
            is_paper=False,
        )
        trades = parse_senate_ptr_html(_FIXTURE_HTML)
        n = store_senate_trades(store=store, run_id=rid, filing=filing, trades=trades)
        assert n == 3

        rows = store.conn.execute(
            "SELECT ticker, chamber, last_name, state_dst, transaction_type "
            "FROM politician_trades WHERE doc_id = ? ORDER BY ticker",
            ["test-doc-1"],
        ).fetchall()
        assert {r[0] for r in rows} == {"NVDA", "MU", "BLK"}
        # All rows tagged chamber='senate' and state derived from the office.
        assert all(r[1] == "senate" for r in rows)
        assert all(r[2] == "Doe" for r in rows)
        assert all(r[3] == "Hawaii" for r in rows)
    finally:
        store.close()
