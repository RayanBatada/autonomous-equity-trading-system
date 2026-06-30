"""Tests for the politician_trades parser.

The PDF parser is intentionally a v0 — the line-merge strategy can confuse
adjacent transactions when one row's description leaks into the next row's
buffer. These tests pin the current behavior so a future v1 cleanup can
demonstrate measurable improvement.
"""

from datetime import date

from sma.ingest.sources.politician_trades import (
    HouseFiling,
    parse_house_index,
    parse_ptr_pdf_text,
)


def test_parse_house_index_picks_up_filing_metadata(tmp_path):
    """The XML index parser extracts the basic filing metadata. Used by
    `iter_recent_house_ptrs` to identify which doc IDs to download."""
    import io
    import zipfile
    xml = (
        '<?xml version="1.0"?>\n'
        '<FinancialDisclosure>\n'
        '  <Member>\n'
        '    <Last>Aaron</Last><First>Richard</First>\n'
        '    <FilingType>P</FilingType>\n'
        '    <StateDst>MI04</StateDst>\n'
        '    <Year>2026</Year>\n'
        '    <FilingDate>4/15/2026</FilingDate>\n'
        '    <DocID>20034201</DocID>\n'
        '  </Member>\n'
        '  <Member>\n'
        '    <Last>Smith</Last><First>Jane</First>\n'
        '    <FilingType>A</FilingType>\n'
        '    <Year>2026</Year>\n'
        '    <FilingDate>5/1/2026</FilingDate>\n'
        '    <DocID>20034999</DocID>\n'
        '  </Member>\n'
        '</FinancialDisclosure>\n'
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("2026FD.xml", xml)
    filings = parse_house_index(buf.getvalue(), 2026)
    assert len(filings) == 2
    ptr = next(f for f in filings if f.filing_type == "P")
    assert ptr.doc_id == "20034201"
    assert ptr.last_name == "Aaron"
    assert ptr.first_name == "Richard"
    assert ptr.state_dst == "MI04"
    assert ptr.filing_year == 2026
    assert ptr.filing_date == date(2026, 4, 15)


def test_parse_ptr_pdf_text_extracts_clean_single_line_transaction():
    """The simplest case: one transaction row with a parenthesized ticker
    and a clean amount range — the parser should extract it correctly."""
    pdf_text = (
        "ID Owner Asset Transaction Date Notification Amount Cap.\n"
        "Apple Inc. - Common Stock (AAPL) [ST] P 03/16/2026 03/20/2026 $1,001 - $15,000\n"
    )
    trades = parse_ptr_pdf_text(pdf_text)
    assert len(trades) == 1
    t = trades[0]
    assert t.ticker == "AAPL"
    assert t.asset_type == "ST"
    assert t.transaction_type == "P"
    assert t.transaction_date == date(2026, 3, 16)
    assert t.amount_min == 1001.0
    assert t.amount_max == 15000.0


def test_parse_ptr_pdf_text_handles_partial_sales():
    """`S (partial)` transaction-type marker should be preserved verbatim."""
    pdf_text = (
        "Berkshire Hathaway Inc. (BRK.B) [ST] S (partial) "
        "01/05/2026 01/10/2026 $50,001 - $100,000\n"
    )
    trades = parse_ptr_pdf_text(pdf_text)
    assert len(trades) == 1
    assert trades[0].ticker == "BRK.B"
    assert trades[0].transaction_type == "S (partial)"


def test_parse_ptr_pdf_text_strips_null_bytes_from_columns():
    """pdfplumber emits NULL bytes in some column-header rows because of
    font-subsetting tricks. Parser strips them before regex matches."""
    pdf_text = (
        "Apple Inc. (AAPL) [ST] P 03/16/2026 03/20/2026 $1\x00,001 - $15\x00,000\n"
    )
    trades = parse_ptr_pdf_text(pdf_text)
    assert len(trades) == 1
    assert trades[0].amount_min == 1001.0


def test_parse_ptr_pdf_text_skips_header_rows():
    """`Filing ID #...` and column-header rows must not produce trades."""
    pdf_text = (
        "Filing ID #20034201\n"
        "ID Owner Asset Transaction Date Notification Amount Cap.\n"
        "Apple Inc. (AAPL) [ST] P 03/16/2026 03/20/2026 $1,001 - $15,000\n"
    )
    trades = parse_ptr_pdf_text(pdf_text)
    assert len(trades) == 1
    assert trades[0].ticker == "AAPL"


def test_house_filing_dataclass_is_frozen():
    f = HouseFiling(
        doc_id="X",
        last_name="A", first_name="B",
        state_dst=None,
        filing_year=2026,
        filing_date=date(2026, 1, 1),
        filing_type="P",
    )
    # Frozen dataclass — assignments raise.
    import dataclasses
    assert dataclasses.is_dataclass(f)
    try:
        f.doc_id = "Y"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("HouseFiling should be frozen")
