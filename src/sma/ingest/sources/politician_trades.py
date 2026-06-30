"""Politician-trade disclosure ingest source.

Pulls Periodic Transaction Reports (PTRs) from the House Clerk's public
disclosure feed and parses each PDF for ticker/transaction metadata.
Senate is structurally similar but the EFD search requires session
cookies — deferred until v1.

Data flow:

  1. Download <year>FD.zip from disclosures-clerk.house.gov.
     Contains an XML index (<year>FD.xml) of all financial filings.
  2. Filter the index to FilingType='P' (PTR — the trade disclosures).
  3. For each PTR, skip if `politician_disclosure_docs` already records
     it as parsed; otherwise download the PDF from
     /public_disc/ptr-pdfs/<year>/<doc_id>.pdf.
  4. Extract text via pdfplumber.
  5. Regex-parse for transactions: ticker in parens, asset type in
     square brackets, transaction type, dates, amount range. Skip
     non-equity rows (mutual funds, options, bonds without tickers).
  6. INSERT into `politician_trades`. Record the disclosure outcome
     in `politician_disclosure_docs` so subsequent runs are idempotent.

Known limitations of v0 (2026-05-09):
  - PDF formats vary; the regex covers the most common single-line
    transaction layout. Multi-line entries or unusual broker formats
    may be skipped (the disclosure_docs row records `parse_status` so
    misses are visible).
  - No name-to-ticker resolution for assets without a parenthesized
    ticker (e.g. "Putnam Sustainable Future Fund"). Those rows are
    inserted with ticker=NULL and asset_description preserved.
  - Senate PTRs are not yet ingested.
  - This source is NOT yet wired into `sma.ingest.run` — invoke
    manually via `python -m sma.ingest.sources.politician_trades`
    until we've verified parsed-data quality on a real disclosure
    sample.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import requests
from loguru import logger

HOUSE_FD_ZIP_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
HOUSE_PTR_PDF_URL = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"
HTTP_TIMEOUT = 30
HTTP_USER_AGENT = "sma-trader/1.0 (research; contact: your-email@example.com)"

# Compiled regexes used by the line parser. Each PTR row in the PDF text
# typically takes the shape:
#   <Asset description> (<TICKER>) [<TYPE>]   <action>   <txn_date>  <notif_date>  $X - $Y
# Some rows include a "(partial)" suffix on the action; some don't have
# the ticker in parens at all. We capture what we can and let the caller
# decide whether to insert a row with ticker=NULL.
_TICKER_RE = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,5})\)")
_ASSET_TYPE_RE = re.compile(r"\[([A-Z]{2,3})\]")
_DATE_RE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")
_AMOUNT_RE = re.compile(
    r"\$([0-9,]+)(?:\.\d+)?\s*-\s*\$([0-9,]+)(?:\.\d+)?"
)
_TXN_TYPE_RE = re.compile(r"\b([PSE])(?:\s*\(partial\))?\b")


@dataclass(frozen=True)
class ParsedTrade:
    ticker: str | None
    asset_description: str
    asset_type: str | None
    transaction_type: str  # 'P', 'S', 'S (partial)', 'E'
    transaction_date: date
    amount_min: float | None
    amount_max: float | None


@dataclass(frozen=True)
class HouseFiling:
    doc_id: str
    last_name: str
    first_name: str
    state_dst: str | None
    filing_year: int
    filing_date: date | None
    filing_type: str  # 'P' for PTR, 'A' annual, 'C' candidate, etc.


def fetch_house_index(year: int, *, session: requests.Session | None = None) -> bytes:
    """Download <year>FD.zip and return its raw bytes.

    Use a fresh session with a real User-Agent — the House Clerk webserver
    rejects requests from unidentified clients (403)."""
    s = session or requests.Session()
    url = HOUSE_FD_ZIP_URL.format(year=year)
    resp = s.get(url, headers={"User-Agent": HTTP_USER_AGENT}, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.content


def parse_house_index(zip_bytes: bytes, year: int) -> list[HouseFiling]:
    """Extract all filings from the year's FD.zip XML index. Returns
    every filing (PTR + annual + candidate); caller filters by type."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        xml_name = f"{year}FD.xml"
        if xml_name not in zf.namelist():
            raise RuntimeError(f"{xml_name} not in House FD zip")
        with zf.open(xml_name) as f:
            tree = ET.parse(f)
    root = tree.getroot()
    out: list[HouseFiling] = []
    for member in root.findall("Member"):
        doc_id = member.findtext("DocID") or ""
        if not doc_id:
            continue
        filing_date_str = member.findtext("FilingDate") or ""
        filing_date_parsed: date | None = None
        if filing_date_str:
            try:
                filing_date_parsed = datetime.strptime(filing_date_str, "%m/%d/%Y").date()
            except ValueError:
                filing_date_parsed = None
        out.append(HouseFiling(
            doc_id=doc_id,
            last_name=member.findtext("Last") or "",
            first_name=member.findtext("First") or "",
            state_dst=member.findtext("StateDst") or None,
            filing_year=year,
            filing_date=filing_date_parsed,
            filing_type=member.findtext("FilingType") or "",
        ))
    return out


def fetch_ptr_pdf(doc_id: str, year: int, *, session: requests.Session | None = None) -> bytes:
    """Download a single PTR PDF. Returns the raw bytes."""
    s = session or requests.Session()
    url = HOUSE_PTR_PDF_URL.format(year=year, doc_id=doc_id)
    resp = s.get(url, headers={"User-Agent": HTTP_USER_AGENT}, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.content


def parse_ptr_pdf_text(pdf_text: str) -> list[ParsedTrade]:
    """Parse the text of a House PTR PDF for transactions.

    Naive line-based parser. Each transaction is on a "row" delimited
    roughly by the Amount column. We split on lines containing both a
    date and an amount range — those are transaction rows.
    """
    out: list[ParsedTrade] = []
    # Split into lines and merge wrapped lines into row-shaped strings.
    raw_lines = [ln.rstrip() for ln in pdf_text.splitlines() if ln.strip()]

    # A transaction row contains an amount range; merge any continuations
    # into the previous line until the next amount range is hit. Header /
    # column-label / certify-signature lines never contain a transaction so
    # we exclude them from the buffer outright (otherwise their text leaks
    # into the next merged row and breaks tickers/types via spurious matches).
    def _is_header_line(line: str) -> bool:
        s = line.replace("\x00", "")
        return (
            "Filing ID" in s
            or s.startswith("ID Owner Asset")
            or s.startswith("Name:")
            or s.startswith("Status:")
            or s.startswith("State/District:")
            or s.startswith("Digitally Signed")
            or "I CERTIFY" in s
            or "STOCK Act" in s
        )

    rows: list[str] = []
    buf: list[str] = []
    for raw_ln in raw_lines:
        # Strip NULL bytes pre-emptively — pdfplumber injects them in some
        # column-header rows because of font-subsetting tricks. Strip first
        # so amount-detection on the line itself sees clean text.
        ln = raw_ln.replace("\x00", "")
        if _is_header_line(ln):
            continue
        buf.append(ln)
        if _AMOUNT_RE.search(ln):
            rows.append(" ".join(buf))
            buf = []
    # Tail (no amount line) is dropped.

    for row in rows:
        amount_match = _AMOUNT_RE.search(row)
        if not amount_match:
            continue
        amount_min = float(amount_match.group(1).replace(",", ""))
        amount_max = float(amount_match.group(2).replace(",", ""))

        # Transaction type: 'P', 'S', 'S (partial)', 'E'
        tt_match = _TXN_TYPE_RE.search(row)
        if not tt_match:
            continue
        transaction_type = tt_match.group(1)
        if "(partial)" in row[tt_match.start() : tt_match.start() + 25].lower():
            transaction_type = f"{transaction_type} (partial)"

        # Transaction date: take the FIRST date in the row (PDF format is
        # transaction_date first, then notification_date).
        date_match = _DATE_RE.search(row)
        if not date_match:
            continue
        m, d, y = date_match.groups()
        try:
            transaction_date = date(int(y), int(m), int(d))
        except ValueError:
            continue

        ticker_match = _TICKER_RE.search(row)
        ticker = ticker_match.group(1) if ticker_match else None
        asset_type_match = _ASSET_TYPE_RE.search(row)
        asset_type = asset_type_match.group(1) if asset_type_match else None

        # Asset description = everything before the transaction-type/date area.
        # Stop at the asset-type bracket or the transaction-type marker.
        desc = row
        for cut in (asset_type_match, ticker_match):
            if cut is not None:
                desc = row[: cut.end()]
                break
        # Clean up multiple spaces
        desc = re.sub(r"\s{2,}", " ", desc).strip()

        out.append(ParsedTrade(
            ticker=ticker,
            asset_description=desc[:500],  # bound row length
            asset_type=asset_type,
            transaction_type=transaction_type,
            transaction_date=transaction_date,
            amount_min=amount_min,
            amount_max=amount_max,
        ))
    return out


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF blob via pdfplumber."""
    import pdfplumber  # local import — heavy dep
    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def iter_recent_house_ptrs(
    *, year: int, since: date | None = None,
    session: requests.Session | None = None,
) -> Iterator[HouseFiling]:
    """Yield every PTR filing for `year` (optionally since `since`)."""
    s = session or requests.Session()
    zip_bytes = fetch_house_index(year, session=s)
    filings = parse_house_index(zip_bytes, year)
    for f in filings:
        if f.filing_type != "P":
            continue
        if since is not None and (f.filing_date is None or f.filing_date < since):
            continue
        yield f


def already_parsed(store, doc_id: str, chamber: str) -> bool:
    row = store.conn.execute(
        "SELECT 1 FROM politician_disclosure_docs "
        "WHERE doc_id = ? AND chamber = ? AND parse_status = 'ok'",
        [doc_id, chamber],
    ).fetchone()
    return row is not None


def store_disclosure_outcome(
    *, store, doc_id: str, chamber: str, filing_year: int, filing_date: date | None,
    parse_status: str, error: str | None, transactions_parsed: int,
) -> None:
    now_ts = datetime.now(UTC).replace(tzinfo=None)
    store.conn.execute(
        """
        INSERT INTO politician_disclosure_docs
        (doc_id, chamber, filing_year, filing_date, parse_status, error,
         transactions_parsed, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (doc_id, chamber) DO UPDATE SET
            parse_status = EXCLUDED.parse_status,
            error = EXCLUDED.error,
            transactions_parsed = EXCLUDED.transactions_parsed,
            ingested_at = EXCLUDED.ingested_at
        """,
        [doc_id, chamber, filing_year, filing_date, parse_status, error,
         transactions_parsed, now_ts],
    )


def store_trades(
    *, store, run_id: int, filing: HouseFiling, trades: list[ParsedTrade],
) -> int:
    """Insert parsed trades. Returns count actually inserted (after dedup)."""
    inserted = 0
    for t in trades:
        try:
            store.conn.execute(
                """
                INSERT OR IGNORE INTO politician_trades
                (doc_id, chamber, last_name, first_name, state_dst, filing_date,
                 transaction_date, ticker, asset_description, asset_type,
                 transaction_type, amount_min, amount_max, run_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [filing.doc_id, "house", filing.last_name, filing.first_name,
                 filing.state_dst, filing.filing_date,
                 t.transaction_date, t.ticker, t.asset_description, t.asset_type,
                 t.transaction_type, t.amount_min, t.amount_max, run_id],
            )
            inserted += 1
        except Exception as e:
            logger.warning(
                "skip duplicate or bad trade row in doc {} ticker={} desc={!r}: {}",
                filing.doc_id, t.ticker, t.asset_description[:50], e,
            )
    return inserted


def run_ingest(
    *, store, run_id: int, year: int, since: date | None = None, limit: int | None = None,
) -> dict:
    """Ingest recent House PTRs into the store. Returns a summary dict."""
    session = requests.Session()
    summary = {"filings_seen": 0, "filings_parsed": 0, "filings_skipped": 0,
               "trades_inserted": 0, "errors": 0}

    filings = list(iter_recent_house_ptrs(year=year, since=since, session=session))
    summary["filings_seen"] = len(filings)
    if limit is not None:
        filings = filings[-limit:]

    for f in filings:
        if already_parsed(store, f.doc_id, "house"):
            summary["filings_skipped"] += 1
            continue
        try:
            pdf_bytes = fetch_ptr_pdf(f.doc_id, year, session=session)
            text = extract_pdf_text(pdf_bytes)
            trades = parse_ptr_pdf_text(text)
            inserted = store_trades(store=store, run_id=run_id, filing=f, trades=trades)
            store_disclosure_outcome(
                store=store, doc_id=f.doc_id, chamber="house",
                filing_year=year, filing_date=f.filing_date,
                parse_status="ok", error=None, transactions_parsed=inserted,
            )
            summary["filings_parsed"] += 1
            summary["trades_inserted"] += inserted
            logger.info(
                "house PTR doc_id={} ({} {}): {} trades parsed",
                f.doc_id, f.first_name, f.last_name, inserted,
            )
        except Exception as e:
            summary["errors"] += 1
            store_disclosure_outcome(
                store=store, doc_id=f.doc_id, chamber="house",
                filing_year=year, filing_date=f.filing_date,
                parse_status="parse_error", error=str(e)[:500],
                transactions_parsed=0,
            )
            logger.warning("failed to ingest doc_id={}: {}", f.doc_id, e)
    return summary


def aggregate_net_dollar_flow(
    *, store, ticker: str, asof: date, lookback_days: int = 30,
) -> float:
    """Net politician dollar flow into `ticker` over the lookback window.

    Buys add (amount_min + amount_max) / 2; sells subtract. Returns 0 if
    no disclosed trades match.
    """
    start = asof - __import__("datetime").timedelta(days=lookback_days)
    row = store.conn.execute(
        """
        SELECT COALESCE(SUM(
            CASE
                WHEN transaction_type = 'P' THEN (amount_min + amount_max) / 2.0
                WHEN transaction_type LIKE 'S%' THEN -1 * (amount_min + amount_max) / 2.0
                ELSE 0
            END
        ), 0.0)
        FROM politician_trades
        WHERE ticker = ?
          AND transaction_date >= ?
          AND transaction_date <= ?
        """,
        [ticker, start, asof],
    ).fetchone()
    return float(row[0]) if row else 0.0


# CLI for manual testing.
def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Ingest House politician trade PTRs")
    parser.add_argument("--year", type=int, default=datetime.now(UTC).year)
    parser.add_argument("--since", type=str, default=None,
                        help="Only ingest filings on/after YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of filings processed (newest first).")
    parser.add_argument("--db", type=Path, default=Path("data/sma.duckdb"))
    args = parser.parse_args()

    since = date.fromisoformat(args.since) if args.since else None

    from sma.ingest.store import Store
    from sma.locks import writer_lock
    with writer_lock(label="politician_ingest"):
        store = Store(path=str(args.db)).connect()
        try:
            rid = store.allocate_run_id()
            summary = run_ingest(store=store, run_id=rid, year=args.year,
                                 since=since, limit=args.limit)
            print(f"summary: {summary}")
        finally:
            store.conn.close()


if __name__ == "__main__":
    _main()
