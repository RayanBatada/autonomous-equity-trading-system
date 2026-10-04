"""Senate Periodic Transaction Report (PTR) ingest source.

Senate disclosures are at https://efdsearch.senate.gov/. Unlike the House
Clerk's bulk FD.zip, Senate has no bulk export — we have to scrape:

  1. GET  /search/home/                    — get csrfmiddlewaretoken
  2. POST /search/home/  (prohibition_agreement=1)  — accept TOS
  3. POST /search/report/data/  (report_types=[11]) — paginated JSON
  4. GET  /search/view/ptr/<uuid>/         — HTML transactions table
     GET  /search/view/paper/<uuid>/       — scanned PDF (skipped: needs OCR)

Schema is shared with the House source — both write into
`politician_trades` with `chamber='senate'` and `politician_disclosure_docs`
with the same doc_id (the UUID at the end of the URL).

Anti-bot notes:
  - Site is Akamai-fronted; plain `curl` returns 403. Use a real browser UA.
  - Polite rate limit: 1.25–2 s between requests. We sleep 1.5s.
  - If a /search/view/... GET redirects back to /search/home/, the session
    expired silently — caller should re-run create_senate_session() and retry.

See `specs/2026-04-30-ticker-on-demand-spec.md` for the related design and
the parallel House implementation in `politician_trades.py`.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date

import requests
from loguru import logger
from lxml import etree
from lxml import html as lxml_html

# ---- URLs ------------------------------------------------------------------

SENATE_BASE = "https://efdsearch.senate.gov"
SENATE_HOME = f"{SENATE_BASE}/search/home/"
SENATE_SEARCH = f"{SENATE_BASE}/search/"
SENATE_DATA = f"{SENATE_BASE}/search/report/data/"

# Browser-shape UA — Akamai blocks default `python-requests/*`.
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
HTTP_TIMEOUT = 30
POLITE_SLEEP_SECONDS = 1.5

# Report-type code for "Periodic Transaction Report" in the EFD search.
# Other codes: 7=Annual, 10=Amendment, 1=New Filer, etc. PTR is always 11.
REPORT_TYPE_PTR = 11
PAGE_SIZE = 100


# ---- Datatypes -------------------------------------------------------------


@dataclass(frozen=True)
class SenateFiling:
    """One row of the Senate search results, pre-parse.

    `doc_id` is the UUID at the end of the disclosure URL. `is_paper`
    distinguishes the HTML-table format (parseable here) from the scanned
    PDF format (requires OCR, currently skipped).
    """
    doc_id: str
    first_name: str
    last_name: str
    office: str
    filing_date: date | None
    is_paper: bool


# ---- Session / TOS dance ---------------------------------------------------


def _extract_csrf_from_html(html_text: str) -> str:
    """Pull the csrfmiddlewaretoken value out of a Django form HTML page."""
    m = re.search(
        r'name=["\']csrfmiddlewaretoken["\']\s+value=["\']([^"\']+)["\']',
        html_text,
    )
    if not m:
        raise RuntimeError(
            "csrfmiddlewaretoken not found in Senate EFD home page HTML — "
            "Django form layout may have changed."
        )
    return m.group(1)


def create_senate_session() -> requests.Session:
    """Return a requests.Session that has accepted the EFD TOS.

    Mutates the session's cookies + default headers so subsequent search
    calls succeed. The TOS acceptance is per-session; cookies expire and
    callers will need to recreate the session every few hours.
    """
    s = requests.Session()
    s.headers.update({
        "User-Agent": BROWSER_UA,
        "DNT": "1",
        "Origin": SENATE_BASE,
    })

    # 1. GET landing page — establishes cookies + gives us the form CSRF token.
    r = s.get(SENATE_HOME, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    form_csrf = _extract_csrf_from_html(r.text)

    # 2. POST acceptance. Referer required (Django's CSRF middleware checks it).
    r = s.post(
        SENATE_HOME,
        data={
            "csrfmiddlewaretoken": form_csrf,
            "prohibition_agreement": "1",
        },
        headers={"Referer": SENATE_HOME},
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()

    # 3. Pin the cookie-CSRF on the session for subsequent POSTs to
    #    /search/report/data/ (Django's double-submit pattern: cookie value
    #    sent both as X-CSRFToken header and inside the form body).
    cookie_csrf = s.cookies.get("csrftoken") or s.cookies.get("csrf")
    if not cookie_csrf:
        raise RuntimeError(
            "csrftoken cookie not present after TOS acceptance — "
            "EFD session setup failed."
        )
    s.headers.update({
        "X-CSRFToken": cookie_csrf,
        "Referer": SENATE_SEARCH,
    })
    return s


# ---- Search ----------------------------------------------------------------


def _parse_search_row(row: list) -> SenateFiling | None:
    """Decode one DataTables row → SenateFiling.

    Row shape (current EFD schema, May 2026):
        [first_name, last_name, office, link_html, date_received]
    """
    if len(row) < 5:
        logger.warning("senate search row malformed (len < 5): %r", row)
        return None
    first, last, office, link_html, date_received = row[:5]

    # link_html is something like:
    #   '<a href="/search/view/ptr/<uuid>/">Periodic Transaction Report</a>'
    # or paper-filed equivalent under /search/view/paper/<uuid>/.
    href_match = re.search(r'href=["\']([^"\']+)["\']', link_html)
    if not href_match:
        logger.warning("senate search row has no href: %r", link_html)
        return None
    href = href_match.group(1)

    is_paper = "/view/paper/" in href
    uuid_match = re.search(r"/view/(?:ptr|paper)/([^/]+)/?", href)
    if not uuid_match:
        logger.warning("senate href doesn't match /view/(ptr|paper)/UUID/: %r", href)
        return None
    doc_id = uuid_match.group(1)

    # date_received may include time, e.g. "05/19/2026 14:32:11"; only the date matters.
    filing_date: date | None = None
    date_match = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", str(date_received))
    if date_match:
        mm, dd, yyyy = (int(x) for x in date_match.groups())
        try:
            filing_date = date(yyyy, mm, dd)
        except ValueError:
            filing_date = None

    return SenateFiling(
        doc_id=doc_id,
        first_name=_strip_html(str(first)),
        last_name=_strip_html(str(last)),
        office=_strip_html(str(office)),
        filing_date=filing_date,
        is_paper=is_paper,
    )


def _strip_html(s: str) -> str:
    """Strip HTML tags (the EFD search occasionally wraps names in <a> tags)."""
    return re.sub(r"<[^>]+>", "", s).strip()


def iter_senate_ptrs(
    *,
    session: requests.Session,
    submitted_start: date,
    submitted_end: date | None = None,
    page_size: int = PAGE_SIZE,
) -> Iterator[SenateFiling]:
    """Yield every Senate PTR submitted in [submitted_start, submitted_end].

    Paginated via DataTables-style POST. Sleeps POLITE_SLEEP_SECONDS between
    pages to stay under the unofficial 1 req/sec ceiling.
    """
    cookie_csrf = session.cookies.get("csrftoken") or session.cookies.get("csrf")
    if not cookie_csrf:
        raise RuntimeError("session has no csrftoken cookie — recreate it")

    start = 0
    end_str = ""
    if submitted_end is not None:
        end_str = submitted_end.strftime("%m/%d/%Y 23:59:59")
    while True:
        payload = {
            "start": str(start),
            "length": str(page_size),
            "report_types": f"[{REPORT_TYPE_PTR}]",
            "filer_types": "[]",
            "submitted_start_date": submitted_start.strftime("%m/%d/%Y 00:00:00"),
            "submitted_end_date": end_str,
            "candidate_state": "",
            "senator_state": "",
            "office_id": "",
            "first_name": "",
            "last_name": "",
            "csrfmiddlewaretoken": cookie_csrf,
        }
        r = session.post(SENATE_DATA, data=payload, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        body = r.json()
        rows = body.get("data") or []
        if not rows:
            return
        for row in rows:
            parsed = _parse_search_row(row)
            if parsed is not None:
                yield parsed
        start += len(rows)
        total = int(body.get("recordsFiltered") or body.get("recordsTotal") or 0)
        if total and start >= total:
            return
        time.sleep(POLITE_SLEEP_SECONDS)


# ---- Per-filing fetch + parse ---------------------------------------------


def fetch_senate_ptr_html(
    doc_id: str, *, session: requests.Session,
) -> tuple[str, str]:
    """Return (response_url, html_text) for a single PTR.

    Caller should check `response_url == SENATE_HOME` — that's the soft
    session-expired signal (redirect to TOS page). Recreate the session
    and retry on hit.
    """
    url = f"{SENATE_BASE}/search/view/ptr/{doc_id}/"
    r = session.get(url, timeout=HTTP_TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    return r.url, r.text


# Importing here so the file can be imported without sma.ingest.sources
# .politician_trades in case of an import cycle (unlikely but defensive).
from sma.ingest.sources.politician_trades import (  # noqa: E402
    ParsedTrade,
    already_parsed,
    store_disclosure_outcome,
)

# Amount strings on the Senate page can be "$1,001 - $15,000" OR
# "$1,000,000 +" (open-ended high band, denoted with a +). Capture both.
_AMOUNT_RANGE_RE = re.compile(
    r"\$([0-9,]+)(?:\.\d+)?\s*[-–]\s*\$([0-9,]+)(?:\.\d+)?"
)
_AMOUNT_OPEN_RE = re.compile(r"\$([0-9,]+)(?:\.\d+)?\s*\+")
_TRANSACTION_DATE_RE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")


def _parse_amount_range(text: str) -> tuple[float | None, float | None]:
    """Parse Senate amount cell. Returns (min, max). Open-ended ("$1M+")
    returns (1_000_000, None) so the caller knows the upper bound is unknown."""
    t = text.strip()
    m = _AMOUNT_RANGE_RE.search(t)
    if m:
        return float(m.group(1).replace(",", "")), float(m.group(2).replace(",", ""))
    m = _AMOUNT_OPEN_RE.search(t)
    if m:
        return float(m.group(1).replace(",", "")), None
    return None, None


def _normalize_senate_order_type(s: str) -> str:
    """Map Senate's 'Purchase'/'Sale (Partial)'/'Sale (Full)'/'Exchange' to
    the same single-char codes the House parser emits: 'P', 'S', 'S (partial)', 'E'."""
    s_lower = s.strip().lower()
    if "purchase" in s_lower:
        return "P"
    if "sale" in s_lower:
        return "S (partial)" if "partial" in s_lower else "S"
    if "exchange" in s_lower:
        return "E"
    return s.strip()[:32]  # preserve raw string for unrecognised types


def parse_senate_ptr_html(html_text: str) -> list[ParsedTrade]:
    """Parse the transactions table out of a Senate HTML PTR view.

    Expected column order (post-2018 schema):
        [#, Transaction Date, Owner, Ticker, Asset Name, Asset Type,
         Order Type, Amount, Comment]

    Skips non-Stock asset types so we don't pollute the politician_trades
    table with options/bonds/mutual-fund rows whose ticker mapping is fuzzy
    (matches the House source's behavior).
    """
    tree = lxml_html.fromstring(html_text)
    # The transactions table is the only/first table inside the .table-responsive
    # wrapper or the page body. Find ALL data rows.
    rows: list[ParsedTrade] = []
    for tr in tree.xpath("//table//tbody/tr"):
        cells = [_text(c) for c in tr.xpath("./td")]
        if len(cells) < 8:
            continue
        # Column indices per the documented schema.
        (_, txn_date_str, _owner, ticker_raw, asset_name,
         asset_type, order_type, amount_cell) = cells[:8]

        if asset_type.strip().lower() not in ("stock", "common stock", "stock option"):
            # Skip mutual funds, bonds, ETFs (still useful but not tracked yet).
            continue

        date_match = _TRANSACTION_DATE_RE.search(txn_date_str)
        if not date_match:
            continue
        mm, dd, yyyy = (int(x) for x in date_match.groups())
        try:
            txn_date = date(yyyy, mm, dd)
        except ValueError:
            continue

        ticker_clean = ticker_raw.strip().upper() or None
        # The page uses "--" or empty cell when there's no ticker (private
        # placement, partnership interest). Treat as None — caller's
        # aggregator already filters ticker IS NOT NULL.
        if ticker_clean in {"--", "-", "—", "N/A", ""}:
            ticker_clean = None

        amount_min, amount_max = _parse_amount_range(amount_cell)

        rows.append(ParsedTrade(
            ticker=ticker_clean,
            asset_description=asset_name.strip()[:500],
            asset_type=asset_type.strip()[:32] or None,
            transaction_type=_normalize_senate_order_type(order_type),
            transaction_date=txn_date,
            amount_min=amount_min,
            amount_max=amount_max,
        ))
    return rows


def _text(el) -> str:
    """Whitespace-normalized text of a lxml element (recursively)."""
    if el is None:
        return ""
    raw = etree.tostring(el, method="text", encoding="unicode")
    return re.sub(r"\s+", " ", raw).strip()


# ---- Storage helpers (reuse House's) --------------------------------------


def store_senate_trades(
    *, store, run_id: int, filing: SenateFiling, trades: list[ParsedTrade],
) -> int:
    """Insert parsed Senate trades. Returns count inserted (after dedup).

    Uses the same `politician_trades` table as the House source with
    chamber='senate'. office (e.g. 'Hawaii, Junior Senator') is parsed into
    a state hint stored in state_dst when present.
    """
    state_dst = _office_to_state(filing.office)
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
                [
                    filing.doc_id, "senate", filing.last_name, filing.first_name,
                    state_dst, filing.filing_date,
                    t.transaction_date, t.ticker, t.asset_description, t.asset_type,
                    t.transaction_type, t.amount_min, t.amount_max, run_id,
                ],
            )
            inserted += 1
        except Exception as e:
            logger.warning(
                "skip senate trade row doc={} ticker={} desc={!r}: {}",
                filing.doc_id, t.ticker, t.asset_description[:50], e,
            )
    return inserted


def _office_to_state(office: str) -> str | None:
    """Best-effort: pull the state from a Senate 'office' string like
    'Hawaii, Junior Senator' or 'WV - Senator'. Returns None if unparseable."""
    if not office:
        return None
    s = office.strip()
    # Comma format: "Hawaii, Junior Senator"
    if "," in s:
        return s.split(",", 1)[0].strip()[:40] or None
    # Dash format: "WV - Senator"
    if " - " in s:
        return s.split(" - ", 1)[0].strip()[:40] or None
    return s[:40] or None


# ---- Top-level run ---------------------------------------------------------


def run_senate_ingest(
    *,
    store,
    run_id: int,
    submitted_start: date,
    submitted_end: date | None = None,
    limit: int | None = None,
    session: requests.Session | None = None,
) -> dict:
    """Ingest Senate PTRs submitted in [submitted_start, submitted_end].

    Mirrors the House source's run_ingest signature. Paper-filed
    disclosures (scanned PDFs) are skipped — they require OCR and account
    for <5% of disclosures empirically. Idempotent: skips doc_ids already
    parsed.
    """
    summary = {
        "filings_seen": 0, "filings_parsed": 0, "filings_skipped": 0,
        "filings_paper_skipped": 0,
        "trades_inserted": 0, "errors": 0,
    }
    s = session or create_senate_session()

    filings_iter = iter_senate_ptrs(
        session=s, submitted_start=submitted_start, submitted_end=submitted_end,
    )
    filings = list(filings_iter)
    summary["filings_seen"] = len(filings)
    if limit is not None:
        filings = filings[:limit]

    for f in filings:
        if f.is_paper:
            summary["filings_paper_skipped"] += 1
            # Record the skip so we don't re-try every run.
            store_disclosure_outcome(
                store=store, doc_id=f.doc_id, chamber="senate",
                filing_year=f.filing_date.year if f.filing_date else submitted_start.year,
                filing_date=f.filing_date,
                parse_status="paper_filed_skipped", error="scanned PDF; OCR not implemented",
                transactions_parsed=0,
            )
            continue
        if already_parsed(store, f.doc_id, "senate"):
            summary["filings_skipped"] += 1
            continue
        try:
            response_url, html_text = fetch_senate_ptr_html(f.doc_id, session=s)
            if response_url == SENATE_HOME:
                # Session expired mid-iteration — recreate and retry once.
                logger.info("senate session expired; recreating + retrying doc={}", f.doc_id)
                s = create_senate_session()
                response_url, html_text = fetch_senate_ptr_html(f.doc_id, session=s)
            trades = parse_senate_ptr_html(html_text)
            inserted = store_senate_trades(
                store=store, run_id=run_id, filing=f, trades=trades,
            )
            store_disclosure_outcome(
                store=store, doc_id=f.doc_id, chamber="senate",
                filing_year=f.filing_date.year if f.filing_date else submitted_start.year,
                filing_date=f.filing_date,
                parse_status="ok", error=None, transactions_parsed=inserted,
            )
            summary["filings_parsed"] += 1
            summary["trades_inserted"] += inserted
            logger.info(
                "senate PTR doc={} ({} {}): {} trades parsed",
                f.doc_id, f.first_name, f.last_name, inserted,
            )
            time.sleep(POLITE_SLEEP_SECONDS)
        except Exception as e:
            summary["errors"] += 1
            store_disclosure_outcome(
                store=store, doc_id=f.doc_id, chamber="senate",
                filing_year=f.filing_date.year if f.filing_date else submitted_start.year,
                filing_date=f.filing_date,
                parse_status="parse_error", error=str(e)[:500],
                transactions_parsed=0,
            )
            logger.warning("failed to ingest senate doc={}: {}", f.doc_id, e)
    return summary


# ---- CLI -------------------------------------------------------------------


def _main() -> None:
    import argparse
    from datetime import timedelta
    from pathlib import Path

    from sma.ingest.store import Store
    from sma.locks import writer_lock

    parser = argparse.ArgumentParser(description="Ingest Senate PTRs")
    parser.add_argument("--start", type=str, default=None,
                        help="Earliest submitted_date to scan, YYYY-MM-DD. "
                             "Default: today - lookback-days.")
    parser.add_argument("--end", type=str, default=None,
                        help="Latest submitted_date (inclusive), YYYY-MM-DD; default today")
    parser.add_argument("--lookback-days", type=int, default=30,
                        help="When --start omitted, scan back N days from today. "
                             "Default 30 covers a typical weekly cron with margin.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of disclosures processed (oldest first).")
    parser.add_argument("--db", type=Path, default=Path("data/sma.duckdb"))
    args = parser.parse_args()

    today = date.today()
    if args.start:
        submitted_start = date.fromisoformat(args.start)
    else:
        submitted_start = today - timedelta(days=args.lookback_days)
    submitted_end = date.fromisoformat(args.end) if args.end else None

    # 2026-08-23: the Mac was off through Sunday morning; when it booted at
    # 19:11 ET the watchdog/launchd caught up senate-ingest, house-ingest,
    # and a stale backup within the same second — all racing the single
    # writer lock. House won; senate exhausted the default 30s timeout and
    # exited 1, skipping the week's Senate PTR ingest. The two jobs' nominal
    # Sunday fire times are already staggered 60 minutes (see
    # src/sma/schedule.py) specifically to avoid this — that staggering is
    # irrelevant to a boot-catchup, which fires every overdue job at once
    # regardless of nominal spacing. This job is not urgent (a weekly
    # disclosure refresh, kickable up to 8h late — see schedule.py's
    # late_kick_max_hours=8.0 for this label), so patience beats racing:
    # mirrors src/sma/backup/runner.py's writer_lock(..., timeout_s=900.0).
    with writer_lock(label="senate_ingest", timeout_s=900.0):
        store = Store(path=str(args.db)).connect()
        try:
            rid = store.allocate_run_id()
            summary = run_senate_ingest(
                store=store, run_id=rid,
                submitted_start=submitted_start,
                submitted_end=submitted_end,
                limit=args.limit,
            )
            print(f"summary: {summary}")
        finally:
            store.conn.close()
        # Sentinel INSIDE the writer lock (serialization contract, like every
        # other scheduled job) and only after success — without it the watchdog
        # could never see this Sunday job as done and re-kicked it at every
        # checkpoint (review 2026-07-20 HIGH).
        from datetime import UTC, datetime

        from sma.sentinels import write_sentinel
        write_sentinel(
            label="com.sma.senate-ingest.weekly",
            asof=date.today(),
            payload={
                "label": "com.sma.senate-ingest.weekly",
                "asof": date.today().isoformat(),
                "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "run_id": rid,
                "summary": str(summary),
            },
        )


if __name__ == "__main__":
    _main()
