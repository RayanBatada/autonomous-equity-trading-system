"""SEC EDGAR filings ingestion.

EDGAR keys companies by 10-digit CIK, not ticker. We download the SEC's
public ticker -> CIK mapping once per run, then fetch each company's recent
filings via the submissions API.

SEC requires a User-Agent header that includes a real contact. They will
block requests without one.
"""

from datetime import date, datetime

import httpx
from loguru import logger

from sma.ingest.sources.base import IngestResult
from sma.ingest.store import Store

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# Domestic 10-K/10-Q/8-K plus the foreign-private-issuer equivalents 20-F
# (annual report) and 6-K (periodic furnished report). Without these, FPIs
# that don't file domestic forms have ZERO filings data -- confirmed against
# live EDGAR submissions 2026-08-05: ARM/ASML/SPOT/TSM file only 20-F/6-K
# (no 10-K/10-Q/8-K at all), so they were entirely invisible to the filings
# features and thesis pipeline. 40-F (Canadian MJDS issuers) was considered
# but not added: SHOP is the only Canadian-domiciled name in the universe and
# its live filing history shows it as a US domestic filer (10-K/10-Q/8-K,
# most recently 2026-08-05) with 40-F/6-K only historical/legacy, so no
# universe name currently needs it. Amendment variants (20-F/A, 6-K/A) are
# intentionally excluded, matching the existing 10-K/A-etc. exclusion.
TRACKED_FORMS = ("10-K", "10-Q", "8-K", "20-F", "6-K")


class EdgarFilingsSource:
    name = "edgar"

    def __init__(self, user_agent: str, max_filings_per_ticker: int = 20):
        self.user_agent = user_agent
        self.max_filings_per_ticker = max_filings_per_ticker
        self._client = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
        )
        self._ticker_to_cik: dict[str, int] | None = None

    def _load_ticker_map(self) -> dict[str, int]:
        if self._ticker_to_cik is not None:
            return self._ticker_to_cik
        resp = self._client.get(TICKERS_URL)
        resp.raise_for_status()
        data = resp.json()
        out: dict[str, int] = {}
        for _, row in data.items():
            t = (row.get("ticker") or "").upper()
            cik = row.get("cik_str")
            if t and cik is not None:
                out[t] = int(cik)
        self._ticker_to_cik = out
        return out

    def fetch(
        self,
        tickers: list[str],
        asof_date: date,
        store: Store,
        run_id: int,
    ) -> IngestResult:
        try:
            tmap = self._load_ticker_map()
        except Exception as e:
            logger.error("edgar ticker map fetch failed: {}", e)
            return IngestResult(self.name, 0, "error", str(e))

        rows = []
        for t in tickers:
            cik = tmap.get(t)
            if cik is None:
                logger.info("edgar: no CIK for {}, skipping", t)
                continue
            try:
                url = SUBMISSIONS_URL_TEMPLATE.format(cik=cik)
                resp = self._client.get(url)
                if resp.status_code != 200:
                    logger.warning("edgar submissions {} returned {}", t, resp.status_code)
                    continue
                data = resp.json()
            except Exception as e:
                logger.warning("edgar submissions failed for {}: {}", t, e)
                continue

            recent = data.get("filings", {}).get("recent", {}) or {}
            accession_nos = recent.get("accessionNumber", []) or []
            forms = recent.get("form", []) or []
            filing_dates = recent.get("filingDate", []) or []
            primary_docs = recent.get("primaryDocument", []) or []

            # Filter by tracked form type FIRST, then take the most-recent N
            # matches. Slicing to max_filings_per_ticker before filtering
            # silently dropped every tracked filing for issuers whose most
            # recent filings are dominated by an untracked form (e.g. Morgan
            # Stanley's 20 most-recent EDGAR filings are all 424B2 debt
            # prospectuses) -- found in the 2026-08-03 data audit.
            #
            # No extra HTTP request is needed to widen the search: the
            # submissions "recent" block already returns up to ~1000 of the
            # filer's most recent filings in a single response (older
            # filings paginate into separate files under filings.files,
            # which we don't fetch), so scanning the whole list already
            # in hand is enough to find 20 matching forms even for
            # prospectus-heavy issuers. (checked against SEC docs 2026-08-03)
            matched = 0
            for i in range(len(accession_nos)):
                form = forms[i]
                if form not in TRACKED_FORMS:
                    continue
                accession = accession_nos[i]
                accession_compact = accession.replace("-", "")
                primary = primary_docs[i] if i < len(primary_docs) else ""
                filing_url = (
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                    f"{accession_compact}/{primary}"
                )
                try:
                    filed_at = datetime.fromisoformat(filing_dates[i])
                except ValueError:
                    continue
                rows.append((
                    t, form, filed_at, accession, filing_url, None, run_id,
                ))
                matched += 1
                if matched >= self.max_filings_per_ticker:
                    break

        if rows:
            store.conn.executemany(
                "INSERT OR REPLACE INTO filings "
                "(ticker, filing_type, filed_at, accession_no, url, summary, run_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return IngestResult(self.name, len(rows), "ok", None)
