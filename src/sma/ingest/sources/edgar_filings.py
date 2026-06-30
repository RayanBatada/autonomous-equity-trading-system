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

            limit = min(len(accession_nos), self.max_filings_per_ticker)
            for i in range(limit):
                form = forms[i]
                if form not in ("10-K", "10-Q", "8-K"):
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

        if rows:
            store.conn.executemany(
                "INSERT OR REPLACE INTO filings "
                "(ticker, filing_type, filed_at, accession_no, url, summary, run_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return IngestResult(self.name, len(rows), "ok", None)
