"""Source protocol and IngestResult.

Each source under sources/ exposes one function:

    def fetch(tickers, asof_date, store, run_id) -> IngestResult: ...

The runner calls this, isolates exceptions per source, and writes the result
to ingest_log.
"""

from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol

from sma.ingest.store import Store

Status = Literal["ok", "circuit_open", "rate_limited", "error", "skipped"]


@dataclass(frozen=True)
class IngestResult:
    source: str
    rows_inserted: int
    status: Status
    error: str | None


class Source(Protocol):
    name: str

    def fetch(
        self,
        tickers: list[str],
        asof_date: date,
        store: Store,
        run_id: int,
    ) -> IngestResult: ...
