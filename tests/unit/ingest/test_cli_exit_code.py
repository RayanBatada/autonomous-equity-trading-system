"""Regression: the ingest CLI's exit-code step imported a NON-EXISTENT `get_job`
from sma.schedule and crashed EVERY run with ImportError (after prices/sentinel
were written) — the recurring `ingest exit 1`. The real export is `get`. The
existing tests call ingest_run() directly and never reached this CLI step.
"""

from unittest.mock import MagicMock

from sma.ingest.__main__ import _ingest_quality_blocking


def test_ingest_quality_blocking_uses_valid_schedule_lookup():
    report = MagicMock()
    report.blocking_failures.return_value = ["all_tickers_have_price"]

    # Must NOT raise ImportError (the bug) and must pass the job's real waivers.
    out = _ingest_quality_blocking(report)

    assert out == ["all_tickers_have_price"]
    (waivers,), _ = report.blocking_failures.call_args
    assert isinstance(waivers, frozenset)


def test_ingest_quality_blocking_empty_when_no_blocking():
    report = MagicMock()
    report.blocking_failures.return_value = []
    assert _ingest_quality_blocking(report) == []
