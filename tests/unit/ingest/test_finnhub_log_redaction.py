"""The Finnhub key never reaches a log (flaw hunt 2026-10-01 D1).

requests' ConnectionError text carries the full request URL, and the Finnhub
client puts the key in the query string (token=...). fetch_with_retry and
the earnings-calendar path logged str(e) as is: 726 lines of
~/Library/Logs/sma/ingest.err.log held the key on 2026-10-04.
"""

from loguru import logger

from sma.ingest.sources._finnhub_retry import fetch_with_retry, redact_secrets

KEY = "d0abcDEF123secretKEY456"
URL_ERR = (
    "HTTPSConnectionPool(host='api.finnhub.io', port=443): Max retries exceeded with url: "
    f"/api/v1/company-news?symbol=HAL&from=2026-10-01&to=2026-10-02&token={KEY} "
    "(Caused by NameResolutionError(...))"
)


def test_redact_secrets_strips_token_and_keeps_the_rest():
    out = redact_secrets(URL_ERR)
    assert KEY not in out
    assert "token=REDACTED" in out
    assert "symbol=HAL" in out


def test_redact_secrets_handles_other_key_params():
    assert KEY not in redact_secrets(f"https://x/?apiKey={KEY}&a=1")
    assert KEY not in redact_secrets(f"https://x/?api_key={KEY}")


def test_fetch_with_retry_never_logs_the_key():
    lines = []
    sink = logger.add(lambda m: lines.append(str(m)), level="DEBUG")
    try:
        def call():
            raise ConnectionError(URL_ERR)

        assert fetch_with_retry(
            "finnhub_news", "HAL", call, is_retryable=lambda e: False, reason="x"
        ) is None
    finally:
        logger.remove(sink)
    assert lines
    assert not any(KEY in line for line in lines)
