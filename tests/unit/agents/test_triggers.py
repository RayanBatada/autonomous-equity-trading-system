from datetime import date

import pytest

from sma.agents.triggers import TriggerConfig, refresh_order, tickers_needing_refresh
from sma.ingest.store import Store


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    # Seed price history (today + yesterday for return calc)
    s.conn.execute(
        """
        INSERT INTO prices (ticker, date, open, high, low, close, adj_close, volume, source, run_id)
        VALUES
            ('AAPL', DATE '2026-04-23', 100, 100, 100, 100, 100, 1000, 'yfinance', 1),
            ('AAPL', DATE '2026-04-24', 100, 100, 100, 102, 102, 1000, 'yfinance', 1),
            ('TSLA', DATE '2026-04-23', 100, 100, 100, 100, 100, 1000, 'yfinance', 1),
            ('TSLA', DATE '2026-04-24', 100, 100, 100, 108, 108, 1000, 'yfinance', 1),
            ('MSFT', DATE '2026-04-23', 100, 100, 100, 100, 100, 1000, 'yfinance', 1),
            ('MSFT', DATE '2026-04-24', 100, 100, 100, 99,  99,  1000, 'yfinance', 1)
        """
    )
    yield s
    s.close()


def test_quiet_day_returns_only_price_movers(store):
    """No earnings, no filings; only TSLA crosses 5% price move."""
    out = tickers_needing_refresh(
        store, date(2026, 4, 24), ["AAPL", "TSLA", "MSFT"], TriggerConfig()
    )
    # AAPL +2%, TSLA +8%, MSFT -1%; 5% threshold catches only TSLA
    assert out == ["TSLA"]


def test_dual_source_uses_yfinance_precedence_not_alpaca(store):
    """When both yfinance + alpaca rows exist for a ticker-day, the price-move
    trigger must use the canonical (yfinance) row — not compare a yfinance row
    against an alpaca row of the SAME day. AAPL is +2% via yfinance (no trigger);
    a divergent alpaca 4/24 row must not manufacture a false move."""
    store.conn.execute(
        """
        INSERT INTO prices (ticker, date, open, high, low, close, adj_close, volume, source, run_id)
        VALUES ('AAPL', DATE '2026-04-24', 110, 110, 110, 110, 110, 1000, 'alpaca', 1)
        """
    )
    out = tickers_needing_refresh(
        store, date(2026, 4, 24), ["AAPL", "TSLA", "MSFT"], TriggerConfig()
    )
    assert "AAPL" not in out  # yfinance 102 vs 100 = +2%, below 5%
    assert out == ["TSLA"]


def test_earnings_release_triggers(store):
    """Released earnings (eps_actual IS NOT NULL) trigger; future-scheduled don't."""
    store.conn.execute(
        """
        INSERT INTO earnings
            (ticker, report_date, eps_estimate, eps_actual,
             revenue_estimate, revenue_actual, source, run_id)
        VALUES
            ('AAPL', DATE '2026-04-24', 1.5, 1.7, 100, 105, 'finnhub', 1)
        """
    )
    out = tickers_needing_refresh(
        store, date(2026, 4, 24), ["AAPL", "MSFT"], TriggerConfig()
    )
    assert "AAPL" in out
    assert "MSFT" not in out


def test_scheduled_earnings_without_actual_does_not_trigger(store):
    """Calendar-only earnings (eps_actual IS NULL) is upcoming, not a trigger."""
    store.conn.execute(
        """
        INSERT INTO earnings
            (ticker, report_date, eps_estimate, eps_actual,
             revenue_estimate, revenue_actual, source, run_id)
        VALUES
            ('AAPL', DATE '2026-04-24', 1.5, NULL, 100, NULL, 'finnhub', 1)
        """
    )
    out = tickers_needing_refresh(
        store, date(2026, 4, 24), ["AAPL"], TriggerConfig()
    )
    # Only the price-move trigger could include AAPL; AAPL is at +2%, below 5% threshold.
    assert "AAPL" not in out


def test_8k_filing_triggers(store):
    store.conn.execute(
        """
        INSERT INTO filings (ticker, filed_at, filing_type, accession_no, url, run_id)
        VALUES ('AAPL', TIMESTAMP '2026-04-24 14:00:00', '8-K', 'A1', 'http://x', 1)
        """
    )
    out = tickers_needing_refresh(
        store, date(2026, 4, 24), ["AAPL"], TriggerConfig()
    )
    assert "AAPL" in out


def test_filing_type_outside_material_set_does_not_trigger(store):
    """A non-material filing (e.g. SC 13G) does NOT trigger by default."""
    store.conn.execute(
        """
        INSERT INTO filings (ticker, filed_at, filing_type, accession_no, url, run_id)
        VALUES ('AAPL', TIMESTAMP '2026-04-24 14:00:00', 'SC 13G', 'A1', 'http://x', 1)
        """
    )
    out = tickers_needing_refresh(
        store, date(2026, 4, 24), ["AAPL"], TriggerConfig()
    )
    # AAPL at +2% < 5% threshold, no earnings, non-material filing -> no trigger
    assert "AAPL" not in out


def test_price_threshold_configurable(store):
    """Lower threshold catches more tickers."""
    out = tickers_needing_refresh(
        store, date(2026, 4, 24), ["AAPL", "TSLA", "MSFT"],
        TriggerConfig(price_move_pct=0.01),
    )
    # AAPL +2%, TSLA +8%, MSFT -1% all >= 1% absolute
    assert set(out) == {"AAPL", "TSLA", "MSFT"}


def test_filing_types_configurable(store):
    """Custom material_filing_types includes SC 13G when explicitly listed."""
    store.conn.execute(
        """
        INSERT INTO filings (ticker, filed_at, filing_type, accession_no, url, run_id)
        VALUES ('AAPL', TIMESTAMP '2026-04-24 14:00:00', 'SC 13G', 'A1', 'http://x', 1)
        """
    )
    out = tickers_needing_refresh(
        store, date(2026, 4, 24), ["AAPL"],
        TriggerConfig(material_filing_types=["SC 13G"]),
    )
    assert "AAPL" in out


def test_universe_filter_applied(store):
    """Tickers triggered but not in universe are excluded."""
    out = tickers_needing_refresh(
        store, date(2026, 4, 24), ["MSFT"], TriggerConfig()
    )
    # TSLA crosses 5% threshold but not in universe; MSFT is below threshold
    assert "TSLA" not in out
    assert out == []


def test_dedups_when_multiple_triggers_fire(store):
    """One ticker with both earnings + price move appears once in output."""
    store.conn.execute(
        """
        INSERT INTO earnings
            (ticker, report_date, eps_estimate, eps_actual,
             revenue_estimate, revenue_actual, source, run_id)
        VALUES ('TSLA', DATE '2026-04-24', 1.0, 1.5, 50, 60, 'finnhub', 1)
        """
    )
    out = tickers_needing_refresh(
        store, date(2026, 4, 24), ["TSLA"], TriggerConfig()
    )
    assert out == ["TSLA"]  # exactly one entry


def test_returns_sorted_list(store):
    """Output is a sorted list, not a set, for deterministic downstream behavior."""
    store.conn.execute(
        """
        INSERT INTO earnings
            (ticker, report_date, eps_estimate, eps_actual,
             revenue_estimate, revenue_actual, source, run_id)
        VALUES
            ('AAPL', DATE '2026-04-24', 1.0, 1.5, 50, 60, 'finnhub', 1),
            ('MSFT', DATE '2026-04-24', 1.0, 1.5, 50, 60, 'finnhub', 1)
        """
    )
    out = tickers_needing_refresh(
        store, date(2026, 4, 24), ["TSLA", "MSFT", "AAPL"], TriggerConfig()
    )
    assert out == sorted(out)
    assert "AAPL" in out and "MSFT" in out and "TSLA" in out


# ---------------------------------------------------------------------------
# refresh_order: held names must never be starved (2026-07-29)
# ---------------------------------------------------------------------------


def test_refresh_order_puts_held_names_first():
    """Held tickers must come FIRST, ahead of event-triggered ones.

    The agents run has a hard 19:58 ET deadline cutoff that stops STARTING new
    tickers (it must release the writer lock before decide fires at 20:00). Any
    held name sitting in the tail of the list simply never gets a thesis.
    """
    order = refresh_order(
        triggered=["AAPL", "ZZZ"], held=["MU", "DKNG"], universe=["AAPL", "ZZZ", "MU", "DKNG"]
    )
    assert order[:2] == ["DKNG", "MU"], f"held names must lead, got {order}"
    assert set(order) == {"AAPL", "ZZZ", "MU", "DKNG"}


def test_refresh_order_dedupes_a_held_name_that_also_triggered():
    """A held name that ALSO has an event trigger appears exactly once."""
    order = refresh_order(triggered=["MU"], held=["MU"], universe=["MU", "AAPL"])
    assert order == ["MU"]


def test_refresh_order_drops_held_names_outside_the_universe():
    """A position in a delisted/removed name must not be sent to the LLM."""
    order = refresh_order(triggered=["AAPL"], held=["DELISTED"], universe=["AAPL"])
    assert order == ["AAPL"]


def test_refresh_order_is_deterministic():
    """Same inputs -> same order, regardless of input ordering."""
    a = refresh_order(triggered=["B", "A"], held=["Z", "Y"], universe=["A", "B", "Y", "Z"])
    b = refresh_order(triggered=["A", "B"], held=["Y", "Z"], universe=["A", "B", "Y", "Z"])
    assert a == b
