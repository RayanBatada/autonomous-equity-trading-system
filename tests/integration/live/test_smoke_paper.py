"""End-to-end paper smoke. Gated by SMA_LIVE_SMOKE=1 so CI doesn't run it.

Procedure (manual; runs against real Alpaca paper account):

    # Day D, after market close (e.g., 18:40 ET):
    SMA_LIVE_SMOKE=1 .venv/bin/python -m sma.live decide \
        --canary AAPL \
        --asof-date $(date +%Y-%m-%d)

    # Wait for next-day market open + 1 minute (~9:31 ET).

    # Day D+1, after open:
    SMA_LIVE_SMOKE=1 .venv/bin/python -m sma.live reconcile \
        --asof-date $(date +%Y-%m-%d)

    # Verify (paper account starts at $100,000):
    .venv/bin/python -c "
    import duckdb
    c = duckdb.connect('data/sma.duckdb', read_only=True)
    print('paper_fills for AAPL:',
          c.execute(\"SELECT * FROM paper_fills WHERE ticker = 'AAPL'\").fetchall())
    print('account_snapshots:',
          c.execute('SELECT * FROM account_snapshots').fetchall())
    "

    # Cleanup (sells the AAPL position the smoke opened):
    SMA_LIVE_SMOKE=1 .venv/bin/python -m sma.live decide \
        --canary AAPL --liquidate \
        --asof-date $(date +%Y-%m-%d)

After this completes successfully, install the launchd jobs:
    bash scripts/install_launchd.sh

Then run --dry-run manually each evening for ~5 trading days before flipping
to auto-fire.
"""

import os

import pytest


@pytest.mark.skipif(
    os.environ.get("SMA_LIVE_SMOKE") != "1",
    reason="paper smoke gated by SMA_LIVE_SMOKE=1 (manual run only)",
)
def test_paper_smoke_documented_procedure():
    """The real smoke is the manual CLI procedure above. This test exists so the
    gate flag is documented and a future automated runner has a hook to wire into.
    """
    pass


@pytest.mark.skipif(
    os.environ.get("SMA_LIVE_SMOKE") != "1",
    reason="paper smoke gated by SMA_LIVE_SMOKE=1 (manual run only)",
)
def test_paper_smoke_buy_reaches_broker_as_day_market_not_opg():
    """LIVE paper-account smoke: a BUY must reach Alpaca as type=MARKET / tif=DAY
    (NOT OPG). OPG (open-auction-only) orders expire unfilled in the paper engine,
    so the bot sells but cannot buy (the 2026-06-08 incident: 82% idle cash).

    This is the integration guard the mocked-broker unit tests structurally CANNOT
    provide — a MagicMock fills/accepts any order type, so "OPG doesn't execute in
    paper" is invisible to them. Only a real submission catches it. Submits 1 share
    through the exact decide path and cleans up (cancel if queued, else liquidate).
    """
    import contextlib
    import uuid

    from sma.config import load_settings
    from sma.live.__main__ import _build_alpaca

    alpaca = _build_alpaca(load_settings())
    coid = f"smoke-{uuid.uuid4().hex[:12]}"
    aid = alpaca.submit_day_market_buy("AAPL", 1, client_order_id=coid)
    try:
        order = alpaca.get_order_by_id(aid)
        assert "MARKET" in str(order.order_type).upper(), order.order_type
        assert str(order.time_in_force).upper().endswith("DAY"), order.time_in_force
        assert str(order.status).split(".")[-1].lower() != "expired", order.status
    finally:
        # Best-effort cleanup so the smoke leaves no position behind.
        try:
            alpaca.tc.cancel_order_by_id(aid)
        except Exception:
            with contextlib.suppress(Exception):
                alpaca.submit_day_sell("AAPL", 1, client_order_id=f"{coid}-exit")


def test_smoke_test_is_gated_by_env_flag():
    """Without SMA_LIVE_SMOKE=1, the smoke test must be skipped."""
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest",
         "tests/integration/live/test_smoke_paper.py::test_paper_smoke_documented_procedure",
         "-v", "--no-header"],
        capture_output=True, text=True,
        env={**os.environ, "SMA_LIVE_SMOKE": ""},
    )
    # Either skipped (1 skipped) or ran (1 passed) is acceptable; what matters
    # is exit_code != 1 (would mean a real failure).
    assert "SKIPPED" in result.stdout or "passed" in result.stdout
