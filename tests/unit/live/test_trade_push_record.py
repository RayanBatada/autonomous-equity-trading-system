"""record_trade_push (2026-09-04): persists tonight's push to
data/sentinels/com.sma.trade-push-<asof>.json via the existing atomic
sentinel writer, so sma.live.reconcile._detect_trade_push_drift can verify
it against booked fills later -- ntfy.sh's own cache expires in 12h and
nothing else records what was actually pushed. See sma.live.trade_push's
module docstring and __main__.py's decide flow (persisted right after the
send attempt, regardless of whether it succeeded).

tests/conftest.py's autouse _isolate_sentinel_dir fixture points
SMA_SENTINEL_DIR at a per-test temp dir, so these tests never touch the real
data/sentinels/.
"""

from datetime import date

from sma.live.trade_push import TRADE_PUSH_SENTINEL_LABEL, record_trade_push
from sma.sentinels import read_sentinel


def test_record_trade_push_persists_all_fields():
    asof = date(2026, 9, 4)
    orders = [{
        "ticker": "LLY", "side": "SELL", "shares": 1,
        "decide_price": 1160.08, "order_notional": 1160.08,
        "order_pct_of_equity": 0.01, "full_exit": False,
    }]

    record_trade_push(
        asof=asof,
        title="SMA trades - Fri 9/4",
        message="2026-09-04\nSELL LLY — 1.0% of equity\nEquity $116,175 | 11 positions",
        orders=orders,
        equity=116_174.92,
        delivered=True,
    )

    sentinel = read_sentinel(label=TRADE_PUSH_SENTINEL_LABEL, asof=asof)
    assert sentinel is not None
    assert sentinel["label"] == TRADE_PUSH_SENTINEL_LABEL
    assert sentinel["asof"] == "2026-09-04"
    assert sentinel["title"] == "SMA trades - Fri 9/4"
    assert "SELL LLY" in sentinel["message"]
    assert sentinel["delivered"] is True
    assert sentinel["equity"] == 116_174.92
    assert sentinel["orders"] == orders
    assert sentinel["completed_at"].endswith("Z")


def test_record_trade_push_persists_even_when_delivered_false():
    """The whole point: ntfy.sh's own cache is gone in 12h either way, so
    what was CLAIMED must be recorded regardless of whether send_ntfy
    actually got it out the door."""
    asof = date(2026, 9, 4)
    record_trade_push(
        asof=asof,
        title="SMA trades - Fri 9/4",
        message="2026-09-04\nBUY AAPL — 5.0% of equity\nEquity $100,000 | 3 positions",
        orders=[{"ticker": "AAPL", "side": "BUY", "shares": 25,
                  "decide_price": 200.0, "order_notional": 5000.0,
                  "order_pct_of_equity": 0.05, "full_exit": False}],
        equity=100_000.0,
        delivered=False,
    )

    sentinel = read_sentinel(label=TRADE_PUSH_SENTINEL_LABEL, asof=asof)
    assert sentinel is not None
    assert sentinel["delivered"] is False


def test_record_trade_push_empty_orders_persists_fine():
    asof = date(2026, 9, 4)
    record_trade_push(
        asof=asof,
        title="SMA trades - Fri 9/4",
        message=(
            "2026-09-04\nNo trades tonight — book unchanged (5 holds)\n"
            "Equity $50,000 | 5 positions"
        ),
        orders=[],
        equity=50_000.0,
        delivered=True,
    )

    sentinel = read_sentinel(label=TRADE_PUSH_SENTINEL_LABEL, asof=asof)
    assert sentinel is not None
    assert sentinel["orders"] == []
