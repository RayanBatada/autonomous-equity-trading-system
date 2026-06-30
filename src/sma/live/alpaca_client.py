"""Thin wrapper around alpaca-py TradingClient.

Normalizes types (Alpaca returns string fields like equity/cash; we want
floats), provides domain-specific submit helpers (DAY_OPG buy / DAY sell),
and isolates the alpaca-py SDK from the rest of sma.live so swapping brokers
later is a one-file change.
"""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetCalendarRequest,
    GetOrdersRequest,
    MarketOrderRequest,
)

ET = ZoneInfo("America/New_York")
NEXT_SESSION_LOOKAHEAD_DAYS = 10


class AlpacaClient:
    def __init__(self, *, trading_client: TradingClient):
        self.tc = trading_client

    @classmethod
    def paper_from_env(cls, api_key: str, secret_key: str) -> "AlpacaClient":
        """Construct against Alpaca's paper-trading endpoint."""
        tc = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
        return cls(trading_client=tc)

    def get_account(self) -> dict:
        a = self.tc.get_account()
        return {
            "equity": float(a.equity),
            "cash": float(a.cash),
            "buying_power": float(a.buying_power),
            "long_market_value": float(getattr(a, "long_market_value", 0.0)),
            "trading_blocked": bool(getattr(a, "trading_blocked", False)),
            "account_blocked": bool(getattr(a, "account_blocked", False)),
        }

    def get_positions(self) -> dict[str, dict]:
        positions = self.tc.get_all_positions()
        return {
            p.symbol: {
                "shares": int(float(p.qty)),
                "cost_basis": float(p.avg_entry_price),
            }
            for p in positions
        }

    def submit_day_opg_buy(
        self, ticker: str, shares: int, *, client_order_id: str | None = None
    ) -> str:
        req = MarketOrderRequest(
            symbol=ticker,
            qty=shares,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.OPG,
            client_order_id=client_order_id,
        )
        order = self.tc.submit_order(req)
        return order.id

    def submit_day_market_buy(
        self, ticker: str, shares: int, *, client_order_id: str | None = None
    ) -> str:
        """DAY market BUY. Used as a fallback when the OPG window is closed
        (Alpaca rejects OPG outside 19:00-09:28 ET). DAY orders submitted
        overnight are queued for the next regular session and typically fill
        near the open auction price — close enough to OPG for our purposes."""
        req = MarketOrderRequest(
            symbol=ticker,
            qty=shares,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        order = self.tc.submit_order(req)
        return order.id

    def submit_day_sell(
        self, ticker: str, shares: int, *, client_order_id: str | None = None
    ) -> str:
        req = MarketOrderRequest(
            symbol=ticker,
            qty=shares,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        order = self.tc.submit_order(req)
        return order.id

    def submit_market_sell(
        self, ticker: str, shares: int, *, client_order_id: str | None = None
    ) -> str:
        """Same as submit_day_sell — kept as a separate name for stop-loss-sweep
        callsites that read better as 'market sell'."""
        return self.submit_day_sell(ticker, shares, client_order_id=client_order_id)

    def get_order_by_client_order_id(self, client_order_id: str) -> tuple[str, str] | None:
        """Return `(order_id, status)` for `client_order_id`, or None ONLY when the
        broker confirms no such order exists (HTTP 404). Used to recover from a
        crash between Alpaca accepting an order and the DB recording its id.

        FAIL CLOSED: any other error (network, auth, rate-limit, 5xx) re-raises —
        the caller must NOT treat a lookup failure as "no order", because that
        would resubmit an order that may already be live (double-submit).
        `status` is lower-cased (e.g. 'filled', 'accepted', 'rejected')."""
        try:
            order = self.tc.get_order_by_client_id(client_order_id)
        except APIError as e:
            if getattr(e, "status_code", None) == 404:
                return None  # broker confirms no such order
            raise
        if order is None:
            return None
        status = str(getattr(order.status, "value", order.status)).lower()
        return (str(order.id), status)

    def next_session_date(self, *, today: date) -> date:
        """Return the date of the next trading session strictly after `today`.

        For a Mon-Thu today, returns tomorrow. For Friday, returns Monday
        (skipping weekend). For day-before-holiday, skips the holiday.
        Used in pre-flight to verify next_session.date == tomorrow before
        submitting DAY_OPG orders that would otherwise sit overnight.
        """
        start = today + timedelta(days=1)
        end = today + timedelta(days=NEXT_SESSION_LOOKAHEAD_DAYS)
        req = GetCalendarRequest(start=start, end=end)
        cal = self.tc.get_calendar(filters=req)
        if not cal:
            raise RuntimeError(
                f"Alpaca returned empty calendar for {start}..{end} "
                f"(next session window after {today})"
            )
        return cal[0].date

    def sessions_between(self, *, start: date, end: date) -> list[date]:
        """Return trading session dates in [start, end] inclusive.

        Used by preflight staleness check to verify a long calendar gap was
        composed of holidays/weekends only — non-empty result means real
        trading sessions were missed and resume should abort.
        """
        if start > end:
            return []
        req = GetCalendarRequest(start=start, end=end)
        cal = self.tc.get_calendar(filters=req)
        return [c.date for c in cal]

    def get_order_by_id(self, order_id: str):
        """Fetch a single order by its Alpaca order id. Reconcile uses this to
        match orders to their decide asof via the intended_orders row, instead of
        a submission-date window — OPG-queued/catch-up/weekend orders submit on a
        DIFFERENT calendar day than their decide asof, so a date window returns
        the wrong orders and misses these."""
        return self.tc.get_order_by_id(order_id)

    def get_orders_for_date(self, target_date: date) -> list:
        """Return all orders (filled, canceled, or open) for the ET trading day
        spanning `target_date`. Uses ET-midnight-to-midnight bounds so daytime
        UTC gives both sides of the day correctly."""
        after = datetime.combine(target_date, time.min, tzinfo=ET)
        until = after + timedelta(days=1)
        req = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            after=after,
            until=until,
            limit=500,
        )
        return self.tc.get_orders(filter=req)
