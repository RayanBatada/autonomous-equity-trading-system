"""Thin wrapper around alpaca-py TradingClient.

Normalizes types (Alpaca returns string fields like equity/cash; we want
floats), provides domain-specific submit helpers (DAY_OPG buy / DAY sell),
and isolates the alpaca-py SDK from the rest of sma.live so swapping brokers
later is a one-file change.

Retry (2026-08-24): every READ-ONLY method below is wrapped with
`_retry_on_transient_network_error`, which retries a call up to 3 times
(~2/4/8s backoff) ONLY on requests.exceptions.ConnectionError/Timeout.
alpaca-py's RESTClient._one_request calls requests.Session.request()
directly (see alpaca.common.rest); a DNS/connection failure raises those
UNWRAPPED, before any HTTP response exists and before it can be turned into
an alpaca.common.exceptions.APIError. That makes ConnectionError/Timeout a
reliable "never reached Alpaca's servers" signal -- safe to retry blind. An
APIError (a real 4xx/5xx response) is NEVER retried here and always
propagates unchanged -- see the decorator's docstring below.

2026-08-24: reconcile crashed hard on a transient DNS blip hitting GET
/v2/account (roving laptop between networks) -- get_account had zero retry
protection, and one blip killed the whole run. The Finnhub ingest sources
already have this class of armor (sma.ingest.sources._finnhub_retry); this
gives the broker client the same treatment, at the method boundary rather
than per-callsite, so nothing added here in the future goes un-armored.

submit_day_opg_buy / submit_day_market_buy / submit_day_sell /
submit_market_sell are deliberately NOT wrapped: for an order submission, a
ConnectionError does not reliably mean "never reached the server" (the
request bytes may already be out when the connection drops), so a blind
retry risks double-submitting a real order. Order submission stays
single-attempt at the HTTP layer for both ConnectionError/Timeout and
APIError -- a rejected or ambiguous order must not be resubmitted blindly.
"""

import functools
import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from time import sleep as _time_sleep
from zoneinfo import ZoneInfo

import requests
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetCalendarRequest,
    GetOrdersRequest,
    GetPortfolioHistoryRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)
from loguru import logger

from sma.live.quantity import as_qty

ET = ZoneInfo("America/New_York")
NEXT_SESSION_LOOKAHEAD_DAYS = 10

# Retry knobs for transient network failures on read-only Alpaca calls.
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_S = (2.0, 4.0, 8.0)
_TRANSIENT_NETWORK_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)

# Default (connect, read) timeout for every Alpaca HTTP call (flaw hunt
# 2026-10-01 A2). alpaca-py passes no timeout to requests, so a connection
# that died after it was established blocked forever: the watchdog hung from
# 15:53 on 2026-10-01 until a manual kickstart, and in ingest or decide the
# same hang holds the writer lock. The read value bounds the wait between
# bytes, not the whole response, so a slow but live response is unaffected.
# A timeout raises requests.exceptions.Timeout: read-only calls retry it
# (above); a submit that times out is recorded as submission_failed by decide
# and adopted by client_order_id at reconcile if the broker did accept it.
ALPACA_HTTP_TIMEOUT_S = (10.0, 60.0)


def with_default_timeout(rest_client, timeout=ALPACA_HTTP_TIMEOUT_S):
    """Give an alpaca-py RESTClient (TradingClient, StockHistoricalDataClient)
    a default timeout on its requests.Session. A caller's explicit timeout
    wins. Idempotent. Returns the client."""
    session = getattr(rest_client, "_session", None)
    if session is None or getattr(session, "_sma_default_timeout", None) is not None:
        return rest_client
    inner = session.request

    @functools.wraps(inner)
    def request(method, url, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = timeout
        return inner(method, url, **kwargs)

    session.request = request
    session._sma_default_timeout = timeout
    return rest_client


# Patchable sleep for retry tests. NOT named `time` -- `time` above is
# datetime.time, used throughout session_close_equity/session_window.
_SLEEP = _time_sleep


def _retry_on_transient_network_error(fn):
    """Decorator for READ-ONLY AlpacaClient methods: retry up to
    _RETRY_ATTEMPTS times (~2/4/8s backoff) on a TRANSIENT network failure
    (DNS resolution / connection refused / timeout) that happened BEFORE any
    HTTP response was received -- see the module docstring for why
    requests.exceptions.ConnectionError/Timeout specifically are that signal,
    and why submit_* methods do not get this decorator. Any other exception
    (notably alpaca.common.exceptions.APIError, a real HTTP response) is
    never caught here and propagates on the first attempt, unchanged.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        label = fn.__qualname__
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                return fn(*args, **kwargs)
            except _TRANSIENT_NETWORK_EXCEPTIONS as e:
                if attempt == _RETRY_ATTEMPTS:
                    raise
                delay = _RETRY_BACKOFF_S[attempt - 1]
                logger.warning(
                    "{}: transient network error on attempt {}/{} ({!r}); retrying in {:.0f}s",
                    label,
                    attempt,
                    _RETRY_ATTEMPTS,
                    e,
                    delay,
                )
                _SLEEP(delay)
        raise AssertionError("unreachable")

    return wrapper


def _session_dt(day: date, value) -> datetime:
    """Normalize an Alpaca Calendar open/close field to an ET-aware datetime.

    Alpaca sends the ET wall clock; alpaca-py parses it into a NAIVE datetime
    (current) or a `datetime.time` (older releases). Neither is UTC, so both
    are localized to ET rather than converted.
    """
    if isinstance(value, datetime):
        return value.replace(tzinfo=ET) if value.tzinfo is None else value.astimezone(ET)
    return datetime.combine(day, value, tzinfo=ET)


def _assert_fractional_is_day(shares: float, tif: TimeInForce, ticker: str) -> None:
    """Alpaca accepts a fractional quantity ONLY on time_in_force=DAY.

    Verified 2026-08-27 against the "Fractional orders (USD)" TIF matrix on
    docs.alpaca.markets/us/docs/orders-at-alpaca (page updated 2026-08-10):
    DAY yes for market/limit/stop/stop-limit; GTC, IOC, FOK, OPG and CLS all no.

    alpaca-py does NOT enforce this — `MarketOrderRequest(qty=0.5,
    time_in_force=TimeInForce.GTC)` constructs happily and is only rejected at
    the broker, i.e. at 18:35 on a night when the whole batch then fails. Check
    it here, where it is a loud local error instead of a silent no-trade night.
    """
    if float(shares) == int(shares):
        return
    if tif is not TimeInForce.DAY:
        raise ValueError(
            f"fractional qty {shares} for {ticker} requires time_in_force=DAY; "
            f"got {tif}. Alpaca rejects fractional GTC/IOC/FOK/OPG/CLS."
        )


class QuoteUnavailableError(RuntimeError):
    """No usable IEX quote or last trade to price a marketable limit off."""


@dataclass(frozen=True)
class RefPrice:
    """Reference prices for one name at one moment. `source` is 'quote' when
    the IEX NBBO-of-IEX was usable, 'trade' when it was missing/too wide and
    the latest IEX trade stood in for both sides."""

    bid: float
    ask: float
    source: str

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True)
class LimitOrderResult:
    order_id: str
    ticker: str
    side: str            # 'BUY' | 'SELL'
    qty: float
    limit_price: float
    ref: RefPrice
    client_order_id: str | None = None


@dataclass
class WorkingOrder:
    """One order the session sweep is responsible for. `stage` is 'limit'
    (first attempt), 'reprice' (the one re-priced limit) or 'market'
    (fallback). `done` flips once the order is terminal and accounted for."""

    order_id: str
    ticker: str
    side: str
    qty: float
    stage: str = "limit"
    filled_qty: float = 0.0
    status: str = "new"
    done: bool = False
    history: list = field(default_factory=list)


_TERMINAL = frozenset({"filled", "canceled", "expired", "rejected", "done_for_day"})


def _tick(price: float) -> float:
    # Alpaca / Reg NMS sub-penny rule: $0.01 at or above $1, $0.0001 below.
    return 0.01 if price >= 1.0 else 0.0001


def marketable_limit_price(side: str, ref: RefPrice, offset_bps: float) -> float:
    """ask + offset for a BUY, rounded UP to the tick; bid - offset for a SELL,
    rounded DOWN. Rounding away from the touch keeps the order marketable."""
    side = side.upper()
    if side == "BUY":
        raw = ref.ask * (1.0 + offset_bps / 10_000.0)
        t = _tick(raw)
        return round(math.ceil(raw / t - 1e-9) * t, 4)
    if side == "SELL":
        raw = ref.bid * (1.0 - offset_bps / 10_000.0)
        t = _tick(raw)
        return round(max(math.floor(raw / t + 1e-9) * t, t), 4)
    raise ValueError(f"side must be BUY or SELL, got {side!r}")


def _order_status(order) -> str:
    return str(getattr(order.status, "value", order.status)).lower()


class AlpacaClient:
    def __init__(
        self,
        *,
        trading_client: TradingClient,
        paper: bool = True,
        data_client=None,
        api_key: str | None = None,
        secret_key: str | None = None,
    ):
        self.tc = trading_client
        #: False only when this client is pointed at the REAL-money endpoint.
        #: Read by the real-money preflight and stamped into its log lines.
        self.paper = paper
        # Market-data client (IEX quotes for the intraday sessions). Built
        # lazily so jobs that never price a limit order never construct it.
        self._data_client = data_client
        self._data_keys = (api_key, secret_key)

    @property
    def data(self):
        if self._data_client is None:
            from alpaca.data.historical import StockHistoricalDataClient

            key, secret = self._data_keys
            if not key or not secret:
                raise RuntimeError("AlpacaClient has no market-data credentials")
            self._data_client = with_default_timeout(StockHistoricalDataClient(key, secret))
        return self._data_client

    @classmethod
    def paper_from_env(cls, api_key: str, secret_key: str) -> "AlpacaClient":
        """Construct against Alpaca's paper-trading endpoint."""
        tc = with_default_timeout(TradingClient(api_key=api_key, secret_key=secret_key, paper=True))
        return cls(trading_client=tc, paper=True, api_key=api_key, secret_key=secret_key)

    @classmethod
    def live_from_env(cls, api_key: str, secret_key: str) -> "AlpacaClient":
        """Construct against Alpaca's REAL-MONEY endpoint (api.alpaca.markets).

        Never call this directly from a job. Go through
        `sma.live.real_money.build_alpaca_client`, which will not hand back a
        live client unless every gate in `live.real_money` is satisfied.

        Live and paper credentials are NOT interchangeable at Alpaca — a paper
        key against this endpoint fails authentication rather than quietly
        trading the wrong book.
        """
        tc = with_default_timeout(
            TradingClient(api_key=api_key, secret_key=secret_key, paper=False)
        )
        return cls(trading_client=tc, paper=False, api_key=api_key, secret_key=secret_key)

    @_retry_on_transient_network_error
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

    @_retry_on_transient_network_error
    def get_positions(self) -> dict[str, dict]:
        positions = self.tc.get_all_positions()
        return {
            p.symbol: {
                # as_qty, not int(): Alpaca reports a fractional position as
                # "2.5" and int() would destroy half a share irrecoverably —
                # nothing downstream ever re-reads p.qty. Returns a plain int
                # for the whole-share book we hold today.
                "shares": as_qty(p.qty),
                "cost_basis": float(p.avg_entry_price),
            }
            for p in positions
        }

    def submit_day_opg_buy(
        self, ticker: str, shares: float, *, client_order_id: str | None = None
    ) -> str:
        _assert_fractional_is_day(shares, TimeInForce.OPG, ticker)
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
        self, ticker: str, shares: float, *, client_order_id: str | None = None
    ) -> str:
        """DAY market BUY. Used as a fallback when the OPG window is closed
        (Alpaca rejects OPG outside 19:00-09:28 ET). DAY orders submitted
        overnight are queued for the next regular session and typically fill
        near the open auction price — close enough to OPG for our purposes.

        Accepts a fractional `shares`: DAY market is exactly the one
        combination Alpaca allows a fractional quantity on."""
        _assert_fractional_is_day(shares, TimeInForce.DAY, ticker)
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
        self, ticker: str, shares: float, *, client_order_id: str | None = None
    ) -> str:
        _assert_fractional_is_day(shares, TimeInForce.DAY, ticker)
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
        self, ticker: str, shares: float, *, client_order_id: str | None = None
    ) -> str:
        """Same as submit_day_sell — kept as a separate name for stop-loss-sweep
        callsites that read better as 'market sell'."""
        return self.submit_day_sell(ticker, shares, client_order_id=client_order_id)

    @_retry_on_transient_network_error
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

    @_retry_on_transient_network_error
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

    @_retry_on_transient_network_error
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

    @_retry_on_transient_network_error
    def session_window(self, *, day: date) -> tuple[datetime, datetime] | None:
        """`day`'s regular-session (open, close) as ET-aware datetimes, or None
        when `day` is not an NYSE trading session.

        Callers that must know WHEN the session actually runs need this rather
        than `sessions_between`, which only answers "is this a trading day".
        Half-days are the reason: 2026-11-27 and 2026-12-24 open at 09:30 and
        close at 13:00, so any fixed 16:00 assumption is wrong on those days.

        Alpaca returns Calendar.open/close as NAIVE datetimes carrying the ET
        wall clock (probed live 2026-08-16: 2026-11-27 open 09:30, close
        13:00), so they are localized to ET here, never assumed UTC. Older
        alpaca-py typed these as `datetime.time`; both shapes are handled.
        """
        cal = self.tc.get_calendar(filters=GetCalendarRequest(start=day, end=day))
        for c in cal:
            if c.date != day:
                continue
            return (_session_dt(day, c.open), _session_dt(day, c.close))
        return None

    @_retry_on_transient_network_error
    def session_close_equity(self, *, day: date) -> tuple[float, str] | None:
        """Account equity at `day`'s session close, from portfolio-history.

        get_account().equity is a LIVE mark: after ~16:00 ET it prices the book
        off after-hours quotes, so an evening read is not that day's close. On
        2026-07-29 that read was off by thousands; on 2026-08-12 a 20:16 read
        stored 102,516.39 against a true close of 102,343.20 (0.17%).

        Two sources, best first:
        1. The published 1D bar for `day`. Authoritative, but Alpaca does not
           post it promptly — measured absent at 21:20 ET on the same session,
           so the evening reconcile essentially never gets it.
        2. The LAST 1Min point inside `day`'s regular session. Validated
           against published closes on 2026-08-10/11: within 0.012% and 0.008%,
           two orders of magnitude better than the after-hours read.

        Tier 2 used to demand an EXACT (16, 0) ET stamp (2026-08-16 fix). That
        returned None on any half-day — 2026-11-27 and 2026-12-24 close at
        13:00, so no 16:00 point exists — and on any session with a missing
        final bar, sending the caller back to the polluted get_account read
        that this method exists to avoid. The session bounds come from the
        trading calendar, so the shortened close is handled by the calendar
        rather than by a constant.

        Returns (equity, source_label), or None if neither is available yet —
        callers then fall back to get_account and record that they did.
        """

        def _et_date(ts: int) -> date:
            return datetime.fromtimestamp(ts, tz=UTC).astimezone(ET).date()

        try:
            daily = self.tc.get_portfolio_history(
                GetPortfolioHistoryRequest(period="1W", timeframe="1D")
            )
            for ts, eq in zip(daily.timestamp, daily.equity, strict=False):
                if eq is not None and _et_date(ts) == day:
                    return (float(eq), "portfolio_history_daily")
        except APIError:
            pass

        try:
            window = self.session_window(day=day)
        except Exception:  # noqa: BLE001 — a calendar blip must not lose the close
            window = (
                datetime.combine(day, time(9, 30), tzinfo=ET),
                datetime.combine(day, time(16, 0), tzinfo=ET),
            )
        if window is None:
            return None  # the calendar says `day` was not a session at all
        session_open, session_close = window

        try:
            minutes = self.tc.get_portfolio_history(
                GetPortfolioHistoryRequest(
                    start=datetime.combine(day, time(0, 0)),
                    end=datetime.combine(day, time(23, 59)),
                    timeframe="1Min",
                )
            )
        except APIError:
            return None

        # LAST point inside regular hours. Scanning for the max stamp rather
        # than taking the tail means a series that also carries pre/post-market
        # points (or arrives unordered) still resolves to the close.
        latest_t = None
        latest_eq = None
        for ts, eq in zip(minutes.timestamp, minutes.equity, strict=False):
            if eq is None:
                continue
            t = datetime.fromtimestamp(ts, tz=UTC).astimezone(ET)
            if t.date() != day or t < session_open or t > session_close:
                continue
            if latest_t is None or t > latest_t:
                latest_t, latest_eq = t, eq
        if latest_eq is None:
            return None
        return (float(latest_eq), "portfolio_history_1min_close")

    @_retry_on_transient_network_error
    def close_position(self, ticker: str) -> str | None:
        """Fully liquidate `ticker`, fraction included.

        `DELETE /v2/positions/{symbol}` with no qty/percentage closes the whole
        position, which is the only path that reliably leaves NO dust behind on
        a fractional holding: a computed SELL qty can miss by an ulp, and the
        remainder is then a position too small to be worth another order.

        Returns the broker order id, or None if the SDK does not surface one.
        """
        order = self.tc.close_position(ticker)
        return str(getattr(order, "id", "") or "") or None

    def cancel_order(self, order_id: str) -> None:
        """Cancel a still-open order by its Alpaca id. Used by the stop-loss sweep
        to pull last night's pending decide order for a name it is exiting, so the
        queued OPG buy can't re-open the position at the same open (the live
        equivalent of the simulator's _exited_today guard). Raises on a broker
        error (e.g. the order already filled/terminal) — the caller decides."""
        self.tc.cancel_order_by_id(order_id)

    @_retry_on_transient_network_error
    def list_open_orders(self) -> list[dict]:
        """Open (unfilled) orders as [{'id','symbol','side','qty'}, ...]. Used by
        the 09:25 pre-open guard to find queued SELLs that would oversell the live
        book (open a short) before they fill at the 09:30 open."""
        orders = self.tc.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
        )
        if len(orders) >= 500:
            logger.warning("list_open_orders hit the 500 cap — open-order read may be truncated")
        out = []
        for o in orders:
            out.append(
                {
                    "id": str(o.id),
                    "symbol": o.symbol,
                    "side": str(getattr(o.side, "value", o.side)).upper(),
                    "qty": as_qty(o.qty or 0),
                    # OPEN includes partially_filled: only the UNFILLED remainder would
                    # still execute, so the oversell gate must measure qty - filled.
                    "filled_qty": as_qty(getattr(o, "filled_qty", 0) or 0),
                }
            )
        return out

    @_retry_on_transient_network_error
    def cancel_all_open_orders(self) -> None:
        """Cancel ALL open orders. Used by the 09:25 pre-open divergence guard to
        pull the evening's queued decide orders when the live broker book has
        diverged from our ledger (a broker-side wipe/glitch), so nothing fills on
        the bad book at the 09:30 open. Raises on broker error — the caller pages."""
        self.tc.cancel_orders()

    @_retry_on_transient_network_error
    def get_order_by_id(self, order_id: str):
        """Fetch a single order by its Alpaca order id. Reconcile uses this to
        match orders to their decide asof via the intended_orders row, instead of
        a submission-date window — OPG-queued/catch-up/weekend orders submit on a
        DIFFERENT calendar day than their decide asof, so a date window returns
        the wrong orders and misses these."""
        return self.tc.get_order_by_id(order_id)

    @_retry_on_transient_network_error
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

    # ---- intraday sessions: quotes, marketable limits, sweep (2026-09-26) ----
    #
    # Nothing on the 20:00 decide / open path calls anything below. The
    # sessions (sma.live.session) are the only callers, and they ship off.

    @_retry_on_transient_network_error
    def get_latest_quote(self, ticker: str) -> dict:
        """Latest IEX quote as {'bid','ask','bid_size','ask_size','ts'}. IEX is
        the free feed; its quote is IEX's own top of book, not the NBBO, so it
        can be missing or wide on thinner names (callers check the spread)."""
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockLatestQuoteRequest

        sym = ticker.replace("-", ".")
        resp = self.data.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=sym, feed=DataFeed.IEX)
        )
        q = resp[sym] if isinstance(resp, dict) else resp
        return {
            "bid": float(getattr(q, "bid_price", 0) or 0),
            "ask": float(getattr(q, "ask_price", 0) or 0),
            "bid_size": float(getattr(q, "bid_size", 0) or 0),
            "ask_size": float(getattr(q, "ask_size", 0) or 0),
            "ts": getattr(q, "timestamp", None),
        }

    @_retry_on_transient_network_error
    def get_latest_trade_price(self, ticker: str) -> float:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockLatestTradeRequest

        sym = ticker.replace("-", ".")
        resp = self.data.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=sym, feed=DataFeed.IEX)
        )
        t = resp[sym] if isinstance(resp, dict) else resp
        return float(getattr(t, "price", 0) or 0)

    def reference_price(self, ticker: str, *, max_spread_bps: float = 50.0) -> RefPrice:
        """Quote if it is two-sided, uncrossed and no wider than
        `max_spread_bps`; else the latest IEX trade for both sides; else raise
        QuoteUnavailableError. Never prices off a zero or crossed book."""
        try:
            q = self.get_latest_quote(ticker)
        except Exception as e:  # noqa: BLE001 - fall through to the trade print
            logger.warning("quote read failed for {}: {!r}; trying last trade", ticker, e)
            q = {"bid": 0.0, "ask": 0.0}
        bid, ask = q["bid"], q["ask"]
        if bid > 0 and ask > 0 and ask >= bid:
            spread_bps = (ask - bid) / ((ask + bid) / 2.0) * 10_000.0
            if spread_bps <= max_spread_bps:
                return RefPrice(bid=bid, ask=ask, source="quote")
            logger.info(
                "{} IEX quote {:.4f}/{:.4f} is {:.0f}bp wide (> {:.0f}); using last trade",
                ticker, bid, ask, spread_bps, max_spread_bps,
            )
        px = self.get_latest_trade_price(ticker)
        if px > 0:
            return RefPrice(bid=px, ask=px, source="trade")
        raise QuoteUnavailableError(f"no usable IEX quote or trade for {ticker}")

    def submit_marketable_limit(
        self,
        ticker: str,
        side: str,
        shares_or_notional: float,
        *,
        offset_bps: float,
        tif: TimeInForce = TimeInForce.DAY,
        is_notional: bool = False,
        fractional: bool = False,
        fractional_precision: int = 9,
        max_spread_bps: float = 50.0,
        client_order_id: str | None = None,
        ref: RefPrice | None = None,
    ) -> LimitOrderResult:
        """Marketable limit: BUY at ask + offset_bps, SELL at bid - offset_bps,
        off a fresh IEX quote (or `ref` when the caller already priced it).

        `shares_or_notional` is a share count, or dollars when is_notional=True
        (converted to shares at the LIMIT price, so the order can never cost
        more than the notional). Whole shares unless `fractional`. DAY is the
        default and the only TIF Alpaca accepts a fractional qty on.

        Like every submit_* here: single attempt, never retried."""
        side = side.upper()
        if ref is None:
            ref = self.reference_price(ticker, max_spread_bps=max_spread_bps)
        limit = marketable_limit_price(side, ref, offset_bps)
        if is_notional:
            raw = float(shares_or_notional) / limit
            if fractional:
                scale = 10 ** fractional_precision
                qty = math.floor(raw * scale) / scale
            else:
                qty = float(math.floor(raw))
        else:
            qty = float(shares_or_notional)
        if qty <= 0:
            raise ValueError(
                f"{ticker}: {shares_or_notional} {'USD' if is_notional else 'shares'} "
                f"rounds to zero shares at limit {limit}"
            )
        qty = int(qty) if float(qty).is_integer() else qty
        _assert_fractional_is_day(qty, tif, ticker)
        req = LimitOrderRequest(
            symbol=ticker.replace("-", "."),
            qty=qty,
            side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
            time_in_force=tif,
            limit_price=limit,
            client_order_id=client_order_id,
        )
        order = self.tc.submit_order(req)
        return LimitOrderResult(
            order_id=str(order.id),
            ticker=ticker,
            side=side,
            qty=qty,
            limit_price=limit,
            ref=ref,
            client_order_id=client_order_id,
        )

    def submit_day_market(
        self, ticker: str, side: str, shares: float, *, client_order_id: str | None = None
    ) -> str:
        """DAY market order either side. The session sweep's market fallback."""
        if side.upper() == "BUY":
            return str(self.submit_day_market_buy(ticker, shares, client_order_id=client_order_id))
        return str(self.submit_day_sell(ticker, shares, client_order_id=client_order_id))

    def _wait_terminal(self, order_id: str, *, sleep_fn, attempts: int = 10, delay_s: float = 1.0):
        """Poll an order until it is terminal (a cancel is asynchronous at
        Alpaca: 'pending_cancel' can still fill). Returns the last order read."""
        order = self.get_order_by_id(order_id)
        for _ in range(attempts):
            if _order_status(order) in _TERMINAL:
                return order
            sleep_fn(delay_s)
            order = self.get_order_by_id(order_id)
        return order

    def sweep_unfilled(
        self,
        working: list[WorkingOrder],
        *,
        session: str,
        deadline: datetime,
        after_minutes: float,
        offset_bps: float,
        reprice_once: bool,
        fallback: str,
        max_spread_bps: float = 50.0,
        fractional: bool = False,
        now_fn=None,
        sleep_fn=None,
        poll_s: float = 15.0,
        on_replace=None,
        coid_fn=None,
    ) -> list[WorkingOrder]:
        """Cancel-and-replace sweep for the session's resting limit orders.

        Round 1, at min(now + after_minutes, deadline): cancel whatever is
        still open, wait for the cancel to settle, then for the unfilled
        remainder EITHER re-price once at a fresh quote (reprice_once) OR go
        straight to the fallback. Round 2 (only after a re-price), at
        min(now + after_minutes, deadline): cancel again and send the
        remainder as a DAY market order when fallback='market'. With
        fallback='leave' the last limit is NOT cancelled: it rests,
        price-bounded, until its DAY expiry and reconcile records any fill.

        `on_replace(old: WorkingOrder, new: WorkingOrder, detail: dict)` is
        called after every replacement is ACCEPTED so the caller can write its
        audit row; `coid_fn(old, stage)` supplies the replacement's
        deterministic client_order_id. Returns every order touched (originals
        and replacements), each with its final status and filled_qty.
        """
        now_fn = now_fn or (lambda: datetime.now(ET))
        sleep_fn = sleep_fn or _SLEEP
        all_orders: list[WorkingOrder] = list(working)
        if fallback not in ("market", "leave"):
            raise ValueError(f"fallback must be 'market' or 'leave', got {fallback!r}")

        def _refresh(w: WorkingOrder) -> None:
            o = self.get_order_by_id(w.order_id)
            w.status = _order_status(o)
            w.filled_qty = float(as_qty(getattr(o, "filled_qty", 0) or 0))
            if w.status in _TERMINAL:
                w.done = True

        def _wait_until(t: datetime, live: list[WorkingOrder]) -> None:
            while True:
                for w in live:
                    if not w.done:
                        try:
                            _refresh(w)
                        except Exception as e:  # noqa: BLE001 - keep polling others
                            logger.warning("sweep[{}]: refresh {} failed: {!r}",
                                           session, w.order_id, e)
                if all(w.done for w in live):
                    return
                remaining = (t - now_fn()).total_seconds()
                if remaining <= 0:
                    return
                sleep_fn(min(poll_s, remaining))

        def _cancel_remainder(w: WorkingOrder) -> float:
            """Cancel w if still open; return the unfilled remainder (0 if the
            order filled in the meantime)."""
            if not w.done:
                try:
                    self.cancel_order(w.order_id)
                except Exception as e:  # noqa: BLE001 - may have just filled
                    logger.info("sweep[{}]: cancel {} raised {!r}; re-reading",
                                session, w.order_id, e)
                o = self._wait_terminal(w.order_id, sleep_fn=sleep_fn)
                w.status = _order_status(o)
                w.filled_qty = float(as_qty(getattr(o, "filled_qty", 0) or 0))
                w.done = w.status in _TERMINAL
                if not w.done:
                    # Cancel never settled: do NOT replace, or the remainder
                    # could fill twice. Leave it to the DAY expiry + reconcile.
                    logger.warning("sweep[{}]: {} still {} after cancel; not replacing",
                                   session, w.order_id, w.status)
                    return 0.0
            if w.status == "rejected":
                return 0.0  # the broker refused it; a replacement would be too
            rem = w.qty - w.filled_qty
            if not fractional:
                rem = float(math.floor(rem + 1e-9))
            return rem if rem > 1e-9 else 0.0

        def _replace(w: WorkingOrder, stage: str, rem: float) -> WorkingOrder | None:
            coid = coid_fn(w, stage) if coid_fn else None
            rem_q = int(rem) if float(rem).is_integer() else rem
            try:
                if stage == "reprice":
                    res = self.submit_marketable_limit(
                        w.ticker, w.side, rem_q, offset_bps=offset_bps,
                        max_spread_bps=max_spread_bps, client_order_id=coid,
                    )
                    new = WorkingOrder(order_id=res.order_id, ticker=w.ticker, side=w.side,
                                       qty=rem_q, stage="reprice")
                    detail = {"limit_price": res.limit_price, "ref": res.ref,
                              "client_order_id": coid}
                else:
                    oid = self.submit_day_market(w.ticker, w.side, rem_q, client_order_id=coid)
                    new = WorkingOrder(order_id=str(oid), ticker=w.ticker, side=w.side,
                                       qty=rem_q, stage="market")
                    detail = {"limit_price": None, "ref": None, "client_order_id": coid}
            except Exception as e:  # noqa: BLE001 - one name must not sink the sweep
                logger.error("sweep[{}]: {} replacement for {} {} failed: {!r}",
                             session, stage, w.side, w.ticker, e)
                w.history.append({"replace_failed": stage, "error": repr(e)})
                return None
            w.history.append({"replaced_by": new.order_id, "stage": stage})
            all_orders.append(new)
            if on_replace is not None:
                on_replace(w, new, detail)
            return new

        def _round(live: list[WorkingOrder], stage: str) -> list[WorkingOrder]:
            out = []
            for w in live:
                rem = _cancel_remainder(w)
                if rem <= 0:
                    continue
                new = _replace(w, stage, rem)
                if new is not None:
                    out.append(new)
            return out

        start = now_fn()
        first = min(start + timedelta(minutes=after_minutes), deadline)
        _wait_until(first, working)
        if all(w.done for w in working):
            return all_orders

        if reprice_once:
            repriced = _round([w for w in working if not w.done or w.filled_qty < w.qty],
                              "reprice")
            if not repriced:
                return all_orders
            second = min(now_fn() + timedelta(minutes=after_minutes), deadline)
            _wait_until(second, repriced)
            if all(w.done for w in repriced):
                return all_orders
            pending = repriced
        else:
            pending = [w for w in working if not w.done or w.filled_qty < w.qty]

        if fallback == "market":
            _round(pending, "market")
        # 'leave': the last limit keeps resting until its DAY expiry. It is
        # price-bounded, and reconcile records whatever fills later.
        return all_orders
