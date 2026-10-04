"""Replay: "what would decide have done on date X?" -- read-only.

Motivation (2026-09-01): this question keeps needing answers by hand -- the
thesis-veto study (~/.sma-pit/thesis-veto-study-2026-08-05) spent hours
reconstructing past decide nights with a bespoke script, and the 2026-08-31
missed-rebalance counterfactual (network outage that evening -- decide never
ran) was approximated with a quick script that could not replicate the real
selection logic (sector-neutralization, hold_rank hysteresis, the theses
overlay, the rebalance dead zone, min_hold). This module runs the REAL code
path -- the same strategy class, the same risk-rail pipeline, the same order
translation `sma.live.decide.decide_once` uses every night -- against stored
predictions/theses for a past `asof`, and returns what it WOULD have ordered.

Guarantees:
  * Never submits an order. `decide_once` is always called with
    `dry_run=True` (its own tested contract: the submit loop and every DB
    write below it live strictly after that branch's early return), and the
    AlpacaClient handed to it is additionally wrapped in `_NoSubmitAlpaca`,
    which raises on any `submit_*`/`get_order_by_client_order_id` call. Two
    independent reasons the guarantee holds, not one.
  * Never writes a sentinel: this module never imports/calls
    `sma.sentinels.write_sentinel`.
  * Never mutates the DB: the DuckDB connection built here
    (`replay_connection`) is a fresh `:memory:` connection with the real
    database ATTACHed READ_ONLY -- there is no writable handle in this
    module at all, so a stray `INSERT`/`UPDATE` would fail at the DuckDB
    layer even if one were (incorrectly) added later.
  * Never needs the writer_lock: every connection opened here is read-only.

Fidelity trick (the veto study's proven pattern): strategy construction is
IDENTICAL to the real decide job (`sma.live.__main__._build_decide_strategy`,
reused directly) except the `Predictor` is swapped for
`StoredPredictionsPredictor`, which returns whatever the real predict job
already wrote to the `predictions` table for `asof` instead of invoking the
model. Point-in-time correctness for theses (a rerun agent could otherwise
write a thesis dated `asof` but created well after `asof`'s real decide
would have run) is enforced by `replay_connection`: it ATTACHes the real db
READ_ONLY under a `live` schema, defines `theses` locally as a view filtered
to `created_at < asof 20:00 ET`, and sets `search_path='main, live'` so
every OTHER unqualified table name (prices, earnings, paper_fills,
account_snapshots, predictions, politician_trades, ...) falls through to the
real `live.*` table untouched.
"""

from __future__ import annotations

import time as _time_module
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from datetime import time as time_cls
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb

from sma.backtest.strategies.base import StrategyDecision
from sma.db_connect import _LOCK_HINTS
from sma.live.alpaca_client import AlpacaClient
from sma.live.decide import (
    CatastrophicLossAbortError,
    _clean_snapshot_equities,
    _drawdown_from_peak,
    _last_prices_per_ticker,
    _latest_buy_dates,
    _load_prices,
    decide_once,
)
from sma.live.orders import STALE_PRICE_DAYS, Order
from sma.live.quantity import QTY_EPS, is_zero
from sma.live.sizing import SizingPolicy
from sma.risk.drawdown import check_drawdown
from sma.risk.earnings_blackout import in_earnings_blackout
from sma.risk.rails import RiskRails
from sma.risk.sector_cap import check_sector_cap
from sma.sectors import sector_for

ET = ZoneInfo("America/New_York")
DEFAULT_TARGET = "ret_30d_forward"
# Matches the real decide job's scheduled fire time (see __main__.py's
# module docstring: "decide -- 18:35 ET" historically, 20:00 ET currently).
# Theses/fills timestamped AT OR AFTER this wall-clock on `asof` could not
# have been visible to a real decide run that evening and are excluded.
DECIDE_TIME_ET = time_cls(20, 0)

BOOK_MODES = ("current", "asof", "empty")


class SubmitBlockedError(RuntimeError):
    """Raised if replay's read-only pipeline ever reaches an order-submission
    call. Should be unreachable (decide_once's dry_run=True branch returns
    before the submit loop) -- this is defense in depth, not the primary
    guarantee. See the module docstring."""


class _NoSubmitAlpaca:
    """Wraps an AlpacaClient-like object (or a `_SyntheticBook`) so any
    order-submission call raises instead of doing anything. `get_account`
    and `get_positions` -- the only two methods `decide_once` calls on its
    dry_run=True path -- pass through untouched."""

    _BLOCKED = (
        "submit_day_opg_buy",
        "submit_day_market_buy",
        "submit_day_sell",
        "submit_market_sell",
        "get_order_by_client_order_id",
    )

    def __init__(self, inner):
        self._inner = inner

    def get_account(self) -> dict:
        return self._inner.get_account()

    def get_positions(self) -> dict[str, dict]:
        return self._inner.get_positions()

    def __getattr__(self, name):
        if name in self._BLOCKED:
            def _blocked(*_args, **_kwargs):
                raise SubmitBlockedError(
                    f"replay is read-only: AlpacaClient.{name} must never be "
                    "called from a replay run"
                )
            return _blocked
        raise AttributeError(name)


@dataclass
class _SyntheticBook:
    """Duck-types the two AlpacaClient methods decide_once's dry-run path
    calls, for `--book asof` / `--book empty` (no real broker read)."""

    equity: float
    cash: float
    positions: dict[str, dict]

    def get_account(self) -> dict:
        return {
            "equity": self.equity,
            "cash": self.cash,
            "buying_power": self.cash,
            "long_market_value": self.equity - self.cash,
            "trading_blocked": False,
            "account_blocked": False,
        }

    def get_positions(self) -> dict[str, dict]:
        return dict(self.positions)


class StoredPredictionsPredictor:
    """Predictor duck-type: `predict_for(asof_date, universe)` returns
    whatever the real predict job already wrote to `predictions` for that
    date, instead of running the model. The proven fidelity trick from the
    thesis-veto study -- same interface XGBoostTopKStrategy expects
    (`sma.model.predictor.Predictor.predict_for`), so it drops in with no
    strategy-side change.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection, target: str = DEFAULT_TARGET):
        self._conn = conn
        self._target = target

    def predict_for(self, asof_date: date, universe: list[str]) -> dict[str, float]:
        rows = self._conn.execute(
            # ARG_MAX(value, computed_at) picks the most-recently-computed row
            # per ticker in case predict ran more than once for the same date.
            "SELECT ticker, ARG_MAX(predicted_value, computed_at) "
            "FROM predictions WHERE asof_date = ? AND target = ? "
            "AND ticker = ANY(?) GROUP BY ticker",
            [asof_date, self._target, list(universe)],
        ).fetchall()
        return {t: float(v) for t, v in rows if v is not None}


class _ConnShim:
    """Duck-types `sma.ingest.store.Store` for the one attribute every
    caller here reads: `.conn`."""

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn


def replay_connection(
    db_path: Path,
    *,
    asof: date,
    decide_time_et: time_cls = DECIDE_TIME_ET,
    retries: int = 12,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
) -> duckdb.DuckDBPyConnection:
    """Fresh READ_ONLY connection to `db_path` with `theses` point-in-time
    filtered to `created_at < asof @ decide_time_et ET`. Every other table
    (prices, earnings, paper_fills, account_snapshots, predictions,
    politician_trades, ...) passes through unfiltered via `search_path`.

    Retries on a transient writer-lock conflict, same backoff as
    `sma.db_connect.read_only_connect` (which this mirrors but cannot reuse
    directly -- that helper opens a direct file connection; this one needs
    an in-memory connection to ATTACH the file under an alias so `theses`
    can be locally shadowed by a filtered view while everything else still
    resolves against the real table).
    """
    # `theses.created_at` is DuckDB's `DEFAULT CURRENT_TIMESTAMP`, written by
    # a process running on this ET-timezone host into a plain (tz-naive)
    # TIMESTAMP column -- it is naive ET wall-clock, NOT UTC (confirmed
    # against real data: real theses for an asof night land at created_at
    # ~19:45-19:58, right before the 20:00 ET decide fire; interpreting that
    # as UTC would place thesis-writing at ~15:45 ET, mid-session, which
    # doesn't happen).
    #
    # UPDATE 2026-09-27: `paper_fills.filled_at` is now ALSO confirmed naive
    # ET wall-clock, not UTC as this comment previously claimed (verified
    # against the Alpaca orders endpoint: a fill stored as 09:32:29 is
    # 13:32:29Z at the broker -- duckdb's Python client converts alpaca-py's
    # tz-aware UTC datetime to this host's local [ET] timezone before storing
    # it into a naive TIMESTAMP column; see `sma.live.reconcile._record_fills`
    # and `sma.live.decide._latest_buy_dates`). So the two columns share ONE
    # convention after all. `reconstruct_book_asof` below compares its naive-ET
    # cutoff directly (the old ET -> UTC conversion was fixed 2026-10-01), and
    # decide's min-hold entry dates take an explicit `asof` cutoff, so replay
    # never sees a fill a real decide run that night could not have seen.
    cutoff_naive_et = datetime.combine(asof, decide_time_et)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            conn = duckdb.connect(":memory:")
            conn.execute(f"ATTACH '{db_path}' AS live (READ_ONLY)")
            # CREATE VIEW cannot be a prepared statement (DuckDB rejects a `?`
            # parameter in DDL), so the cutoff is inlined as a literal. Safe:
            # it is an internally-computed ISO timestamp, never user input.
            conn.execute(
                "CREATE VIEW theses AS SELECT * FROM live.theses "
                f"WHERE created_at < TIMESTAMP '{cutoff_naive_et.isoformat(sep=' ')}'"
            )
            conn.execute("SET search_path = 'main,live'")
            return conn
        except Exception as e:  # noqa: BLE001 - inspect message, re-raise non-lock
            if not any(hint in str(e) for hint in _LOCK_HINTS):
                raise
            last = e
            _time_module.sleep(min(base_delay * (2**attempt), max_delay))
    assert last is not None
    raise last


def _latest_snapshot_equity(conn, *, asof: date) -> tuple[float, float]:
    """(equity, cash) from the most recent account_snapshots row with
    asof_date <= asof. (0.0, 0.0) when there is no history at all yet."""
    row = conn.execute(
        "SELECT equity, cash FROM account_snapshots WHERE asof_date <= ? "
        "ORDER BY asof_date DESC LIMIT 1",
        [asof],
    ).fetchone()
    if row is None or row[0] is None:
        return 0.0, 0.0
    return float(row[0]), float(row[1] if row[1] is not None else 0.0)


def reconstruct_book_asof(
    conn,
    *,
    asof: date,
    decide_time_et: time_cls = DECIDE_TIME_ET,
) -> dict[str, dict]:
    """Positions reconstructed from cumulative `paper_fills`, counting only
    fills with `filled_at` strictly before `asof @ decide_time_et`.

    `filled_at` is naive ET wall-clock (see the convention note in
    `replay_connection` above), so the cutoff is the naive ET timestamp and
    is compared directly. Until 2026-10-01 it was converted ET -> naive UTC
    first, which pushed the effective cutoff 4-5 hours late (to midnight ET
    in summer); no real fill lands in that window, but it was wrong.

    This is a SIMPLE reconstruction, not a broker-accurate one -- documented
    limits:
      * cost_basis is a running weighted-average-cost approximation (BUY
        updates the average; SELL leaves it unchanged). It is only ever READ
        as decide's fallback dollar-value for a ticker with no current price
        (`sma.live.decide._to_dollars`), so it essentially never affects a
        replayed decision -- but it is not Alpaca's own lot accounting
        (no wash-sale/commission adjustments).
      * If reconcile has not yet run for a real batch that already filled at
        the broker, those fills are simply absent from `paper_fills` and
        this reconstruction under-counts the book -- exactly the situation
        found live on 2026-09-01 (Friday 2026-08-28's own decide batch had
        not been reconciled as of this writing). Use `--book current` for a
        broker-accurate reconstruction when reconcile is behind.
    """
    cutoff_naive_et = datetime.combine(asof, decide_time_et)
    rows = conn.execute(
        "SELECT ticker, side, filled_shares, fill_price, filled_at "
        "FROM paper_fills WHERE filled_shares > 0 AND filled_at IS NOT NULL "
        "ORDER BY ticker, filled_at"
    ).fetchall()
    shares: dict[str, float] = {}
    wavg_cost: dict[str, float] = {}
    for ticker, side, qty, price, filled_at in rows:
        if filled_at is None or filled_at >= cutoff_naive_et:
            continue
        cur = shares.get(ticker, 0.0)
        if side == "BUY":
            new = cur + qty
            prev_cost = wavg_cost.get(ticker, price)
            wavg_cost[ticker] = (
                ((prev_cost * cur) + (price * qty)) / new if new > 0 else price
            )
            shares[ticker] = new
        else:
            shares[ticker] = cur - qty
    return {
        t: {"shares": s, "cost_basis": wavg_cost.get(t, 0.0)}
        for t, s in shares.items()
        if s > QTY_EPS
    }


def build_book(
    *,
    mode: str,
    conn,
    asof: date,
    alpaca: AlpacaClient | None = None,
) -> object:
    """Return an object exposing `.get_account()` / `.get_positions()` for
    the requested `mode`. NOT yet wrapped in `_NoSubmitAlpaca` -- callers do
    that once, at the call site that hands it to `decide_once`."""
    if mode not in BOOK_MODES:
        raise ValueError(f"book must be one of {BOOK_MODES}; got {mode!r}")
    if mode == "current":
        if alpaca is None:
            raise ValueError("--book current requires a live AlpacaClient")
        return alpaca
    if mode == "empty":
        equity, _cash = _latest_snapshot_equity(conn, asof=asof)
        return _SyntheticBook(equity=equity, cash=equity, positions={})
    # mode == "asof"
    positions = reconstruct_book_asof(conn, asof=asof)
    equity, _snapshot_cash = _latest_snapshot_equity(conn, asof=asof)
    prices = _last_prices_per_ticker(
        _load_prices(
            _ConnShim(conn), list(positions), start=asof - timedelta(days=10), end=asof,
        ),
        list(positions),
    )
    market_value = sum(
        prices[t][0] * p["shares"] if t in prices else p["cost_basis"] * p["shares"]
        for t, p in positions.items()
    )
    cash = max(0.0, equity - market_value)
    return _SyntheticBook(equity=equity, cash=cash, positions=positions)


@dataclass
class ReplayDecision:
    """One ticker's outcome, for the human-readable report."""

    ticker: str
    action: str              # HOLD | ENTRY | EXIT | RESIZE | BLOCKED_ENTRY
    rail: str                # what decided it -- "target", "min_hold", "sector_cap", ...
    prior_weight: float | None
    target_weight: float | None
    side: str | None = None
    shares: float | None = None


@dataclass
class ReplayResult:
    asof: date
    book: str
    aborted: bool
    aborted_reason: str | None
    decisions: list[ReplayDecision] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)
    summary: str = ""


def _rail_for_blocked_increase(
    *,
    ticker: str,
    target_weight: float,
    rails: RiskRails,
    current_drawdown: float,
    asof: date,
    upcoming_earnings: dict[str, list[date]],
    sector_exposure_pct: dict[str, float],
    current_weight: float,
) -> str:
    """Best-effort label for why an exposure-increasing decision (a new buy
    or an add) was dropped by `sma.risk.pipeline.apply`. Approximate: it
    checks each rail against the PRE-TRADE sector exposure rather than
    pipeline.apply's exact incrementally-updated running exposure (which
    depends on the order every other decision in the batch was processed
    in) -- good enough to label the dominant reason for a report, not
    guaranteed to match apply()'s internal bookkeeping order for a batch
    with several same-sector trades stacked together.
    """
    triggered, _ = check_drawdown(rails=rails, current_drawdown=current_drawdown)
    if triggered:
        return "drawdown"
    if in_earnings_blackout(ticker, asof, upcoming_earnings):
        return "earnings_blackout"
    sector = sector_for(ticker)
    post_sector = sector_exposure_pct.get(sector, 0.0) - current_weight + target_weight
    triggered, _ = check_sector_cap(
        rails=rails, sector=sector, sector_exposure_after=post_sector,
    )
    if triggered:
        return "sector_cap"
    if target_weight > rails.max_position_pct:
        return "position_cap"
    return "risk_pipeline (reason not resolved by replay's approximate re-check)"


def _rail_for_missing_exit(
    *,
    ticker: str,
    held_shares: float,
    rails: RiskRails,
    asof: date,
    position_entry_dates: dict[str, date],
    last_price: float | None,
    adv_dollars: dict[str, float],
    sizing: SizingPolicy,
) -> str:
    """Best-effort label for why a held name the model dropped (not in
    raw_decisions) did not produce a SELL order. translate()'s force-sell
    branch has exactly two ways to suppress it: min_hold, or the ADV
    participation cap trimming it to zero (a full exit is exempt from the
    min-notional floor, so that rail cannot be the cause)."""
    min_hold = rails.min_hold_days if rails else 0
    entry = position_entry_dates.get(ticker)
    if min_hold > 0 and entry is not None:
        held_days = (asof - entry).days
        if 0 <= held_days < min_hold:
            return f"min_hold ({held_days}d held, min_hold={min_hold}d)"
    if sizing.max_participation_of_adv > 0 and last_price and last_price > 0:
        adv = adv_dollars.get(ticker, 0.0)
        if adv > 0:
            cap_qty = sizing.participation_cap_qty(adv_dollars=adv, price=last_price)
            if cap_qty is not None and (is_zero(cap_qty) or cap_qty <= 0):
                return "adv_participation_cap (trimmed to zero)"
    return "risk_pipeline (reason not resolved by replay's approximate re-check)"


def _missing_order_reasons(
    *,
    decisions: list[StrategyDecision],
    order_by: dict[str, Order],
    book_positions: dict[str, dict],
    last_prices: dict[str, tuple[float, date]],
    asof: date,
    account_equity: float,
    cash: float,
    rails: RiskRails,
    sizing: SizingPolicy,
    adv_dollars: dict[str, float],
    current_drawdown: float,
) -> dict[str, str]:
    """For every ticker in `decisions` (post-rails) that produced NO order --
    `orders` is `translate()`'s REAL output, already computed by the real
    decide_once call this replay made -- figure out which of translate()'s
    OWN gates dropped it: missing/stale price, the rebalance dead zone, the
    ADV participation cap, the min-notional floor, or the cash floor's
    highest-target-weight-first funding queue (2026-09-01 finding: on the
    2026-08-31 replay, DKNG and ISRG both PASSED the risk pipeline but never
    got a BUY order -- the account simply ran out of cash funding higher-
    priority resizes first; the earlier version of this report had no way to
    say that and fell back to an unhelpful "reason not resolved").

    This mirrors `sma.live.orders.translate`'s sequencing closely enough to
    explain its real output for a report; it is a SEPARATE re-implementation
    (not shared code), so a future translate() change could silently drift
    this out of sync with it -- covered by the regression test's own
    reproduction of a real night, not by an exact-equality test against
    translate()'s internals.
    """
    stale_threshold = asof - timedelta(days=STALE_PRICE_DAYS)
    dead_zone = rails.rebalance_dead_zone_pct if rails else 0.0
    reasons: dict[str, str] = {}
    buy_queue: list[tuple[float, str, float]] = []  # (-target_weight, ticker, cost)

    for d in decisions:
        if d.ticker in order_by:
            continue
        held = book_positions.get(d.ticker, {}).get("shares", 0.0)
        entry = last_prices.get(d.ticker)
        price, price_date = entry if entry else (None, None)
        fresh = entry is not None and price > 0 and price_date >= stale_threshold
        if not fresh:
            reasons[d.ticker] = (
                "missing/stale price (translate cannot size a new entry)"
                if held <= 0 else
                "missing/stale price (translate froze this reduction)"
            )
            continue
        target_shares = sizing.target_qty(d.target_weight, account_equity, price)
        delta = target_shares - held
        if abs(delta) <= QTY_EPS:
            reasons[d.ticker] = "target≈current (no rebalance)"
            continue
        if held > 0 and abs(delta) <= dead_zone * held:
            reasons[d.ticker] = f"rebalance dead-zone ({dead_zone:.0%})"
            continue
        if delta < 0:
            # A reduction that's neither at-target nor dead-zoned and still
            # produced no order must be the force-sell path's min_hold veto
            # -- handled by `_rail_for_missing_exit`'s caller, not here.
            continue
        cost = price * delta
        if sizing.max_participation_of_adv > 0:
            adv = adv_dollars.get(d.ticker, 0.0)
            if adv > 0:
                cap_qty = sizing.participation_cap_qty(adv_dollars=adv, price=price)
                if cap_qty is not None and (is_zero(cap_qty) or cap_qty <= 0):
                    reasons[d.ticker] = "adv_participation_cap (trimmed to zero)"
                    continue
        if sizing.min_order_notional > 0 and cost < sizing.min_order_notional:
            reasons[d.ticker] = (
                f"min_notional (${cost:,.2f} < ${sizing.min_order_notional:,.2f})"
            )
            continue
        buy_queue.append((-d.target_weight, d.ticker, cost))

    if buy_queue:
        floor_dollars = 0.0
        if rails is not None and rails.cash_floor_pct > 0:
            from sma.risk.derisk import derisk_cash_floor
            eff_floor = derisk_cash_floor(
                rails.cash_floor_pct, current_drawdown,
                start=rails.drawdown_derisk_start,
                slope=rails.drawdown_derisk_slope,
                cap=rails.drawdown_derisk_cap,
            )
            floor_dollars = eff_floor * account_equity
        haircut = rails.sell_proceeds_haircut if rails is not None else 1.0
        sell_proceeds = haircut * sum(
            (o.last_price or 0.0) * o.shares for o in order_by.values() if o.side == "SELL"
        )
        running_cash = cash + sell_proceeds
        for _neg_weight, ticker, cost in sorted(buy_queue):
            if running_cash - cost < floor_dollars:
                reasons[ticker] = (
                    f"cash_floor (${running_cash:,.0f} available, needs "
                    f"${cost:,.0f}, floor ${floor_dollars:,.0f} — higher-"
                    "priority buys were funded first)"
                )
            else:
                running_cash -= cost
                # translate() DID fund this one; if it's still missing an
                # order, replay's re-check is genuinely wrong somewhere.
                reasons.setdefault(
                    ticker, "risk_pipeline (reason not resolved by replay's approximate re-check)",
                )
    return reasons


def _build_report(
    *,
    asof: date,
    rails: RiskRails,
    sizing: SizingPolicy,
    book_positions: dict[str, dict],
    account_equity: float,
    cash: float,
    raw_decisions: list[StrategyDecision],
    decisions: list[StrategyDecision],
    orders: list[Order],
    conn,
    universe: list[str],
) -> list[ReplayDecision]:
    raw_by = {d.ticker: d for d in raw_decisions}
    dec_by = {d.ticker: d for d in decisions}
    order_by = {o.ticker: o for o in orders}

    prices = _last_prices_per_ticker(
        _load_prices(_ConnShim(conn), universe, start=asof - timedelta(days=10), end=asof),
        universe,
    )
    last_price_of = {t: p[0] for t, p in prices.items()}

    from sma.live.decide import _load_upcoming_earnings_from_store, _sector_exposure
    upcoming_earnings = _load_upcoming_earnings_from_store(
        store=_ConnShim(conn), start=asof, end=asof + timedelta(days=14),
    )
    sector_exposure_pct = _sector_exposure(
        book_positions, prices, account_equity, sector_for,
    )
    # Same asof cutoff decide_once uses, so the "exit blocked: min_hold"
    # label agrees with the order translate() actually produced.
    position_entry_dates = _latest_buy_dates(
        store=_ConnShim(conn), tickers=set(book_positions), asof=asof,
    )

    def current_weight(ticker: str) -> float:
        if account_equity <= 0:
            return 0.0
        pos = book_positions.get(ticker)
        if pos is None:
            return 0.0
        price = last_price_of.get(ticker, pos.get("cost_basis", 0.0))
        return (price * pos["shares"]) / account_equity

    # Real current_drawdown -- the SAME computation decide_once does
    # internally (garbage-filtered snapshot history, peak vs today's
    # equity) -- rather than a hardcoded 0.0, so the drawdown rail's
    # attribution is not silently wrong whenever the book is actually in a
    # drawdown.
    clean_snapshots = _clean_snapshot_equities(_ConnShim(conn), asof)
    peak_equity = max([e for _, e in clean_snapshots] + [account_equity])
    current_drawdown = _drawdown_from_peak(peak_equity, account_equity)

    adv_dollars = _adv_dollars(conn, universe, asof, sizing)

    # Tickers the risk pipeline ACCEPTED (in dec_by) but translate() still
    # gave no order to (2026-09-01 finding: a decision can pass every risk
    # rail and still get cut by translate()'s own gates -- most notably the
    # cash floor's funding queue, e.g. 2026-08-31's DKNG/ISRG).
    missing_reasons = _missing_order_reasons(
        decisions=decisions, order_by=order_by, book_positions=book_positions,
        last_prices=prices, asof=asof, account_equity=account_equity, cash=cash,
        rails=rails, sizing=sizing, adv_dollars=adv_dollars,
        current_drawdown=current_drawdown,
    )

    out: list[ReplayDecision] = []
    all_tickers = set(book_positions) | set(raw_by) | set(dec_by) | set(order_by)
    for ticker in sorted(all_tickers):
        held_shares = book_positions.get(ticker, {}).get("shares", 0.0)
        held = held_shares > QTY_EPS
        prior_w = current_weight(ticker) if held else None

        if ticker in order_by:
            o = order_by[ticker]
            if not held and o.side == "BUY":
                action = "ENTRY"
            elif o.full_exit:
                action = "EXIT"
            else:
                action = "RESIZE"
            rail = "adv_participation_cap (trimmed)" if o.capped_by else "target"
            out.append(ReplayDecision(
                ticker=ticker, action=action, rail=rail,
                prior_weight=prior_w,
                target_weight=dec_by[ticker].target_weight if ticker in dec_by else None,
                side=o.side, shares=o.shares,
            ))
            continue

        if held:
            if ticker in dec_by:
                rail = missing_reasons.get(ticker, "target≈current (no rebalance)")
                out.append(ReplayDecision(
                    ticker=ticker, action="HOLD", rail=rail,
                    prior_weight=prior_w, target_weight=dec_by[ticker].target_weight,
                ))
            elif ticker in raw_by:
                rail = _rail_for_blocked_increase(
                    ticker=ticker, target_weight=raw_by[ticker].target_weight,
                    rails=rails, current_drawdown=current_drawdown, asof=asof,
                    upcoming_earnings=upcoming_earnings,
                    sector_exposure_pct=sector_exposure_pct,
                    current_weight=prior_w or 0.0,
                )
                out.append(ReplayDecision(
                    ticker=ticker, action="HOLD", rail=f"rails blocked re-buy: {rail}",
                    prior_weight=prior_w, target_weight=raw_by[ticker].target_weight,
                ))
            else:
                rail = _rail_for_missing_exit(
                    ticker=ticker, held_shares=held_shares, rails=rails, asof=asof,
                    position_entry_dates=position_entry_dates,
                    last_price=last_price_of.get(ticker), adv_dollars=adv_dollars,
                    sizing=sizing,
                )
                out.append(ReplayDecision(
                    ticker=ticker, action="HOLD", rail=f"model dropped, exit blocked: {rail}",
                    prior_weight=prior_w, target_weight=None,
                ))
            continue

        # Not held, not ordered: only interesting if the model wanted it.
        if ticker in dec_by:
            rail = missing_reasons.get(
                ticker, "risk_pipeline (reason not resolved by replay's approximate re-check)",
            )
            out.append(ReplayDecision(
                ticker=ticker, action="BLOCKED_ENTRY", rail=rail,
                prior_weight=None, target_weight=dec_by[ticker].target_weight,
            ))
        elif ticker in raw_by:
            rail = _rail_for_blocked_increase(
                ticker=ticker, target_weight=raw_by[ticker].target_weight,
                rails=rails, current_drawdown=current_drawdown, asof=asof,
                upcoming_earnings=upcoming_earnings,
                sector_exposure_pct=sector_exposure_pct,
                current_weight=0.0,
            )
            out.append(ReplayDecision(
                ticker=ticker, action="BLOCKED_ENTRY", rail=rail,
                prior_weight=None, target_weight=raw_by[ticker].target_weight,
            ))
    return out


def _adv_dollars(conn, universe, asof, sizing: SizingPolicy) -> dict[str, float]:
    if sizing.max_participation_of_adv <= 0:
        return {}
    rows = conn.execute(
        """
        SELECT ticker, AVG(close * volume) FROM (
            SELECT ticker, close, volume,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices
            WHERE ticker = ANY($tickers) AND date <= $asof
              AND close IS NOT NULL AND volume IS NOT NULL
        ) t
        WHERE rn <= $lookback
        GROUP BY ticker
        """,
        {"tickers": list(universe), "asof": asof, "lookback": sizing.adv_lookback_days},
    ).fetchall()
    return {t: float(a) for t, a in rows if a is not None}


def replay_once(
    *,
    asof: date,
    db_path: Path,
    universe: list[str],
    rails: RiskRails,
    sizing: SizingPolicy | None = None,
    settings=None,
    book: str = "current",
    use_theses: bool = True,
    alpaca: AlpacaClient | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> ReplayResult:
    """Run the real decide selection + rails pipeline for `asof`, read-only.

    `conn` lets tests inject a pre-built point-in-time connection (e.g. one
    built against an in-memory fixture DB rather than a real file); when
    omitted, a fresh one is opened via `replay_connection(db_path, asof=asof)`.
    """
    from sma.live.__main__ import _build_decide_strategy  # local import:
    # avoids a module-load cycle (that module imports plenty of live.*
    # submodules). Same builder the real decide job uses (sleeves included).

    owns_conn = conn is None
    pit_conn = conn or replay_connection(db_path, asof=asof)
    try:
        store_shim = _ConnShim(pit_conn)
        predictor = StoredPredictionsPredictor(pit_conn)
        strategy = _build_decide_strategy(
            universe, use_theses=use_theses, db=str(db_path), store=store_shim,
            settings=settings, predictor=predictor,
        )
        book_obj = build_book(mode=book, conn=pit_conn, asof=asof, alpaca=alpaca)
        guarded_alpaca = _NoSubmitAlpaca(book_obj)

        try:
            result = decide_once(
                asof=asof,
                store=store_shim,
                alpaca=guarded_alpaca,
                universe=universe,
                strategy=strategy,
                sector_for=sector_for,
                rails=rails,
                dry_run=True,
                sizing=sizing or SizingPolicy(),
            )
        except CatastrophicLossAbortError as e:
            return ReplayResult(
                asof=asof, book=book, aborted=True, aborted_reason=str(e),
                summary=f"{asof.isoformat()}: would have ABORTED — {e}",
            )

        book_positions = guarded_alpaca.get_positions()
        account = guarded_alpaca.get_account()
        account_equity = account["equity"]
        decisions = _build_report(
            asof=asof, rails=rails, sizing=sizing or SizingPolicy(),
            book_positions=book_positions, account_equity=account_equity,
            cash=account["cash"],
            raw_decisions=result.raw_decisions or [],
            decisions=result.decisions or [],
            orders=result.orders or [],
            conn=pit_conn, universe=universe,
        )
        orders = result.orders or []
        entries = sum(1 for d in decisions if d.action == "ENTRY")
        exits = sum(1 for d in decisions if d.action == "EXIT")
        resizes = sum(1 for d in decisions if d.action == "RESIZE")
        summary = (
            f"{asof.isoformat()} [{book}]: {len(orders)} order(s) — "
            f"{entries} entr{'y' if entries == 1 else 'ies'}, "
            f"{exits} exit{'s' if exits != 1 else ''}, "
            f"{resizes} resize{'s' if resizes != 1 else ''} "
            f"(equity ${account_equity:,.0f}, {len(book_positions)} held)"
        )
        return ReplayResult(
            asof=asof, book=book, aborted=False, aborted_reason=None,
            decisions=decisions, orders=orders, summary=summary,
        )
    finally:
        if owns_conn:
            pit_conn.close()
