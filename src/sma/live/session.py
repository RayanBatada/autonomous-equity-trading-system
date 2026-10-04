"""Intraday trade sessions: python -m sma.live session --name midday|close [--dry-run]

A session trades the aggregate target book for one time of day with
marketable-limit orders (strategy-expansion.md section 3). The 20:00 decide /
09:30 open path is a separate job and is not touched by anything here.

One run:
  1. writer_lock(label="session-<name>")
  2. the market must be open today and the ET clock inside the session window
     (`live.sessions.<name>.window`, clipped to 5 minutes before a half-day
     close). --dry-run reports the window verdict but carries on.
  3. load data/state/session_targets-<name>-<asof>.json, a flat
     {ticker: target_notional_usd}. No file -> "no targets for session", exit 0.
     That is the default state: nothing writes these files yet (the sleeve
     framework will). Only tickers IN the file are traded; a held name that is
     absent is left alone, and a target of 0 sells the whole position.
  4. `live.sessions.<name>.enabled` must be true (default false) unless
     --dry-run.
  5. diff against live positions at the IEX reference price, skip deltas under
     `live.execution.min_trade_notional`, never sell more than is held, never
     buy on margin (buys are scaled to cash + this session's sell proceeds),
     skip any name that already has an open order.
  6. sells then buys as marketable limits (AlpacaClient.submit_marketable_limit),
     then AlpacaClient.sweep_unfilled: re-price once, then market or leave.
  7. sentinel data/sentinels/com.sma.live.session.<name>-<date>.json on every
     non-error exit (status: ok | no_targets | disabled | market_closed |
     out_of_window | already_ran). --dry-run writes nothing.

Audit trail: every order gets an intended_orders row BEFORE it is submitted,
source 'session-<name>' (replacements 'session-<name>-reprice' /
'session-<name>-market'), session=<name>, order_type limit|market,
last_price = the arrival mid the order was priced off (the fill-quality
benchmark), limit_price. Reconcile records the fills like any other order.
"""

from __future__ import annotations

import json
import math
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import click
from loguru import logger

from sma.live.alpaca_client import (
    AlpacaClient,
    QuoteUnavailableError,
    RefPrice,
    WorkingOrder,
)
from sma.live.orders import client_order_id

ET = ZoneInfo("America/New_York")
SESSION_NAMES = ("midday", "close")
STATE_DIR = Path("data/state")
# A session never runs into the last minutes before the close (half-days too).
_CLOSE_BUFFER = timedelta(minutes=5)


def sentinel_label(name: str) -> str:
    return f"com.sma.live.session.{name}"


def targets_path(name: str, asof: date, state_dir: Path = STATE_DIR) -> Path:
    return Path(state_dir) / f"session_targets-{name}-{asof.isoformat()}.json"


def load_targets(path: Path) -> dict[str, float] | None:
    """{TICKER: target_notional} or None when the file does not exist.
    Malformed content raises: a half-written book must not trade."""
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object of ticker -> notional")
    out: dict[str, float] = {}
    for k, v in raw.items():
        val = float(v)
        if not math.isfinite(val) or val < 0:
            raise ValueError(f"{path}: target for {k!r} must be a finite notional >= 0")
        out[str(k).strip().upper()] = val
    return out


@dataclass
class PlannedOrder:
    ticker: str
    side: str
    qty: float
    ref: RefPrice
    held_shares: float
    target_notional: float

    @property
    def notional(self) -> float:
        return self.qty * self.ref.mid


def _floor_qty(q: float, *, fractional: bool, precision: int) -> float:
    if q <= 0:
        return 0.0
    if not fractional:
        return float(math.floor(q + 1e-9))
    scale = 10**precision
    return math.floor(q * scale + 1e-9) / scale


def plan_orders(
    *,
    targets: dict[str, float],
    positions: dict[str, dict],
    ref_fn,
    cash: float,
    min_trade_notional: float,
    fractional: bool = False,
    precision: int = 9,
    open_order_tickers: frozenset[str] = frozenset(),
    price_buffer_bps: float = 0.0,
) -> tuple[list[PlannedOrder], list[dict]]:
    """Diff the target book against live positions. Returns (orders, skipped),
    sells first. Pure apart from `ref_fn(ticker) -> RefPrice`."""
    sells: list[PlannedOrder] = []
    buys: list[PlannedOrder] = []
    skipped: list[dict] = []
    for ticker in sorted(targets):
        target = targets[ticker]
        if ticker in open_order_tickers:
            skipped.append({"ticker": ticker, "reason": "open order already working"})
            continue
        held = float(positions.get(ticker, {}).get("shares", 0) or 0)
        try:
            ref = ref_fn(ticker)
        except QuoteUnavailableError as e:
            skipped.append({"ticker": ticker, "reason": str(e)})
            continue
        delta = target - held * ref.mid
        if target == 0 and held > 0:
            sells.append(PlannedOrder(ticker, "SELL", held, ref, held, target))
            continue
        if abs(delta) < min_trade_notional:
            skipped.append({"ticker": ticker, "reason": f"delta ${delta:,.2f} under minimum"})
            continue
        qty = _floor_qty(abs(delta) / ref.mid, fractional=fractional, precision=precision)
        if delta < 0:
            qty = min(qty, held)
        if qty <= 0:
            skipped.append({"ticker": ticker, "reason": "rounds to zero shares"})
            continue
        (buys if delta > 0 else sells).append(
            PlannedOrder(ticker, "BUY" if delta > 0 else "SELL", qty, ref, held, target)
        )

    # Cash check at the worst price each order can print: a buy at its limit
    # (ask + offset), a sell at its limit (bid - offset).
    buf = price_buffer_bps / 10_000.0
    budget = max(cash, 0.0) + sum(o.qty * o.ref.bid * (1 - buf) for o in sells)
    want = sum(o.qty * o.ref.ask * (1 + buf) for o in buys)
    if buys and want > budget:
        scale = budget / want
        logger.warning(
            "session: buys want ${:,.0f} but cash + sells = ${:,.0f}; scaling buys by {:.3f}",
            want, budget, scale,
        )
        scaled = []
        for o in buys:
            q = _floor_qty(o.qty * scale, fractional=fractional, precision=precision)
            if q > 0:
                o.qty = q
                scaled.append(o)
            else:
                skipped.append({"ticker": o.ticker, "reason": "scaled to zero by cash limit"})
        buys = scaled
    return sells + buys, skipped


def session_bounds(
    cfg_window: tuple, day: date, market: tuple[datetime, datetime]
) -> tuple[datetime, datetime]:
    """The configured ET window on `day`, clipped to the actual session (open,
    close - 5 min) so a half-day's 13:00 close is never crossed."""
    start = datetime.combine(day, cfg_window[0], tzinfo=ET)
    end = datetime.combine(day, cfg_window[1], tzinfo=ET)
    m_open, m_close = market
    return max(start, m_open), min(end, m_close - _CLOSE_BUFFER)


@contextmanager
def _db(path: str):
    """Short-lived writable connection. The writer lock is held for the whole
    session, but the DuckDB file is opened only around each write so the
    dashboard's readers are not locked out during the sweep's waits."""
    from sma.ingest.store import Store

    store = Store(path=path).connect()
    try:
        yield store
    finally:
        store.close()


def _insert_intended(store, *, asof, ticker, side, qty, arrival, source, session, order_type,
                     limit_price, run_id, alpaca_order_id=None) -> str:
    iid = str(uuid.uuid4())
    store.conn.execute(
        "INSERT INTO intended_orders (intended_order_id, asof_date, ticker, side, "
        "target_shares, target_weight, last_price, source, alpaca_order_id, status, run_id, "
        "session, order_type, limit_price) "
        "VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, 'submitted', ?, ?, ?, ?)",
        [iid, asof, ticker, side, qty, arrival, source, alpaca_order_id, run_id, session,
         order_type, limit_price],
    )
    return iid


def _sentinel(name: str, asof: date, payload: dict) -> None:
    from sma.sentinels import write_sentinel

    base = {
        "label": sentinel_label(name),
        "asof": asof.isoformat(),
        "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    write_sentinel(label=sentinel_label(name), asof=asof, payload={**base, **payload})


def run_session(
    *,
    name: str,
    dry_run: bool,
    db: str,
    config: str,
    alpaca: AlpacaClient | None = None,
    now_fn=None,
    sleep_fn=None,
    state_dir: Path = STATE_DIR,
    echo=click.echo,
) -> dict:
    from sma.config import load_settings
    from sma.live.real_money import (
        build_alpaca_client,
        build_gate,
        force_dry_run,
        preflight_real_money,
    )
    from sma.locks import writer_lock

    if name not in SESSION_NAMES:
        raise click.BadParameter(f"--name must be one of {SESSION_NAMES}")
    now_fn = now_fn or (lambda: datetime.now(ET))
    settings = load_settings(config_path=Path(config))
    scfg = getattr(settings.live.sessions, name)
    ecfg = settings.live.execution
    sizing = settings.live.sizing
    gate = build_gate(settings)
    dry = force_dry_run(dry_run, gate)
    now = now_fn()
    asof = now.date()

    def _finish(status: str, msg: str, **extra) -> dict:
        echo(f"session[{name}] {asof.isoformat()}: {msg}")
        payload = {"status": status, "detail": msg, "dry_run": dry, **extra}
        if not dry:
            _sentinel(name, asof, payload)
        return payload

    with writer_lock(label=f"session-{name}"):
        if alpaca is None:
            alpaca = build_alpaca_client(settings, gate=gate)
        market = alpaca.session_window(day=asof)
        window_note = None
        if market is None:
            window_note = "market closed today"
        else:
            start, end = session_bounds(scfg.bounds(), asof, market)
            if not (start <= now < end):
                window_note = (f"outside session window {start:%H:%M}-{end:%H:%M} ET "
                               f"(now {now:%H:%M})")
        if window_note and not dry:
            status = "market_closed" if market is None else "out_of_window"
            return _finish(status, f"skipped: {window_note}")
        if window_note:
            echo(f"session[{name}] dry-run note: {window_note}; planning anyway")
            end = now + timedelta(minutes=30)

        path = targets_path(name, asof, state_dir)
        targets = load_targets(path)
        if targets is None:
            return _finish("no_targets", f"no targets for session ({path} absent)")
        if not scfg.enabled and not dry:
            return _finish("disabled", f"live.sessions.{name}.enabled is false; "
                           f"{len(targets)} targets ignored")

        if gate.armed and not dry:
            preflight_real_money(alpaca=alpaca, gate=gate)
        account = alpaca.get_account()
        if account.get("trading_blocked") or account.get("account_blocked"):
            raise click.ClickException(f"session[{name}]: account is blocked; not trading")
        positions = alpaca.get_positions()
        open_tickers = frozenset(o["symbol"].replace(".", "-")
                                 for o in alpaca.list_open_orders())
        orders, skipped = plan_orders(
            targets=targets,
            positions=positions,
            ref_fn=lambda t: alpaca.reference_price(t, max_spread_bps=ecfg.max_quote_spread_bps),
            cash=float(account.get("cash", 0.0)),
            min_trade_notional=ecfg.min_trade_notional,
            fractional=sizing.fractional_shares,
            precision=sizing.fractional_precision,
            open_order_tickers=open_tickers,
            price_buffer_bps=ecfg.limit_offset_bps,
        )
        for o in orders:
            echo(f"  {o.side:<4} {o.ticker:<6} {o.qty:>10g} @~{o.ref.mid:.2f} "
                 f"({o.ref.source}) held {o.held_shares:g} target ${o.target_notional:,.0f}")
        for s in skipped:
            echo(f"  skip {s['ticker']}: {s['reason']}")
        if dry:
            echo(f"session[{name}] DRY RUN: {len(orders)} order(s) planned, none submitted")
            return {"status": "dry_run", "planned": len(orders), "skipped": skipped,
                    "orders": [(o.side, o.ticker, o.qty) for o in orders]}

        source = f"session-{name}"
        with _db(db) as store:
            already = store.conn.execute(
                "SELECT COUNT(*) FROM intended_orders WHERE asof_date = ? AND source = ?",
                [asof, source],
            ).fetchone()[0]
            if already:
                return _finish("already_ran", f"{already} {source} order(s) already on file "
                               "for today; not trading twice")
            run_id = store.allocate_run_id()
            working: list[WorkingOrder] = []
            failed = 0
            arrival: dict[str, float] = {}
            for o in orders:
                coid = client_order_id(asof, o.ticker, o.side, source=source)
                iid = _insert_intended(
                    store, asof=asof, ticker=o.ticker, side=o.side, qty=o.qty,
                    arrival=o.ref.mid, source=source, session=name, order_type="limit",
                    limit_price=None, run_id=run_id,
                )
                try:
                    res = alpaca.submit_marketable_limit(
                        o.ticker, o.side, o.qty, offset_bps=ecfg.limit_offset_bps,
                        ref=o.ref, client_order_id=coid,
                        fractional_precision=sizing.fractional_precision,
                    )
                except Exception as e:  # noqa: BLE001 - one name must not sink the book
                    logger.error("session[{}]: submit {} {} failed: {!r}",
                                 name, o.side, o.ticker, e)
                    store.conn.execute(
                        "UPDATE intended_orders SET status = 'submission_failed', error = ? "
                        "WHERE intended_order_id = ?", [str(e)[:500], iid])
                    failed += 1
                    continue
                store.conn.execute(
                    "UPDATE intended_orders SET alpaca_order_id = ?, limit_price = ? "
                    "WHERE intended_order_id = ?", [res.order_id, res.limit_price, iid])
                arrival[o.ticker] = o.ref.mid
                working.append(WorkingOrder(order_id=res.order_id, ticker=o.ticker,
                                            side=o.side, qty=res.qty))

        def _on_replace(old: WorkingOrder, new: WorkingOrder, detail: dict) -> None:
            with _db(db) as st:
                _insert_intended(
                    st, asof=asof, ticker=new.ticker, side=new.side, qty=new.qty,
                    arrival=arrival.get(new.ticker), source=f"{source}-{new.stage}",
                    session=name, order_type="market" if new.stage == "market" else "limit",
                    limit_price=detail.get("limit_price"), run_id=run_id,
                    alpaca_order_id=new.order_id,
                )

        def _coid(old: WorkingOrder, stage: str) -> str:
            return client_order_id(asof, old.ticker, old.side, source=f"{source}-{stage}")

        touched = alpaca.sweep_unfilled(
            working,
            session=name,
            deadline=end,
            after_minutes=ecfg.sweep_after_minutes,
            offset_bps=ecfg.limit_offset_bps,
            reprice_once=ecfg.reprice_once,
            fallback=ecfg.fallback,
            max_spread_bps=ecfg.max_quote_spread_bps,
            fractional=sizing.fractional_shares,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
            on_replace=_on_replace,
            coid_fn=_coid,
        ) if working else []

        summary = [
            {"ticker": w.ticker, "side": w.side, "qty": w.qty, "stage": w.stage,
             "order_id": w.order_id, "status": w.status, "filled_qty": w.filled_qty}
            for w in touched
        ]
        return _finish(
            "ok",
            f"{len(working)} submitted, {failed} failed, {len(skipped)} skipped, "
            f"{sum(1 for w in touched if w.stage != 'limit')} replacement(s)",
            run_id=run_id, targets_file=str(path), planned=len(orders),
            submitted=len(working), failed=failed, skipped=skipped, orders=summary,
        )


@click.command("session")
@click.option("--name", required=True, type=click.Choice(SESSION_NAMES))
@click.option("--dry-run", is_flag=True, default=False,
              help="Plan against live positions and quotes; submit and write nothing.")
@click.option("--db", default="data/sma.duckdb", type=click.Path())
@click.option("--config", default="config.yaml", type=click.Path(exists=True))
def session_cmd(name, dry_run, db, config):
    """Intraday trade session (midday ~10:35 / close ~15:45 ET). Off by default."""
    from sma.ingest.notify import notify_failure

    try:
        run_session(name=name, dry_run=dry_run, db=db, config=config)
    except click.ClickException:
        raise
    except Exception as e:
        notify_failure(title=f"SMA {name} session FAILED", message=repr(e)[:500])
        raise
