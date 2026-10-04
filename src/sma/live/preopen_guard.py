"""Pre-open safety guard (09:25 ET), motivated by incident 2026-07-07.

That night Alpaca cleared all 11 paper positions overnight (a broker-side glitch,
no sells). At the 09:30 open the previous evening's queued decide orders filled
against the empty book — a queued `QCOM SELL 6` sold a QCOM the account no longer
held, opening a SHORT — because the bot's other rails only run at 16:30 (reconcile
ledger-drift) and 20:00 (decide catastrophic-loss abort). Nothing checked BEFORE
the open.

Two layers, both pre-open, both independent of the (default-OFF) price exits:

1. Per-ticker OVERSELL gate (the harm-preventer). Cancel any queued SELL whose
   quantity exceeds what the broker currently holds — that order would open/deepen
   a short. Uses ONLY live Alpaca data (open orders vs the live book), so it never
   false-halts, needs no ledger, works at ANY scale (even a single zeroed name),
   and is symbol-safe (both sides are Alpaca's own symbols). This directly stops
   the 7/07 QCOM-short.

2. Book-wide WIPE halt (the catastrophe alarm). If a large fraction of the
   positions our paper_fills ledger holds are absent from the broker AND live
   equity has collapsed vs the last account snapshot, cancel ALL queued orders and
   page — a whole-book wipe. The equity corroboration is what makes this safe: a
   stale/lagging ledger shows "missing" positions but equity is intact, so it does
   NOT halt (avoiding a false halt on a missed reconcile).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

from sma.live.quantity import QTY_EPS, as_qty, is_zero


@dataclass
class PreOpenDivergence:
    is_divergent: bool
    missing_names: list[str] = field(default_factory=list)
    missing_fraction: float = 0.0
    ledger_positions: int = 0
    broker_positions: int = 0
    detail: str = ""


@dataclass
class PreOpenGuardResult:
    oversell_cancelled: list[dict] = field(default_factory=list)
    wipe_halt: bool = False
    divergence: PreOpenDivergence | None = None


def _canon(symbol: str) -> str:
    """Canonical form for cross-source symbol matching (BRK.B / BRK-B / brk.b)."""
    return symbol.upper().replace("-", "").replace(".", "")


def find_oversell_orders(*, open_orders: list[dict], broker: dict[str, float]) -> list[dict]:
    """Queued SELL orders that would sell MORE than the broker currently holds —
    each would open/deepen a short (the 7/07 phantom-short). `open_orders` items
    are {'id','symbol','side','qty'}; `broker` is {symbol: shares}. Live-data only,
    so no false positives on a normal morning (a queued full-exit sells exactly the
    held qty; only an overnight holdings drop makes qty > held)."""
    broker_c = {_canon(t): as_qty(v) for t, v in broker.items()}
    out = []
    for o in open_orders:
        if str(o.get("side", "")).upper() != "SELL":
            continue
        held = max(broker_c.get(_canon(o["symbol"]), 0), 0)
        # Only the UNFILLED remainder would still execute at the open.
        remaining = as_qty(o.get("qty", 0)) - as_qty(o.get("filled_qty", 0) or 0)
        # QTY_EPS slack: under fractional sizing `qty - filled` on a partially
        # filled order lands float-ulps above the held quantity, and a bare `>`
        # would cancel a perfectly legitimate full exit every morning.
        if remaining > held + QTY_EPS:
            out.append({**o, "held": held, "remaining": remaining})
    return out


def check_preopen_divergence(
    *,
    ledger: dict[str, float],
    broker: dict[str, float],
    min_missing_fraction: float = 0.5,
    min_ledger_positions: int = 2,
) -> PreOpenDivergence:
    """Book-wide: how much of the ledger's long book is absent from the broker.

    A ledger long is "missing" when the broker holds <= 0 of it (zeroed/flipped) —
    a small share-count drift does NOT count. Divergent iff the ledger holds >=
    `min_ledger_positions` longs AND >= `min_missing_fraction` of them are missing.
    Symbols are canonicalised so a dot/dash convention mismatch is not read as
    missing. Pure + side-effect free.
    """
    ledger_longs = {t: n for t, n in ledger.items() if n > 0}
    broker_c = {_canon(t): v for t, v in broker.items()}
    missing = sorted(t for t in ledger_longs if broker_c.get(_canon(t), 0) <= 0)
    frac = (len(missing) / len(ledger_longs)) if ledger_longs else 0.0
    is_divergent = len(ledger_longs) >= min_ledger_positions and frac >= min_missing_fraction
    detail = (
        f"{len(missing)}/{len(ledger_longs)} ledger positions ({frac:.0%}) absent "
        f"from the broker book: {', '.join(missing[:15])}"
    )
    return PreOpenDivergence(
        is_divergent=is_divergent,
        missing_names=missing,
        missing_fraction=frac,
        ledger_positions=len(ledger_longs),
        broker_positions=sum(1 for v in broker.values() if v > 0),
        detail=detail,
    )


def ledger_net_positions(store) -> dict[str, float]:
    """Net shares per ticker from paper_fills (BUY - SELL), non-zero only."""
    return {
        t: as_qty(n or 0)
        for t, n in store.conn.execute(
            "SELECT ticker, "
            "SUM(CASE WHEN UPPER(side) = 'BUY' THEN filled_shares "
            "ELSE -filled_shares END) "
            "FROM paper_fills GROUP BY ticker"
        ).fetchall()
        if not is_zero(as_qty(n or 0))
    }


def last_snapshot_equity(store) -> float | None:
    """Equity from the most recent account_snapshots row, or None if none exists."""
    row = store.conn.execute(
        "SELECT equity FROM account_snapshots ORDER BY asof_date DESC LIMIT 1"
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def run_preopen_guard(
    *,
    store,
    broker_positions: dict,
    alpaca,
    notify_fn,
    min_missing_fraction: float = 0.5,
    min_ledger_positions: int = 2,
    equity_crash_fraction: float = 0.3,
) -> PreOpenGuardResult:
    """Run both pre-open layers at 09:25. `broker_positions` is the already-fetched
    `alpaca.get_positions()` map ({ticker: {"shares": int, ...}}), reused so we
    don't re-hit the broker for the book. Returns a result; the caller halts the
    day's sweep only when `wipe_halt`.
    """
    broker = {t: as_qty(info.get("shares", 0)) for t, info in broker_positions.items()}

    # Layer 1 — per-ticker oversell gate (cancel phantom-short sells before the open)
    try:
        open_orders = alpaca.list_open_orders()
    except Exception:  # noqa: BLE001 — don't let a read error skip layer 2
        open_orders = []
    # A transient/partial position read (Alpaca intermittently returns an empty or
    # short list on a 200) can make a genuinely-held name look sold. If the FIRST
    # read implies ANY oversell, re-confirm with ONE re-fetch before cancelling, so
    # a read glitch can't strand a legitimate exit (not just the whole-empty case).
    if find_oversell_orders(open_orders=open_orders, broker=broker):
        try:
            refetched = alpaca.get_positions()
            broker = {t: as_qty(info.get("shares", 0)) for t, info in refetched.items()}
        except Exception:  # noqa: BLE001 — keep the original read on a failed re-fetch
            pass
    oversell = find_oversell_orders(open_orders=open_orders, broker=broker)
    cancelled: list[dict] = []
    for o in oversell:
        try:
            alpaca.cancel_order(o["id"])
            cancelled.append(o)
        except Exception:  # noqa: BLE001 — page regardless (fail loud)
            cancelled.append({**o, "cancel_failed": True})

    # Layer 2 — book-wide wipe halt, corroborated by an equity collapse
    ledger = ledger_net_positions(store)
    div = check_preopen_divergence(
        ledger=ledger,
        broker=broker,
        min_missing_fraction=min_missing_fraction,
        min_ledger_positions=min_ledger_positions,
    )
    equity_crashed = False
    snap = last_snapshot_equity(store)
    if div.is_divergent and snap is not None and snap > 0:
        try:
            broker_equity = float(alpaca.get_account()["equity"])
            equity_crashed = broker_equity <= snap * (1 - equity_crash_fraction)
        except Exception:  # noqa: BLE001 — can't confirm equity → don't escalate to a full halt
            equity_crashed = False
    wipe_halt = div.is_divergent and equity_crashed

    if wipe_halt:
        # Best-effort cancel: a broker error here must NEVER stop the page below —
        # a wipe we couldn't cancel orders for is exactly when a human must hear.
        with contextlib.suppress(Exception):
            alpaca.cancel_all_open_orders()
        notify_fn(
            title="SMA HALTED: broker book wiped (pre-open)",
            message=(
                f"09:25 guard: {div.detail}, AND live equity collapsed vs the last "
                f"snapshot (>= {equity_crash_fraction:.0%} drop) — a broker-side "
                f"wipe (see incident 2026-07-07). ALL queued orders cancelled; the "
                f"day is halted. Verify the account before resuming."
            ),
        )
    elif cancelled:
        n_short = sum(1 for c in cancelled if not c.get("cancel_failed"))
        names = ", ".join(f"{c['symbol']}({c['qty']}>{c['held']})" for c in cancelled[:15])
        notify_fn(
            title="SMA pre-open: cancelled oversell orders (broker holdings dropped)",
            message=(
                f"09:25 guard cancelled {n_short} queued SELL(s) that exceeded the "
                f"live broker holdings (would open a short — the 7/07 failure mode): "
                f"{names}. The broker book lost shares overnight without a sale — "
                f"verify the account."
            ),
        )
    elif div.is_divergent:
        # Book looks wiped but equity did NOT collapse (cash-heavy wipe, or a stale/
        # lagging ledger) and no oversell order needed cancelling. Not a trading
        # halt — but a large book-vs-ledger gap should never be fully silent.
        notify_fn(
            title="SMA pre-open: positions missing but equity intact — verify",
            message=(
                f"09:25 guard: {div.detail}, but live equity did NOT collapse. Likely "
                f"a stale ledger (missed reconcile) or a cash-heavy broker wipe — not "
                f"halting, but verify the account/ledger."
            ),
        )

    return PreOpenGuardResult(
        oversell_cancelled=cancelled, wipe_halt=wipe_halt, divergence=div,
    )
