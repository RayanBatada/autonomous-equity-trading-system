"""Backtest simulator: walks forward day-by-day through prices, applies a
strategy's target weights, executes at next-day open with slippage, enforces
risk rails, and returns a BacktestResult.

Lookahead prevention: strategy.decide(D, prices) only ever sees prices with
date <= D. Fills happen at D+1 open. Mark-to-market uses D's close.

Pure function. Same inputs => same outputs. No I/O except an optional git
SHA fetch for provenance.
"""

import inspect
import logging
import subprocess
from datetime import date

import pandas as pd

from sma.backtest.fees import FeeSchedule
from sma.backtest.metrics import (
    annualized_return,
    calmar,
    max_drawdown,
    sharpe,
    sortino,
    total_return,
)
from sma.backtest.result import BacktestResult
from sma.backtest.risk import RiskRails, check_order
from sma.backtest.slippage import SlippageModel, apply_slippage
from sma.backtest.strategies.base import Strategy, StrategyDecision
from sma.live.sizing import SizingPolicy

logger = logging.getLogger(__name__)

#: Starting capital for a backtest when the caller does not say otherwise.
#: Every research script and the evaluate CLI pin this same number so results
#: stay comparable across studies; it matches the paper account's original
#: $100k deposit. It is a PARAMETER everywhere — pass `initial_cash=` to run a
#: study at $50, $10k or $10M (see tests/unit/backtest/test_capital_scale.py).
DEFAULT_INITIAL_CASH: float = 100_000.0


class LookaheadLeakError(Exception):
    """Raised when a strategy returns a StrategyDecision whose asof_date does not
    match the current simulation date, indicating an attempt to use future data."""


def _current_git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _compute_adv_dollars(
    prices: pd.DataFrame,
    ticker: str,
    asof_date: date,
    lookback: int = 20,
) -> float:
    """Average daily dollar volume over the `lookback` trading days STRICTLY
    BEFORE `asof_date`.

    `asof_date` is the fill day, and fills occur at its OPEN — so that day's
    full-session close/volume is not yet known and must be excluded (using
    `<= asof_date` leaks fill-day volume, inflating ADV and understating
    slippage on high-volume days). If fewer than 5 prior days available, fall
    back to the most recent prior day's price * volume.
    """
    sub = prices[(prices["ticker"] == ticker) & (prices["date"] < asof_date)]
    if sub.empty:
        return 0.0
    sub = sub.sort_values("date").tail(lookback)
    if len(sub) < 5:
        last = sub.iloc[-1]
        return float(last["close"]) * float(last["volume"])
    dollar_volumes = sub["close"] * sub["volume"]
    return float(dollar_volumes.mean())


def _monthly_returns(
    daily_returns: list[float], dates: list[date]
) -> tuple[list[float], list[str]]:
    """Compound daily returns into monthly returns.

    Returns (values, period_labels). Labels parallel values, formatted "YYYY-MM".
    """
    if not daily_returns:
        return [], []
    by_month: dict[tuple[int, int], float] = {}
    order: list[tuple[int, int]] = []
    for d, r in zip(dates, daily_returns, strict=True):
        key = (d.year, d.month)
        if key not in by_month:
            by_month[key] = 1.0
            order.append(key)
        by_month[key] *= (1.0 + r)
    values = [by_month[k] - 1.0 for k in order]
    labels = [f"{y:04d}-{m:02d}" for y, m in order]
    return values, labels


def _adj_open(row) -> float:
    """Open in ADJUSTED space: open * adj_close/close.

    The sim marks-to-market at adj_close; fills must live on the same
    split/dividend-adjusted scale or a 2:1 split between fill and mark fakes
    a -50% position loss and dividends leak into P&L attribution (audit
    simulator.py:176). Falls back to the raw open when close/adj_close are
    unusable (defensive; loaders filter adj_close IS NOT NULL).
    """
    o = float(row["open"])
    c = float(row["close"])
    ac = float(row["adj_close"])
    if c <= 0 or ac <= 0:
        return o
    return o * (ac / c)


def simulate(
    *,
    strategy: Strategy,
    universe: list[str],
    prices: pd.DataFrame,
    sector_map: dict[str, str],
    earnings_blackouts: dict[str, list[date]] | None = None,
    window_name: str,
    start_date: date,
    end_date: date,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    slippage_model: SlippageModel | None = None,
    fees: FeeSchedule | None = None,
    sizing: SizingPolicy | None = None,
    rails: RiskRails | None = None,
    seed: int | None = None,
    membership: dict[str, date] | None = None,
    extra_cash_floor_by_date: dict[date, float] | None = None,
) -> BacktestResult:
    """Walk forward through trading dates and simulate a strategy.

    Args:
        strategy: object with .name and .decide(asof_date, prices) -> list[StrategyDecision].
        universe: tickers eligible for trading. Decisions outside universe are skipped.
        prices: DataFrame with columns [ticker, date, open, close, adj_close, volume].
        sector_map: ticker -> GICS sector for sector-exposure risk checks.
        earnings_blackouts: ticker -> list of upcoming earnings dates. Buys are
            suppressed when the fill date falls within the blackout window before
            any of those dates. Sells are unaffected. Pass None to disable.
        window_name: "train", "val", or "test"; recorded on the result.
        start_date / end_date: inclusive bounds for the simulation.
        initial_cash: starting account value. Defaults to DEFAULT_INITIAL_CASH.
        slippage_model: defaults to SlippageModel().
        sizing: capital-scale sizing policy. Defaults to SizingPolicy(), which is
            whole shares — exactly what this simulator has always done. Pass
            SizingPolicy(fractional_shares=True) to model a small account that
            can hold a fraction of a share; without it a $50 backtest correctly
            but uselessly reports that nothing was ever bought.
        fees: regulatory sell-side fee schedule. Defaults to FeeSchedule(), which
            is ALL ZERO — paper trading charges no regulatory fees, so a zero
            schedule is what keeps the backtest comparable to the paper track
            record. Pass real rates only when modelling a real-money account.
        rails: defaults to RiskRails().
        seed: optional seed recorded on the result for provenance.
        extra_cash_floor_by_date: optional {date: floor} overlay — on a listed
            date the effective cash floor is max(drawdown-derisk floor, this).
            The dispersion de-risk research seam (2026-07-20): detectors are
            precomputed OUTSIDE the sim (point-in-time, prices-only) and fed in
            as a floor series, so the sim stays deterministic and the money
            path untouched. None = inert.

    Returns:
        BacktestResult with metrics and daily/monthly returns.
    """
    if slippage_model is None:
        slippage_model = SlippageModel()
    if rails is None:
        rails = RiskRails()
    if fees is None:
        fees = FeeSchedule()
    if sizing is None:
        sizing = SizingPolicy()

    universe_set = set(universe)

    # Trading dates within window (sorted, unique).
    all_dates = sorted({
        d for d in prices["date"].unique()
        if start_date <= d <= end_date
    })

    # State.
    cash: float = initial_cash
    positions_shares: dict[str, float] = {}     # ticker -> shares
    cost_basis: dict[str, float] = {}            # avg buy price per ticker
    entry_dates: dict[str, date] = {}            # ticker -> date first opened
    # Post-entry PEAK price (adjusted space) per open position, for the trailing
    # stop. Seeded at the entry fill and raised by each subsequent adj_close;
    # NEVER includes a day's own open/high before that day's exit check, so the
    # trailing stop can't fire on lookahead (a fresh high never triggers itself).
    peak_price: dict[str, float] = {}            # ticker -> running peak
    holding_days_per_trade: list[int] = []       # filled when a position closes
    realized_round_trips: list[float] = []       # P&L per round trip (for hit rate)
    trades: list[dict] = []                       # one entry per filled buy/sell

    daily_returns: list[float] = []
    daily_dates: list[date] = []
    equity_curve: list[float] = [initial_cash]
    peak_equity: float = initial_cash

    num_trades: int = 0
    pending_orders: list[StrategyDecision] = []  # decisions made on D, executed on D+1

    def _equity_at_close(d: date) -> float:
        # Use adj_close for mark-to-market so dividends are captured in returns.
        eq = cash
        for tkr, sh in positions_shares.items():
            row = prices[(prices["ticker"] == tkr) & (prices["date"] == d)]
            if row.empty:
                # No price on this day; use last known adj_close.
                prior = prices[(prices["ticker"] == tkr) & (prices["date"] < d)]
                if prior.empty:
                    continue
                px = float(prior.sort_values("date").iloc[-1]["adj_close"])
            else:
                px = float(row.iloc[0]["adj_close"])
            eq += sh * px
        return eq

    # Decide the strategy.decide() call convention ONCE. Previously this used a
    # try/except TypeError that retried with a positional fundamentals arg — but
    # that also caught (and masked) a TypeError raised *inside* decide(), e.g. a
    # crashing tilt() under tilt_strict, silently retrying and scoring the broken
    # proposal as the untilted baseline. Inspect the signature instead so an
    # internal TypeError propagates instead of being mistaken for a sig mismatch.
    _decide_params = inspect.signature(strategy.decide).parameters
    _decide_requires_fundamentals = (
        "fundamentals" in _decide_params
        and _decide_params["fundamentals"].default is inspect.Parameter.empty
    )
    # Pass the live held set so a hysteresis-aware strategy can keep names in its
    # rank buffer instead of churning. Only when the strategy accepts the kwarg.
    _decide_accepts_holdings = "current_holdings" in _decide_params

    for i, d in enumerate(all_dates):
        # Tickers force-exited by a Step-0 price exit TODAY. A still-standing
        # target from yesterday's decision must NOT re-buy them at the SAME open
        # (that would undo the exit as a no-op round-trip and make any
        # take-profit/trailing backtest meaningless). They may be re-bought on a
        # LATER day if the model still wants them. Stays empty when all exit
        # knobs are 0, so default behavior is byte-identical to the baseline.
        _exited_today: set[str] = set()
        # ---- Step 0: PRICE-EXIT scan (before pending orders; simulates automatic
        # triggers at the open). Three independent, default-OFF risk exits, each
        # evaluated against today's OPEN in adjusted space:
        #   * take-profit   — open >= entry * (1 + take_profit_pct)   (bank gains)
        #   * trailing stop — open <= peak  * (1 - trailing_stop_pct) (lock in a
        #                     winner that rolled over; peak is the post-entry high)
        #   * fixed stop    — open <= entry * (1 - stop_loss_pct)     (legacy)
        # ORDERING vs min_hold: a price exit is a RISK exit and fires even inside
        # rails.min_hold_days (matching the pre-existing fixed stop, which never
        # consulted min_hold). min_hold only protects rank-rotation force-sells
        # (Step 1b), not hard risk exits. When all three knobs are 0 this whole
        # block is a no-op — identical to the prior stop-only baseline.
        for tkr in list(positions_shares.keys()):
            row = prices[(prices["ticker"] == tkr) & (prices["date"] == d)]
            if row.empty:
                continue
            today_open = _adj_open(row.iloc[0])
            basis = cost_basis.get(tkr, today_open)
            if basis <= 0:
                continue
            peak = peak_price.get(tkr, basis)
            pct_loss = (basis - today_open) / basis
            # Priority: take-profit (bank the win) > trailing stop > fixed stop.
            # Each guarded by its own knob so a 0 default disables it.
            exit_reason: str | None = None
            if rails.take_profit_pct > 0 and today_open >= basis * (1 + rails.take_profit_pct):
                exit_reason = "take_profit"
            elif (
                rails.trailing_stop_pct > 0
                and peak > 0
                and today_open <= peak * (1 - rails.trailing_stop_pct)
            ):
                exit_reason = "trailing_stop"
            # stop_loss_pct <= 0 DISABLES the stop. Without this guard,
            # `pct_loss >= 0.0` fired on every flat/down position at the open,
            # liquidating break-even holdings and confounding no-stop baselines.
            elif rails.stop_loss_pct > 0 and pct_loss >= rails.stop_loss_pct:
                exit_reason = "stop_loss"
            if exit_reason is None:
                continue
            shares = positions_shares[tkr]
            # A price exit is a market SELL — apply slippage like any other fill,
            # don't model it as frictionless (which overstated the stop's
            # backtested value). It MUST execute, so fall back to the raw open
            # if the slippage model rejects the order on size-vs-ADV.
            adv = _compute_adv_dollars(prices, tkr, d)
            try:
                exit_px = apply_slippage(today_open, "sell", adv, slippage_model)
            except ValueError:
                exit_px = today_open
            proceeds = shares * exit_px - fees.sell_fees(shares=shares, price=exit_px)
            cash += proceeds
            realized = (exit_px - basis) * shares
            realized_round_trips.append(realized)
            trades.append({
                "date": d, "ticker": tkr, "action": exit_reason,
                "shares": shares, "price": exit_px, "value": proceeds,
            })
            if tkr in entry_dates:
                holding_days_per_trade.append((d - entry_dates[tkr]).days)
                del entry_dates[tkr]
            del positions_shares[tkr]
            del cost_basis[tkr]
            peak_price.pop(tkr, None)
            _exited_today.add(tkr)
            num_trades += 1
            logger.debug("%s triggered for %s on %s at %.2f", exit_reason, tkr, d, today_open)

        # ---- Step 1: execute any orders pending from prior day (fill at today's open).
        # Live-parity rotation freeing (Codex 2026-06-11): the live risk
        # pipeline frees sector room for held names the model DROPPED — their
        # force-sell lands in this same batch (Step 1b) — unless min_hold
        # protects the exit. Without this the sim rejected the rotation buy
        # live accepts and completed rotations a day late (return drag).
        _batch_tickers = {dec.ticker for dec in pending_orders}
        _dropped_today: set[str] = set()
        if _batch_tickers:
            for _tkr in positions_shares:
                if _tkr in _batch_tickers:
                    continue
                if rails.min_hold_days > 0 and _tkr in entry_dates:
                    _hd = (d - entry_dates[_tkr]).days
                    if 0 <= _hd < rails.min_hold_days:
                        continue  # exit will be min-hold-vetoed; room stays reserved
                _dropped_today.add(_tkr)

        # Live-parity cash (Codex 2026-06-16): live translate sums ALL sell
        # proceeds (running_cash = cash + sell_proceeds) BEFORE gating buys on
        # the cash floor. The sim force-sells in Step 1b (after buys), so a
        # rotation buy funded by a drop was wrongly floor-blocked. Project the
        # force-sell proceeds now so the buy floor-check + trim see them, just
        # like live. Step 1b still executes the actual sells (cash nets out).
        projected_fs_proceeds = 0.0
        for _tkr in _dropped_today:
            _held = positions_shares.get(_tkr, 0.0)
            if _held <= 0:
                continue
            _row = prices[(prices["ticker"] == _tkr) & (prices["date"] == d)]
            if _row.empty:
                continue
            try:
                _fp = apply_slippage(
                    _adj_open(_row.iloc[0]), "sell",
                    _compute_adv_dollars(prices, _tkr, d), slippage_model,
                )
            except ValueError:
                continue
            projected_fs_proceeds += _held * _fp

        for decision in pending_orders:
            tkr = decision.ticker
            if tkr not in universe_set:
                logger.debug("skip decision for %s: not in universe", tkr)
                continue
            row = prices[(prices["ticker"] == tkr) & (prices["date"] == d)]
            if row.empty:
                logger.debug("skip decision for %s on %s: no price row", tkr, d)
                continue
            open_px = _adj_open(row.iloc[0])

            # Account value pre-trade for sizing.
            account_value = _equity_at_close(all_dates[i - 1]) if i > 0 else cash
            current_position_dollars = positions_shares.get(tkr, 0.0) * open_px
            target_dollars = decision.target_weight * account_value
            order_dollars = target_dollars - current_position_dollars

            if abs(order_dollars) < 1.0:
                continue  # no meaningful change

            # Rebalance dead-zone (live orders.py parity): skip small rebalances
            # of an EXISTING position. New entries (current==0) bypass it; full
            # exits are handled by force-sell (Step 1b), not here.
            if (
                current_position_dollars > 0
                and rails.rebalance_dead_zone_pct > 0
                and abs(order_dollars)
                < rails.rebalance_dead_zone_pct * current_position_dollars
            ):
                continue

            side = "buy" if order_dollars > 0 else "sell"

            # A ticker force-exited by a Step-0 price exit earlier TODAY must not
            # be re-bought at the same open by a still-standing target from
            # yesterday — that would undo the exit as a no-op round-trip. It may
            # be re-bought on a LATER day if the model still wants it. Same-day
            # SELLs are unaffected (the price exit already flattened the position,
            # so `held <= 0` short-circuits any sell below anyway).
            if side == "buy" and tkr in _exited_today:
                logger.debug(
                    "skip same-day re-entry of %s on %s: price-exited earlier today",
                    tkr, d,
                )
                continue

            # Risk checks (only meaningful on increases; sells reduce exposure).
            if side == "buy":
                # Build current dollar positions and sector exposure as of pre-trade.
                current_positions_dollars: dict[str, float] = {}
                sector_exposure: dict[str, float] = {}
                for held_tkr, held_sh in positions_shares.items():
                    held_row = prices[(prices["ticker"] == held_tkr) & (prices["date"] == d)]
                    if held_row.empty:
                        prior = prices[(prices["ticker"] == held_tkr) & (prices["date"] < d)]
                        if prior.empty:
                            continue
                        held_px = float(prior.sort_values("date").iloc[-1]["adj_close"])
                    else:
                        held_px = _adj_open(held_row.iloc[0])
                    held_value = held_sh * held_px
                    current_positions_dollars[held_tkr] = held_value
                    if held_tkr in _dropped_today:
                        # Force-sold later this batch (Step 1b): its sector
                        # room is freed for this batch's buys (live parity).
                        continue
                    held_sector = sector_map.get(held_tkr, "Unknown")
                    sector_exposure[held_sector] = (
                        sector_exposure.get(held_sector, 0.0) + held_value / account_value
                    )

                if peak_equity > 0:
                    current_drawdown = max(0.0, (peak_equity - account_value) / peak_equity)
                else:
                    current_drawdown = 0.0
                ok, reason = check_order(
                    rails=rails,
                    ticker=tkr,
                    order_dollars=order_dollars,
                    account_value=account_value,
                    current_positions=current_positions_dollars,
                    sector=sector_map.get(tkr, "Unknown"),
                    sector_exposure=sector_exposure,
                    current_drawdown=current_drawdown,
                )
                if not ok:
                    logger.debug("risk reject for %s on %s: %s", tkr, d, reason)
                    continue

                if earnings_blackouts is not None:
                    from sma.backtest.earnings_blackout import in_earnings_blackout
                    if in_earnings_blackout(tkr, d, earnings_blackouts):
                        logger.debug("earnings blackout: skipping buy of %s on %s", tkr, d)
                        continue

            # Compute fill price after slippage.
            adv = _compute_adv_dollars(prices, tkr, d)
            try:
                fill_px = apply_slippage(open_px, side, adv, slippage_model)
            except ValueError as exc:
                logger.debug("slippage reject for %s on %s: %s", tkr, d, exc)
                continue

            if side == "buy":
                # Cash floor, drawdown-scaled (2026-06-15): in a deepening
                # drawdown the effective floor rises, cutting exposure as a
                # persistent momentum crash develops. slope=0 -> static floor.
                from sma.risk.derisk import derisk_cash_floor
                _dd = (
                    max(0.0, (peak_equity - account_value) / peak_equity)
                    if peak_equity > 0 else 0.0
                )
                _eff_floor = derisk_cash_floor(
                    rails.cash_floor_pct, _dd,
                    start=rails.drawdown_derisk_start,
                    slope=rails.drawdown_derisk_slope,
                    cap=rails.drawdown_derisk_cap,
                )
                if extra_cash_floor_by_date:
                    # Research overlay (dispersion de-risk): highest floor wins.
                    _eff_floor = max(
                        _eff_floor, extra_cash_floor_by_date.get(d, 0.0)
                    )
                min_cash_after = _eff_floor * account_value
                # cash + incoming force-sell proceeds (live parity), not bare cash
                available_cash = cash + projected_fs_proceeds
                max_spendable = max(0.0, available_cash - min_cash_after)
                if max_spendable <= 0:
                    logger.debug(
                        "cash floor reject for %s on %s: no spendable cash above floor", tkr, d
                    )
                    continue
                order_dollars = min(order_dollars, max_spendable)
                # Whole shares by default; cash from rounding stays in cash.
                # Fractional mode truncates to the policy's precision instead,
                # which is what makes a small-account backtest meaningful.
                shares_to_buy = sizing.target_qty(1.0, order_dollars, fill_px)
                if shares_to_buy <= 0:
                    continue
                if sizing.min_order_notional > 0 and (
                    shares_to_buy * fill_px < sizing.min_order_notional
                ):
                    continue
                cost = shares_to_buy * fill_px
                if cost > available_cash:
                    # Can't afford even with incoming force-sell proceeds. Trim.
                    shares_to_buy = sizing.target_qty(1.0, available_cash, fill_px)
                    if shares_to_buy <= 0:
                        continue
                    cost = shares_to_buy * fill_px
                # Spend now; Step 1b adds the actual force-sell proceeds so cash
                # nets out by end of batch (may dip negative intra-batch, as in
                # live where buys+sells submit together). available_cash is
                # recomputed per buy from the updated cash, so the shared
                # projected proceeds are consumed correctly without double-count.
                cash -= cost + fees.buy_fees(shares=shares_to_buy, price=fill_px)
                prev_shares = positions_shares.get(tkr, 0.0)
                prev_basis = cost_basis.get(tkr, 0.0)
                new_shares = prev_shares + shares_to_buy
                # Weighted avg cost basis.
                cost_basis[tkr] = (prev_basis * prev_shares + fill_px * shares_to_buy) / new_shares
                positions_shares[tkr] = new_shares
                if tkr not in entry_dates:
                    entry_dates[tkr] = d
                # Seed / raise the post-entry peak. A brand-new position starts
                # at its fill; adding to an existing one keeps the higher peak so
                # the trailing stop still references the true post-entry high.
                peak_price[tkr] = max(peak_price.get(tkr, 0.0), fill_px)
                trades.append({
                    "date": d, "ticker": tkr, "action": "buy",
                    "shares": shares_to_buy, "price": fill_px, "value": cost,
                })
                num_trades += 1
            else:  # sell
                held = positions_shares.get(tkr, 0.0)
                if held <= 0:
                    continue
                # Sell shares to bring position to target_dollars.
                shares_to_sell_value = -order_dollars  # positive
                if sizing.fractional_shares:
                    # No `+ 1`: that existed only to compensate for the floor
                    # below, and a whole extra share is a large error once the
                    # position itself can be a fraction of one.
                    shares_to_sell = min(held, shares_to_sell_value / fill_px)
                else:
                    shares_to_sell = min(held, int(shares_to_sell_value // fill_px) + 1)
                # Don't oversell.
                if shares_to_sell > held:
                    shares_to_sell = held
                if shares_to_sell <= 0:
                    continue
                proceeds = (
                    shares_to_sell * fill_px
                    - fees.sell_fees(shares=shares_to_sell, price=fill_px)
                )
                cash += proceeds
                # Round-trip P&L: realized = (fill_px - cost_basis) * shares_to_sell
                basis = cost_basis.get(tkr, fill_px)
                realized = (fill_px - basis) * shares_to_sell
                realized_round_trips.append(realized)
                positions_shares[tkr] = held - shares_to_sell
                if positions_shares[tkr] <= 0:
                    # Closed out: track holding days.
                    if tkr in entry_dates:
                        holding_days_per_trade.append((d - entry_dates[tkr]).days)
                        del entry_dates[tkr]
                    positions_shares.pop(tkr, None)
                    cost_basis.pop(tkr, None)
                    peak_price.pop(tkr, None)
                trades.append({
                    "date": d, "ticker": tkr, "action": "sell",
                    "shares": shares_to_sell, "price": fill_px, "value": proceeds,
                })
                num_trades += 1

        # ---- Step 1b: force-sell held names the strategy DROPPED (mirrors live
        # translate() keep_set). A top-K strategy holds only its current picks;
        # without this the backtest held dropped names forever, making the
        # stop-loss the de-facto sole exit and corrupting turnover/returns.
        # min_hold protects full exits of positions younger than min_hold_days.
        # ONLY when there IS a target portfolio: an EMPTY decision set means a
        # pipeline gap (no model/predictions that day), not "go to cash" — live
        # translate()'s paranoia rail holds in that case rather than liquidating.
        decision_tickers = {dec.ticker for dec in pending_orders}
        for tkr in (list(positions_shares.keys()) if decision_tickers else []):
            if tkr in decision_tickers:
                continue
            held = positions_shares.get(tkr, 0.0)
            if held <= 0:
                continue
            if rails.min_hold_days > 0 and tkr in entry_dates:
                held_days = (d - entry_dates[tkr]).days
                if 0 <= held_days < rails.min_hold_days:
                    continue  # min_hold protects this full exit (live parity)
            row = prices[(prices["ticker"] == tkr) & (prices["date"] == d)]
            if row.empty:
                continue
            open_px = _adj_open(row.iloc[0])
            adv = _compute_adv_dollars(prices, tkr, d)
            try:
                fill_px = apply_slippage(open_px, "sell", adv, slippage_model)
            except ValueError:
                continue
            proceeds = held * fill_px - fees.sell_fees(shares=held, price=fill_px)
            cash += proceeds
            basis = cost_basis.get(tkr, fill_px)
            realized_round_trips.append((fill_px - basis) * held)
            if tkr in entry_dates:
                holding_days_per_trade.append((d - entry_dates[tkr]).days)
                del entry_dates[tkr]
            positions_shares.pop(tkr, None)
            cost_basis.pop(tkr, None)
            peak_price.pop(tkr, None)
            trades.append({
                "date": d, "ticker": tkr, "action": "sell",
                "shares": held, "price": fill_px, "value": proceeds,
            })
            num_trades += 1

        pending_orders = []

        # ---- Step 2: ask strategy for new decisions (using only data up to and including D).
        visible_prices = prices[prices["date"] <= d]
        _hold_kw = (
            {"current_holdings": set(positions_shares.keys())}
            if _decide_accepts_holdings else {}
        )
        if _decide_requires_fundamentals:
            decisions = strategy.decide(d, visible_prices, None, **_hold_kw)
        else:
            decisions = strategy.decide(d, visible_prices, **_hold_kw)
        if decisions:
            for dec in decisions:
                if dec.asof_date != d:
                    raise LookaheadLeakError(
                        f"ticker={dec.ticker}: decision claims asof_date={dec.asof_date} "
                        f"but current sim date is {d}"
                    )
            if membership is not None:
                # Point-in-time selection: drop names not yet in the universe as
                # of decision day D (added later). Features still score over the
                # full universe (parity with how the model was trained); only the
                # tradable SELECTION is restricted, so the backtest can't buy
                # hindsight-added names. Names with no recorded date are kept.
                decisions = [
                    dec for dec in decisions
                    # Fail closed: a name absent from the membership map is NOT
                    # tradable. An explicit None date is legacy "always present".
                    if dec.ticker in membership
                    and (membership[dec.ticker] is None or membership[dec.ticker] <= d)
                ]
            pending_orders.extend(decisions)

        # ---- Step 2b: raise each open position's post-entry peak with today's
        # adj_close (the mark price). This is the ONLY place peaks grow from
        # market data; by updating at the CLOSE it stays strictly historical for
        # the next day's open-time trailing-stop check (no lookahead). A no-op
        # when trailing_stop is off, but cheap enough to always maintain.
        for tkr in positions_shares:
            prow = prices[(prices["ticker"] == tkr) & (prices["date"] == d)]
            if prow.empty:
                continue
            ac = float(prow.iloc[0]["adj_close"])
            if ac > peak_price.get(tkr, 0.0):
                peak_price[tkr] = ac

        # ---- Step 3: mark-to-market at end of day, compute daily return.
        eq = _equity_at_close(d)
        prev_eq = equity_curve[-1]
        ret = (eq - prev_eq) / prev_eq if prev_eq > 0 else 0.0
        daily_returns.append(ret)
        daily_dates.append(d)
        equity_curve.append(eq)
        if eq > peak_equity:
            peak_equity = eq

    # ---- Step 4: compute metrics.
    monthly, monthly_periods = _monthly_returns(daily_returns, daily_dates)

    # Include positions still open at end of backtest so daily-rebalance strategies
    # report meaningful holding periods rather than near-zero.
    for _tkr, entry_d in entry_dates.items():
        holding_days_per_trade.append((end_date - entry_d).days)

    if holding_days_per_trade:
        avg_hold = sum(holding_days_per_trade) / len(holding_days_per_trade)
    else:
        avg_hold = 0.0

    if realized_round_trips:
        hr = sum(1 for p in realized_round_trips if p > 0) / len(realized_round_trips)
    else:
        hr = 0.0

    return BacktestResult(
        strategy_name=strategy.name,
        window=window_name,  # type: ignore[arg-type]
        start_date=start_date,
        end_date=end_date,
        sharpe=sharpe(daily_returns),
        sortino=sortino(daily_returns),
        calmar=calmar(daily_returns),
        max_drawdown=max_drawdown(daily_returns),
        total_return=total_return(daily_returns),
        annualized_return=annualized_return(daily_returns),
        hit_rate=hr,
        num_trades=num_trades,
        avg_holding_days=avg_hold,
        daily_returns=daily_returns,
        monthly_returns=monthly,
        code_commit=_current_git_sha(),
        data_run_id=0,
        seed=seed,
        monthly_periods=monthly_periods,
        trades=trades,
    )
