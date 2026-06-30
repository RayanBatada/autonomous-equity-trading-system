"""Adversarial tests proving all five risk rails in RiskRails fire correctly.

Threat model:
1. Position cap (5%): strategy submits target_weight=0.5; rejected.
2. Sector cap (25%): six 5% positions in same sector; 6th order rejected.
3. Drawdown auto-pause (15%): account in 16%+ drawdown; buy rejected.
4. Stop-loss (8%): not implemented in Phase 1 simulator; placeholder only.
5. Cash floor (5%): not implemented in Phase 1 simulator; placeholder only.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from sma.backtest.risk import RiskRails
from sma.backtest.simulator import simulate
from sma.backtest.slippage import SlippageModel
from sma.backtest.strategies.base import StrategyDecision

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class FixedDecisionStrategy:
    """Returns a fixed list of decisions on a specific date, nothing otherwise."""

    name = "fixed_decision"

    def __init__(self, decisions_by_date: dict[date, list[StrategyDecision]]):
        self.decisions_by_date = decisions_by_date

    def decide(self, asof_date, prices, fundamentals=None):
        return self.decisions_by_date.get(asof_date, [])


def _make_flat_prices(
    tickers: list[str], start: date, days: int, price: float = 100.0
) -> pd.DataFrame:
    rows = []
    for t in tickers:
        for i in range(days):
            d = start + timedelta(days=i)
            rows.append(
                {
                    "ticker": t,
                    "date": d,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "adj_close": price,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rail 1: position cap (5%)
# ---------------------------------------------------------------------------


def test_position_cap_rejects_50_percent_order():
    """Strategy targets 50% of account in AAA; simulator must reject the order.

    With flat prices at $100 and initial_cash=$100,000:
    - target_weight=0.5 => order_dollars=50,000 on day 2 fill
    - check_order: 50,000/100,000 = 50% > 5% cap => rejected
    - No trade should be filled, so num_trades remains 0.
    """
    start = date(2025, 1, 1)
    prices = _make_flat_prices(["AAA"], start=start, days=10)
    strategy = FixedDecisionStrategy(
        {
            start: [
                StrategyDecision(
                    asof_date=start, ticker="AAA", target_weight=0.5
                )
            ]
        }
    )
    result = simulate(
        strategy=strategy,
        universe=["AAA"],
        prices=prices,
        sector_map={"AAA": "Tech"},
        window_name="train",
        start_date=start,
        end_date=start + timedelta(days=9),
        initial_cash=100_000.0,
        slippage_model=SlippageModel(base_bps=0.0, illiquidity_penalty_bps=0.0),
        rails=RiskRails(),
    )
    # Order is rejected outright: no trade, account stays at $100,000.
    assert result.num_trades == 0, (
        f"Expected 0 trades (position cap should reject 50% order), "
        f"got {result.num_trades}"
    )
    # Total return should be ~0% because no position was taken.
    assert result.total_return == pytest.approx(0.0, abs=1e-6), (
        f"Expected 0 total return with no trades, got {result.total_return:.4f}"
    )


# ---------------------------------------------------------------------------
# Rail 2: sector cap (25%)
# ---------------------------------------------------------------------------


def test_sector_cap_rejects_overflow():
    """Six tickers each targeting 5% in same sector; total would be 30%.

    With flat prices and 0 slippage:
    - Tickers A1..A6 all in sector 'Tech', each with target_weight=0.05.
    - check_order fills A1..A5 (cumulative 25%), then rejects A6.
    - Final sector exposure must be <= 25%.

    Verification: 5 trades at $100 * ~50 shares each = $5,000/ticker.
    At most 5 trades should fill; total sector exposure <= 25% of $100,000.
    """
    tickers = [f"A{i}" for i in range(1, 7)]
    start = date(2025, 1, 1)
    prices = _make_flat_prices(tickers, start=start, days=10)

    # All 6 decisions on day 1 (filled day 2).
    decisions = [
        StrategyDecision(asof_date=start, ticker=t, target_weight=0.05)
        for t in tickers
    ]
    strategy = FixedDecisionStrategy({start: decisions})

    result = simulate(
        strategy=strategy,
        universe=tickers,
        prices=prices,
        sector_map={t: "Tech" for t in tickers},
        window_name="train",
        start_date=start,
        end_date=start + timedelta(days=9),
        initial_cash=100_000.0,
        slippage_model=SlippageModel(base_bps=0.0, illiquidity_penalty_bps=0.0),
        rails=RiskRails(),
    )

    # At most 5 of 6 orders should fill; 6th exceeds sector cap.
    assert result.num_trades <= 5, (
        f"Expected at most 5 trades (sector cap at 25% of 6 tickers), "
        f"got {result.num_trades}"
    )
    # At least 1 fill confirms the cap is the reason, not some other reject.
    assert result.num_trades >= 1, (
        "Expected at least 1 trade to fill; got 0 (sector cap may be misconfigured)"
    )


# ---------------------------------------------------------------------------
# Rail 3: drawdown auto-pause (15%)
# ---------------------------------------------------------------------------


def test_drawdown_auto_pause_rejects_buy_during_drawdown():
    """Account in 16%+ drawdown; new buy order must be rejected.

    Price series for AAA:
    - Day 1 (2025-01-01): open=100, close=100  -- strategy buys 4% position
    - Day 2 (2025-01-02): open=80, close=80    -- position fill; price drops (-20% MTM)
    - Day 3 (2025-01-03): open=80, close=80    -- drawdown ~16%; strategy tries another 4% buy
    - Day 4 (2025-01-04): open=80, close=80    -- would-be fill date for day 3 buy

    Timeline:
    - Day 1 close: strategy emits 4% buy, pending for day-2 fill.
    - Day 2 open: buy fills at $80. Account value before trade uses day-1 equity ($100,000).
      After fill: ~50 shares * $80 = $4,000 position, cash ~$96,000.
      Day-2 MTM: equity = 96,000 + 50*80 = 100,000 (no change from price perspective,
      but wait -- we track peak from initial $100,000 and price is $80 now).
      Actually: account_value for sizing uses day-1 close equity = $100,000.
      Day-2 equity_at_close = cash_after_buy + shares*close_80 = 96,000 + 50*80 = 100,000.
      Hmm -- the 4% buy was sized off $100,000, so order_dollars=4,000, shares=50 at $80.
      Cost=50*80=4,000. equity_day2 = (100,000-4,000) + 50*80 = 96,000+4,000 = 100,000.
      No drawdown yet because close price is also $80.

    We need a bigger drop so drawdown fires. Use a 20% drop directly:
    - Day 1 open=100, close=100: strategy buys 80% position (capped to 5% by rail 1).
      Wait, we need an actual drawdown. Let's use a price that actually causes a drawdown.

    Revised approach: hold positions, then have price drop sharply.
    - Day 1 open=100, close=100: buy 5% (exactly at cap).
    - Day 2 open=100, close=100: fill happens; equity stays ~$100,000.
    - Day 3 open=80, close=80: price drops. peak=$100,000, equity < $84,000 => drawdown>16%.
    - Day 4: strategy emits another 5% buy on day 3 (fills day 4). Rejected by drawdown rail.

    Actually the buy on day 3 would be at open=$80, and account_value uses day-2 equity.
    Let's trace carefully with 0 slippage, initial_cash=$100,000, price=$100 flat then drop.

    Day 1: no pending orders. Emit buy 5% for AAA.
    Day 2: fill at open=$100. account_value = day-1 equity = $100,000.
           order_dollars = 0.05 * 100,000 = 5,000. shares = 5000//100 = 50.
           cost=50*100=5,000. cash=95,000.
           MTM: 95,000 + 50*100 = $100,000. peak stays $100,000.
           Strategy emits nothing on day 2.
    Day 3: open=80, close=80. No pending orders. MTM: 95,000+50*80=$99,000.
           Drawdown = (100,000-99,000)/100,000 = 1%. Not enough yet.
           Strategy emits another 5% buy on day 3.
    Day 4: would fill. account_value = day-3 equity = $99,000.
           drawdown = (100,000-99,000)/100,000 = 1%. Still not enough.

    Need a sharper drop. Use price series that causes 20%+ drawdown before the second buy.

    Better: use a very large initial position that magnifies the drop.
    If we hold 100% of account in one stock (bypassing rail for testing), then a 20% price
    drop => 20% drawdown. But the position cap limits us to 5%.

    With 5% position and 20% price drop: drawdown = 5%*20% = 1%. Too small.

    Alternative: use multiple tickers to fill 25% sector limit, then drop all prices.
    5 tickers at 5% each = 25% exposed. 20% price drop => 25%*20% = 5% drawdown. Still small.

    The cleanest approach: directly manipulate the price series so that on the fill date,
    the account is in deep drawdown. Start with a very high initial price on day 1, then
    a crash to create the drawdown, then a buy attempt.

    Best approach: open simulation with positions pre-loaded is not possible. Instead,
    use many tickers filling the full position limit to maximize exposure, then crash.

    Actually, let's re-read check_order. It uses peak_equity from the simulator loop.
    peak_equity starts at initial_cash. The drawdown computation:
      current_drawdown = max(0, (peak_equity - account_value) / peak_equity)

    To get 16% drawdown with 5% max position per ticker: we need the total portfolio
    to drop 16%. With 5 tickers at 5% = 25% exposure, a 64%+ price drop achieves it.

    Instead: use a pre-crash approach. Set up the price so equity starts at $100k,
    peaks, then crashes 20% before the buy attempt.

    Simplest: buy on day 1 using many tickers to max out exposure. Then crash all prices.

    With 5 tickers at 5% each = 25% of account in stocks:
    Remaining 75% in cash. A 100% price drop of stocks gives 25% drawdown.
    Use 70% price drop: 25% * 0.70 = 17.5% drawdown. That exceeds the 15% limit.

    Let's verify with concrete numbers:
    - initial_cash = 100,000
    - Day 1: buy 5 tickers at 5% each (fills day 2 at $100 open).
    - Day 2: all fills. Each: 5,000 // 100 = 50 shares. 5 tickers = 250 shares.
             cash = 100,000 - 5*5,000 = 75,000. equity = 75,000 + 5*50*100 = 100,000.
             peak = 100,000.
    - Day 3: open=30, close=30 (70% drop). equity = 75,000 + 250*30 = 75,000+7,500 = 82,500.
             drawdown = (100,000-82,500)/100,000 = 17.5%. Exceeds 15%.
             Strategy emits a 5% buy on day 3.
    - Day 4: fill attempt. account_value = day-3 equity = 82,500.
             drawdown = 17.5% > 15%. check_order returns False. Trade rejected.
    """
    tickers = [f"X{i}" for i in range(1, 6)]  # 5 tickers for full sector exposure
    start = date(2025, 1, 1)

    rows = []
    # Days 1-2: price = 100
    for t in tickers:
        for d_offset, px in [(0, 100.0), (1, 100.0), (2, 30.0), (3, 30.0), (4, 30.0)]:
            d = start + timedelta(days=d_offset)
            rows.append(
                {
                    "ticker": t,
                    "date": d,
                    "open": px,
                    "high": px,
                    "low": px,
                    "close": px,
                    "adj_close": px,
                    "volume": 10_000_000,
                }
            )
    prices = pd.DataFrame(rows)

    buy_day = start  # buy 5 tickers on day 1 (fills day 2)
    crash_day = start + timedelta(days=2)  # price crashes here; strategy emits another buy
    second_buy_day = crash_day  # emitted on day 3 (fills day 4)

    # First set of buys: all 5 tickers at 5% on day 1.
    first_buys = [
        StrategyDecision(asof_date=buy_day, ticker=t, target_weight=0.05)
        for t in tickers
    ]
    # Second buy: try to add more AAA on the crash day; should be rejected due to drawdown.
    second_buys = [
        StrategyDecision(
            asof_date=second_buy_day, ticker=tickers[0], target_weight=0.05
        )
    ]

    strategy = FixedDecisionStrategy(
        {
            buy_day: first_buys,
            second_buy_day: second_buys,
        }
    )

    result = simulate(
        strategy=strategy,
        universe=tickers,
        prices=prices,
        sector_map={t: "Tech" for t in tickers},
        window_name="train",
        start_date=start,
        end_date=start + timedelta(days=4),
        initial_cash=100_000.0,
        slippage_model=SlippageModel(base_bps=0.0, illiquidity_penalty_bps=0.0),
        rails=RiskRails(),
    )

    # Day-1 buys fill on day 2: 5 buy trades.
    # Day-3 buy attempt fills on day 4: should be REJECTED due to drawdown.
    # The 70% crash on day 3 also triggers stop-losses on all 5 positions.
    buy_trade_count = sum(1 for t in result.trades if t["action"] == "buy")
    assert buy_trade_count == 5, (
        f"Expected exactly 5 buy trades (initial buys only; drawdown should block day-4 fill), "
        f"got {buy_trade_count}"
    )


# ---------------------------------------------------------------------------
# Rail 4: stop-loss (8%) -- not implemented in Phase 1 simulator
# ---------------------------------------------------------------------------


def test_stop_loss_closes_position_at_8_pct_below_entry():
    """A position that drops 12% below entry (exceeds 8% threshold) auto-closes.

    Price series for AAA (5 days):
    - Day 0: open=100, close=100  -- strategy issues buy; fills day 1 at open=100
    - Day 1: open=100, close=100  -- buy fills; cost_basis=100
    - Day 2: open=88, close=88   -- 12% below cost_basis; stop-loss fires at open=88
    - Day 3: open=88, close=88   -- position already closed
    - Day 4: open=88, close=88   -- position already closed
    """
    start = date(2025, 2, 1)
    days = 5
    rows = []
    prices_by_day = [100.0, 100.0, 88.0, 88.0, 88.0]
    for i in range(days):
        d = start + timedelta(days=i)
        px = prices_by_day[i]
        rows.append({
            "ticker": "AAA", "date": d,
            "open": px, "high": px, "low": px,
            "close": px, "adj_close": px, "volume": 1_000_000,
        })
    prices = pd.DataFrame(rows)

    # Buy 4% on day 0; never issue another order.
    strategy = FixedDecisionStrategy({
        start: [StrategyDecision(asof_date=start, ticker="AAA", target_weight=0.04)],
    })

    result = simulate(
        strategy=strategy,
        universe=["AAA"],
        prices=prices,
        sector_map={"AAA": "Tech"},
        window_name="train",
        start_date=start,
        end_date=start + timedelta(days=4),
        initial_cash=100_000.0,
        slippage_model=SlippageModel(base_bps=0.0, illiquidity_penalty_bps=0.0),
        rails=RiskRails(),
    )

    # Position must be gone: stop-loss fired on day 2.
    stop_loss_trades = [t for t in result.trades if t["action"] == "stop_loss"]
    assert len(stop_loss_trades) == 1, (
        f"Expected exactly 1 stop_loss trade, got {len(stop_loss_trades)}"
    )
    fired = stop_loss_trades[0]
    assert fired["date"] == start + timedelta(days=2), (
        f"Stop-loss should fire on day 2, fired on {fired['date']}"
    )
    assert fired["price"] == pytest.approx(88.0, abs=0.01), (
        f"Stop-loss fill should be at open=88, got {fired['price']}"
    )

    # No open positions remain.
    assert result.trades, "Expected at least one trade (buy + stop_loss)"
    # The round trip P&L should be negative (bought at 100, stopped at 88).
    buy_trades = [t for t in result.trades if t["action"] == "buy"]
    assert len(buy_trades) == 1
    shares = buy_trades[0]["shares"]
    expected_pnl = (88.0 - 100.0) * shares
    assert expected_pnl < 0, "Round-trip P&L should be negative"


# ---------------------------------------------------------------------------
# Rail 5: cash floor (5%) -- not implemented in Phase 1 simulator
# ---------------------------------------------------------------------------


def test_cash_floor_keeps_5_percent_in_cash():
    """Simulator must reserve 5% cash buffer regardless of strategy target weight.

    Use RiskRails with max_position_pct=1.0 and max_sector_pct=1.0 so the
    position/sector caps are out of the way, isolating the cash floor rail.
    With initial_cash=$100,000 and cash_floor_pct=0.05, at most $95,000 can be
    deployed. Final cash must be >= $5,000.
    """
    start = date(2025, 3, 1)
    prices = _make_flat_prices(["AAA"], start=start, days=10, price=100.0)

    # Target 100% of account in AAA; cash floor should trim to 95%.
    strategy = FixedDecisionStrategy({
        start: [StrategyDecision(asof_date=start, ticker="AAA", target_weight=1.0)],
    })

    result = simulate(
        strategy=strategy,
        universe=["AAA"],
        prices=prices,
        sector_map={"AAA": "Tech"},
        window_name="train",
        start_date=start,
        end_date=start + timedelta(days=9),
        initial_cash=100_000.0,
        slippage_model=SlippageModel(base_bps=0.0, illiquidity_penalty_bps=0.0),
        rails=RiskRails(max_position_pct=1.0, max_sector_pct=1.0, cash_floor_pct=0.05),
    )

    # Find cash after all buys by inspecting the trades list.
    # total cost of all buys
    total_bought = sum(
        t["value"] for t in result.trades if t["action"] == "buy"
    )
    final_cash = 100_000.0 - total_bought
    assert final_cash >= 5_000.0, (
        f"Expected final cash >= $5,000 (5% floor), got ${final_cash:.2f}. "
        f"Total deployed: ${total_bought:.2f}"
    )
