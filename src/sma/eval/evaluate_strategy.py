"""Public evaluation entry point.

The single function the autoresearch agent in Phase 6 is allowed to call.
The agent imports from `sma.eval` and is forbidden from editing anything
under `sma.backtest`. This module is intentionally thin: it loads data,
delegates to the simulator, returns the BacktestResult.

Three guarantees this module enforces:

1. The function returns a BacktestResult with a single scalar primary
   metric (sharpe), so the agent can do keep/discard on result.sharpe.
2. Given (strategy code SHA, window, seed), the output is byte-identical
   across calls. The simulator is deterministic; this module loads the
   same data the same way each time.
3. Calling with window='test' requires an explicit opt-in flag so the
   agent cannot accidentally peek at the held-out window.
"""

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from sma.backtest.earnings_blackout import load_upcoming_earnings
from sma.backtest.result import BacktestResult
from sma.backtest.risk import RiskRails
from sma.backtest.simulator import DEFAULT_INITIAL_CASH, simulate
from sma.backtest.slippage import SlippageModel
from sma.backtest.strategies.base import Strategy
from sma.backtest.windows import WindowName, window_dates
from sma.db_connect import read_only_connect
from sma.sectors import sector_map_for

DEFAULT_DB_PATH = Path("data/sma.duckdb")


def _load_prices_for_window(
    db_path: Path,
    universe: list[str],
    start_date: date,
    end_date: date,
    *,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Load OHLCV from DuckDB for the given universe and window.

    Prefers yfinance source (it has split-adjusted adj_close). Falls back
    to alpaca if yfinance is missing for a (ticker, date).

    If `conn` is provided, reuses it (no new connection opened). This is
    required when the caller holds a writable connection elsewhere in the
    same process — DuckDB rejects mixed read/write configs on the same
    file (autoresearch loop hit this 2026-05-18).
    """
    if conn is None:
        if not db_path.exists():
            raise FileNotFoundError(
                f"No DuckDB at {db_path}; run `python -m sma.ingest run` first."
            )
        # read_only_connect (2026-08-05 audit): retry a transient lock overlap
        # instead of crashing (this path is also used by predict backfills).
        owned_conn = read_only_connect(db_path)
    else:
        owned_conn = None
        # Defensive: ensure caller-passed conn isn't None at this point.
        # If they passed a conn, we DON'T close it (caller manages lifecycle).
    use_conn = conn or owned_conn
    try:
        # Use yfinance preferentially. For a ticker/date with multiple sources,
        # take the yfinance row.
        df = use_conn.execute("""
            SELECT ticker, date, open, high, low, close, adj_close, volume
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY ticker, date
                           ORDER BY CASE source WHEN 'yfinance' THEN 0
                                                 WHEN 'alpaca' THEN 1
                                                 ELSE 2 END
                       ) AS rn
                FROM prices
                WHERE ticker = ANY($tickers)
                  AND date BETWEEN $start_date AND $end_date
                  AND adj_close IS NOT NULL
            ) t
            WHERE rn = 1
            ORDER BY ticker, date
        """, {"tickers": list(universe), "start_date": start_date, "end_date": end_date}).df()
    finally:
        if owned_conn is not None:
            owned_conn.close()

    # DuckDB returns date as datetime.date; pandas may convert it to Timestamp.
    # Normalize to date.
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def evaluate_strategy(
    *,
    strategy: Strategy,
    window: WindowName,
    universe: list[str],
    seed: int | None = None,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    db_path: Path = DEFAULT_DB_PATH,
    slippage_model: SlippageModel | None = None,
    rails: RiskRails | None = None,
    i_promise_this_is_a_promotion_decision: bool = False,
    membership: dict[str, date] | str | None = None,
) -> BacktestResult:
    """Evaluate a strategy on the given window.

    `window`:
      - "train": for model fitting and feature engineering iteration.
      - "val":   for hyperparameter / strategy tuning. The autoresearch
                 agent in Phase 6 optimizes Sharpe on this window.
      - "test":  the held-out promotion gate. Touched ONCE before any
                 paper-trade promotion decision. Requires
                 `i_promise_this_is_a_promotion_decision=True`.

    Returns a BacktestResult. Use `result.sharpe` as the single scalar
    optimization signal.
    """
    if window not in ("train", "val", "test"):
        raise ValueError(f"Unknown window: {window!r}")
    # Survivorship is an EXPLICIT choice (audit evaluate_strategy.py:109): the
    # old silent default traded the full CURRENT universe over history. The
    # universe.yaml added-dates are vault-addition dates (2026-04+), so
    # auto-loading them would instead zero out historical windows — callers
    # must pick one deliberately.
    if membership is None:
        raise ValueError(
            "evaluate_strategy requires an explicit `membership`: pass a "
            "point-in-time {ticker: first-tradable-date} map, or "
            "membership='current' to knowingly accept survivorship bias "
            "(trades today's universe across the whole window)."
        )
    if membership == "current":
        membership = None  # simulator: None = no membership filtering
    if window == "test" and not i_promise_this_is_a_promotion_decision:
        raise PermissionError(
            "Evaluating against the test window requires "
            "`i_promise_this_is_a_promotion_decision=True`. The test window "
            "is touched once before paper-trade promotion. Use 'val' for tuning."
        )

    start_date, end_date = window_dates(window)
    prices = _load_prices_for_window(db_path, universe, start_date, end_date)
    earnings = load_upcoming_earnings(db_path, start_date, end_date)

    return simulate(
        strategy=strategy,
        universe=universe,
        prices=prices,
        sector_map=sector_map_for(universe),
        earnings_blackouts=earnings,
        window_name=window,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        slippage_model=slippage_model or SlippageModel(),
        rails=rails or RiskRails(),
        seed=seed,
        # Point-in-time membership (opt-in). Default None = full-universe
        # (current behavior). NOTE: activating it on our windows is time-gated —
        # most are pre-2026-04-25 (the universe's birth) so a point-in-time set
        # would be near-empty/degenerate. Pass this only for post-birth windows
        # until the dynamic windows roll forward. See sma.ingest.universe.
        membership=membership,
    )
