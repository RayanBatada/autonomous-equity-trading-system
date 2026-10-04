"""CLI entry point for the backtest harness.

Run as `python -m sma.backtest <subcommand>`.
"""

import sys
from datetime import date
from pathlib import Path

import click

from sma.backtest.overfit import detect_overfit
from sma.backtest.result import BacktestResult
from sma.backtest.simulator import DEFAULT_INITIAL_CASH
from sma.backtest.strategies.base import Strategy
from sma.eval.evaluate_strategy import _load_prices_for_window, evaluate_strategy
from sma.ingest.universe import load_universe

_UNIVERSE_PATH = Path(__file__).parent.parent / "universe.yaml"
_KNOWN_STRATEGIES = ("buy_and_hold_spy", "equal_weight", "random_long", "xgb_top_k")
_DEFAULT_DB_PATH = Path("data/sma.duckdb")
_DEFAULT_MODELS_DIR = Path("models_artifacts")


def _get_universe() -> list[str]:
    return load_universe(_UNIVERSE_PATH)


def _build_strategy(
    name: str,
    seed: int | None,
    universe: list[str],
    use_theses: bool = False,
    sector_neutralize: float = 0.0,
) -> Strategy:
    from sma.backtest.strategies.buy_and_hold_spy import BuyAndHoldSPYStrategy
    from sma.backtest.strategies.equal_weight import EqualWeightStrategy
    from sma.backtest.strategies.random_long import RandomLongStrategy

    if name == "buy_and_hold_spy":
        return BuyAndHoldSPYStrategy()
    if name == "equal_weight":
        return EqualWeightStrategy(universe=universe)
    if name == "random_long":
        return RandomLongStrategy(universe=universe, seed=seed if seed is not None else 0)
    if name == "xgb_top_k":
        from sma.backtest.strategies.xgb_top_k import XGBoostTopKStrategy
        from sma.ingest.store import Store
        from sma.model.predictor import Predictor
        predictor = Predictor(
            models_dir=_DEFAULT_MODELS_DIR,
            db_path=_DEFAULT_DB_PATH,
        )
        store = (
            Store(path=_DEFAULT_DB_PATH).connect(read_only=True)
            if use_theses else None
        )
        return XGBoostTopKStrategy(
            predictor=predictor,
            universe=universe,
            use_theses=use_theses,
            store=store,
            sector_neutralize=sector_neutralize,
        )
    raise click.BadParameter(
        f"Unknown strategy: {name}. Known: {', '.join(_KNOWN_STRATEGIES)}"
    )


def _print_result(result: BacktestResult) -> None:
    commit = (result.code_commit or "")[:8]
    click.echo(f"strategy:          {result.strategy_name}")
    click.echo(f"window:            {result.window}")
    click.echo(f"start_date:        {result.start_date}")
    click.echo(f"end_date:          {result.end_date}")
    click.echo(f"sharpe:            {result.sharpe:.4f}")
    click.echo(f"sortino:           {result.sortino:.4f}")
    click.echo(f"calmar:            {result.calmar:.4f}")
    click.echo(f"max_drawdown:      {result.max_drawdown:.2%}")
    click.echo(f"total_return:      {result.total_return:.2%}")
    click.echo(f"annualized_return: {result.annualized_return:.2%}")
    click.echo(f"hit_rate:          {result.hit_rate:.2%}")
    click.echo(f"num_trades:        {result.num_trades}")
    click.echo(f"avg_holding_days:  {result.avg_holding_days:.2f}")
    click.echo(f"code_commit:       {commit}")
    click.echo(f"data_run_id:       {result.data_run_id}")
    click.echo(f"seed:              {result.seed}")


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--strategy", required=True)
@click.option("--window", type=click.Choice(["train", "val", "test"]), required=True)
@click.option("--seed", type=int, default=None)
@click.option("--i-promise-this-is-a-promotion-decision", is_flag=True, default=False)
@click.option("--use-theses", is_flag=True, default=False,
              help="(xgb_top_k only) Layer Phase 4 LLM theses on top of the quant strategy.")
@click.option("--sector-neutralize", type=click.FloatRange(0.0, 1.0), default=0.0,
              show_default=True,
              help="(xgb_top_k only) Sector-relative scoring strength λ in [0,1]; "
                   "0=plain top-K, 1=rank on within-sector relative strength.")
@click.option("--initial-cash", type=click.FloatRange(min=0.0, min_open=True),
              default=DEFAULT_INITIAL_CASH, show_default=True,
              help="Starting capital. Was hardcoded to $100k; it is a parameter "
                   "because whole-share sizing behaves very differently at $50, "
                   "$10k and $10M — see live.sizing in config.yaml.")
def evaluate(
    strategy: str,
    window: str,
    seed: int | None,
    i_promise_this_is_a_promotion_decision: bool,
    use_theses: bool,
    sector_neutralize: float,
    initial_cash: float,
) -> None:
    universe = _get_universe()
    s = _build_strategy(
        strategy, seed, universe, use_theses=use_theses, sector_neutralize=sector_neutralize
    )
    # Rails consistent with the strategy contract: xgb_top_k is "k positions
    # x target_weight each" (10 x 10%). The default RiskRails cap of 5%
    # REJECTED every >5% decision, so the backtest silently measured only the
    # <=5% TAIL of the top-K — an inverse-selected portfolio (found 2026-06-10
    # when the gross-preserving tilt zeroed all trades).
    from sma.backtest.risk import eval_rails_for
    eval_rails = eval_rails_for(s)
    try:
        result = evaluate_strategy(
            # research CLI over historical windows: current-universe (biased)
            # by design until a true point-in-time membership map exists
            membership="current",
            strategy=s,
            initial_cash=initial_cash,
            window=window,  # type: ignore[arg-type]
            universe=universe,
            rails=eval_rails,
            seed=seed,
            i_promise_this_is_a_promotion_decision=i_promise_this_is_a_promotion_decision,
        )
    except PermissionError as exc:
        raise click.ClickException(str(exc)) from exc
    _print_result(result)


@cli.command("detect-overfit")
@click.option("--strategy", required=True)
@click.option("--seed", type=int, default=None)
def detect_overfit_cmd(strategy: str, seed: int | None) -> None:
    universe = _get_universe()
    from sma.backtest.risk import eval_rails_for

    s_train = _build_strategy(strategy, seed, universe)
    train_result = evaluate_strategy(
        membership="current",  # research CLI: survivorship accepted explicitly
        strategy=s_train,
        window="train",
        universe=universe,
        rails=eval_rails_for(s_train),  # NOT default rails (inverse-selection)
        seed=seed,
    )

    s_val = _build_strategy(strategy, seed, universe)
    val_result = evaluate_strategy(
        membership="current",  # research CLI: survivorship accepted explicitly
        strategy=s_val,
        window="val",
        universe=universe,
        rails=eval_rails_for(s_val),
        seed=seed,
    )

    report = detect_overfit(train_result, val_result)
    click.echo(report.summary())

    if not report.passed:
        raise SystemExit(1)


def _spy_ground_truth_return(db_path: Path) -> tuple[float, date, date]:
    """Return (ground_truth_return, first_trading_day, last_trading_day) for SPY in 2025.

    Uses the first and last trading days actually present in the price table
    rather than assuming Jan 1 / Dec 31 are trading days.
    """
    prices = _load_prices_for_window(db_path, ["SPY"], date(2025, 1, 1), date(2025, 12, 31))
    spy = prices[prices["ticker"] == "SPY"].sort_values("date")
    if spy.empty:
        raise ValueError("No SPY data found in 2025 in the database.")
    first_day = spy.iloc[0]["date"]
    last_day = spy.iloc[-1]["date"]
    first_price = float(spy.iloc[0]["adj_close"])
    last_price = float(spy.iloc[-1]["adj_close"])
    return (last_price / first_price) - 1.0, first_day, last_day


def _spy_simulated_return(db_path: Path) -> float:
    """Run BuyAndHoldSPYStrategy(target_weight=1.0) over 2025 SPY data, return total_return.

    The simulator itself executes fills in adjusted space (open * adj_close /
    close, see simulator._adj_open, 2026-06-10) so no caller-side price
    normalization is needed — doing it here too would double-adjust (the
    +1.05% SPY drift caught by this very gate).
    """
    from sma.backtest.risk import RiskRails
    from sma.backtest.simulator import DEFAULT_INITIAL_CASH, simulate
    from sma.backtest.slippage import SlippageModel
    from sma.backtest.strategies.buy_and_hold_spy import BuyAndHoldSPYStrategy

    prices = _load_prices_for_window(db_path, ["SPY"], date(2025, 1, 1), date(2025, 12, 31))
    spy = prices[prices["ticker"] == "SPY"].sort_values("date")
    if spy.empty:
        raise ValueError("No SPY data found in 2025 in the database.")
    first_day = spy.iloc[0]["date"]
    last_day = spy.iloc[-1]["date"]

    result = simulate(
        strategy=BuyAndHoldSPYStrategy(target_weight=1.0),
        universe=["SPY"],
        prices=prices,
        sector_map={"SPY": "ETF"},
        window_name="verify",
        start_date=first_day,
        end_date=last_day,
        initial_cash=DEFAULT_INITIAL_CASH,
        slippage_model=SlippageModel(),
        rails=RiskRails(
            max_position_pct=1.0,
            max_sector_pct=1.0,
            cash_floor_pct=0.0,
            stop_loss_pct=1.0,  # disable stop-loss for buy-and-hold ground-truth check
        ),
        seed=None,
    )
    return result.total_return


@cli.command()
def smoke() -> None:
    universe = _get_universe()
    strategies = list(_KNOWN_STRATEGIES)
    windows = ["train", "val"]

    # Column widths
    col_strategy = 20
    col_window = 6
    col_sharpe = 8
    col_return = 12
    col_drawdown = 12
    col_trades = 10

    header = (
        f"{'strategy':<{col_strategy}}  "
        f"{'window':<{col_window}}  "
        f"{'sharpe':>{col_sharpe}}  "
        f"{'total_ret':>{col_return}}  "
        f"{'max_dd':>{col_drawdown}}  "
        f"{'num_trades':>{col_trades}}"
    )
    sep = "-" * len(header)
    click.echo(sep)
    click.echo(header)
    click.echo(sep)

    for strat_name in strategies:
        for window in windows:
            s = _build_strategy(strat_name, None, universe)
            result = evaluate_strategy(
            # research CLI over historical windows: current-universe (biased)
            # by design until a true point-in-time membership map exists
            membership="current",
                strategy=s,
                window=window,  # type: ignore[arg-type]
                universe=universe,
            )
            row = (
                f"{result.strategy_name:<{col_strategy}}  "
                f"{result.window:<{col_window}}  "
                f"{result.sharpe:>{col_sharpe}.4f}  "
                f"{result.total_return:>{col_return}.2%}  "
                f"{result.max_drawdown:>{col_drawdown}.2%}  "
                f"{result.num_trades:>{col_trades}}"
            )
            click.echo(row)

    click.echo(sep)


@cli.command("verify-spy")
@click.option("--db", type=click.Path(), default=str(_DEFAULT_DB_PATH), show_default=True)
def verify_spy(db: str) -> None:
    """Check that the simulator's SPY 2025 return matches the ground-truth adj_close return."""
    db_path = Path(db)
    ground_truth, first_day, last_day = _spy_ground_truth_return(db_path)
    simulated = _spy_simulated_return(db_path)
    diff = abs(simulated - ground_truth)

    click.echo(f"date_range:      {first_day} to {last_day}")
    click.echo(f"ground_truth:    {ground_truth:.4%}")
    click.echo(f"simulated:       {simulated:.4%}")
    click.echo(f"abs_diff:        {diff:.4%}")

    if diff < 0.01:
        click.echo("result:          PASS (diff < 1%)")
    else:
        click.echo("result:          FAIL (diff >= 1%)")
        sys.exit(1)


if __name__ == "__main__":
    cli()
