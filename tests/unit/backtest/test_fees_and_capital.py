"""Regulatory fees and configurable starting capital.

Paper charges no regulatory fees, so the default schedule is all-zero and the
backtest is unchanged. The 2026 rates exist as an opt-in for real-money studies.
"""

import pytest

from sma.backtest.fees import (
    FINRA_TAF_MAX_PER_TRADE_2026,
    FINRA_TAF_PER_SHARE_2026,
    SEC_FEE_PER_MILLION_2026,
    FeeSchedule,
)
from sma.backtest.simulator import DEFAULT_INITIAL_CASH


def test_default_schedule_is_free():
    f = FeeSchedule()
    assert f.is_zero
    assert f.sell_fees(shares=1_000_000, price=1_000.0) == 0.0
    assert f.buy_fees(shares=1_000_000, price=1_000.0) == 0.0


def test_real_2026_rates_match_the_published_schedule():
    f = FeeSchedule.real_money_2026()
    assert f.sec_fee_per_million == SEC_FEE_PER_MILLION_2026 == 20.60
    assert f.finra_taf_per_share == FINRA_TAF_PER_SHARE_2026 == 0.000195
    assert f.finra_taf_max_per_trade == FINRA_TAF_MAX_PER_TRADE_2026 == 9.79


def test_worked_round_trip_costs_about_a_quarter_basis_point():
    """200 shares at $50 = $10,000 principal.
    SEC $0.2060 + TAF $0.0390 + CAT $0.0006 = $0.2456, about 0.25 bps."""
    f = FeeSchedule.real_money_2026()
    assert f.sell_fees(shares=200, price=50.0) == pytest.approx(0.2456, abs=1e-4)
    assert f.buy_fees(shares=200, price=50.0) == pytest.approx(0.0006, abs=1e-6)


def test_sec_and_taf_are_sell_side_only():
    f = FeeSchedule.real_money_2026()
    buy = f.buy_fees(shares=1000, price=100.0)
    assert buy == pytest.approx(1000 * 0.000003)     # CAT only


def test_taf_cap_binds_on_a_huge_fill():
    f = FeeSchedule.real_money_2026()
    huge = f.sell_fees(shares=10_000_000, price=1.0)
    uncapped_taf = 10_000_000 * FINRA_TAF_PER_SHARE_2026
    assert uncapped_taf > FINRA_TAF_MAX_PER_TRADE_2026
    # sec + capped taf + cat
    want = 10_000_000 * 1.0 * 20.60 / 1e6 + 9.79 + 10_000_000 * 0.000003
    assert huge == pytest.approx(want)


def test_fees_handle_fractional_quantities():
    f = FeeSchedule.real_money_2026()
    assert f.sell_fees(shares=0.5, price=100.0) > 0
    assert f.sell_fees(shares=0.5, price=100.0) < f.sell_fees(shares=1.0, price=100.0)


def test_initial_cash_is_a_named_parameter_not_a_magic_number():
    assert DEFAULT_INITIAL_CASH == 100_000.0
    import inspect

    from sma.backtest.simulator import simulate
    sig = inspect.signature(simulate)
    assert sig.parameters["initial_cash"].default == DEFAULT_INITIAL_CASH
    assert sig.parameters["fees"].default is None


def test_evaluate_cli_exposes_initial_cash():
    from sma.backtest.__main__ import evaluate
    names = {p.name for p in evaluate.params}
    assert "initial_cash" in names
    opt = next(p for p in evaluate.params if p.name == "initial_cash")
    assert opt.default == DEFAULT_INITIAL_CASH


# ------------------------------------------------ fractional simulation ------

def _tiny_market():
    """Two names, one cheap and one expensive, over a handful of sessions."""
    import pandas as pd
    rows = []
    for i, d in enumerate(pd.bdate_range("2026-01-05", periods=12).date):
        rows += [
            {"ticker": "CHEAP", "date": d, "open": 10.0 + i * 0.1,
             "high": 11.0, "low": 9.0, "close": 10.0 + i * 0.1,
             "adj_close": 10.0 + i * 0.1, "volume": 5_000_000},
            {"ticker": "PRICEY", "date": d, "open": 900.0 + i,
             "high": 950.0, "low": 880.0, "close": 900.0 + i,
             "adj_close": 900.0 + i, "volume": 3_000_000},
        ]
    return pd.DataFrame(rows)


def _run(initial_cash, sizing=None):
    from sma.backtest.risk import RiskRails
    from sma.backtest.simulator import simulate
    from sma.backtest.strategies.equal_weight import EqualWeightStrategy

    prices = _tiny_market()
    return simulate(
        strategy=EqualWeightStrategy(universe=["CHEAP", "PRICEY"]),
        universe=["CHEAP", "PRICEY"],
        prices=prices,
        sector_map={"CHEAP": "Industrials", "PRICEY": "Health Care"},
        window_name="val",
        start_date=prices["date"].min(),
        end_date=prices["date"].max(),
        initial_cash=initial_cash,
        # 50% per name is the point of the test; the default 5% cap would
        # reject both decisions and every run would be flat cash.
        rails=RiskRails(max_position_pct=0.5, max_sector_pct=1.0,
                        cash_floor_pct=0.0, stop_loss_pct=0.0),
        sizing=sizing,
    )


def test_default_sizing_is_bit_identical_to_passing_nothing():
    from sma.live.sizing import SizingPolicy
    a = _run(100_000.0)
    b = _run(100_000.0, sizing=SizingPolicy())
    assert a.total_return == b.total_return
    assert a.sharpe == b.sharpe


def test_whole_share_backtest_silently_drops_the_expensive_name_at_fifty():
    """The measured failure, reproduced in the sim. A $25 slot buys 2 shares of
    a $10 name and ZERO of a $900 one, so half the book never exists — and the
    backtest reports a perfectly respectable return for the half that did."""
    r = _run(50.0)
    bought = {t["ticker"] for t in r.trades if t["action"] == "buy"}
    assert bought == {"CHEAP"}
    assert "PRICEY" not in bought


def test_fractional_backtest_holds_both_names_at_fifty_dollars():
    from sma.live.sizing import SizingPolicy
    r = _run(50.0, sizing=SizingPolicy(fractional_shares=True))
    bought = {t["ticker"] for t in r.trades if t["action"] == "buy"}
    assert bought == {"CHEAP", "PRICEY"}
    pricey = next(t for t in r.trades if t["ticker"] == "PRICEY")
    assert 0 < pricey["shares"] < 1        # a genuine fraction of a share


def test_fractional_and_whole_agree_at_a_large_account():
    """Fractional is a small-account fix, not a strategy change: at $10M the two
    modes differ only by the share-rounding residue, which is immaterial."""
    from sma.live.sizing import SizingPolicy
    whole = _run(10_000_000.0)
    frac = _run(10_000_000.0, sizing=SizingPolicy(fractional_shares=True))
    assert frac.total_return == pytest.approx(whole.total_return, abs=1e-4)
