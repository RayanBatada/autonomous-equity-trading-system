import pytest

from sma.backtest.slippage import SlippageModel, apply_slippage


def test_default_slippage_is_5bps_on_liquid_ticker():
    model = SlippageModel()
    fill_price = apply_slippage(
        intended_price=100.0,
        side="buy",
        adv_dollars=50_000_000,  # $50M ADV: very liquid
        model=model,
    )
    # 5 bps = 0.05% = 0.0005; buying pays MORE
    assert fill_price == pytest.approx(100.0 * 1.0005, rel=1e-9)


def test_buying_pays_more_selling_receives_less():
    model = SlippageModel()
    buy_price = apply_slippage(intended_price=100.0, side="buy",
                                adv_dollars=50_000_000, model=model)
    sell_price = apply_slippage(intended_price=100.0, side="sell",
                                 adv_dollars=50_000_000, model=model)
    assert buy_price > 100.0
    assert sell_price < 100.0
    # Symmetric magnitudes
    assert (buy_price - 100.0) == pytest.approx(100.0 - sell_price, rel=1e-9)


def test_low_liquidity_widens_slippage():
    model = SlippageModel()
    liquid = apply_slippage(intended_price=100.0, side="buy",
                             adv_dollars=50_000_000, model=model)
    # Above min_adv_dollars floor (5M) but below illiquidity threshold (10M)
    illiquid = apply_slippage(intended_price=100.0, side="buy",
                               adv_dollars=7_000_000, model=model)
    # Illiquid should pay more
    assert illiquid > liquid


def test_slippage_below_min_liquidity_threshold_raises():
    """Tickers below the universe min ADV cap should be rejected, not just penalized."""
    model = SlippageModel(min_adv_dollars=5_000_000)
    with pytest.raises(ValueError, match="below minimum liquidity"):
        apply_slippage(intended_price=100.0, side="buy",
                       adv_dollars=1_000_000, model=model)


def test_custom_bps_override():
    model = SlippageModel(base_bps=20.0)  # very wide
    p = apply_slippage(intended_price=100.0, side="buy",
                       adv_dollars=50_000_000, model=model)
    assert p == pytest.approx(100.0 * 1.0020, rel=1e-9)


def test_zero_slippage_model_returns_intended_price():
    model = SlippageModel(base_bps=0.0, illiquidity_penalty_bps=0.0)
    p_buy = apply_slippage(intended_price=100.0, side="buy",
                           adv_dollars=50_000_000, model=model)
    p_sell = apply_slippage(intended_price=100.0, side="sell",
                            adv_dollars=50_000_000, model=model)
    assert p_buy == 100.0
    assert p_sell == 100.0


def test_slippage_model_is_immutable():
    model = SlippageModel()
    with pytest.raises(Exception):
        model.base_bps = 999  # type: ignore
