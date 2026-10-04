"""Regression: codex HIGH finding (2026-05-12) — `_build_rails` previously
forgot to copy `min_hold_days` from `LiveRails`, so the live decide path
saw `RiskRails.min_hold_days == 0` regardless of config. The advertised
churn-protection rail was silently OFF.

These tests assert every LiveRails field reaches RiskRails so that a
future field-addition can't silently regress the same way.
"""

from types import SimpleNamespace

from sma.config import LiveRails
from sma.live.__main__ import _build_rails


def _settings_with_rails(**overrides) -> SimpleNamespace:
    """Make a minimal stand-in for the real Settings object."""
    rails = LiveRails(**overrides)
    return SimpleNamespace(live=SimpleNamespace(rails=rails))


def test_build_rails_passes_min_hold_days_to_riskrails():
    settings = _settings_with_rails(min_hold_days=3)
    rails = _build_rails(settings)
    assert rails.min_hold_days == 3, (
        "_build_rails must thread LiveRails.min_hold_days through to "
        f"RiskRails (got {rails.min_hold_days})"
    )


def test_build_rails_defaults_min_hold_to_one_when_config_default():
    settings = _settings_with_rails()  # LiveRails default is 1
    rails = _build_rails(settings)
    assert rails.min_hold_days == 1, (
        "LiveRails defaults min_hold_days=1; the live RiskRails should "
        f"inherit that (got {rails.min_hold_days})"
    )


def test_build_rails_can_disable_min_hold_via_config():
    settings = _settings_with_rails(min_hold_days=0)
    rails = _build_rails(settings)
    assert rails.min_hold_days == 0


def test_build_rails_threads_price_exit_knobs():
    """The 2026-07-01 price exits must reach RiskRails (same regression class
    as min_hold_days once did): a config value silently dropped = a rail off."""
    settings = _settings_with_rails(trailing_stop_pct=0.15, take_profit_pct=0.30)
    rails = _build_rails(settings)
    assert rails.trailing_stop_pct == 0.15
    assert rails.take_profit_pct == 0.30


def test_build_rails_price_exits_default_off():
    rails = _build_rails(_settings_with_rails())
    assert rails.trailing_stop_pct == 0.0
    assert rails.take_profit_pct == 0.0


def test_build_rails_no_config_falls_back_to_safe_default():
    """When the settings has no .live attribute (legacy path), defaults
    apply. The legacy default uses stop_loss_pct=0; min_hold_days is the
    RiskRails dataclass default (0) since no config object is consulted."""
    settings = SimpleNamespace(live=None)
    rails = _build_rails(settings)
    assert rails.stop_loss_pct == 0.0


def test_build_strategy_honors_live_strategy_config():
    """Ship of the 2026-06-12 corrected-eval winner (val sharpe +1.60 vs
    -1.58 baseline): sector-relative scoring λ=1.0 + rank hysteresis
    hold_rank=30 + min_hold 7d. Strategy params must come from
    settings.live.strategy, not hardcodes."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from sma.live.__main__ import _build_strategy

    settings = SimpleNamespace(
        live=SimpleNamespace(
            strategy=SimpleNamespace(k=15, hold_rank=30, sector_neutralize=1.0)
        )
    )
    with patch("sma.model.predictor.Predictor") as _p:
        _p.return_value = MagicMock()
        s = _build_strategy(
            ["AAPL"], use_theses=False, db="x.duckdb", store=MagicMock(),
            settings=settings,
        )
    assert s.k == 15
    assert s.hold_rank == 30
    assert s.sector_neutralize == 1.0


def test_build_strategy_threads_min_score_conviction_floor():
    """The entry conviction floor must reach the strategy from config; a dropped
    value would silently buy the full top-K on weak days."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from sma.live.__main__ import _build_strategy

    settings = SimpleNamespace(
        live=SimpleNamespace(
            strategy=SimpleNamespace(
                k=15, hold_rank=30, sector_neutralize=1.0, min_score=0.01
            )
        )
    )
    with patch("sma.model.predictor.Predictor") as _p:
        _p.return_value = MagicMock()
        s = _build_strategy(
            ["AAPL"], use_theses=False, db="x.duckdb", store=MagicMock(),
            settings=settings,
        )
    assert s.min_score == 0.01


def test_build_strategy_min_score_defaults_off():
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from sma.live.__main__ import _build_strategy

    with patch("sma.model.predictor.Predictor") as _p:
        _p.return_value = MagicMock()
        s = _build_strategy(
            ["AAPL"], use_theses=False, db="x.duckdb", store=MagicMock(),
            settings=SimpleNamespace(live=SimpleNamespace()),
        )
    assert s.min_score is None


def test_build_strategy_defaults_without_config_block():
    """Old configs without live.strategy keep prior behavior (k=15, no
    hysteresis, no sector-neutralize)."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from sma.live.__main__ import _build_strategy

    with patch("sma.model.predictor.Predictor") as _p:
        _p.return_value = MagicMock()
        s = _build_strategy(
            ["AAPL"], use_theses=False, db="x.duckdb", store=MagicMock(),
            settings=SimpleNamespace(live=SimpleNamespace()),
        )
    assert s.k == 15
    assert s.hold_rank == 15  # hysteresis off (hold_rank == k)
    assert s.sector_neutralize == 0.0
