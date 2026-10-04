"""Gates between this bot and real money. Every test here is a "must refuse"."""

import contextlib
from unittest.mock import MagicMock, patch

import pytest

from sma.live.real_money import (
    RealMoneyGate,
    RealMoneyRefusedError,
    build_alpaca_client,
    build_gate,
    check_equity_ceiling,
    force_dry_run,
    preflight_real_money,
)


def _settings(**real_money):
    s = MagicMock()
    s.secrets.alpaca_api_key = "k"
    s.secrets.alpaca_api_secret = "s"
    defaults = dict(enabled=False, real_money_ack=False, max_real_equity=1000.0, dry_run=False)
    defaults.update(real_money)
    for k, v in defaults.items():
        setattr(s.live.real_money, k, v)
    return s


def test_default_gate_is_not_armed():
    assert RealMoneyGate().armed is False
    assert build_gate(_settings()).armed is False


def test_enabled_alone_does_not_arm():
    assert RealMoneyGate(enabled=True).armed is False
    assert "real_money_ack" in RealMoneyGate(enabled=True).refusal_reason()


def test_ack_alone_does_not_arm():
    assert RealMoneyGate(real_money_ack=True).armed is False


def test_both_gates_arm():
    assert RealMoneyGate(enabled=True, real_money_ack=True).armed is True


def test_missing_real_money_block_falls_back_to_defaults():
    s = MagicMock()
    s.live.real_money = None
    assert build_gate(s).armed is False


def test_equity_ceiling_refuses_above_the_cap():
    gate = RealMoneyGate(enabled=True, real_money_ack=True, max_real_equity=100.0)
    with pytest.raises(RealMoneyRefusedError, match="exceeds"):
        check_equity_ceiling(equity=101.0, gate=gate)
    check_equity_ceiling(equity=100.0, gate=gate)      # at the cap is fine


def test_equity_ceiling_is_inert_on_paper():
    check_equity_ceiling(equity=10_000_000.0, gate=RealMoneyGate())


def test_unarmed_config_builds_a_paper_client():
    with patch("sma.live.real_money.AlpacaClient") as cls:
        cls.paper_from_env.return_value = MagicMock(paper=True)
        build_alpaca_client(_settings())
    cls.paper_from_env.assert_called_once()
    cls.live_from_env.assert_not_called()


def test_armed_config_builds_a_live_client():
    with patch("sma.live.real_money.AlpacaClient") as cls:
        cls.live_from_env.return_value = MagicMock(paper=False)
        build_alpaca_client(_settings(enabled=True, real_money_ack=True))
    cls.live_from_env.assert_called_once()
    cls.paper_from_env.assert_not_called()


def test_missing_credentials_raise():
    s = _settings()
    s.secrets.alpaca_api_key = ""
    with pytest.raises(ValueError, match="ALPACA_API_KEY"):
        build_alpaca_client(s)


def test_preflight_refuses_armed_gate_holding_a_paper_client():
    """The dangerous inverse: an operator who believes they are live and is not."""
    gate = RealMoneyGate(enabled=True, real_money_ack=True)
    with pytest.raises(RealMoneyRefusedError, match="PAPER endpoint"):
        preflight_real_money(alpaca=MagicMock(paper=True), gate=gate)


def test_preflight_refuses_live_client_with_unarmed_gate():
    with pytest.raises(RealMoneyRefusedError, match="not armed"):
        preflight_real_money(alpaca=MagicMock(paper=False), gate=RealMoneyGate())


def test_preflight_passes_within_the_ceiling():
    gate = RealMoneyGate(enabled=True, real_money_ack=True, max_real_equity=1000.0)
    client = MagicMock(paper=False)
    client.get_account.return_value = {"equity": 50.0}
    assert preflight_real_money(alpaca=client, gate=gate) == 50.0


def test_preflight_refuses_above_the_ceiling():
    gate = RealMoneyGate(enabled=True, real_money_ack=True, max_real_equity=1000.0)
    client = MagicMock(paper=False)
    client.get_account.return_value = {"equity": 118_000.0}
    with pytest.raises(RealMoneyRefusedError, match="max_real_equity"):
        preflight_real_money(alpaca=client, gate=gate)


def test_preflight_is_a_noop_on_paper():
    assert preflight_real_money(alpaca=MagicMock(paper=True), gate=RealMoneyGate()) == 0.0


def test_config_dry_run_forces_dry_run():
    """live.real_money.dry_run forces dry-run even without --dry-run."""
    assert force_dry_run(False, RealMoneyGate(dry_run=True)) is True


def test_config_dry_run_never_clears_an_explicit_flag():
    """It can only ADD safety — --dry-run always wins."""
    assert force_dry_run(True, RealMoneyGate(dry_run=False)) is True
    assert force_dry_run(True, RealMoneyGate(dry_run=True)) is True


def test_dry_run_default_is_off():
    assert force_dry_run(False, RealMoneyGate()) is False


def test_decide_applies_the_forced_dry_run(monkeypatch):
    """End-to-end through _decide_impl: config dry_run must reach decide_once."""
    from sma.live import __main__ as m

    seen = {}
    settings = _settings(dry_run=True)
    monkeypatch.setattr(m, "load_settings", lambda **kw: settings)
    monkeypatch.setattr(m, "load_universe", lambda p: ["AAPL"])
    monkeypatch.setattr(m, "_build_alpaca", lambda s: MagicMock(paper=True))
    monkeypatch.setattr(m, "run_preflight", lambda **kw: None)

    class _Lock:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(m, "writer_lock", lambda **kw: _Lock())
    monkeypatch.setattr(m, "Store", lambda **kw: MagicMock(**{"connect.return_value": MagicMock()}))
    monkeypatch.setattr(m, "_build_strategy", lambda *a, **k: MagicMock())

    def fake_decide_once(**kw):
        seen["dry_run"] = kw["dry_run"]
        return MagicMock(failed=0, submitted=0, skipped=0, dry_run=True,
                         decisions_after_rails=0)

    monkeypatch.setattr(m, "decide_once", fake_decide_once)
    # Downstream bookkeeping (sentinels, notify) is not what this asserts.
    with contextlib.suppress(Exception):
        m._decide_impl(None, False, None, True, ":memory:", "config.yaml", "u.yaml")
    assert seen.get("dry_run") is True


def test_non_bool_config_values_never_arm_real_money():
    """`is True`, not truthiness. A hand-edited YAML string, a stray object or a
    test double must never route an order to the live endpoint."""
    for bogus in ("true", "false", 1, [1], object(), MagicMock()):
        g = RealMoneyGate(enabled=bogus, real_money_ack=bogus)
        assert g.armed is False, f"{bogus!r} armed real money"


def test_build_gate_coerces_bogus_values_to_safe_defaults():
    s = MagicMock()          # every attribute is a truthy MagicMock
    gate = build_gate(s)
    assert gate.enabled is False
    assert gate.real_money_ack is False
    assert gate.armed is False
    assert gate.max_real_equity == 1000.0
    assert gate.dry_run is False


def test_build_gate_reads_real_values():
    s = _settings(enabled=True, real_money_ack=True, max_real_equity=60.0, dry_run=True)
    gate = build_gate(s)
    assert gate.armed is True
    assert gate.max_real_equity == 60.0
    assert gate.dry_run is True
