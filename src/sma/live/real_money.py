"""Gates between this bot and real money.

Design rule: the default of every gate is "paper", and reaching the real-money
endpoint requires THREE independent affirmative facts, not one. A single typo, a
bad merge, a copied config or an env var set in the wrong shell must not be able
to route an order to `api.alpaca.markets`.

    live.real_money.enabled          = True    # route to the live endpoint
    live.real_money.real_money_ack   = True    # a human said yes, on purpose
    account equity                  <= live.real_money.max_real_equity

The equity ceiling is the one that matters most. It is not a risk limit on any
single trade — it is a limit on the SIZE OF THE EXPERIMENT. A $50 trial that
silently becomes a $50,000 one because a config was copied to another machine is
the failure this exists to make impossible, and it is checked against the live
broker's own equity read, at the live endpoint, before the first order.

Account facts verified against Alpaca's docs on 2026-08-27 (see
docs/REAL_MONEY_CHECKLIST.md for the citations) that shape what a tiny real
account can safely do:
  * Alpaca opens NO cash accounts — every account is a margin account. Below
    $2,000 equity it is a LIMITED margin account: 1x buying power, no shorting,
    but trading on unsettled funds IS allowed. A good-faith violation is a
    cash-account concept and therefore cannot occur. Alpaca publishes no GFV
    policy at all; the "3 strikes / 90 days" rule people quote is another
    broker's convention.
  * FINRA retired the pattern-day-trader rule outright on 2026-06-04 (Reg
    Notice 26-10): no day-trade counting, no $25,000 minimum. `rails.min_hold_days
    = 7` already made a same-day round trip impossible here, so this bot could
    not have tripped the old rule either way — but the rule is simply gone.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from sma.live.alpaca_client import AlpacaClient
from sma.live.sizing import _flag, _num


class RealMoneyRefusedError(Exception):  # noqa: N818 - "Refused" reads better at the call site
    """Real-money submission was requested but a gate is not satisfied. Always
    fail CLOSED: refuse to trade rather than fall back to paper silently, so an
    operator who believes they are live is never quietly not live (and vice
    versa)."""


@dataclass(frozen=True)
class RealMoneyGate:
    enabled: bool = False
    real_money_ack: bool = False
    max_real_equity: float = 1000.0
    dry_run: bool = False

    @property
    def armed(self) -> bool:
        """True when config asks for real money. Says nothing about equity —
        that needs a broker read (see `check_equity_ceiling`).

        `is True`, not truthiness. Arming real-money submission is the one place
        in this system where a merely-truthy value must not count: the string
        "false" from a hand-edited YAML is truthy, and so is every test double.
        """
        return self.enabled is True and self.real_money_ack is True

    def refusal_reason(self) -> str | None:
        if not self.enabled:
            return "live.real_money.enabled is false"
        if not self.real_money_ack:
            return (
                "live.real_money.real_money_ack is false — set it explicitly to "
                "acknowledge that real orders will be placed with real money"
            )
        return None


def build_gate(settings) -> RealMoneyGate:
    """Read the gates out of config, coercing every field.

    Same reasoning as sizing.build_sizing_policy, with more at stake: a value
    that is not literally a bool must never arm real-money submission, and a
    max_real_equity that is not a real number must fall back to the
    conservative default rather than compare as something surprising.
    """
    cfg = getattr(getattr(settings, "live", None), "real_money", None)
    if cfg is None:
        return RealMoneyGate()
    return RealMoneyGate(
        enabled=_flag(getattr(cfg, "enabled", False), False),
        real_money_ack=_flag(getattr(cfg, "real_money_ack", False), False),
        max_real_equity=_num(getattr(cfg, "max_real_equity", 1000.0), 1000.0),
        dry_run=_flag(getattr(cfg, "dry_run", False), False),
    )


def build_alpaca_client(settings, *, gate: RealMoneyGate | None = None) -> AlpacaClient:
    """The ONLY place a live-endpoint client is constructed.

    Returns a paper client unless the config gates are armed. Never falls back
    silently: an armed-but-unsatisfiable config raises.
    """
    api_key = settings.secrets.alpaca_api_key
    secret_key = settings.secrets.alpaca_api_secret
    if not api_key or not secret_key:
        raise ValueError(
            "ALPACA_API_KEY / ALPACA_API_SECRET must be set in .env or environment."
        )
    gate = gate if gate is not None else build_gate(settings)
    if not gate.armed:
        return AlpacaClient.paper_from_env(api_key=api_key, secret_key=secret_key)
    logger.warning(
        "REAL MONEY: routing to Alpaca's live endpoint (max_real_equity=${:,.2f}, "
        "dry_run={}). Live and paper API keys are NOT interchangeable — if these "
        "credentials are paper keys, authentication will fail here rather than "
        "trade the wrong book.",
        gate.max_real_equity, gate.dry_run,
    )
    return AlpacaClient.live_from_env(api_key=api_key, secret_key=secret_key)


def force_dry_run(cli_dry_run: bool, gate: RealMoneyGate) -> bool:
    """Resolve the effective dry-run flag.

    `live.real_money.dry_run` can only ever ADD safety: it forces dry-run on,
    and never clears an explicit `--dry-run`. This is the knob you leave set for
    the first week of a real-money account — the job authenticates against the
    live endpoint, reads the real account, sizes real orders, logs exactly what
    it would submit, and submits nothing.
    """
    return bool(cli_dry_run) or gate.dry_run is True


def check_equity_ceiling(*, equity: float, gate: RealMoneyGate) -> None:
    """Refuse to submit real orders when the account is larger than the
    experiment was authorised for. No-op on a paper account."""
    if not gate.armed:
        return
    if equity > gate.max_real_equity:
        raise RealMoneyRefusedError(
            f"REAL-MONEY REFUSED: live equity ${equity:,.2f} exceeds "
            f"live.real_money.max_real_equity ${gate.max_real_equity:,.2f}. "
            "Raise the ceiling deliberately if this account is meant to be this "
            "big; do not raise it to get past this message."
        )


def preflight_real_money(*, alpaca: AlpacaClient, gate: RealMoneyGate) -> float:
    """Full real-money preflight. Returns the equity it verified.

    Order matters: the endpoint check comes first, because an armed gate that
    somehow ended up holding a PAPER client is the dangerous inverse mistake —
    an operator who thinks they are trading real money and is not.
    """
    if not gate.armed:
        reason = gate.refusal_reason()
        if alpaca.paper:
            return 0.0
        raise RealMoneyRefusedError(
            f"client is pointed at the LIVE endpoint but the gates are not armed "
            f"({reason}); refusing to trade"
        )
    if alpaca.paper:
        raise RealMoneyRefusedError(
            "real-money gates are armed but the client is pointed at the PAPER "
            "endpoint; refusing to trade rather than trade the wrong book"
        )
    equity = float(alpaca.get_account().get("equity", 0.0))
    check_equity_ceiling(equity=equity, gate=gate)
    logger.warning(
        "REAL MONEY preflight PASSED: live equity ${:,.2f} <= ceiling ${:,.2f}",
        equity, gate.max_real_equity,
    )
    return equity


#: Legacy-friendly alias; the module's own name for the failure is "refused".
RealMoneyRefused = RealMoneyRefusedError
