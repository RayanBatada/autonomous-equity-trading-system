"""Configuration loading for the SMA ingest pipeline.

Reads non-secret defaults from a YAML file and secrets from environment
variables. Returns a single immutable Settings object that the rest of the
application receives via dependency injection.
"""

from datetime import time
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from sma.features.parallel import default_feature_workers


class FinnhubLimit(BaseModel):
    requests_per_minute: int


class NewsAPILimit(BaseModel):
    requests_per_day: int


class EdgarLimit(BaseModel):
    requests_per_second: int


class RateLimits(BaseModel):
    finnhub: FinnhubLimit
    newsapi: NewsAPILimit
    edgar: EdgarLimit


class Retries(BaseModel):
    max: int
    base_delay: float
    jitter: float


class CircuitBreaker(BaseModel):
    failures_to_open: int
    cooldown_minutes: int


class IngestConfig(BaseModel):
    default_lookback_days: int
    rate_limits: RateLimits
    retries: Retries
    circuit_breaker: CircuitBreaker
    # Cumulative seconds ONE source run may spend sleeping between per-ticker
    # retries. Ingest holds the global writer lock, so an unbounded 429 storm
    # over a 267-name universe could add hours of lock hold and starve
    # predict/decide (the 8/4 no-trade mechanism). See
    # sma.ingest.sources._finnhub_retry.RetrySleepBudget.
    retry_sleep_budget_s: float = 180.0
    # no_split_inconsistency quality check (2026-10-01): flag a day where the
    # yfinance and alpaca close-to-close returns differ by more than this.
    split_inconsistency_threshold: float = Field(default=0.20, gt=0.0, le=1.0)


class Secrets(BaseSettings):
    """Loaded from environment + .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    finnhub_api_key: str = Field(alias="FINNHUB_API_KEY")
    newsapi_key: str = Field(alias="NEWSAPI_KEY")
    alpaca_api_key: str = Field(alias="ALPACA_API_KEY")
    alpaca_api_secret: str = Field(alias="ALPACA_API_SECRET")
    alpaca_base_url: str = Field(alias="ALPACA_BASE_URL")
    edgar_user_agent: str = Field(alias="EDGAR_USER_AGENT")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")


class AgentTriggers(BaseModel):
    price_move_pct: float = 0.05
    material_filing_types: list[str] = ["8-K", "10-K", "10-Q"]


class AgentsConfig(BaseModel):
    daily_budget_usd: float = 0.50
    warn_threshold_pct: int = 80
    triggers: AgentTriggers = AgentTriggers()
    haiku_model_id: str = "claude-haiku-4-5-20251001"


class ModelConfig(BaseModel):
    """Model-training knobs, read by the weekly retrain CLI.

    Deliberately does NOT import sma.model.ensemble for its default: sma.config
    is imported by ingest, live, and the agents, and that import would drag
    xgboost into all of them. The number is duplicated here and pinned to
    ensemble.DEFAULT_ENSEMBLE_SEEDS by a test instead. sma.features.parallel is
    a different matter — it is pure stdlib, so importing it here costs nothing.
    """

    # Seed ensemble size: N models identical except random_state, scored as the
    # mean of their predictions. 10 is the ensemble-rank study's adopted arm
    # (B_ens, ADOPT-CANDIDATE-VARIANCE, 2026-08-21) — adopted to cut the
    # seed lottery (variance of mean IC down to 0.12-0.17x single-seed), NOT
    # for a mean-IC gain, which it did not demonstrate. 1 restores the exact
    # single-model behaviour that trained every artifact before 2026-08-24.
    ensemble_seeds: int = Field(default=10, ge=1, le=50)
    # Processes the per-asof feature build fans out over. The feature build is
    # the cost centre of the project — 65-80 min of every weekly retrain and
    # the dominant cost of every walk-forward study — and the asof axis is
    # embarrassingly parallel, so this is the one knob that buys wall clock
    # back. Output is BIT-IDENTICAL at any value (proven against the serial
    # path on a real prod-DB slice); 1 is the exact serial path and builds no
    # pool at all. The default is machine-dependent by design:
    # min(4, cpu_count - 1), which leaves a core for the dashboard and the
    # evening trading jobs and caps worker RSS (each worker holds its own copy
    # of the price and news frames).
    feature_workers: int = Field(
        default_factory=default_feature_workers, ge=1, le=32,
    )


class LiveRails(BaseModel):
    """Phase 5 risk-rail config. stop_loss_pct=0 by default per rail diagnostic.

    The _pct fields are FRACTIONS in [0, 1]; bounds reject a typo like
    stop_loss_pct=8.0 (800%) that would silently disable the rail.
    """
    stop_loss_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    # 2026-07-01: smarter price exits (default OFF pending the both-windows
    # backtest gate). trailing_stop exits when price drops N below the
    # post-entry peak; take_profit exits when price rises N above entry.
    trailing_stop_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    take_profit_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    cash_floor_pct: float = Field(default=0.05, ge=0.0, le=1.0)
    # 2026-07-01: live-only haircut on ESTIMATED same-day sell proceeds when
    # sizing rotation buys (see orders.translate + risk.rails). 1.0 = legacy.
    sell_proceeds_haircut: float = Field(default=1.0, gt=0.0, le=1.0)
    max_sector_pct: float = Field(default=0.25, ge=0.0, le=1.0)
    max_drawdown_pct: float = Field(default=0.15, ge=0.0, le=1.0)
    max_position_pct: float = Field(default=0.05, ge=0.0, le=1.0)
    # 2026-06-16: drawdown-scaled de-risk. An initial validation run showed
    # slope 1.5 improving val sharpe/return/maxDD, but that result was later
    # REVERSED: it was a simulator-bug artifact (the backtest gated buys
    # BEFORE same-batch force-sell proceeds while live adds them first).
    # After fixing that parity bug, slope 1.5 HURT (sharpe -0.11 vs +0.26
    # off) — see key-learnings.md. Do not enable on the strength of the
    # debunked number; needs a better regime signal + fresh multi-window
    # validation. slope=0 disables.
    drawdown_derisk_start: float = Field(default=0.05, ge=0.0, le=1.0)
    drawdown_derisk_slope: float = Field(default=0.0, ge=0.0, le=20.0)
    drawdown_derisk_cap: float = Field(default=0.60, ge=0.0, le=1.0)
    # 2026-05-12: block full SELLs of positions held <N calendar days. The
    # 5/5 run did 13 round-trip wash trades (BUY at open + SELL same evening
    # → -$197 slippage). Default 1 prevents same-day reversals while still
    # allowing the model to act on overnight info. Applies to both
    # force-sells AND in-decision liquidations (target_weight=0); partial
    # reductions still fire. Set to 0 to disable.
    min_hold_days: int = 1
    # 2026-05-14: skip partial rebalances smaller than this fraction of the
    # current position. Defends against equity-shrink-driven trims: when
    # OTHER positions fell, account_equity drops, target_value drops, every
    # position gets trimmed even though nothing about THAT name changed —
    # paying spread/slippage to flatten into a drawdown. Default 0.10 means
    # a 5% equity-drift drift won't trigger any rebalance (since 5% < 10%
    # of any single position). New entries (held=0) and full exits
    # (force-sell or target_weight=0) are NEVER blocked; only partials.
    rebalance_dead_zone_pct: float = Field(default=0.10, ge=0.0, le=1.0)


class LiveDecide(BaseModel):
    """Pre-flight ingest-completion polling config."""
    ingest_completion_max_retries: int = 20
    ingest_completion_retry_seconds: int = 60


class LiveDrift(BaseModel):
    """Reconcile drift-detection thresholds (spec §7 Layer 3). Fractions [0, 1]."""
    buy_miss_alert_threshold_pct: float = Field(default=0.20, ge=0.0, le=1.0)
    partial_fill_alert_threshold_pct: float = Field(default=0.90, ge=0.0, le=1.0)
    catastrophic_loss_alert_pct: float = Field(default=0.10, ge=0.0, le=1.0)
    catastrophic_loss_abort_pct: float = Field(default=0.30, ge=0.0, le=1.0)
    # 2026-07-30: catastrophe insurance recommended by the money-path review.
    # catastrophic_loss_abort_pct above only compares today vs YESTERDAY's
    # snapshot, so it is unreachable by a slow bleed spread across many
    # sub-threshold days (worst real single day so far: -5.1%) even though
    # peak-to-trough drawdown has already hit -17.6% without tripping it.
    # This second, independent threshold compares today's equity against the
    # running (garbage-filtered) equity PEAK instead of yesterday's snapshot —
    # see decide_once's second abort check. Explicitly approved new field;
    # this does NOT relax the do-not-touch rule on the existing values above,
    # which stay pinned to their validated numbers.
    catastrophic_peak_drawdown_abort_pct: float = Field(default=0.25, ge=0.0, le=1.0)
    # 2026-09-04: trade-push accuracy check (post-mortem: a decide push
    # overstated an order's size 5x -- the LLY sell rendered "5.7% of equity"
    # for a trade that actually notional'd ~1.0% -- caught only by a human
    # eyeballing the push against Alpaca fills). RELATIVE tolerance (not a
    # flat fraction like the thresholds above): reconcile.
    # _detect_trade_push_drift compares |realized_pct - pushed_pct| /
    # |pushed_pct| against this bound. 0.20 is generous for ordinary
    # overnight price moves between the decide-time mark and the next
    # session's fill, while still catching a wrong-by-multiples bug like
    # the LLY one.
    trade_push_pct_tolerance_pct: float = Field(default=0.20, ge=0.0, le=1.0)


class LivePreOpenGuard(BaseModel):
    """09:25 ET pre-open broker-vs-ledger divergence halt (incident 2026-07-07:
    Alpaca wiped all paper positions overnight; the open then traded on the empty
    book). Halt the day if >= `min_missing_fraction` of the ledger's long
    positions are absent from the live broker book (only when the ledger holds at
    least `min_ledger_positions`, so a near-empty book never false-halts)."""
    enabled: bool = True
    min_missing_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    min_ledger_positions: int = Field(default=2, ge=1)
    # A book-wide wipe HALT requires equity to also collapse by >= this fraction
    # vs the last snapshot (a stale ledger shows "missing" positions but intact
    # equity → no false halt). The per-ticker oversell gate needs no such gate.
    equity_crash_fraction: float = Field(default=0.3, ge=0.0, le=1.0)


class LiveStrategy(BaseModel):
    """xgb_top_k parameters for LIVE decide (2026-06-12 corrected-eval ship:
    sector-relative scoring + rank hysteresis took val sharpe -1.58 -> +1.60
    combined with rails.min_hold_days=7; see ab_wave3/ab_wave4 scripts)."""
    k: int = Field(default=15, ge=1, le=50)
    # None -> hold_rank=k (hysteresis off). 2x k keeps held names while they
    # stay in the top-2k, cutting churn on the 30d signal.
    hold_rank: int | None = Field(default=None, ge=1, le=100)
    # 0 = plain top-K; 1 = rank purely on within-sector relative strength.
    sector_neutralize: float = Field(default=0.0, ge=0.0, le=1.0)
    # Entry conviction floor: only BUY a new name whose RAW model score (the
    # predicted 30d forward return, before tilts/sector-demean) clears this bar,
    # so weak-signal days hold cash instead of filling every slot with mediocre
    # names. None = off (buy the full top-K). Applies to new buys only; held
    # names keep their rank-hysteresis eligibility.
    min_score: float | None = Field(default=None)


class SizingConfig(BaseModel):
    """Capital-scale sizing knobs (2026-08-27 money-path scale audit).

    EVERY default here is the no-op that reproduces the pre-2026-08-27 live
    behaviour bit-for-bit. The audit that motivated them, run against the real
    2026-08-27 decide output on the production DB (fresh book at each equity):

        equity      names bought / decided   gross deployed   worst order / 20d ADV
        $50               0 / 9                    0.0%       n/a  (nothing fillable)
        $500              3 / 9                   20.6%       0.0001%
        $10,000           8 / 9                   61.7%       0.001%
        $25,000           9 / 9                   70.5%       0.003%
        $100,000          9 / 9                   76.6%       0.013%
        $1,000,000        9 / 9                   77.1%       0.135%
        $10,000,000       9 / 9                   77.1%       1.347%

    Target gross was 77.1% at every scale, so the whole shortfall below ~$25k
    is share-rounding: floor(w*E/px) drops to 0 for any name whose price
    exceeds w*E. On 2026-08-27 that killed LLY ($1,176/share) at $10k and
    everything but the three cheapest names at $500.
    """

    # Fractional / notional sizing. False = today's floor()-to-whole-shares.
    # True lets a $50 account hold all k names. Alpaca supports fractional
    # quantities on DAY market orders only (never OPG — retired here anyway).
    fractional_shares: bool = False

    # Decimal places a fractional quantity is TRUNCATED to (never rounded up:
    # rounding up would over-commit cash the cash-floor check already sized).
    # Alpaca accepts up to 9 decimal places on fractional qty.
    fractional_precision: int = Field(default=9, ge=0, le=9)

    # Minimum dollar notional per order. Alpaca's own floor for a fractional
    # order is $1; below that the broker rejects. 0.0 = off (no floor applied,
    # which is the pre-2026-08-27 behaviour). Set to >= 1.0 whenever
    # fractional_shares is on, or a slot smaller than $1 becomes a guaranteed
    # broker rejection every night. A FULL EXIT is never dropped by this rail —
    # a dust position must always be liquidatable.
    min_order_notional: float = Field(default=0.0, ge=0.0)

    # Cap a single order at this fraction of the name's 20-day average daily
    # DOLLAR volume. 0.0 = off (default; today's behaviour). The overflow is
    # not cancelled, it simply is not ordered tonight and the next rebalance
    # re-computes the same delta, so a large book walks into a position over
    # several days instead of printing the whole thing into one open auction.
    #
    # Why the default is OFF rather than 0.02: across all 666 real orders this
    # bot has ever placed, the largest was 0.0197% of the name's ADV — a 2% cap
    # is ~100x above anything that has ever traded, so it would be a no-op
    # today. But it is NOT provably a no-op: NANC (the politician-flow thematic
    # ETF, in the universe since 2026-05-09) trades ~$0.4M/day, and a 10% slot
    # at the current ~$120k equity would be ~3% of its ADV — over a 2% cap. It
    # has never been picked, so enabling this cannot be called bit-identical.
    # Ship the capability, leave it off, flip it deliberately (see below).
    max_participation_of_adv: float = Field(default=0.0, ge=0.0, le=1.0)

    # Trading days of history behind the ADV estimate used by the cap above.
    adv_lookback_days: int = Field(default=20, ge=1, le=250)


class FeeModel(BaseModel):
    """Regulatory sell-side fees. Paper trading charges NONE of these, which is
    why every default is 0.0 — a paper backtest must keep matching the paper
    account. The real published rates live in the config.yaml comment and in
    docs/REAL_MONEY_CHECKLIST.md; set them only when trading real money."""

    # SEC Section 31 fee, sells only, dollars per $1,000,000 of principal.
    # The SEC re-sets this rate annually.
    sec_fee_per_million: float = Field(default=0.0, ge=0.0)
    # FINRA Trading Activity Fee, sells only, dollars per share.
    finra_taf_per_share: float = Field(default=0.0, ge=0.0)
    # Per-trade cap on the TAF (0 = uncapped).
    finra_taf_max_per_trade: float = Field(default=0.0, ge=0.0)


class RealMoneyConfig(BaseModel):
    """LIVE (real-money) routing. Every gate must be crossed deliberately.

    Submitting a real order requires ALL of:
      enabled=True  AND  real_money_ack=True  AND  equity <= max_real_equity
    Any one of them unset routes to the paper endpoint. dry_run=True logs the
    orders it WOULD have sent and submits nothing, at either endpoint.
    """

    enabled: bool = False
    # Explicit human acknowledgement. Deliberately separate from `enabled` so
    # that no single typo, merge or default can arm real-money submission.
    real_money_ack: bool = False
    # Hard ceiling on the equity this bot will trade with real money. A $50
    # experiment can never silently become a $50,000 one: preflight REFUSES to
    # submit when live equity exceeds this.
    max_real_equity: float = Field(default=1000.0, ge=0.0)
    # Log would-be orders, submit nothing. Independent of the gates above.
    dry_run: bool = False
    fees: FeeModel = FeeModel()


# ---- sessions / execution / intraday (2026-09-26, strategy-expansion.md §3) ----
# Every default below is OFF or inert. The 20:00 decide / 09:30 open path does
# not read any of it.


def _parse_hhmm(s: str) -> time:
    h, m = s.strip().split(":")
    return time(int(h), int(m))


class SessionWindow(BaseModel):
    """One intraday trade session. `window` is "HH:MM-HH:MM" ET wall clock."""

    enabled: bool = False
    window: str = "10:30-11:00"

    @field_validator("window")
    @classmethod
    def _window_ordered(cls, v: str) -> str:
        start, end = (_parse_hhmm(x) for x in v.split("-"))
        if not start < end:
            raise ValueError(f"session window {v!r}: start must be before end")
        return v

    def bounds(self) -> tuple[time, time]:
        start, end = (_parse_hhmm(x) for x in self.window.split("-"))
        return start, end


class LiveSessions(BaseModel):
    midday: SessionWindow = SessionWindow(window="10:30-11:00")
    close: SessionWindow = SessionWindow(window="15:40-15:55")


class LiveExecution(BaseModel):
    """Marketable-limit execution for the intraday sessions (never the open)."""

    # Limit priced at ask + offset (buy) / bid - offset (sell).
    limit_offset_bps: float = Field(default=10.0, ge=0.0, le=500.0)
    # Minutes a limit may rest before the sweep cancels it.
    sweep_after_minutes: float = Field(default=5.0, gt=0.0, le=60.0)
    # On the first sweep, re-price the remainder once at a fresh quote.
    reprice_once: bool = True
    # After the last limit attempt: "market" sends the remainder as a DAY
    # market order, "leave" lets it go unfilled.
    fallback: Literal["market", "leave"] = "market"
    # A quote wider than this is not trusted as a limit reference (IEX can be
    # thin); the latest trade is used instead.
    max_quote_spread_bps: float = Field(default=50.0, gt=0.0)
    # Skip deltas smaller than this many dollars.
    min_trade_notional: float = Field(default=50.0, ge=0.0)


_INTRADAY_DEFAULT_TICKERS = [
    "SPY", "XLK", "XLF", "XLV", "XLE", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
]


class IntradayConfig(BaseModel):
    """1-minute IEX bars into prices_intraday (python -m sma.ingest intraday)."""

    tickers: list[str] = Field(default_factory=lambda: list(_INTRADAY_DEFAULT_TICKERS))
    # Also fetch every name the paper_fills ledger currently holds.
    include_held: bool = True

    @field_validator("tickers")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("intraday.tickers must not be empty")
        return v


class LiveConfig(BaseModel):
    alpaca_paper_endpoint: str = "https://paper-api.alpaca.markets"
    alpaca_live_endpoint: str = "https://api.alpaca.markets"
    sizing: SizingConfig = SizingConfig()
    real_money: RealMoneyConfig = RealMoneyConfig()
    rails: LiveRails = LiveRails()
    use_theses: bool = True
    decide: LiveDecide = LiveDecide()
    drift: LiveDrift = LiveDrift()
    strategy: LiveStrategy = LiveStrategy()
    preopen_guard: LivePreOpenGuard = LivePreOpenGuard()
    sessions: LiveSessions = LiveSessions()
    execution: LiveExecution = LiveExecution()


class NotifyConfig(BaseModel):
    """Operator-facing push notifications. Never affects trading behaviour —
    everything here is a side channel that reads decide's already-final
    output after orders are submitted."""

    # 2026-08-31: nightly ntfy push of tonight's SUBMITTED orders (as target
    # weights, not share counts) to Rayan's phone, so he can manually mirror
    # them on his own account at any size. Default TRUE — it's a
    # notification, not a trading change, so it's opt-out rather than
    # opt-in. See sma.live.trade_push and docs/REAL_MONEY_CHECKLIST.md
    # "Mirroring signals manually".
    trade_pushes: bool = True


class SleeveConfig(BaseModel):
    """One sleeve in `strategies.sleeves` (see sma.strategies.allocator).
    `name` must be registered in sma.strategies.registry; that check runs
    when decide builds the sleeves, not here, so loading config never
    imports the model stack."""
    name: str
    capital_fraction: float = Field(ge=0.0, le=1.0)
    mode: Literal["live", "shadow"] = "live"
    enabled: bool = True


def _default_sleeves() -> list[SleeveConfig]:
    return [SleeveConfig(name="xgb_momentum", capital_fraction=1.0, mode="live", enabled=True)]


class StrategiesSettings(BaseModel):
    """Multi-strategy sleeves (2026-09-26). Default is the incumbent alone at
    100% capital, which reproduces the pre-sleeve decide byte for byte
    (tests/integration/live/test_sleeve_golden.py)."""
    sleeves: list[SleeveConfig] = Field(default_factory=_default_sleeves)

    @field_validator("sleeves")
    @classmethod
    def _valid_allocation(cls, v: list[SleeveConfig]) -> list[SleeveConfig]:
        # Single source of the rules (live fractions <= 1.0, unique names,
        # >= 1 live sleeve). Lazy import keeps config light.
        from sma.strategies.allocator import validate_sleeves

        validate_sleeves(v)
        return v


class Settings(BaseModel):
    ingest: IngestConfig
    sources_enabled: list[str]
    secrets: Secrets
    agents: AgentsConfig = AgentsConfig()
    live: LiveConfig = LiveConfig()
    model: ModelConfig = ModelConfig()
    notify: NotifyConfig = NotifyConfig()
    intraday: IntradayConfig = IntradayConfig()
    strategies: StrategiesSettings = StrategiesSettings()


def load_settings(config_path: Path | str = "config.yaml") -> Settings:
    """Load YAML config + environment secrets into a Settings object.

    Raises on missing required secrets so we fail loud at startup rather than
    failing partway through a daily run.
    """
    cfg_path = Path(config_path)
    raw = yaml.safe_load(cfg_path.read_text())
    return Settings(
        ingest=IngestConfig(**raw["ingest"]),
        sources_enabled=raw["sources_enabled"],
        secrets=Secrets(),  # populated from env / .env
        agents=AgentsConfig(**raw.get("agents", {})),
        live=LiveConfig(**raw.get("live", {})),
        model=ModelConfig(**raw.get("model", {})),
        notify=NotifyConfig(**raw.get("notify", {})),
        intraday=IntradayConfig(**(raw.get("intraday") or {})),
        strategies=StrategiesSettings(**(raw.get("strategies") or {})),
    )


def load_model_config(config_path: Path | str = "config.yaml") -> ModelConfig:
    """Read just the `model:` block, tolerantly.

    The model CLI has never called load_settings(), so it has never needed API
    secrets in its environment — the 04:00 retrain runs from launchd with a
    minimal env (PATH and TZ only). Reading one training knob must not change
    that, which is why this exists instead of a load_settings() call.

    Anything unreadable — file absent, YAML malformed, block not a mapping, a
    value out of bounds — falls back to defaults rather than raising: an
    optional knob must never be the reason the weekly retrain dies. The CLI
    logs the value it actually resolved, so a bad config surfaces there.
    """
    try:
        raw = yaml.safe_load(Path(config_path).read_text()) or {}
        block = raw.get("model") or {}
        if not isinstance(block, dict):
            return ModelConfig()
        return ModelConfig(**block)
    except Exception:  # noqa: BLE001 - an optional knob must never abort a retrain
        return ModelConfig()
