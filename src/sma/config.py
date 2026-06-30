"""Configuration loading for the SMA ingest pipeline.

Reads non-secret defaults from a YAML file and secrets from environment
variables. Returns a single immutable Settings object that the rest of the
application receives via dependency injection.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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


class LiveRails(BaseModel):
    """Phase 5 risk-rail config. stop_loss_pct=0 by default per rail diagnostic.

    The _pct fields are FRACTIONS in [0, 1]; bounds reject a typo like
    stop_loss_pct=8.0 (800%) that would silently disable the rail.
    """
    stop_loss_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    cash_floor_pct: float = Field(default=0.05, ge=0.0, le=1.0)
    max_sector_pct: float = Field(default=0.25, ge=0.0, le=1.0)
    max_drawdown_pct: float = Field(default=0.15, ge=0.0, le=1.0)
    max_position_pct: float = Field(default=0.05, ge=0.0, le=1.0)
    # 2026-06-16: drawdown-scaled de-risk (validated slope 1.5: better val
    # sharpe/return/maxDD). slope=0 disables.
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


class LiveConfig(BaseModel):
    alpaca_paper_endpoint: str = "https://paper-api.alpaca.markets"
    rails: LiveRails = LiveRails()
    use_theses: bool = True
    decide: LiveDecide = LiveDecide()
    drift: LiveDrift = LiveDrift()
    strategy: LiveStrategy = LiveStrategy()


class Settings(BaseModel):
    ingest: IngestConfig
    sources_enabled: list[str]
    secrets: Secrets
    agents: AgentsConfig = AgentsConfig()
    live: LiveConfig = LiveConfig()


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
    )
