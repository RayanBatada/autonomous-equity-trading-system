"""CLI: python -m sma.research <subcommand>

Subcommands:
  ticker    — full research report for a single ticker (markdown to stdout)
"""

from __future__ import annotations

from datetime import date as date_cls
from pathlib import Path

import click

from sma.config import load_settings
from sma.ingest.store import Store
from sma.ingest.universe import load_universe
from sma.research.report import build_report
from sma.sectors import sector_for

DEFAULT_DB = "data/sma.duckdb"
DEFAULT_CONFIG = "config.yaml"
DEFAULT_UNIVERSE = "src/sma/universe.yaml"


@click.group()
def cli() -> None:
    """On-demand ticker research."""


def _maybe_alpaca_position(ticker: str, *, config_path: Path) -> dict | None:
    """Best-effort Alpaca position lookup.

    Returns None and prints a warning on any failure — research must work
    offline (e.g. during a network outage or without API keys configured).
    """
    try:
        settings = load_settings(config_path=config_path)
        key = settings.secrets.alpaca_api_key
        secret = settings.secrets.alpaca_secret_key
        if not key or not secret:
            return None
        from sma.live.alpaca_client import AlpacaClient

        client = AlpacaClient.paper_from_env(api_key=key, secret_key=secret)
        positions = client.get_positions()
        return positions.get(ticker.upper())
    except Exception as e:
        click.echo(f"# warning: Alpaca position lookup failed: {e}", err=True)
        return None


def _maybe_refresh_thesis(*, ticker: str, asof: date_cls, config_path: Path, db: str) -> None:
    """Invoke the agents thesis pipeline for this ticker.

    Uses the same machinery as `python -m sma.agents thesis --ticker X --asof Y`,
    minus the writer-lock dance (we already hold it implicitly: this command
    blocks while the LLM round-trips run).
    """
    from sma.agents.__main__ import _build_context, _build_pipeline
    from sma.locks import writer_lock

    settings = load_settings(config_path=config_path)
    with writer_lock(label="research-refresh"):
        store = Store(path=db).connect()
        try:
            pipeline = _build_pipeline(settings, store)
            run_id = store.allocate_run_id()
            ctx = _build_context(store, ticker.upper(), asof)
            click.echo(
                f"# refreshing thesis for {ticker.upper()} (asof={asof.isoformat()}, "
                f"news={len(ctx.news_rows)}, filings={len(ctx.filing_rows)}) ...",
                err=True,
            )
            out = pipeline.run(ctx, run_id=run_id)
            if out is None:
                click.echo(
                    "# warning: thesis pipeline returned None "
                    "(budget exhausted, no cached fallback).",
                    err=True,
                )
        finally:
            store.close()


@cli.command()
@click.argument("ticker")
@click.option("--asof", default=None, help="ISO date (default: today)")
@click.option("--refresh", is_flag=True, default=False,
              help="Run the agents thesis pipeline before reporting (paid LLM call)")
@click.option("--no-alpaca", is_flag=True, default=False,
              help="Skip Alpaca position lookup (offline mode)")
@click.option("--config", default=DEFAULT_CONFIG, type=click.Path(exists=True))
@click.option("--universe", default=DEFAULT_UNIVERSE, type=click.Path(exists=True))
@click.option("--db", default=DEFAULT_DB)
@click.option("--output", type=click.Path(), default=None,
              help="Write markdown to PATH instead of stdout")
def ticker(ticker, asof, refresh, no_alpaca, config, universe, db, output):
    """Render a research report for a single TICKER."""
    asof_d = date_cls.fromisoformat(asof) if asof else date_cls.today()
    config_path = Path(config)

    if refresh:
        _maybe_refresh_thesis(
            ticker=ticker, asof=asof_d, config_path=config_path, db=db
        )

    universe_list = load_universe(Path(universe))
    sector = sector_for(ticker.upper())
    position = None if no_alpaca else _maybe_alpaca_position(
        ticker, config_path=config_path
    )

    store = Store(path=db).connect(read_only=True)
    try:
        report = build_report(
            store=store,
            ticker=ticker,
            asof=asof_d,
            universe=universe_list,
            sector=sector,
            position=position,
        )
    finally:
        store.close()

    text = report.to_markdown()
    if output:
        Path(output).write_text(text, encoding="utf-8")
        click.echo(f"Wrote report to {output} ({len(text)} chars).")
    else:
        click.echo(text)


if __name__ == "__main__":
    cli()
