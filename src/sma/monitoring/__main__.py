"""CLI: python -m sma.monitoring check

Runs at 22:30 ET weekdays via launchd (com.sma.monitoring.daily). Reads
critical-job sentinels for today and fires a macOS notification per
missing one. Skips on non-trading days (US holidays).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import click

from sma.config import load_settings
from sma.live.__main__ import _build_alpaca
from sma.monitoring import check_critical_jobs_fired

ET = ZoneInfo("America/New_York")
DEFAULT_CONFIG = "config.yaml"


@click.group()
def cli() -> None:
    """SMA post-pipeline monitoring."""


@cli.command()
@click.option("--config", default=DEFAULT_CONFIG, type=click.Path(exists=True))
def check(config: str) -> None:
    """Verify critical jobs fired today; notify on misses."""
    asof = datetime.now(ET).date()
    settings = load_settings(config_path=Path(config))
    alpaca = _build_alpaca(settings)
    missing = check_critical_jobs_fired(asof=asof, alpaca=alpaca)
    if missing:
        click.echo(f"missing: {', '.join(missing)}")
    else:
        click.echo("ok")


if __name__ == "__main__":
    cli()
