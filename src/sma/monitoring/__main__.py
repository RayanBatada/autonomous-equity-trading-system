"""CLI: python -m sma.monitoring check | weekly-digest

`check` runs at 22:30 ET weekdays via launchd (com.sma.monitoring.daily).
Reads critical-job sentinels for today and fires a macOS notification per
missing one. Skips on non-trading days (US holidays).

`weekly-digest` runs Sunday 18:00 ET (com.sma.weekly-digest.weekly): writes
the week-in-review markdown file + ntfy push. See
sma.monitoring.weekly_digest for the read-only DB/sentinel computation.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import click

from sma.config import load_settings
from sma.live.__main__ import _build_alpaca
from sma.monitoring import (
    check_critical_jobs_fired,
    check_model_staleness,
    check_regime_turn,
)
from sma.sentinels import write_sentinel

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
    # Model staleness pages every evening a retrain has been missed — the miss
    # itself gets one watchdog page and then goes silent while the model ages
    # (2026-07-13: served week-old weights unnoticed until manual audit).
    stale = check_model_staleness(asof=asof)
    # Regime-turn: pages only on an actual crossing (trailing-21-decide-date
    # IC flipping into/out of |t|>=2 significance at the 10d horizon) -- see
    # check_regime_turn's docstring. A steady regime is silent every other
    # evening, unlike the two checks above.
    regime_turn = check_regime_turn(asof=asof)
    # 2026-08-29: write a completion sentinel like every other scheduled job.
    # Found while building the weekly digest's ops section (it reads exactly
    # this sentinel): com.sma.monitoring.daily had NEVER written one since
    # this module was created, so it showed up "missing" every single night
    # in the digest's clean/degraded count -- a false read with nothing
    # actually wrong, and it also meant sma.watchdog re-kicked this job past
    # its deadline every evening even on nights it ran fine. Written last
    # (after every check already ran) so a broken check doesn't prevent the
    # sentinel a healthy run deserves.
    write_sentinel(
        label="com.sma.monitoring.daily",
        asof=asof,
        payload={
            "label": "com.sma.monitoring.daily",
            "asof": asof.isoformat(),
            "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "run_id": int(datetime.now(UTC).timestamp()),
            "missing_critical_jobs": missing,
            "model_stale": stale,
            "regime_turn": regime_turn,
        },
    )
    if missing or stale or regime_turn:
        click.echo(
            f"missing: {', '.join(missing) or '-'} | model_stale: {stale} | "
            f"regime_turn: {regime_turn or '-'}"
        )
    else:
        click.echo("ok")


@cli.command("weekly-digest")
@click.option(
    "--week-ending",
    default=None,
    help="Friday ISO date this digest covers (default: most recent Friday on/before today ET).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Write the markdown file but skip the ntfy push and the completion sentinel.",
)
@click.option("--db", "db_path", default="data/sma.duckdb", type=click.Path())
@click.option("--models-dir", default="models_artifacts", type=click.Path())
@click.option("--universe", "universe_path", default="src/sma/universe.yaml", type=click.Path())
@click.option("--output-dir", default=None, type=click.Path())
def weekly_digest_cmd(
    week_ending: str | None,
    dry_run: bool,
    db_path: str,
    models_dir: str,
    universe_path: str,
    output_dir: str | None,
) -> None:
    """Week-in-review digest: P&L, trading, live signal, and ops health."""
    from sma.monitoring.weekly_digest import (
        DEFAULT_OUTPUT_DIR,
        most_recent_friday,
        run_weekly_digest,
    )

    we = (
        date.fromisoformat(week_ending)
        if week_ending
        else most_recent_friday(datetime.now(ET).date())
    )
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    path = run_weekly_digest(
        week_ending=we,
        db_path=Path(db_path),
        models_dir=Path(models_dir),
        universe_path=Path(universe_path),
        output_dir=out_dir,
        dry_run=dry_run,
    )
    suffix = " (dry-run: no ntfy push, no sentinel)" if dry_run else ""
    click.echo(f"weekly digest written: {path}{suffix}")


if __name__ == "__main__":
    cli()
