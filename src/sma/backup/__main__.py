"""CLI entry point for the backup module. Run as `python -m sma.backup run [...]`."""

from __future__ import annotations

import os
from datetime import date as date_cls
from pathlib import Path

import click
from loguru import logger

from sma.backup.runner import DEFAULT_BACKUP_DIR, DEFAULT_OFFSITE_BACKUP_DIR, run_backup


@click.group()
def cli() -> None:
    pass


@cli.command("run")
@click.option(
    "--asof",
    default=None,
    help="Date for this backup in YYYY-MM-DD format (default: today)",
)
@click.option(
    "--backup-dir",
    default=None,
    help=(
        "Destination directory for backups "
        "(default: SMA_BACKUP_DIR env var, else local ~/sma-backups)"
    ),
)
@click.option(
    "--db",
    default="data/sma.duckdb",
    help="Path to source DuckDB file (default: data/sma.duckdb)",
)
@click.option(
    "--offsite-dir",
    default=None,
    help=(
        "Destination directory for the offsite copy (default: "
        "SMA_OFFSITE_BACKUP_DIR env var, else the iCloud sma-backups folder). "
        "2026-08-24: this had NO override before — any ad-hoc/test run of "
        "this command (e.g. against a worktree's dev DB) silently landed its "
        "offsite copy in the real folder under the real filename. Always "
        "pass this explicitly for anything other than the real scheduled job."
    ),
)
@click.option(
    "--models-dir",
    default="models_artifacts",
    help="Model artifacts dir to include in the backup set (2026-06-11: the "
         "DB-only set left destroyed artifacts unrecoverable)",
)
@click.option(
    "--retain-days",
    default=30,
    type=int,
    help="Keep daily backups for this many days (default: 30)",
)
@click.option(
    "--retain-months",
    default=12,
    type=int,
    help="Keep monthly snapshots for this many months (default: 12)",
)
def run(
    asof: str | None,
    backup_dir: str | None,
    db: str,
    offsite_dir: str | None,
    retain_days: int,
    retain_months: int,
    models_dir: str,
) -> None:
    """Copy data/sma.duckdb to the backup directory and apply retention policy."""
    asof_date = date_cls.fromisoformat(asof) if asof else date_cls.today()

    resolved_backup_dir = (
        Path(backup_dir).expanduser()
        if backup_dir is not None
        else Path(
            os.environ.get(
                "SMA_BACKUP_DIR",
                str(DEFAULT_BACKUP_DIR),
            )
        ).expanduser()
    )

    resolved_offsite_dir = (
        Path(offsite_dir).expanduser()
        if offsite_dir is not None
        else Path(
            os.environ.get(
                "SMA_OFFSITE_BACKUP_DIR",
                str(DEFAULT_OFFSITE_BACKUP_DIR),
            )
        ).expanduser()
    )

    db_path = Path(db)

    logger.info(
        "backup: starting for asof={} db={} backup_dir={} offsite_dir={}",
        asof_date,
        db_path,
        resolved_backup_dir,
        resolved_offsite_dir,
    )

    payload = run_backup(
        models_dir=Path(models_dir),
        db_path=db_path,
        backup_dir=resolved_backup_dir,
        offsite_dir=resolved_offsite_dir,
        asof=asof_date,
        retain_days=retain_days,
        retain_months=retain_months,
    )

    click.echo(
        f"Backup complete: {payload['backup_path']} "
        f"({payload['backup_size_bytes']:,} bytes) "
        f"verified={payload['verified']} "
        f"dailies={payload['retained_dailies']} "
        f"monthlies={payload['retained_monthlies']}"
    )

    if not payload["verified"]:
        logger.error("backup: verification failed; backup may be corrupt")
        from sma.ingest.notify import notify_failure
        notify_failure(
            title="SMA Backup verification FAILED",
            message=f"{payload['backup_path']} could not be read back — may be corrupt.",
        )
        raise SystemExit(1)


@cli.command("restore")
@click.option("--asof", default=None, help="Backup date to restore (default: latest)")
@click.option("--backup-dir", default=None)
@click.option("--db", default="data/sma.duckdb", help="Restore target DB path")
@click.option("--models-dir", default="models_artifacts", help="Restore target models dir")
@click.option(
    "--yes", is_flag=True, default=False,
    help="Required: confirms overwriting live state",
)
def restore_cmd(asof, backup_dir, db, models_dir, yes) -> None:
    """Restore the DB (+ model artifacts when present) from a backup.

    Verifies the backup is a readable DuckDB BEFORE touching live state;
    replaces atomically under writer_lock; moves the previous models dir
    aside (models_artifacts.pre-restore-<ts>) instead of deleting it.
    """
    import shutil
    from datetime import datetime

    from sma.backup.runner import _backup_dates_in_dir, _verify_backup
    from sma.locks import writer_lock

    bdir = Path(backup_dir) if backup_dir else Path(
        os.environ.get("SMA_BACKUP_DIR", "~/sma-backups")
    ).expanduser()
    dates = _backup_dates_in_dir(bdir)
    if not dates:
        raise click.ClickException(f"no backups found in {bdir}")
    target_date = date_cls.fromisoformat(asof) if asof else dates[-1]
    src = bdir / f"sma-{target_date.isoformat()}.duckdb"
    if not src.exists():
        raise click.ClickException(f"no backup for {target_date} in {bdir}")
    if not _verify_backup(src):
        raise click.ClickException(f"backup {src} does not verify as readable DuckDB")
    if not yes:
        raise click.ClickException(
            f"would restore {src.name} over {db} (and models-{target_date} over "
            f"{models_dir} if present). Re-run with --yes to proceed."
        )

    db_path = Path(db)
    with writer_lock(label="restore"):
        tmp = db_path.with_suffix(".duckdb.restore-tmp")
        shutil.copy2(src, tmp)
        tmp.replace(db_path)
        click.echo(f"restored DB {src.name} -> {db_path}")

        models_src = bdir / f"models-{target_date.isoformat()}"
        if models_src.is_dir():
            mdir = Path(models_dir)
            if mdir.is_dir():
                aside = mdir.with_name(
                    f"{mdir.name}.pre-restore-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                )
                mdir.rename(aside)
                click.echo(f"previous models dir moved to {aside}")
            shutil.copytree(models_src, mdir)
            n = sum(1 for f in mdir.rglob("*") if f.is_file())
            click.echo(f"restored {n} model files -> {mdir}")
        else:
            click.echo(f"no models-{target_date} in backup set (DB-only restore)")


# Must stay at the very END of the module: `python -m sma.backup <cmd>` runs the
# file top-to-bottom, so this guard MUST come AFTER every @cli.command is
# registered — otherwise cli() dispatches before the later commands exist and
# they are unreachable when executed as a module (2026-07-04: `restore` was dead).
if __name__ == "__main__":
    cli()
