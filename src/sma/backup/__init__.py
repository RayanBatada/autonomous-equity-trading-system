"""DuckDB backup module.

Daily copy of data/sma.duckdb + models_artifacts/ to a LOCAL folder
(~/sma-backups by default — iCloud sync-during-copy corrupts backups;
override via SMA_BACKUP_DIR for an off-machine target). Retention: keep last retain_days
daily snapshots; keep last retain_months monthly snapshots (last-day-of-month
files); delete everything else.
"""

from sma.backup.runner import run_backup

__all__ = ["run_backup"]
