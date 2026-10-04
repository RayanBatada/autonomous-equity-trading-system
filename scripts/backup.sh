#!/usr/bin/env bash
# Superseded by sma.backup.runner on 2026-06-04 — do NOT run this script.
#
# This script used to copy the production DuckDB directly into iCloud Drive.
# That is the exact corruption path that motivated the switch: iCloud's sync
# daemon can observe (and corrupt) a backup file mid-write when it lands
# straight in a synced folder. sma.backup.runner writes locally, verifies the
# backup is a readable DuckDB, THEN (as of 2026-07-30) copies the finished,
# verified file into the offsite iCloud dir via a temp name + atomic rename
# so the sync daemon only ever sees a complete file.
#
# Kept as a signpost only.
set -euo pipefail

echo "superseded by \`python -m sma.backup run\` (see src/sma/backup/runner.py); " \
     "direct-to-iCloud backups were abandoned 2026-06-04 after sync corruption" >&2
exit 1
