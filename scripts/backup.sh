#!/usr/bin/env bash
# Backup the production DuckDB to a dated copy on iCloud Drive.
# Run manually for now; can be added to launchd later if desired.
#
# Usage: bash scripts/backup.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/data/sma.duckdb"
DEST_DIR="$HOME/Library/Mobile Documents/com~apple~CloudDocs/sma-backups"
TIMESTAMP="$(date +%F-%H%M)"
DEST="$DEST_DIR/sma-$TIMESTAMP.duckdb"

if [ ! -f "$SRC" ]; then
    echo "No DB at $SRC; nothing to back up." >&2
    exit 1
fi

mkdir -p "$DEST_DIR"
cp "$SRC" "$DEST"

# Keep last 14 backups; delete older.
find "$DEST_DIR" -name "sma-*.duckdb" -type f | sort -r | tail -n +15 | xargs -I{} rm "{}"

SIZE=$(du -h "$DEST" | cut -f1)
COUNT=$(find "$DEST_DIR" -name "sma-*.duckdb" -type f | wc -l | tr -d ' ')
echo "Backed up $SRC -> $DEST ($SIZE)"
echo "Total backups in $DEST_DIR: $COUNT (capped at 14)"
