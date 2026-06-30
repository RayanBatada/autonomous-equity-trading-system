#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$REPO/.venv"
HOME_DIR="$HOME"
MODE="${1:-install}"

if [ ! -d "$VENV" ]; then
    echo "Virtualenv not found at $VENV. Run 'uv venv && uv pip install -e .[dev]' first." >&2
    exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs/sma"

case "$MODE" in
    --check)
        "$VENV/bin/python" -m ops.launchd.render_plists --check
        exit $?
        ;;
    --render-only)
        "$VENV/bin/python" -m ops.launchd.render_plists --out-dir "$REPO/ops/launchd"
        echo "Rendered plists to $REPO/ops/launchd/"
        exit 0
        ;;
esac

# Default: render + install
"$VENV/bin/python" -m ops.launchd.render_plists --out-dir "$REPO/ops/launchd"
echo "Rendered plists to $REPO/ops/launchd/"

JOBS=(
    "com.sma.ingest.daily"
    "com.sma.model.retrain.weekly"
    "com.sma.model.predict.daily"
    "com.sma.agents.daily"
    "com.sma.live.decide.daily"
    "com.sma.live.stop-loss.weekday"
    "com.sma.live.reconcile.daily"
    "com.sma.backup.daily"
    "com.sma.autoresearch.nightly"
    "com.sma.monitoring.daily"
    "com.sma.senate-ingest.weekly"
    "com.sma.house-ingest.weekly"
    "com.sma.watchdog"
    "com.sma.dashboard"
)
for label in "${JOBS[@]}"; do
    src="$REPO/ops/launchd/$label.plist"
    dest="$HOME/Library/LaunchAgents/$label.plist"
    [ -f "$src" ] || { echo "Plist missing: $src" >&2; exit 1; }
    cp "$src" "$dest"
    launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$dest"
    echo "Installed $label"
done

# pmset wake schedule (sudo prompt)
echo
echo "Configuring pmset wake schedule (sudo required):"
# pmset repeat is SINGLE-SLOT: a second call REPLACES the first (the old
# two-line version left only the stale Saturday 01:45 wake — the evening
# pipeline wake was silently gone, Codex review 2026-06-11). One daily wake
# covers the money path (ingest 18:30 → decide 20:00). Off-evening jobs
# (Mon 04:00 retrain, Mon 07:00 autoresearch, Sun politician ingests, 22:00
# backup) rely on launchd's missed-job coalescing at next wake + the
# watchdog's alerts — observed working (6/08 retrain coalesced to 16:05).
sudo pmset repeat wakeorpoweron MTWRFSU 18:10:00
echo "  pmset configured: daily wake 18:10 (evening pipeline; other jobs coalesce on wake)"

echo
echo "Done. Schedule (from src/sma/schedule.py manifest):"
"$VENV/bin/python" -c "from sma.schedule import SCHEDULE; [print(f'  {j.label:40} {j.fire_time_et} ET, days={[d.name for d in j.days]}') for j in SCHEDULE]"
echo "  com.sma.watchdog                          hourly 19-22 ET checkpoints + at login"
echo "  com.sma.dashboard                         KeepAlive=true, http://localhost:8765"
echo
echo "AC required: see ops/launchd/README.md"
