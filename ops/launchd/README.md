# Mac scheduling layer

This directory contains macOS-specific scheduling artifacts. The Python core
(`src/sma/**`) does not depend on launchd, pmset, or caffeinate.

## AC required

The schedule assumes the laptop is on AC power. On battery, scheduled jobs will
silently fail to fire when the system sleeps. Keep the laptop plugged in during
the paper-trading proof window.

## Install

    bash ops/launchd/install.sh

This renders plists from `src/sma/schedule.py`, installs them into
`~/Library/LaunchAgents/`, configures `pmset repeat wakeorpoweron`, and prints
the schedule.

## Render only (no install)

    bash ops/launchd/install.sh --render-only

## Verify (drift check)

    bash ops/launchd/install.sh --check

Exits 0 if installed plists match the rendered manifest, 1 on drift.

## Linux migration

When moving to an always-on Linux PC, create `ops/systemd/` with `*.service` +
`*.timer` files generated from the same `src/sma/schedule.py` manifest. The
Python core stays unchanged.

## Intraday sessions and intraday bars (ship disabled)

Three jobs are rendered here but are NOT in `install.sh`'s JOBS list, so a
normal install never loads them and nothing new fires:

| label | fires (ET, Mon-Fri) | command |
|---|---|---|
| `com.sma.ingest.intraday` | 15:41 | `python -m sma.ingest intraday` |
| `com.sma.live.session.midday` | 10:35 | `python -m sma.live session --name midday` |
| `com.sma.live.session.close` | 15:45 | `python -m sma.live session --name close` |

They live in `sma.schedule.OPTIONAL_SCHEDULE`. The watchdog only looks at one
once launchd reports it loaded, so an unloaded job never pages.

To enable one (example: midday). A session trades only when ALL of these hold:
the plist is loaded, `live.sessions.midday.enabled: true` in config.yaml, and a
target file `data/state/session_targets-midday-<date>.json` exists. Try it
first with `python -m sma.live session --name midday --dry-run`.

    cp ops/launchd/com.sma.live.session.midday.plist ~/Library/LaunchAgents/
    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.sma.live.session.midday.plist

To disable again:

    launchctl bootout gui/$(id -u)/com.sma.live.session.midday
    rm ~/Library/LaunchAgents/com.sma.live.session.midday.plist

The intraday bars job needs no config flag: load its plist the same way.
