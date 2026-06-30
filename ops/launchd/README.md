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
