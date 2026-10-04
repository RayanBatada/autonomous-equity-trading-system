# Linux scheduling layer (systemd)

The Linux twin of `ops/launchd/`, for the Mac -> always-on-host migration
(`~/StockMarket/host-migration-runbook.md`). Same manifest
(`src/sma/schedule.py`), same "renderer, not hand-written units" approach.
The Python core (`src/sma/**`) does not depend on systemd any more than it
depends on launchd.

## Render

    python -m ops.systemd.render_units --out-dir ops/systemd/

Writes 29 unit files for the 15 jobs: 13 `SCHEDULE` jobs each get a
`sma-<short>.service` + `sma-<short>.timer` pair (26 files), `sma-watchdog`
gets a pair (2 files), and `sma-dashboard` gets a `.service` only (1 file, no
timer — it's a daemon, not a scheduled job).

## Timezone: read this before provisioning

Every job fires on ET wall-clock time. systemd's `OnCalendar=` (like
launchd's `StartCalendarInterval`) evaluates against the **host's system
local time**, not any `TZ=` set inside a unit. Modern systemd does support an
explicit per-line timezone suffix (`OnCalendar=Mon..Fri 18:30:00
America/New_York`, documented in `systemd.time(7)`), but this generator does
**not** use that form — the target host's exact systemd version isn't
pinned yet, and embedding timezones adds a second, unverified mechanism on
top of the one question that already caused a real incident (the 2026-08-12
boot storm, see below).

**Instead: set the host's system timezone at provision time, before
installing anything here.**

    sudo timedatectl set-timezone America/New_York
    timedatectl   # confirm it stuck

This is runbook Section 1c's own resolution, carried over unchanged. Every
generated unit repeats this assumption in its header comment and also sets
`Environment=TZ=America/New_York` in `[Service]` as defense-in-depth (pins
the Python process's own clock, e.g. `datetime.now()`, exactly like
`ops/launchd/render_plists.py`'s identical comment) — that line does **not**
make `OnCalendar=` fire at the right time by itself. If the host timezone is
ever anything other than America/New_York, every job fires at the wrong
wall-clock moment with no error from systemd.

## Install

Two ways to install the rendered units. **Recommended: `systemctl --user`
under a dedicated unprivileged account**, per the runbook's own Section 2b
research — it's the closest match to launchd's per-user agent model (no job
here needs root, including the watchdog's kickstart calls), and the
generated units carry no `User=`/`Group=` directive so they're valid either
way without editing.

### Recommended: `systemctl --user` (dedicated `sma` account)

    sudo useradd -m -s /bin/bash sma
    sudo loginctl enable-linger sma      # REQUIRED — see below
    sudo -u sma -i
    mkdir -p ~/.config/systemd/user ~/logs/sma
    cd ~/code/Stock-Market-Predictor-Agents
    python -m ops.systemd.render_units --out-dir ~/.config/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable --now sma-*.timer sma-dashboard.service

**Do not skip `loginctl enable-linger sma`.** Without it, every
`systemctl --user` timer silently stops firing the moment the `sma` user's
session ends (e.g. the SSH session used to install them logs out) — the
classic "why didn't my systemd user timer fire on a headless server" trap.
Verify it stuck:

    loginctl show-user sma   # expect Linger=yes

### Alternative: system units

If you'd rather install as root-owned system units (e.g. no interactive
login for `sma` at all), render straight into `/etc/systemd/system/` and add
`User=sma` / `Group=sma` to each `.service`'s `[Service]` block so jobs still
run unprivileged — the generator deliberately leaves those directives out
(the rendered file is otherwise identical either way, so pick your install
mode without re-rendering):

    sudo python -m ops.systemd.render_units --out-dir /etc/systemd/system/
    for f in /etc/systemd/system/sma-*.service; do
      sudo sed -i '/^\[Service\]/a User=sma\nGroup=sma' "$f"
    done
    sudo systemctl daemon-reload
    sudo systemctl enable --now sma-*.timer sma-dashboard.service

No linger step needed here — system units aren't tied to a login session.

## Verify

    systemctl --user list-timers          # (or `sudo systemctl list-timers` for system units)

Confirm all 14 timers (13 `SCHEDULE` jobs + watchdog) show sane `NEXT`/`LEFT`
columns in ET, and `sma-dashboard.service` shows `active (running)`:

    systemctl --user status sma-dashboard.service

Kick a job manually to confirm the unit itself works before trusting the
timer:

    systemctl --user start sma-ingest.service
    journalctl --user -u sma-ingest.service -n 50   # or the append: log files in ~/logs/sma/

## Drift check

    python -m ops.systemd.render_units --check --installed-dir ~/.config/systemd/user

Renders to a tempdir and diffs against whatever's installed at
`--installed-dir` (default `~/.config/systemd/user`; pass
`/etc/systemd/system` if you installed system units instead). Exits 1 on any
difference — run this after every `schedule.py` change, same discipline as
`ops/launchd/install.sh --check`.

## Persistent=true and the 2026-08-12 boot storm

Every generated `.timer` sets `Persistent=true`, systemd's equivalent of
launchd's missed-`StartCalendarInterval`-job catch-up: if the host is off or
asleep when a job's scheduled time passes, it fires once at next boot instead
of being silently skipped. This is not a new risk introduced by this
generator — it's parity with what launchd already ran in production,
including a real incident: on 2026-08-12 the Mac was down most of the day,
rebooted at 20:11 ET, and every missed job fired at boot inside 13 minutes
(ingest through stop-loss all completed by 20:24). That event, and the
"boot storm" ordering/after-hours gaps it exposed, is exactly why this
codebase's application-layer guards exist — dead-zone checks in
`sma/live/__main__.py`, the ingest-sentinel gate in `sma/agents/__main__.py`,
`writer_lock`/`heavy_job_lock` serialization (`src/sma/locks.py`), and
sentinel idempotency (`src/sma/sentinels.py`). Those guards are what make
`Persistent=true` safe here; they were hardened against exactly this
scenario, not a hypothetical one, and they carry over to Linux unchanged
(pure POSIX / DuckDB, no OS coupling per the migration runbook's Section 1e).
Do not turn `Persistent=true` off in an attempt to "fix" a future boot
storm — same lesson as `test_only_watchdog_and_dashboard_run_at_load` on the
launchd side: the fix lives in the job code, not the scheduler config.

## Known remaining item: the watchdog's launchctl calls

**Out of scope for this generator.** `src/sma/watchdog.py` reads
`sma.schedule.SCHEDULE`, checks sentinels, and does deadline math — all pure
Python, no OS coupling. Two spots shell out to launchd specifically and still
need a systemd port before the watchdog can self-heal on Linux:

- `src/sma/watchdog.py:79-93` (`_launchctl_state`) — shells out to
  `launchctl print gui/<uid>/<label>` and parses its `state =` line. Linux
  equivalent: `systemctl --user show sma-<short>.service
  --property=ActiveState --value` (values `active`/`activating`/`inactive`/
  `failed`, not launchd's `running`/`waiting`/`not running` vocabulary —
  `watchdog.py:212`'s skip-check needs its accepted-states set adjusted).
- `src/sma/watchdog.py:211-228` (the kickstart call) — shells out to
  `launchctl kickstart -p gui/<uid>/<label>`. Linux equivalent:
  `systemctl --user start sma-<short>.service`.

Two more spots are cosmetic only (hardcoded `launchctl kickstart` text in
human-facing alerts, not the watchdog's own logic):
`src/sma/live/__main__.py`'s `_rekick_hint()` and
`src/sma/monitoring/__init__.py:74-76`. Wrong instructions in a page are
annoying, not silently dangerous, so they're lower priority than the
watchdog port itself.

This unit generator makes the watchdog port mechanically checkable (unit
names are `sma-<short>.service`, matching `_short_name()` from
`ops/launchd/render_plists.py`, which `ops/systemd/render_units.py` imports
directly so the two can never name a job differently) but does not do the
port. Do it, with tests, before migration day — see
`~/StockMarket/host-migration-runbook.md` Section 2c for the full writeup.

## No caffeinate

`ops/launchd/render_plists.py` bakes `/usr/bin/caffeinate -s` (or `-i` for
the dashboard) into every `ProgramArguments` list, to keep the Mac awake
through a job. These units strip that wrapper entirely (see
`_strip_caffeinate` in `render_units.py`) — there is nothing to keep awake
on a host that never sleeps, and no `pmset repeat wakeorpoweron` step to
mirror either (`ops/launchd/install.sh`'s pmset call has no systemd
equivalent, deliberately — there's no "wake" concept to schedule on a VM
that's always on).

## Intraday sessions and intraday bars (ship disabled)

`render_units.py` also renders `sma-ingest.intraday`, `sma-live.session.midday`
and `sma-live.session.close` (.service + .timer) from
`sma.schedule.OPTIONAL_SCHEDULE`. Do not enable their timers as part of the
migration. To turn one on later (the session also needs
`live.sessions.<name>.enabled: true` and a target file, see
`ops/launchd/README.md`):

    systemctl --user enable --now sma-live.session.midday.timer

The watchdog evaluates an optional job only when `systemctl --user is-enabled`
reports its timer `enabled`.
