"""Weekly "week in review" digest -- P&L, trading activity, live signal, and
ops health for the past week, so it reaches Rayan without him asking.

Runs Sunday 18:00 ET via launchd/systemd (com.sma.weekly-digest.weekly, see
src/sma/schedule.py). Everything here is READ-ONLY against the DB (account_
snapshots, paper_fills, intended_orders) plus sentinels (data/sentinels/) and
model artifacts (models_artifacts/) -- this module never opens a writable
DuckDB connection and never needs the writer_lock. Its only writes are a
markdown file under ~/StockMarket/weekly/ and (on a real, non-dry-run
completion) its own launchd/systemd completion sentinel.

The "week" has two windows, both anchored on `week_ending` (a Friday):
  - the TRADING week (Monday..Friday, 5 days) for P&L / best-worst-day /
    trading-activity questions -- there is no market activity on a weekend
    to report on.
  - the OPS week (Saturday..Friday, 7 days) for the sentinel-health check --
    this covers the prior weekend's backup + senate/house ingests too, since
    those are real scheduled jobs whose misses matter operationally even
    though they don't touch P&L.

Signal section reuses sma.eval.live_ic (trailing_ic_regime /
model_edge_ic_df) verbatim -- the SAME functions the dashboard's Model tab
and sma.monitoring.check_regime_turn already use. No IC math is duplicated
here.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from loguru import logger

from sma import schedule as sched
from sma.db_connect import read_only_connect
from sma.eval import live_ic
from sma.ingest.notify import send_ntfy
from sma.model.persistence import latest_model_for_date
from sma.sentinels import read_sentinel, write_sentinel

DEFAULT_DB_PATH = Path("data/sma.duckdb")
DEFAULT_MODELS_DIR = Path("models_artifacts")
DEFAULT_UNIVERSE_PATH = Path("src/sma/universe.yaml")
DEFAULT_OUTPUT_DIR = Path.home() / "StockMarket" / "weekly"

JOB_LABEL = "com.sma.weekly-digest.weekly"
IC_HORIZONS_DAYS = (10, 20)
IC_WINDOW = live_ic.IC_REGIME_WINDOW  # 21 -- same trailing window as the dashboard
NTFY_CHAR_LIMIT = 600


def most_recent_friday(today: date) -> date:
    """The Friday on/before `today`. date.weekday(): Mon=0 .. Fri=4 .. Sun=6."""
    offset = (today.weekday() - 4) % 7
    return today - timedelta(days=offset)


def digest_filename(week_ending: date) -> str:
    """`YYYY-WW.md` using the ISO week number of `week_ending`."""
    iso = week_ending.isocalendar()
    return f"{iso.year}-{iso.week:02d}.md"


@dataclass(frozen=True)
class IcRead:
    horizon_days: int
    mean: float | None
    t_stat: float | None
    n: int
    level: str


@dataclass(frozen=True)
class WeeklyDigest:
    week_ending: date  # Friday
    week_starting: date  # Monday of the same trading week
    ops_week_starting: date  # Saturday before week_starting

    # P&L
    friday_equity: float | None
    prior_friday_equity: float | None
    week_pnl_pct: float | None
    start_date: date | None
    start_equity: float | None
    since_start_pct: float | None
    best_day: tuple[date, float] | None
    worst_day: tuple[date, float] | None

    # Trading
    orders_submitted: int
    orders_failed: int
    fills_filled: int
    names_entered: tuple[str, ...]
    names_exited: tuple[str, ...]
    position_count: int | None
    cash: float | None

    # Signal
    ic_reads: tuple[IcRead, ...]

    # Ops
    clean_nights: int
    total_nights: int
    degraded_dates: tuple[date, ...]
    problems_by_date: dict[date, tuple[str, ...]]
    model_id: str | None


# ---------------------------------------------------------------------------
# P&L
# ---------------------------------------------------------------------------


def _compute_pnl(
    db_path: Path, *, week_ending: date, prior_friday: date, week_starting: date
) -> dict:
    empty = dict(
        friday_equity=None,
        prior_friday_equity=None,
        week_pnl_pct=None,
        start_date=None,
        start_equity=None,
        since_start_pct=None,
        best_day=None,
        worst_day=None,
    )
    if not Path(db_path).exists():
        return empty

    try:
        con = read_only_connect(db_path)
        try:
            row = con.execute(
                "SELECT equity FROM account_snapshots WHERE asof_date = ?", [week_ending]
            ).fetchone()
            friday_equity = float(row[0]) if row else None

            row = con.execute(
                "SELECT equity FROM account_snapshots WHERE asof_date = ?", [prior_friday]
            ).fetchone()
            prior_equity = float(row[0]) if row else None

            row = con.execute(
                "SELECT asof_date, equity FROM account_snapshots ORDER BY asof_date ASC LIMIT 1"
            ).fetchone()
            start_date, start_equity = (row[0], float(row[1])) if row else (None, None)

            rows = con.execute(
                "SELECT asof_date, equity FROM account_snapshots "
                "WHERE asof_date <= ? ORDER BY asof_date ASC",
                [week_ending],
            ).fetchall()
        finally:
            con.close()
    except Exception as e:  # noqa: BLE001 -- a schema-drift/partial DB must not break the digest
        logger.warning(f"weekly_digest: PnL computation errored: {e!r}")
        return empty

    week_pnl_pct = (
        (friday_equity / prior_equity - 1.0) * 100.0
        if friday_equity is not None and prior_equity
        else None
    )
    since_start_pct = (
        (friday_equity / start_equity - 1.0) * 100.0
        if friday_equity is not None and start_equity
        else None
    )

    # Best/worst day-over-day % change with a date IN the trading week
    # [week_starting, week_ending]. The anchor for the first change (Monday)
    # is whatever snapshot immediately precedes it (normally the prior
    # week's Friday) -- a real day-over-day return, not a fabricated
    # first-day-of-window artifact.
    best_day: tuple[date, float] | None = None
    worst_day: tuple[date, float] | None = None
    changes: list[tuple[date, float]] = []
    for (_d0, e0), (d1, e1) in zip(rows, rows[1:], strict=False):
        if d1 < week_starting or d1 > week_ending or not e0:
            continue
        changes.append((d1, (e1 / e0 - 1.0) * 100.0))
    if changes:
        best_day = max(changes, key=lambda t: t[1])
        worst_day = min(changes, key=lambda t: t[1])

    return dict(
        friday_equity=friday_equity,
        prior_friday_equity=prior_equity,
        week_pnl_pct=week_pnl_pct,
        start_date=start_date,
        start_equity=start_equity,
        since_start_pct=since_start_pct,
        best_day=best_day,
        worst_day=worst_day,
    )


# ---------------------------------------------------------------------------
# Trading activity
# ---------------------------------------------------------------------------


def _compute_trading(db_path: Path, *, start: date, end: date, asof: date) -> dict:
    empty = dict(
        orders_submitted=0,
        orders_failed=0,
        fills_filled=0,
        names_entered=(),
        names_exited=(),
        position_count=None,
        cash=None,
    )
    if not Path(db_path).exists():
        return empty

    try:
        con = read_only_connect(db_path)
        try:
            orders_submitted = con.execute(
                "SELECT COUNT(*) FROM intended_orders "
                "WHERE asof_date BETWEEN ? AND ? AND status = 'submitted'",
                [start, end],
            ).fetchone()[0]
            orders_failed = con.execute(
                "SELECT COUNT(*) FROM intended_orders "
                "WHERE asof_date BETWEEN ? AND ? AND status = 'submission_failed'",
                [start, end],
            ).fetchone()[0]
            fills_filled = con.execute(
                "SELECT COUNT(*) FROM paper_fills "
                "WHERE asof_date BETWEEN ? AND ? AND status = 'filled'",
                [start, end],
            ).fetchone()[0]
            entered = con.execute(
                "SELECT DISTINCT ticker FROM paper_fills "
                "WHERE asof_date BETWEEN ? AND ? AND status = 'filled' "
                "AND UPPER(side) = 'BUY' ORDER BY ticker",
                [start, end],
            ).fetchall()
            exited = con.execute(
                "SELECT DISTINCT ticker FROM paper_fills "
                "WHERE asof_date BETWEEN ? AND ? AND status = 'filled' "
                "AND UPPER(side) = 'SELL' ORDER BY ticker",
                [start, end],
            ).fetchall()
            row = con.execute(
                "SELECT position_count, cash FROM account_snapshots "
                "WHERE asof_date <= ? ORDER BY asof_date DESC LIMIT 1",
                [asof],
            ).fetchone()
        finally:
            con.close()
    except Exception as e:  # noqa: BLE001 -- a schema-drift/partial DB must not break the digest
        logger.warning(f"weekly_digest: trading-activity computation errored: {e!r}")
        return empty

    position_count = int(row[0]) if row and row[0] is not None else None
    cash = float(row[1]) if row and row[1] is not None else None

    return dict(
        orders_submitted=int(orders_submitted),
        orders_failed=int(orders_failed),
        fills_filled=int(fills_filled),
        names_entered=tuple(r[0] for r in entered),
        names_exited=tuple(r[0] for r in exited),
        position_count=position_count,
        cash=cash,
    )


# ---------------------------------------------------------------------------
# Signal: reuses sma.eval.live_ic verbatim, no reimplementation.
# ---------------------------------------------------------------------------


def _compute_ic_read(
    horizon_days: int, db_path: Path, universe_path: Path, week_ending: date
) -> IcRead:
    try:
        if not Path(db_path).exists() or not Path(universe_path).exists():
            return IcRead(horizon_days, None, None, 0, "insufficient")
        ic_df = live_ic.model_edge_ic_df(
            horizon_days, db_path=db_path, universe_path=universe_path
        )
        if ic_df.empty:
            return IcRead(horizon_days, None, None, 0, "insufficient")
        series = (
            ic_df[ic_df["asof_date"] <= week_ending]
            .sort_values("asof_date")
            .set_index("asof_date")["ic"]
        )
        if series.dropna().empty:
            return IcRead(horizon_days, None, None, 0, "insufficient")
        regime = live_ic.trailing_ic_regime(series, window=IC_WINDOW)
        return IcRead(horizon_days, regime["mean"], regime["t_stat"], regime["n"], regime["level"])
    except Exception as e:  # noqa: BLE001 -- a broken IC read must not break the digest
        logger.warning(f"weekly_digest: IC computation errored for horizon={horizon_days}: {e!r}")
        return IcRead(horizon_days, None, None, 0, "insufficient")


# ---------------------------------------------------------------------------
# Ops: sentinel presence per expected job per date, holiday-aware via
# ingest's own holiday_skipped flag (no live Alpaca call -- stays DB/
# sentinel read-only, matching this module's contract).
# ---------------------------------------------------------------------------


def _job_ops_problems(d: date) -> tuple[str, ...]:
    """Problem strings ('<label>: missing' / '<label>: quality failed') for
    every job SCHEDULE expects on `d`. Empty tuple = a fully clean night.

    A market holiday is detected from ingest's own holiday_skipped sentinel
    (written by `sma.ingest run` for exactly this case) -- when set, jobs
    with requires_market_data=True are dropped from the expected set for
    that date, the same guard sma.watchdog applies via a live Alpaca
    calendar call. Reusing the sentinel instead keeps this function DB/
    network-free.
    """
    ingest_sentinel = read_sentinel(label="com.sma.ingest.daily", asof=d)
    is_holiday = bool(ingest_sentinel and ingest_sentinel.get("holiday_skipped"))

    problems: list[str] = []
    for job in sched.SCHEDULE:
        if not sched.runs_today(job.label, asof=d):
            continue
        if job.requires_market_data and is_holiday:
            continue
        check_label = job.liveness_sentinel_label or job.label
        sentinel = read_sentinel(label=check_label, asof=d)
        if sentinel is None:
            problems.append(f"{job.label}: missing")
        elif job.label == "com.sma.ingest.daily" and not sentinel.get("quality", {}).get(
            "passed", True
        ):
            problems.append(f"{job.label}: quality failed")
    return tuple(problems)


def _compute_ops(ops_week_starting: date, week_ending: date) -> dict:
    clean_nights = 0
    total_nights = 0
    degraded_dates: list[date] = []
    problems_by_date: dict[date, tuple[str, ...]] = {}

    d = ops_week_starting
    while d <= week_ending:
        total_nights += 1
        problems = _job_ops_problems(d)
        if problems:
            degraded_dates.append(d)
            problems_by_date[d] = problems
        else:
            clean_nights += 1
        d += timedelta(days=1)

    return dict(
        clean_nights=clean_nights,
        total_nights=total_nights,
        degraded_dates=tuple(degraded_dates),
        problems_by_date=problems_by_date,
    )


def _serving_model_id(models_dir: Path, asof: date) -> str | None:
    try:
        return latest_model_for_date(Path(models_dir), asof).stem
    except FileNotFoundError:
        return None
    except Exception as e:  # noqa: BLE001 -- a bad artifact dir must not break the digest
        logger.warning(f"weekly_digest: model lookup errored: {e!r}")
        return None


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


def compute_weekly_digest(
    week_ending: date,
    *,
    db_path: Path = DEFAULT_DB_PATH,
    models_dir: Path = DEFAULT_MODELS_DIR,
    universe_path: Path = DEFAULT_UNIVERSE_PATH,
) -> WeeklyDigest:
    """Compute the full digest for the trading week ending `week_ending`
    (a Friday). Every sub-computation degrades gracefully on missing/empty
    data (never raises) -- see each helper's own docstring."""
    week_starting = week_ending - timedelta(days=4)  # Monday
    ops_week_starting = week_ending - timedelta(days=6)  # Saturday before that Monday
    prior_friday = week_ending - timedelta(days=7)

    pnl = _compute_pnl(
        Path(db_path),
        week_ending=week_ending,
        prior_friday=prior_friday,
        week_starting=week_starting,
    )
    trading = _compute_trading(
        Path(db_path), start=week_starting, end=week_ending, asof=week_ending
    )
    ic_reads = tuple(
        _compute_ic_read(h, Path(db_path), Path(universe_path), week_ending)
        for h in IC_HORIZONS_DAYS
    )
    ops = _compute_ops(ops_week_starting, week_ending)
    model_id = _serving_model_id(Path(models_dir), week_ending)

    return WeeklyDigest(
        week_ending=week_ending,
        week_starting=week_starting,
        ops_week_starting=ops_week_starting,
        ic_reads=ic_reads,
        model_id=model_id,
        **pnl,
        **trading,
        **ops,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt_money(x: float | None) -> str:
    return f"${x:,.0f}" if x is not None else "n/a"


def _fmt_pct(x: float | None) -> str:
    return f"{x:+.1f}%" if x is not None else "n/a"


def _fmt_ic(read: IcRead) -> str:
    if read.mean is None or read.t_stat is None:
        return f"{read.horizon_days}d not enough live history yet"
    return f"{read.horizon_days}d {read.mean:+.3f} (t={read.t_stat:.1f}, n={read.n}, {read.level})"


def render_markdown(d: WeeklyDigest) -> str:
    """Rayan's-voice markdown recap: numbers first, no em dashes."""
    lines: list[str] = [f"# Week in review: {d.week_ending.isoformat()}", ""]

    lines.append("## P&L")
    if d.friday_equity is not None:
        pnl_line = f"{_fmt_money(d.friday_equity)} Friday close"
        if d.prior_friday_equity is not None:
            pnl_line += (
                f", {_fmt_pct(d.week_pnl_pct)} from last Friday's "
                f"{_fmt_money(d.prior_friday_equity)}"
            )
        if d.since_start_pct is not None and d.start_date is not None:
            pnl_line += (
                f". {_fmt_pct(d.since_start_pct)} since the {_fmt_money(d.start_equity)} "
                f"start on {d.start_date.isoformat()}"
            )
        lines.append(pnl_line + ".")
        if d.best_day is not None and d.worst_day is not None:
            lines.append(
                f"Best day {d.best_day[0].isoformat()} {_fmt_pct(d.best_day[1])}, "
                f"worst day {d.worst_day[0].isoformat()} {_fmt_pct(d.worst_day[1])}."
            )
    else:
        lines.append("No Friday close on record yet.")
    lines.append("")

    lines.append("## Trading")
    orders_line = f"{d.orders_submitted} orders submitted, {d.fills_filled} filled this week"
    if d.orders_failed:
        orders_line += f", {d.orders_failed} failed to submit"
    lines.append(orders_line + ".")
    entered = ", ".join(d.names_entered) if d.names_entered else "none"
    exited = ", ".join(d.names_exited) if d.names_exited else "none"
    lines.append(f"Entered: {entered}. Exited: {exited}.")
    pos = d.position_count if d.position_count is not None else "n/a"
    lines.append(f"{pos} positions, {_fmt_money(d.cash)} cash.")
    lines.append("")

    lines.append("## Signal")
    for read in d.ic_reads:
        lines.append(f"Trailing-{IC_WINDOW} live IC, {_fmt_ic(read)}.")
    lines.append("")

    lines.append("## Ops")
    lines.append(f"{d.clean_nights}/{d.total_nights} nights fully clean this week.")
    for dd in d.degraded_dates:
        lines.append(f"{dd.isoformat()} degraded: {', '.join(d.problems_by_date[dd])}.")
    lines.append(f"Model serving: {d.model_id or 'none found'}.")

    return "\n".join(lines) + "\n"


def render_ntfy_message(d: WeeklyDigest) -> str:
    """P&L headline + clean-nights count + IC read, hard-bounded to
    NTFY_CHAR_LIMIT chars (a defensive cap -- the template alone stays far
    under it for any realistic input, but a huge names_entered/exited list
    is never included here precisely so an unbounded field can't blow the
    push past ntfy's payload)."""
    if d.friday_equity is not None:
        pnl = (
            f"{_fmt_money(d.friday_equity)} ({_fmt_pct(d.week_pnl_pct)} wk, "
            f"{_fmt_pct(d.since_start_pct)} since start)"
        )
    else:
        pnl = "no equity data"

    ic_bits = []
    for read in d.ic_reads:
        if read.mean is None:
            ic_bits.append(f"IC{read.horizon_days} n/a")
        else:
            ic_bits.append(f"IC{read.horizon_days} {read.mean:+.2f} ({read.level})")

    msg = (
        f"Week ending {d.week_ending.isoformat()}: {pnl}. "
        f"{d.clean_nights}/{d.total_nights} nights clean. "
        f"{', '.join(ic_bits)}."
    )
    if len(msg) > NTFY_CHAR_LIMIT:
        msg = msg[: NTFY_CHAR_LIMIT - 3] + "..."
    return msg


def write_digest_file(
    markdown: str, week_ending: date, *, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / digest_filename(week_ending)
    path.write_text(markdown)
    return path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_weekly_digest(
    *,
    week_ending: date,
    db_path: Path = DEFAULT_DB_PATH,
    models_dir: Path = DEFAULT_MODELS_DIR,
    universe_path: Path = DEFAULT_UNIVERSE_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dry_run: bool = False,
    run_date: date | None = None,
    notify_fn=send_ntfy,
) -> Path:
    """Compute + render + write the markdown file; on a real (non-dry-run)
    completion, also push the ntfy summary and write this job's own
    completion sentinel (asof=`run_date`, defaulting to today) so the
    watchdog sees this Sunday's run as done and doesn't keep re-kicking it.

    `dry_run=True` (the manual-verification path) still writes the markdown
    file -- the actual deliverable -- but skips BOTH the ntfy push and the
    sentinel write, matching `sma.live decide --dry-run`'s convention that a
    dry run must not be recorded as a real completed run.
    """
    digest = compute_weekly_digest(
        week_ending,
        db_path=Path(db_path),
        models_dir=Path(models_dir),
        universe_path=Path(universe_path),
    )
    markdown = render_markdown(digest)
    path = write_digest_file(markdown, week_ending, output_dir=Path(output_dir))

    if dry_run:
        logger.info("weekly_digest: dry-run -- skipping ntfy push and sentinel write")
        return path

    with contextlib.suppress(Exception):
        notify_fn(
            render_ntfy_message(digest),
            title=f"SMA week in review: {week_ending.isoformat()}",
            priority="default",
        )

    resolved_run_date = run_date if run_date is not None else datetime.now(UTC).date()
    write_sentinel(
        label=JOB_LABEL,
        asof=resolved_run_date,
        payload={
            "label": JOB_LABEL,
            "asof": resolved_run_date.isoformat(),
            "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "run_id": int(datetime.now(UTC).timestamp()),
            "week_ending": week_ending.isoformat(),
            "output_path": str(path),
        },
    )
    return path
