"""Tests for sma.monitoring.weekly_digest -- the Sunday 18:00 ET
week-in-review digest (com.sma.weekly-digest.weekly).

Layout:
  - PnL/trading/ops computation on a fixture DB with a known week
    (test_compute_* below), including holiday-aware ops and graceful
    degradation on a missing/empty DB.
  - IC-read wiring into sma.eval.live_ic (proves reuse, not duplication).
  - Markdown rendering (a snapshot) and the ntfy message's hard char bound.
  - write_digest_file's filename convention and run_weekly_digest's
    dry-run vs real-run side effects (ntfy push + completion sentinel).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from sma import schedule as sched
from sma.monitoring import weekly_digest as wd
from sma.sentinels import read_sentinel, write_sentinel

WEEK_ENDING = date(2026, 6, 5)  # Friday
WEEK_STARTING = date(2026, 6, 1)  # Monday
OPS_WEEK_STARTING = date(2026, 5, 30)  # Saturday
PRIOR_FRIDAY = date(2026, 5, 29)  # Friday


# ---------------------------------------------------------------------------
# Fixture DB: account_snapshots / paper_fills / intended_orders only (no
# predictions/prices -- proves the IC section degrades to "insufficient"
# rather than crashing when that data isn't there yet).
# ---------------------------------------------------------------------------


def _make_trading_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "t.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE account_snapshots (
            asof_date DATE, equity DOUBLE, cash DOUBLE, buying_power DOUBLE,
            long_market_value DOUBLE, position_count INTEGER,
            total_unrealized_pnl DOUBLE, run_id BIGINT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE paper_fills (
            alpaca_order_id VARCHAR, intended_order_id VARCHAR, asof_date DATE,
            ticker VARCHAR, side VARCHAR, filled_shares DOUBLE,
            fill_price DOUBLE, status VARCHAR, run_id BIGINT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE intended_orders (
            intended_order_id VARCHAR, asof_date DATE, ticker VARCHAR,
            side VARCHAR, target_shares DOUBLE, status VARCHAR, run_id BIGINT
        )
        """
    )

    snapshots = [
        # (date, equity, cash, position_count)
        (date(2026, 4, 30), 90_000.0, 30_000.0, 6),  # earliest -> start
        (PRIOR_FRIDAY, 100_000.0, 25_000.0, 8),
        (date(2026, 6, 1), 101_000.0, 24_000.0, 8),
        (date(2026, 6, 2), 99_000.0, 22_000.0, 9),
        (date(2026, 6, 3), 103_000.0, 18_000.0, 9),
        (date(2026, 6, 4), 102_000.0, 16_000.0, 9),
        (WEEK_ENDING, 105_000.0, 15_000.0, 9),
    ]
    for i, (d, equity, cash, pos) in enumerate(snapshots):
        con.execute(
            "INSERT INTO account_snapshots VALUES (?, ?, ?, ?, ?, ?, NULL, ?)",
            [d, equity, cash, equity * 0.5, equity * 0.5, pos, i],
        )

    fills = [
        # (date, ticker, side, status)
        (date(2026, 6, 1), "AAPL", "BUY", "filled"),
        (date(2026, 6, 1), "TSLA", "SELL", "rejected"),  # excluded: not filled
        (date(2026, 6, 2), "MSFT", "BUY", "filled"),
        (date(2026, 6, 3), "NVDA", "SELL", "filled"),
        (date(2026, 6, 4), "AAPL", "BUY", "filled"),  # dup ticker -> dedup'd
    ]
    for i, (d, ticker, side, status) in enumerate(fills):
        con.execute(
            "INSERT INTO paper_fills VALUES (?, NULL, ?, ?, ?, 10, 100.0, ?, ?)",
            [f"ord{i}", d, ticker, side, status, i],
        )

    orders = [
        (date(2026, 6, 1), "submitted"),
        (date(2026, 6, 2), "submitted"),
        (date(2026, 6, 3), "submitted"),
        (date(2026, 6, 3), "submission_failed"),
        (date(2026, 6, 4), "submitted"),
        (date(2026, 6, 5), "submitted"),
    ]
    for i, (d, status) in enumerate(orders):
        con.execute(
            "INSERT INTO intended_orders VALUES (?, ?, 'AAPL', 'BUY', 1, ?, ?)",
            [f"io{i}", d, status, i],
        )

    con.close()
    return db_path


def test_compute_pnl_week_over_week_and_since_start(tmp_path):
    db_path = _make_trading_db(tmp_path)
    d = wd.compute_weekly_digest(WEEK_ENDING, db_path=db_path, models_dir=tmp_path / "models")

    assert d.friday_equity == 105_000.0
    assert d.prior_friday_equity == 100_000.0
    assert d.week_pnl_pct == pytest.approx(5.0)
    assert d.start_date == date(2026, 4, 30)
    assert d.start_equity == 90_000.0
    assert d.since_start_pct == pytest.approx((105_000.0 / 90_000.0 - 1.0) * 100.0)


def test_compute_best_and_worst_day_within_trading_week(tmp_path):
    db_path = _make_trading_db(tmp_path)
    d = wd.compute_weekly_digest(WEEK_ENDING, db_path=db_path, models_dir=tmp_path / "models")

    # 06-03: 103000/99000 - 1 = +4.04% -- the best day.
    assert d.best_day is not None
    assert d.best_day[0] == date(2026, 6, 3)
    assert d.best_day[1] == pytest.approx((103_000.0 / 99_000.0 - 1.0) * 100.0)

    # 06-02: 99000/101000 - 1 = -1.98% -- the worst day.
    assert d.worst_day is not None
    assert d.worst_day[0] == date(2026, 6, 2)
    assert d.worst_day[1] == pytest.approx((99_000.0 / 101_000.0 - 1.0) * 100.0)


def test_compute_trading_activity(tmp_path):
    db_path = _make_trading_db(tmp_path)
    d = wd.compute_weekly_digest(WEEK_ENDING, db_path=db_path, models_dir=tmp_path / "models")

    assert d.orders_submitted == 5
    assert d.orders_failed == 1
    assert d.fills_filled == 4  # the 'rejected' TSLA fill is excluded
    assert d.names_entered == ("AAPL", "MSFT")  # sorted, deduped
    assert d.names_exited == ("NVDA",)
    assert d.position_count == 9
    assert d.cash == 15_000.0


def test_ic_reads_are_insufficient_without_predictions_table(tmp_path):
    """The fixture DB above has no predictions/prices tables at all -- proves
    a real-world partial DB degrades to 'insufficient' rather than raising."""
    db_path = _make_trading_db(tmp_path)
    d = wd.compute_weekly_digest(WEEK_ENDING, db_path=db_path, models_dir=tmp_path / "models")

    assert {r.horizon_days for r in d.ic_reads} == {10, 20}
    for read in d.ic_reads:
        assert read.mean is None
        assert read.level == "insufficient"


def test_compute_weekly_digest_degrades_gracefully_on_missing_db(tmp_path):
    d = wd.compute_weekly_digest(
        WEEK_ENDING, db_path=tmp_path / "nope.duckdb", models_dir=tmp_path / "models"
    )
    assert d.friday_equity is None
    assert d.week_pnl_pct is None
    assert d.since_start_pct is None
    assert d.best_day is None
    assert d.orders_submitted == 0
    assert d.names_entered == ()
    assert d.position_count is None


# ---------------------------------------------------------------------------
# Ops: sentinel presence per expected job per date, holiday-aware.
# ---------------------------------------------------------------------------


def _expected_labels_for(d: date, *, holiday: bool = False) -> list[str]:
    out = []
    for job in sched.SCHEDULE:
        if not sched.runs_today(job.label, asof=d):
            continue
        if job.requires_market_data and holiday:
            continue
        out.append(job)
    return out


def _write_clean_week(
    *,
    holidays: frozenset[date] = frozenset(),
    skip: frozenset[tuple[str, date]] = frozenset(),
):
    d = OPS_WEEK_STARTING
    while d <= WEEK_ENDING:
        is_holiday = d in holidays
        if is_holiday:
            write_sentinel(
                label="com.sma.ingest.daily",
                asof=d,
                payload={"holiday_skipped": True},
            )
        for job in _expected_labels_for(d, holiday=is_holiday):
            check_label = job.liveness_sentinel_label or job.label
            if (check_label, d) in skip:
                continue
            if job.label == "com.sma.ingest.daily" and is_holiday:
                continue  # already written above
            write_sentinel(
                label=check_label,
                asof=d,
                payload={"label": check_label, "asof": d.isoformat(), "run_id": 1},
            )
        d += timedelta(days=1)


def test_ops_all_clean_when_every_sentinel_present(tmp_path):
    _write_clean_week()
    d = wd.compute_weekly_digest(
        WEEK_ENDING, db_path=tmp_path / "nope.duckdb", models_dir=tmp_path / "models"
    )
    assert d.total_nights == 7
    assert d.clean_nights == 7
    assert d.degraded_dates == ()


def test_ops_flags_missing_sentinel_by_name(tmp_path):
    missing_date = date(2026, 6, 3)
    _write_clean_week(skip=frozenset({("com.sma.live.decide.daily", missing_date)}))

    d = wd.compute_weekly_digest(
        WEEK_ENDING, db_path=tmp_path / "nope.duckdb", models_dir=tmp_path / "models"
    )
    assert d.clean_nights == 6
    assert d.degraded_dates == (missing_date,)
    assert d.problems_by_date[missing_date] == ("com.sma.live.decide.daily: missing",)


def test_ops_quality_failure_is_flagged_even_when_sentinel_exists(tmp_path):
    """ingest DID write a sentinel that day, but quality.passed=False --
    a real failure, distinct from a missing sentinel."""
    bad_date = date(2026, 6, 2)
    _write_clean_week(skip=frozenset({("com.sma.ingest.daily", bad_date)}))
    write_sentinel(
        label="com.sma.ingest.daily",
        asof=bad_date,
        payload={"quality": {"passed": False, "blocking_failures": ["x"]}},
    )

    d = wd.compute_weekly_digest(
        WEEK_ENDING, db_path=tmp_path / "nope.duckdb", models_dir=tmp_path / "models"
    )
    assert d.problems_by_date[bad_date] == ("com.sma.ingest.daily: quality failed",)


def test_ops_holiday_excludes_market_jobs_from_expected_set(tmp_path):
    """A market holiday means predict/agents/decide/stop-loss/reconcile/
    monitoring never fire and never should be counted as missing --
    exactly the guard sma.watchdog applies via a live Alpaca calendar call,
    reproduced here off ingest's own holiday_skipped sentinel."""
    holiday = date(2026, 6, 4)
    _write_clean_week(holidays=frozenset({holiday}))

    d = wd.compute_weekly_digest(
        WEEK_ENDING, db_path=tmp_path / "nope.duckdb", models_dir=tmp_path / "models"
    )
    assert d.clean_nights == 7
    assert holiday not in d.degraded_dates


# ---------------------------------------------------------------------------
# Signal: reuses sma.eval.live_ic -- wiring proof (not a math re-test).
# ---------------------------------------------------------------------------


def _make_universe_yaml(tmp_path: Path, tickers: list[str]) -> Path:
    path = tmp_path / "universe.yaml"
    body = "\n".join(f"    - {t}" for t in tickers)
    path.write_text(f"universe:\n  tickers:\n{body}\n")
    return path


def test_ic_reads_wire_into_live_ic(tmp_path):
    import numpy as np

    rng = np.random.default_rng(7)
    tickers = ["T0", "T1", "T2", "T3", "T4", "T5"]
    n_days = 60
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(n_days)]

    db_path = tmp_path / "ic.duckdb"
    universe_path = _make_universe_yaml(tmp_path, tickers)
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE TABLE predictions (asof_date DATE, ticker VARCHAR, target VARCHAR, "
        "predicted_value DOUBLE, model_id VARCHAR)"
    )
    con.execute(
        "CREATE TABLE prices (ticker VARCHAR, date DATE, open DOUBLE, high DOUBLE, "
        "low DOUBLE, close DOUBLE, adj_close DOUBLE, volume BIGINT, source VARCHAR, "
        "run_id BIGINT)"
    )
    con.execute(
        "CREATE TABLE intended_orders (intended_order_id VARCHAR, asof_date DATE, "
        "ticker VARCHAR, side VARCHAR, target_shares DOUBLE, status VARCHAR, run_id BIGINT)"
    )

    price = {tkr: 100.0 + i * 5 for i, tkr in enumerate(tickers)}
    for d in dates:
        for i, tkr in enumerate(tickers):
            price[tkr] = max(1.0, price[tkr] + (i + 1) * 0.15 + rng.normal(0, 1.0))
            px = price[tkr]
            con.execute(
                "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, 0, 'yfinance', 1)",
                [tkr, d, px, px, px, px, px],
            )

    horizon = 10
    decide_dates = [d for i, d in enumerate(dates) if i + 1 + horizon < n_days]
    assert len(decide_dates) >= 21
    week_ending = decide_dates[-1]

    for d in decide_dates:
        for i, tkr in enumerate(tickers):
            con.execute(
                "INSERT INTO predictions VALUES (?, ?, 'ret_30d_forward', ?, 'm1')",
                [d, tkr, float(i)],
            )
        con.execute(
            "INSERT INTO intended_orders VALUES (gen_random_uuid(), ?, ?, 'BUY', 1, "
            "'submitted', 1)",
            [d, tickers[0]],
        )
    con.close()

    d = wd.compute_weekly_digest(
        week_ending, db_path=db_path, models_dir=tmp_path / "models", universe_path=universe_path
    )
    ic10 = next(r for r in d.ic_reads if r.horizon_days == 10)
    assert ic10.mean is not None
    assert ic10.level == "positive"
    assert ic10.n >= 2

    # This DB has no account_snapshots/paper_fills tables at all (only
    # predictions/prices/intended_orders) -- proves PnL/trading degrade to
    # empty rather than raising on a schema-drift/partial DB.
    assert d.friday_equity is None
    assert d.orders_submitted == 0


# ---------------------------------------------------------------------------
# Model serving lookup
# ---------------------------------------------------------------------------


def test_serving_model_id_from_models_dir(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    stem = "xgb_ret_30d_forward_2026-06-01_abcd1234"
    (models_dir / f"{stem}.pkl").write_bytes(b"")
    (models_dir / f"{stem}.json").write_text('{"created_at": "2026-06-01T04:00:00Z"}')

    d = wd.compute_weekly_digest(
        WEEK_ENDING, db_path=tmp_path / "nope.duckdb", models_dir=models_dir
    )
    assert d.model_id == stem


def test_serving_model_id_none_when_models_dir_missing(tmp_path):
    d = wd.compute_weekly_digest(
        WEEK_ENDING, db_path=tmp_path / "nope.duckdb", models_dir=tmp_path / "does-not-exist"
    )
    assert d.model_id is None


# ---------------------------------------------------------------------------
# Rendering: markdown snapshot + ntfy char bound
# ---------------------------------------------------------------------------


def _fixed_digest() -> wd.WeeklyDigest:
    return wd.WeeklyDigest(
        week_ending=WEEK_ENDING,
        week_starting=WEEK_STARTING,
        ops_week_starting=OPS_WEEK_STARTING,
        friday_equity=105_000.0,
        prior_friday_equity=100_000.0,
        week_pnl_pct=5.0,
        start_date=date(2026, 4, 30),
        start_equity=90_000.0,
        since_start_pct=16.666666666666668,
        best_day=(date(2026, 6, 3), 4.040404040404041),
        worst_day=(date(2026, 6, 2), -1.9801980198019802),
        orders_submitted=5,
        orders_failed=1,
        fills_filled=4,
        names_entered=("AAPL", "MSFT"),
        names_exited=("NVDA",),
        position_count=9,
        cash=15_000.0,
        ic_reads=(
            wd.IcRead(10, 0.05, 2.4, 21, "positive"),
            wd.IcRead(20, -0.02, -0.8, 21, "neutral"),
        ),
        clean_nights=6,
        total_nights=7,
        degraded_dates=(date(2026, 6, 3),),
        problems_by_date={date(2026, 6, 3): ("com.sma.live.decide.daily: missing",)},
        model_id="xgb_ret_30d_forward_2026-06-01_abcd1234",
    )


def test_render_markdown_snapshot():
    text = wd.render_markdown(_fixed_digest())
    assert text == (
        "# Week in review: 2026-06-05\n"
        "\n"
        "## P&L\n"
        "$105,000 Friday close, +5.0% from last Friday's $100,000. "
        "+16.7% since the $90,000 start on 2026-04-30.\n"
        "Best day 2026-06-03 +4.0%, worst day 2026-06-02 -2.0%.\n"
        "\n"
        "## Trading\n"
        "5 orders submitted, 4 filled this week, 1 failed to submit.\n"
        "Entered: AAPL, MSFT. Exited: NVDA.\n"
        "9 positions, $15,000 cash.\n"
        "\n"
        "## Signal\n"
        "Trailing-21 live IC, 10d +0.050 (t=2.4, n=21, positive).\n"
        "Trailing-21 live IC, 20d -0.020 (t=-0.8, n=21, neutral).\n"
        "\n"
        "## Ops\n"
        "6/7 nights fully clean this week.\n"
        "2026-06-03 degraded: com.sma.live.decide.daily: missing.\n"
        "Model serving: xgb_ret_30d_forward_2026-06-01_abcd1234.\n"
    )


def test_render_markdown_never_uses_em_dash():
    assert "—" not in wd.render_markdown(_fixed_digest())


def test_render_markdown_handles_missing_data_without_crashing():
    empty = wd.WeeklyDigest(
        week_ending=WEEK_ENDING,
        week_starting=WEEK_STARTING,
        ops_week_starting=OPS_WEEK_STARTING,
        friday_equity=None,
        prior_friday_equity=None,
        week_pnl_pct=None,
        start_date=None,
        start_equity=None,
        since_start_pct=None,
        best_day=None,
        worst_day=None,
        orders_submitted=0,
        orders_failed=0,
        fills_filled=0,
        names_entered=(),
        names_exited=(),
        position_count=None,
        cash=None,
        ic_reads=(
            wd.IcRead(10, None, None, 0, "insufficient"),
            wd.IcRead(20, None, None, 0, "insufficient"),
        ),
        clean_nights=0,
        total_nights=0,
        degraded_dates=(),
        problems_by_date={},
        model_id=None,
    )
    text = wd.render_markdown(empty)
    assert "No Friday close on record yet." in text
    assert "Entered: none. Exited: none." in text
    assert "n/a positions, n/a cash." in text
    assert "Model serving: none found." in text


def test_render_ntfy_message_within_char_limit_and_has_required_bits():
    msg = wd.render_ntfy_message(_fixed_digest())
    assert len(msg) <= wd.NTFY_CHAR_LIMIT
    assert "$105,000" in msg  # P&L headline
    assert "6/7" in msg  # clean-nights count
    assert "IC10" in msg and "IC20" in msg  # IC read


def test_render_ntfy_message_is_hard_bounded_even_when_limit_is_tiny(monkeypatch):
    """Forces the truncation branch: proves the bound is enforced by code,
    not just satisfied by a template that happens to stay short."""
    monkeypatch.setattr(wd, "NTFY_CHAR_LIMIT", 40)
    msg = wd.render_ntfy_message(_fixed_digest())
    assert len(msg) <= 40


# ---------------------------------------------------------------------------
# write_digest_file / run_weekly_digest orchestration
# ---------------------------------------------------------------------------


def test_digest_filename_uses_iso_week():
    assert wd.digest_filename(WEEK_ENDING) == "2026-23.md"
    assert wd.digest_filename(date(2026, 8, 28)) == "2026-35.md"


def test_write_digest_file_creates_dir(tmp_path):
    out_dir = tmp_path / "weekly"
    path = wd.write_digest_file("hello\n", WEEK_ENDING, output_dir=out_dir)
    assert path == out_dir / "2026-23.md"
    assert path.read_text() == "hello\n"


def test_run_weekly_digest_dry_run_skips_notify_and_sentinel(tmp_path):
    db_path = _make_trading_db(tmp_path)
    out_dir = tmp_path / "weekly"
    notified = []

    path = wd.run_weekly_digest(
        week_ending=WEEK_ENDING,
        db_path=db_path,
        models_dir=tmp_path / "models",
        universe_path=tmp_path / "no-universe.yaml",
        output_dir=out_dir,
        dry_run=True,
        notify_fn=lambda *a, **k: notified.append((a, k)),
    )

    assert path.exists()
    assert "Week in review" in path.read_text()
    assert notified == []
    assert read_sentinel(label=wd.JOB_LABEL, asof=date.today()) is None


def test_run_weekly_digest_real_run_notifies_and_writes_sentinel(tmp_path):
    db_path = _make_trading_db(tmp_path)
    out_dir = tmp_path / "weekly"
    notified = []
    run_date = date(2026, 6, 7)

    path = wd.run_weekly_digest(
        week_ending=WEEK_ENDING,
        db_path=db_path,
        models_dir=tmp_path / "models",
        universe_path=tmp_path / "no-universe.yaml",
        output_dir=out_dir,
        dry_run=False,
        run_date=run_date,
        notify_fn=lambda text, **kw: notified.append((text, kw)),
    )

    assert path.exists()
    assert len(notified) == 1
    text, kwargs = notified[0]
    assert len(text) <= wd.NTFY_CHAR_LIMIT
    assert kwargs.get("title") == "SMA week in review: 2026-06-05"

    sentinel = read_sentinel(label=wd.JOB_LABEL, asof=run_date)
    assert sentinel is not None
    assert sentinel["week_ending"] == "2026-06-05"


def test_run_weekly_digest_never_raises_when_notify_fn_blows_up(tmp_path):
    """A broken notifier must never break a job that already successfully
    wrote the markdown file (matches notify_failure's own
    never-break-the-caller contract elsewhere in this codebase)."""
    db_path = _make_trading_db(tmp_path)
    out_dir = tmp_path / "weekly"

    def _boom(*a, **k):
        raise RuntimeError("ntfy is down")

    path = wd.run_weekly_digest(
        week_ending=WEEK_ENDING,
        db_path=db_path,
        models_dir=tmp_path / "models",
        universe_path=tmp_path / "no-universe.yaml",
        output_dir=out_dir,
        dry_run=False,
        run_date=date(2026, 6, 7),
        notify_fn=_boom,
    )
    assert path.exists()
    assert read_sentinel(label=wd.JOB_LABEL, asof=date(2026, 6, 7)) is not None


def test_most_recent_friday():
    assert wd.most_recent_friday(date(2026, 6, 5)) == date(2026, 6, 5)  # Friday itself
    assert wd.most_recent_friday(date(2026, 6, 6)) == date(2026, 6, 5)  # Saturday
    assert wd.most_recent_friday(date(2026, 6, 7)) == date(2026, 6, 5)  # Sunday
    assert wd.most_recent_friday(date(2026, 6, 8)) == date(2026, 6, 5)  # Monday


# ---------------------------------------------------------------------------
# CLI wiring: `python -m sma.monitoring weekly-digest`
# ---------------------------------------------------------------------------


def test_cli_weekly_digest_dry_run_writes_file_without_notify(tmp_path):
    """--dry-run never calls notify_fn at all (checked directly in
    test_run_weekly_digest_dry_run_skips_notify_and_sentinel above) --
    this test is the CLI-wiring proof: flags parse, defaults resolve, and
    the file lands at the expected path end to end."""
    from click.testing import CliRunner

    from sma.monitoring.__main__ import cli

    db_path = _make_trading_db(tmp_path)
    out_dir = tmp_path / "weekly"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "weekly-digest",
            "--week-ending",
            "2026-06-05",
            "--dry-run",
            "--db",
            str(db_path),
            "--models-dir",
            str(tmp_path / "models"),
            "--universe",
            str(tmp_path / "no-universe.yaml"),
            "--output-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "2026-23.md").exists()


def test_cli_weekly_digest_defaults_week_ending_to_most_recent_friday(tmp_path, monkeypatch):
    """No --week-ending given: falls back to most_recent_friday(today ET),
    proving the CLI wires that default through rather than requiring the
    flag on every Sunday launchd/systemd fire."""
    from click.testing import CliRunner

    from sma.monitoring import __main__ as monitoring_main

    captured = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return tmp_path / "weekly" / "fake.md"

    monkeypatch.setattr("sma.monitoring.weekly_digest.run_weekly_digest", _fake_run)

    runner = CliRunner()
    result = runner.invoke(monitoring_main.cli, ["weekly-digest", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert captured["week_ending"] == wd.most_recent_friday(
        datetime.now(monitoring_main.ET).date()
    )
