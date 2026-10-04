#!/usr/bin/env python
"""Build tests/fixtures/replay_synthetic.duckdb: a GENERATED replay fixture.

Every number in it comes from a seeded random generator; nothing is read
from the production database or any broker. It exists so the replay and
sleeve golden tests can run in a public copy of this repo without
publishing the real 2026-08-28 fixture (real paper fills, account
snapshots and model theses).

Shape mirrors tests/fixtures/replay_2026_08_28.duckdb: 9 sessions of prices
for the whole universe (a yfinance row with adj_close and an alpaca row
without, as ingest writes them), one night of predictions, a few nights of
theses, upcoming earnings for two names (exercises the blackout), an equity
history, and synthetic BUY fills for ten held names, some inside the 7-day
minimum hold.

Deterministic: same seed, same file contents row for row. Rebuild with
    uv run python scripts/build_synthetic_replay_fixture.py
then re-pin EXPECTED_SYNTHETIC_ORDERS in tests/integration/live/test_replay.py
from `python -m sma.live replay --asof-date 2026-08-28 --book asof --db <file>`.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "tests" / "fixtures" / "replay_synthetic.duckdb"
SEED = 20261004
ASOF = date(2026, 8, 28)
SESSIONS = [date(2026, 8, d) for d in (18, 19, 20, 21, 24, 25, 26, 27, 28)]
RUN_ID = 1
MODEL_ID = "synthetic_model_for_tests"


def _weekdays(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def build(out: Path = OUT) -> Path:
    import sma.locks as locks
    from sma.ingest.store import Store
    from sma.ingest.universe import load_universe

    rng = np.random.default_rng(SEED)
    universe = sorted(load_universe(REPO / "src" / "sma" / "universe.yaml"))
    if out.exists():
        out.unlink()

    with tempfile.TemporaryDirectory() as tmp:
        lock = Path(tmp) / ".sma-writer.lock"
        locks.DEFAULT_LOCK_PATH = lock  # never touch the repo's real lock
        with locks.writer_lock(label="build-synthetic-fixture", lock_path=lock):
            store = Store(path=str(out)).connect()
            c = store.conn

            last_close: dict[str, float] = {}
            for t in universe:
                level = float(np.exp(rng.uniform(np.log(15), np.log(600))))
                adv_shares = float(np.exp(rng.uniform(np.log(5e5), np.log(3e7))))
                for d in SESSIONS:
                    r = float(rng.normal(0.0, 0.02))
                    o = level
                    level = level * (1 + r)
                    hi = max(o, level) * (1 + abs(float(rng.normal(0, 0.005))))
                    lo = min(o, level) * (1 - abs(float(rng.normal(0, 0.005))))
                    vol = int(adv_shares * float(rng.uniform(0.6, 1.4)))
                    c.execute(
                        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'yfinance', ?)",
                        [t, d, o, hi, lo, level, level, vol, RUN_ID],
                    )
                    c.execute(
                        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, NULL, ?, 'alpaca', ?)",
                        [t, d, o, hi, lo, level, vol // 30, RUN_ID],
                    )
                last_close[t] = level

            now = datetime(2026, 8, 28, 19, 30)
            for t in universe:
                c.execute(
                    "INSERT INTO predictions VALUES (?, ?, 'ret_30d_forward', ?, ?, ?)",
                    [ASOF, t, float(rng.normal(0.0, 0.05)), MODEL_ID, now],
                )

            thesis_cols = [
                r[0] for r in c.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'theses' ORDER BY ordinal_position"
                ).fetchall()
            ]
            convictions = ["bullish", "neutral", "bearish"]
            for i, d in enumerate([date(2026, 8, 26), date(2026, 8, 27), ASOF]):
                for t in rng.choice(universe, size=20, replace=False):
                    conv = convictions[int(rng.integers(0, 3))]
                    row = {
                        "ticker": str(t), "asof_date": d, "run_id": RUN_ID + i,
                        "news_summary": "Synthetic thesis for tests.",
                        "key_developments": "[]", "notable_filings": "[]",
                        "bull_case": "Synthetic.", "bear_case": "Synthetic.",
                        "asymmetric_risks": "[]", "catalyst_window": "far",
                        "conviction": conv,
                        "score": {"bullish": 0.4, "neutral": 0.0, "bearish": -0.4}[conv],
                        "flags": "[]", "action_hint": "hold",
                        "reasoning": "Synthetic thesis for tests.",
                        "created_at": datetime(d.year, d.month, d.day, 19, 50),
                    }
                    cols = [k for k in thesis_cols if k in row]
                    c.execute(
                        f"INSERT INTO theses ({', '.join(cols)}) "
                        f"VALUES ({', '.join('?' for _ in cols)})",
                        [row[k] for k in cols],
                    )

            for t in rng.choice(universe, size=2, replace=False):
                c.execute(
                    "INSERT INTO earnings (ticker, report_date, eps_estimate, eps_actual, "
                    "revenue_estimate, revenue_actual, source, run_id) "
                    "VALUES (?, ?, 1.0, NULL, NULL, NULL, 'synthetic', ?)",
                    [str(t), date(2026, 8, 31), RUN_ID],
                )

            equity = 100_000.0
            for d in _weekdays(date(2026, 4, 30), ASOF):
                equity *= 1 + float(rng.normal(0.0004, 0.01))
                c.execute(
                    "INSERT INTO account_snapshots (asof_date, equity, cash, buying_power, "
                    "long_market_value, position_count, run_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?, 10, ?, ?)",
                    [d, equity, equity * 0.08, equity * 0.16, equity * 0.92, RUN_ID,
                     datetime(d.year, d.month, d.day, 16, 30)],
                )

            held = [str(t) for t in rng.choice(universe, size=10, replace=False)]
            buy_days = [date(2026, 8, d) for d in (10, 12, 14, 18, 20, 24, 25, 26, 27, 28)]
            for i, (t, d) in enumerate(zip(held, buy_days, strict=True)):
                px = last_close[t]
                shares = float(max(1, int(equity * 0.09 / px)))
                ts = datetime(d.year, d.month, d.day, 9, 32)
                c.execute(
                    "INSERT INTO paper_fills (alpaca_order_id, intended_order_id, asof_date, "
                    "ticker, side, filled_shares, fill_price, commission, fees, status, "
                    "submitted_at, filled_at, run_id) "
                    "VALUES (?, NULL, ?, ?, 'BUY', ?, ?, 0, 0, 'filled', ?, ?, ?)",
                    [f"synthetic-{i}", d - timedelta(days=1), t, shares, px, ts, ts, RUN_ID],
                )
            c.execute("CHECKPOINT")
            store.close()
    return out


if __name__ == "__main__":
    print(f"wrote {build()}")
