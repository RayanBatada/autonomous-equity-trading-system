"""Telegram daily digest of the bot's buy signals — a paper-trading preview so
you can judge signal quality before committing real money.

Reads the DB read-only and sends via sma.ingest.notify.send_telegram (a no-op
without creds, so it's safe to run unconfigured). Intended to run right after
the evening decide job:  python -m sma.digest
"""
from __future__ import annotations

from pathlib import Path

import duckdb

from sma.ingest.notify import send_telegram


def build_digest(
    db_path: str = "data/sma.duckdb",
    models_dir: str = "models_artifacts",
    universe_path: str = "src/sma/universe.yaml",
) -> str:
    """Compose the digest text. Every section is defensive: a missing table or
    column degrades to a note, never an exception — a digest must not crash."""
    con = duckdb.connect(db_path, read_only=True)
    lines: list[str] = ["\U0001F4CA <b>SMA daily digest</b>"]

    try:
        row = con.execute(
            "SELECT asof_date, equity FROM account_snapshots ORDER BY asof_date DESC LIMIT 1"
        ).fetchone()
        if row:
            lines.append(f"equity ${row[1]:,.0f} (as of {row[0]})")
    except Exception:
        pass

    try:
        last = con.execute(
            "SELECT MAX(asof_date) FROM intended_orders WHERE source='decide'"
        ).fetchone()[0]
        rows = con.execute(
            "SELECT ticker, UPPER(side) FROM intended_orders "
            "WHERE asof_date=? AND source='decide' ORDER BY side, ticker",
            [last],
        ).fetchall()
        buys = [t for t, s in rows if s == "BUY"]
        sells = [t for t, s in rows if s == "SELL"]
        lines.append(f"\n<b>decide {last}</b>: {len(buys)} buys, {len(sells)} sells")
        if buys:
            lines.append("\U0001F7E2 BUY: " + ", ".join(buys))
        if sells:
            lines.append("\U0001F534 SELL: " + ", ".join(sells))
    except Exception:
        lines.append("\n(no decide orders found yet)")

    try:
        from sma.ingest.universe import load_universe
        from sma.model.predictor import Predictor

        uni = [t for t in load_universe(universe_path) if t != "SPY"]
        last_px = con.execute("SELECT MAX(date) FROM prices").fetchone()[0]
        scores = Predictor(
            models_dir=Path(models_dir), db_path=Path(db_path)
        ).predict_for(last_px, uni)
        ranked = sorted(scores, key=scores.get, reverse=True)
        lines.append(
            "\n<b>model top-5</b>: "
            + ", ".join(f"{t} ({scores[t]:+.2f})" for t in ranked[:5])
        )
        if "MU" in scores:
            lines.append(f"MU: rank {ranked.index('MU') + 1}/{len(ranked)} ({scores['MU']:+.2f})")
    except Exception:
        lines.append("\n(model scores unavailable)")

    con.close()
    lines.append("\n<i>paper preview — thin/unproven edge, not advice</i>")
    return "\n".join(lines)


def main() -> None:
    msg = build_digest()
    sent = send_telegram(msg)
    print(msg)
    print(f"\n[telegram sent: {sent}]")


if __name__ == "__main__":
    main()
