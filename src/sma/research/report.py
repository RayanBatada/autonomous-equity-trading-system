"""Assemble a single-ticker research report from DuckDB rows.

Pure functions, no I/O beyond the DB connection passed in. The CLI in
``__main__.py`` is the only place that opens connections / calls Alpaca.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from sma.ingest.store import Store


@dataclass
class ReportSection:
    title: str
    body: str  # already-rendered markdown body for the section


@dataclass
class TickerReport:
    ticker: str
    asof_date: date
    in_universe: bool
    sector: str | None
    sections: list[ReportSection]

    def to_markdown(self) -> str:
        header = (
            f"# {self.ticker} — research report\n\n"
            f"- **As of**: {self.asof_date.isoformat()}\n"
            f"- **Sector**: {self.sector or 'unknown'}\n"
            f"- **In universe**: {'yes' if self.in_universe else 'no'}\n\n"
        )
        body = "\n".join(
            f"## {s.title}\n\n{s.body.rstrip()}\n" for s in self.sections
        )
        return header + body


def _fmt_pct(v: float | None) -> str:
    return f"{v * 100:+.2f}%" if v is not None else "—"


def _fmt_num(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _render_quant_section(store: Store, ticker: str, asof: date) -> ReportSection:
    """Latest prediction + 30d price context."""
    # Order by computed_at as the actual tie-breaker (not model_id alphabetic):
    # multiple model batches on the same asof_date should pick the most
    # recently *computed* row, not the lexically-largest model_id suffix.
    pred = store.conn.execute(
        "SELECT asof_date, predicted_value, model_id FROM predictions "
        "WHERE ticker = ? AND asof_date <= ? "
        "ORDER BY asof_date DESC, computed_at DESC LIMIT 1",
        [ticker, asof],
    ).fetchone()

    px_rows = store.conn.execute(
        "SELECT date, adj_close FROM prices "
        "WHERE ticker = ? AND date <= ? AND date >= ? - INTERVAL '30 days' "
        "AND adj_close IS NOT NULL ORDER BY date",
        [ticker, asof, asof],
    ).fetchall()

    lines = []
    if pred:
        lines.append(
            f"- **Latest XGBoost prediction** ({pred[0].isoformat()}, "
            f"model `{pred[2]}`): `{pred[1]:+.4f}`"
        )
    else:
        lines.append("- No XGBoost prediction available (ticker may be outside universe).")

    if len(px_rows) >= 2:
        first, last = px_rows[0][1], px_rows[-1][1]
        ret_30d = (last - first) / first if first else None
        peak = first
        max_dd = 0.0
        for _, c in px_rows:
            if c > peak:
                peak = c
            dd = (c - peak) / peak if peak else 0.0
            if dd < max_dd:
                max_dd = dd
        lines.append(f"- **Last close** ({px_rows[-1][0].isoformat()}): `${last:.2f}`")
        lines.append(f"- **30d return**: {_fmt_pct(ret_30d)}")
        lines.append(f"- **30d max drawdown**: {_fmt_pct(max_dd)}")
    elif px_rows:
        lines.append(f"- **Last close** ({px_rows[-1][0].isoformat()}): `${px_rows[-1][1]:.2f}`")
        lines.append("- 30d window too short for return/drawdown.")
    else:
        lines.append("- No price rows in the 30d window.")

    return ReportSection(title="Quant signal", body="\n".join(lines))


def _render_news_section(store: Store, ticker: str, asof: date, limit: int = 8) -> ReportSection:
    rows = store.conn.execute(
        "SELECT published_at, source, headline FROM news "
        "WHERE ticker = ? AND published_at >= ? - INTERVAL '14 days' "
        "AND published_at < ? + INTERVAL '1 day' "
        "ORDER BY published_at DESC LIMIT ?",
        [ticker, asof, asof, limit],
    ).fetchall()
    if not rows:
        return ReportSection(title="Recent news (14d)", body="_No news rows in the last 14 days._")
    lines = []
    for published_at, source, headline in rows:
        if isinstance(published_at, datetime):
            when = published_at.strftime("%Y-%m-%d")
        else:
            when = str(published_at)
        lines.append(f"- `{when}` (`{source}`) — {headline}")
    return ReportSection(title=f"Recent news ({len(rows)})", body="\n".join(lines))


def _render_theses_section(store: Store, ticker: str, asof: date, limit: int = 3) -> ReportSection:
    # Theses PK is (ticker, asof_date, run_id). When --refresh creates a new
    # row for an already-thesised day, ordering only by asof_date gives an
    # arbitrary winner; tiebreak by run_id DESC so the *latest* run wins.
    rows = store.conn.execute(
        "SELECT asof_date, conviction, score, action_hint, reasoning, "
        "       news_summary, bull_case, bear_case, catalyst_window, flags "
        "FROM theses WHERE ticker = ? AND asof_date <= ? "
        "ORDER BY asof_date DESC, run_id DESC LIMIT ?",
        [ticker, asof, limit],
    ).fetchall()
    if not rows:
        return ReportSection(
            title="LLM theses",
            body="_No thesis rows yet. Run with `--refresh` to generate one._",
        )

    # First row gets full detail; subsequent rows just headlines.
    blocks: list[str] = []
    head = rows[0]
    flags_str = head[9]
    try:
        flags = json.loads(flags_str) if isinstance(flags_str, str) else (flags_str or [])
    except (TypeError, ValueError):
        flags = []
    blocks.append(
        f"### Latest thesis — {head[0].isoformat()}\n\n"
        f"- **Conviction**: `{head[1]}`  **Score**: `{head[2]:+.2f}`  "
        f"**Action hint**: `{head[3]}`\n"
        f"- **Catalyst window**: `{head[8]}`  **Flags**: `{flags or '[]'}`\n\n"
        f"**News summary**: {head[5]}\n\n"
        f"**Bull case**: {head[6]}\n\n"
        f"**Bear case**: {head[7]}\n\n"
        f"**Reasoning**: {head[4]}"
    )
    if len(rows) > 1:
        older = "\n".join(
            f"- `{r[0].isoformat()}` — `{r[1]}` (score `{r[2]:+.2f}`, action `{r[3]}`)"
            for r in rows[1:]
        )
        blocks.append(f"### Earlier theses\n\n{older}")
    return ReportSection(title="LLM theses", body="\n\n".join(blocks))


def _render_politician_section(
    store: Store, ticker: str, asof: date, lookback_days: int = 90,
) -> ReportSection:
    """Show House/Senate disclosures in the last N days."""
    import duckdb

    cutoff = asof - timedelta(days=lookback_days)
    try:
        rows = store.conn.execute(
            "SELECT transaction_date, filing_date, transaction_type, "
            "       amount_min, amount_max, last_name, first_name, chamber "
            "FROM politician_trades WHERE ticker = ? "
            "AND transaction_date >= ? AND transaction_date <= ? "
            "ORDER BY transaction_date DESC LIMIT 25",
            [ticker, cutoff, asof],
        ).fetchall()
    except duckdb.CatalogException:
        # Table missing on very old DBs that haven't run the v5 migration.
        # Narrowed from bare `except Exception` so column-name typos or other
        # query bugs propagate to the caller instead of getting masked as
        # "table not present".
        return ReportSection(
            title="Politician trades (90d)",
            body="_politician_trades table not present (run ingest first)._",
        )
    if not rows:
        return ReportSection(
            title="Politician trades (90d)",
            body="_No politician disclosures in the last 90 days._",
        )
    lines = []
    for transaction_date, _filing_date, txn, amt_min, amt_max, last, first, chamber in rows:
        if hasattr(transaction_date, "isoformat"):
            td = transaction_date.isoformat()
        else:
            td = str(transaction_date)
        name = ", ".join(p for p in [last, first] if p) or "unknown"
        if amt_min is not None and amt_max is not None:
            amount = f"${amt_min:,.0f}–${amt_max:,.0f}"
        else:
            amount = "—"
        lines.append(
            f"- `{td}` — **{txn}** {amount} by {name} ({chamber or '—'})"
        )
    return ReportSection(title=f"Politician trades ({len(rows)})", body="\n".join(lines))


def _render_position_section(position: dict | None) -> ReportSection:
    """Render an Alpaca position dict (qty, avg_entry_price, market_value, unrealized_pl)."""
    if not position:
        return ReportSection(title="Current position", body="_No open position._")
    pl = float(position.get("unrealized_pl", 0))
    pl_sign = "+" if pl >= 0 else "-"
    lines = [
        f"- **Shares**: `{position.get('qty', '?')}`",
        f"- **Avg entry**: `${float(position.get('avg_entry_price', 0)):.2f}`",
        f"- **Market value**: `${float(position.get('market_value', 0)):,.2f}`",
        f"- **Unrealized P&L**: `{pl_sign}${abs(pl):,.2f}` "
        f"(`{float(position.get('unrealized_plpc', 0)) * 100:+.2f}%`)",
    ]
    return ReportSection(title="Current position", body="\n".join(lines))


def build_report(
    *,
    store: Store,
    ticker: str,
    asof: date,
    universe: list[str],
    sector: str | None,
    position: dict | None,
) -> TickerReport:
    """Build a TickerReport. All DB reads happen here; Alpaca position passed in."""
    ticker = ticker.upper()
    in_universe = ticker in {t.upper() for t in universe}
    sections = [
        _render_quant_section(store, ticker, asof),
        _render_position_section(position),
        _render_news_section(store, ticker, asof),
        _render_theses_section(store, ticker, asof),
        _render_politician_section(store, ticker, asof),
    ]
    return TickerReport(
        ticker=ticker,
        asof_date=asof,
        in_universe=in_universe,
        sector=sector,
        sections=sections,
    )
