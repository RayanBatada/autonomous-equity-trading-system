"""Anthropic-API wrapper that proposes new tilt() bodies.

Per the locked design (`specs/2026-04-28-phase-6-autoresearch.md`), the
agent sees the current `active.py`, the recent experiment log, the eval
contract, and live paper-trading performance vs SPY. It returns the FULL
new `active.py` plus a 1-sentence summary of what it changed.

The model is Sonnet 4.6 (`claude-sonnet-4-6`) per Q2 of the locked spec:
high-quality code judgment is worth the $50-100/1000-iter cost. Prompt
caching cuts iteration-2+ cost by ~80% since the constraints + current
active.py + perf snapshot don't change within a run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096


@dataclass(frozen=True)
class AgentProposal:
    """One iteration's output from the LLM agent."""
    active_py_text: str       # complete proposed `active.py` file content
    summary: str              # 1-sentence change description
    cost_usd: float           # Anthropic API spend for this call


# Split into stable (cached) + variable (per-iteration) parts. The stable
# prefix is identical across all iterations within a single run, so the 4.6
# Anthropic prompt-cache ($0.30/MTok reads vs $3/MTok regular input) drops
# per-iteration cost dramatically — ~80% on a 10-iteration run.

_STABLE_HEADER = """\
You are an autoresearch agent rewriting the body of `tilt()` in
`src/sma/strategy/active.py` to improve the strategy's walk-forward
cross-validation Sharpe.

Constraints (loop-runner-enforced; non-compliance → rejected experiment):
  - Edit ONLY `src/sma/strategy/active.py`.
  - Preserve the signature: `tilt(*, asof_date, decisions, ctx) -> list[StrategyDecision]`.
  - `StrategyDecision` is a FROZEN dataclass with EXACTLY three fields:
    `asof_date: date`, `ticker: str`, `target_weight: float`. It has NO other
    fields — do NOT pass any other kwarg (e.g. signal_metadata, score, reason);
    constructing one with an unknown kwarg raises and fails the experiment.
    To change a weight, copy: `dataclasses.replace(d, target_weight=new_w)`
    (import dataclasses at the top of active.py).
  - `TiltContext` fields are FROZEN: quant_scores, theses, portfolio_dollars,
    sector_exposure, sector_for, account_equity, cash.
  - Output `target_weight` for each decision must stay in [0, 0.10]
    (max_position_pct=0.10 in production rails).
  - Tickers may be REMOVED from `decisions`; NEW tickers may NOT be introduced.
  - Function must be deterministic given (asof_date, decisions, ctx).

Eval contract (read-only):
  - Walk-forward CV across 5 val sub-windows W1..W5.
  - monotonicity_score = sum(1 for w if sharpe[w] >= baseline[w] + 0.05).
  - Promotion criterion: mono >= 3 AND overall > baseline + 0.1.
"""


def _build_perf_context(
    *,
    snapshots: list[dict[str, Any]] | None,
    paper_fills_net_count: int | None,
    spy_returns: dict[str, float] | None,
) -> str:
    """Format live paper-trading performance for the agent prompt.

    All inputs nullable — when the autoresearch loop is run on a fresh DB
    (CI, smoke), the perf snapshot is just "(none yet)" and the agent works
    from the CV-eval contract alone.
    """
    if not snapshots:
        return "(no live paper-trading snapshots yet — run on a populated DB to enable this signal)"

    first, last = snapshots[0], snapshots[-1]
    days = (last["date"] - first["date"]).days
    portfolio_ret_pct = ((last["equity"] - first["equity"]) / first["equity"]) * 100
    lines = [
        f"Live paper-trading from {first['date']} → {last['date']} ({days}d):",
        f"  equity: ${first['equity']:,.0f} → ${last['equity']:,.0f} ({portfolio_ret_pct:+.2f}%)",
        f"  current cash: ${last['cash']:,.0f}, positions: {last['position_count']}",
    ]
    if paper_fills_net_count is not None:
        lines.append(f"  paper_fills net long positions: {paper_fills_net_count}")
    if spy_returns is not None and "return_pct" in spy_returns:
        spy_ret_pct = spy_returns["return_pct"]
        alpha_pp = portfolio_ret_pct - spy_ret_pct
        lines.append(f"  SPY benchmark same period: {spy_ret_pct:+.2f}%")
        lines.append(
            f"  → alpha vs SPY: {alpha_pp:+.2f}pp "
            f"({'beating' if alpha_pp >= 0 else 'BEHIND'} the market)"
        )
        if alpha_pp < -1.0:
            lines.append(
                "  PRIORITY: the strategy is materially underperforming SPY. "
                "Tilt edits that close this gap (better risk-adjusted "
                "selection, sharper conviction-weighting, or removing "
                "low-quality picks) score highest."
            )
    return "\n".join(lines)


def build_prompt_blocks(
    *,
    current_active_py: str,
    recent_experiments: list[dict[str, Any]],
    perf_context: str,
) -> list[dict[str, Any]]:
    """Build the user-message content blocks for one autoresearch iteration.

    Returns two text blocks:
      1. Stable prefix (constraints + current active.py + live perf context),
         marked with cache_control for cross-iteration reuse.
      2. Variable suffix (recent experiment log + instruction), uncached.

    The stable block is identical across all iterations within a single
    run (current_active_py is fixed to the baseline, perf_context is
    computed once at run start), so the prompt cache reads it back at ~10%
    of full input cost on iterations 2+.
    """
    if not recent_experiments:
        log_summary = "(no prior experiments on this branch yet)"
    else:
        lines = ["iter | status | mono | overall | summary"]
        for e in recent_experiments[-20:]:
            mono = e.get("monotonicity_score")
            mono_s = f"{mono:.3f}" if isinstance(mono, (int, float)) else "-"
            overall = e.get("sharpe_overall")
            overall_s = f"{overall:.3f}" if isinstance(overall, (int, float)) else "-"
            status = e.get("status") or "?"
            lines.append(
                f"{e['iter_index']:>4} | {status:>11} | {mono_s:>4} | {overall_s:>7} | "
                f"{(e.get('proposal_summary') or '')[:80]}"
            )
        log_summary = "\n".join(lines)

    stable_text = (
        f"{_STABLE_HEADER}\n"
        f"Live performance snapshot (computed at run start):\n"
        f"{perf_context}\n\n"
        f"Current `active.py`:\n"
        f"```python\n{current_active_py}\n```\n"
    )
    variable_text = (
        f"Recent experiment log (last {min(len(recent_experiments), 20)} "
        f"iterations on this branch):\n"
        f"{log_summary}\n\n"
        f"Propose ONE focused edit. Output the FULL new `active.py` file "
        f"content inside a ```python ... ``` block, followed by a 1-sentence "
        f"summary of what you changed."
    )
    return [
        {"type": "text", "text": stable_text, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": variable_text},
    ]


# Legacy single-string prompt builder, kept for tests that pinned the old
# behavior. New callers should use build_prompt_blocks.
def build_prompt(
    *,
    current_active_py: str,
    recent_experiments: list[dict[str, Any]],
    perf_context: str = "(none — using legacy build_prompt without live perf)",
) -> str:
    """Flatten the cache-enabled block structure into a single string."""
    blocks = build_prompt_blocks(
        current_active_py=current_active_py,
        recent_experiments=recent_experiments,
        perf_context=perf_context,
    )
    return "\n\n".join(b["text"] for b in blocks)


def propose(
    *,
    current_active_py: str,
    recent_experiments: list[dict[str, Any]],
    perf_context: str = "(no live performance context provided)",
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> AgentProposal:
    """Call Anthropic; parse the response for the new active.py + summary.

    Uses prompt caching on the stable prefix to cut iteration-2+ cost by
    ~80%. Cache TTL is 5min default; a 10-iteration autoresearch run
    typically completes well under that window.

    Raises:
        RuntimeError: API call fails or response doesn't contain a valid
                      python code block.
    """
    from anthropic import Anthropic
    api_key = _load_anthropic_api_key()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not found in environment OR in .env file. "
            "launchd-fired jobs don't inherit shell env vars — make sure the "
            "key is in the repo-root .env file (the same place sma.config's "
            "Secrets BaseSettings reads it from)."
        )
    client = Anthropic(api_key=api_key)

    blocks = build_prompt_blocks(
        current_active_py=current_active_py,
        recent_experiments=recent_experiments,
        perf_context=perf_context,
    )
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": blocks}],
    )

    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    new_py, summary = _parse_proposal(text)

    # Cost (Sonnet 4.6: $3/MTok input, $15/MTok output as of 2026).
    # Cache reads are billed at ~10% of regular input price; cache writes
    # at ~125% (5-min TTL) — verify by inspecting usage.cache_*_input_tokens.
    usage = resp.usage
    regular_input = getattr(usage, "input_tokens", 0)
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cost = (
        (regular_input / 1_000_000) * 3.0
        + (cache_read / 1_000_000) * 0.30      # 10% of input price
        + (cache_create / 1_000_000) * 3.75    # 125% of input price (5min TTL)
        + (usage.output_tokens / 1_000_000) * 15.0
    )
    return AgentProposal(active_py_text=new_py, summary=summary, cost_usd=cost)


def _load_anthropic_api_key() -> str:
    """Read ANTHROPIC_API_KEY from process env OR fall back to repo-root .env.

    Empirical 2026-05-25: the launchd-fired autoresearch job had ANTHROPIC_API_KEY
    unset (launchd doesn't inherit shell env vars; only PATH + TZ are
    in EnvironmentVariables on the rendered plist). All 10 iterations
    failed with `agent error: ANTHROPIC_API_KEY not set in environment`.

    sma.config.Secrets already uses pydantic-settings with `env_file=".env"`,
    so the .env in the repo root is the canonical place for the key —
    autoresearch just wasn't loading from there. This helper mirrors that
    behavior, preferring an explicit env var (for tests + manual runs that
    export the key directly), then falling back to the .env file relative
    to the working directory (launchd sets WorkingDirectory=repo_root).
    """
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    try:
        from sma.config import Secrets
        secrets = Secrets()
        return (secrets.anthropic_api_key or "").strip()
    except Exception:
        return ""


def _parse_proposal(text: str) -> tuple[str, str]:
    """Pull the python code block + the summary sentence from the agent reply."""
    import re
    m = re.search(r"```python\n(.+?)\n```", text, re.DOTALL)
    if not m:
        raise RuntimeError(
            f"agent reply missing ```python ...``` block. "
            f"Response head: {text[:300]!r}"
        )
    code = m.group(1).strip()
    after = text[m.end():].strip()
    # First non-empty line after the code block is the summary.
    summary_line = next((ln.strip() for ln in after.splitlines() if ln.strip()), "")
    if not summary_line:
        summary_line = "(no summary)"
    return code, summary_line[:500]


# ---- Live performance context helper ---------------------------------------


def compute_live_perf_context(store, *, asof_today: date | None = None) -> str:
    """Query the live DB for a performance snapshot the agent can act on.

    Returns the formatted string ready to splice into the prompt — call
    once at run start, then pass to every `propose()` invocation in the
    iteration loop so the cache prefix stays stable.

    `asof_today` is the cutoff; defaults to today (ET, naive). SPY
    benchmark is computed over the same date range as the account
    snapshots so the alpha figure is apples-to-apples.
    """
    if asof_today is None:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        asof_today = datetime.now(ZoneInfo("America/New_York")).date()

    conn = store.conn

    # Account snapshots — last 90 days. Bound the window so a years-old
    # backfill doesn't dominate the prompt.
    cutoff = asof_today - timedelta(days=90)
    rows = conn.execute(
        """
        SELECT asof_date, equity, cash, position_count
        FROM account_snapshots
        WHERE asof_date >= ?
        ORDER BY asof_date
        """,
        [cutoff],
    ).fetchall()
    snapshots: list[dict[str, Any]] | None = None
    if rows:
        snapshots = [
            {
                "date": r[0],
                "equity": float(r[1] or 0),
                "cash": float(r[2] or 0),
                "position_count": int(r[3] or 0),
            }
            for r in rows
        ]

    # Net paper_fills positions (BUY - SELL by ticker) — sanity-check vs
    # account_snapshots.position_count to surface phantom-sentinel-style
    # divergences.
    paper_fills_net: int | None = None
    try:
        net_rows = conn.execute(
            """
            SELECT ticker, SUM(CASE WHEN side='BUY' THEN filled_shares
                                    ELSE -filled_shares END) AS n
            FROM paper_fills GROUP BY ticker HAVING n > 0
            """,
        ).fetchall()
        paper_fills_net = len(net_rows)
    except Exception:
        paper_fills_net = None

    # SPY return over the same window as snapshots, for the alpha calc.
    spy: dict[str, float] | None = None
    if snapshots:
        try:
            spy_rows = conn.execute(
                """
                SELECT date, adj_close FROM prices
                WHERE ticker = 'SPY' AND date BETWEEN ? AND ?
                ORDER BY date
                """,
                [snapshots[0]["date"], snapshots[-1]["date"]],
            ).fetchall()
            if len(spy_rows) >= 2:
                spy_start, spy_end = float(spy_rows[0][1]), float(spy_rows[-1][1])
                spy = {"return_pct": ((spy_end - spy_start) / spy_start) * 100}
        except Exception:
            spy = None

    return _build_perf_context(
        snapshots=snapshots,
        paper_fills_net_count=paper_fills_net,
        spy_returns=spy,
    )
