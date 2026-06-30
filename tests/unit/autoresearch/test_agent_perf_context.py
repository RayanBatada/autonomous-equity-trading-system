"""Tests for the autoresearch agent's live-perf context + cached prompt blocks.

The proposer is more useful when it knows the strategy's *actual* live gap
vs SPY (paper-trading is currently -1.78% vs SPY +2.48% as of 5/18). And
the cache-control block split is what cuts iteration-2+ cost ~80%.
"""

from __future__ import annotations

from datetime import date

from sma.autoresearch.agent import (
    _build_perf_context,
    build_prompt,
    build_prompt_blocks,
    compute_live_perf_context,
)
from sma.ingest.store import Store

# ---- _build_perf_context ----------------------------------------------------


def test_build_perf_context_handles_empty_snapshots():
    out = _build_perf_context(snapshots=None, paper_fills_net_count=None, spy_returns=None)
    assert "no live paper-trading snapshots yet" in out


def test_build_perf_context_alpha_positive():
    snaps = [
        {"date": date(2026, 5, 1), "equity": 100_000, "cash": 50_000, "position_count": 10},
        {"date": date(2026, 5, 10), "equity": 105_000, "cash": 40_000, "position_count": 12},
    ]
    out = _build_perf_context(
        snapshots=snaps, paper_fills_net_count=12, spy_returns={"return_pct": 3.0},
    )
    # Portfolio +5%, SPY +3% → alpha +2pp
    assert "+5.00%" in out
    assert "SPY benchmark same period: +3.00%" in out
    assert "+2.00pp" in out
    assert "beating" in out


def test_build_perf_context_alpha_negative_triggers_priority_callout():
    """Empirical 5/18 state: portfolio -1.78%, SPY +2.48%, alpha -4.26pp.
    The PRIORITY callout is what nudges the agent toward gap-closing edits."""
    snaps = [
        {"date": date(2026, 4, 30), "equity": 100_000, "cash": 50_000, "position_count": 10},
        {"date": date(2026, 5, 18), "equity": 98_220, "cash": 76_000, "position_count": 3},
    ]
    out = _build_perf_context(
        snapshots=snaps, paper_fills_net_count=3, spy_returns={"return_pct": 2.48},
    )
    assert "-1.78%" in out
    assert "+2.48%" in out
    assert "-4.26pp" in out
    assert "BEHIND" in out
    assert "PRIORITY" in out  # the strategy-gap nudge fires when behind by >1pp


def test_build_perf_context_no_priority_when_small_underperformance():
    """A small lag (0.5pp behind) shouldn't trigger the PRIORITY callout
    — the agent shouldn't over-react to noise."""
    snaps = [
        {"date": date(2026, 5, 1), "equity": 100_000, "cash": 50_000, "position_count": 10},
        {"date": date(2026, 5, 10), "equity": 100_500, "cash": 49_000, "position_count": 11},
    ]
    out = _build_perf_context(
        snapshots=snaps, paper_fills_net_count=11, spy_returns={"return_pct": 1.0},
    )
    assert "PRIORITY" not in out  # 0.5pp behind, under the 1.0pp threshold


# ---- build_prompt_blocks (caching) ------------------------------------------


def test_build_prompt_blocks_returns_two_blocks_with_cache_on_first():
    """The stable prefix gets cache_control; the variable suffix doesn't.
    This is what makes iteration 2+ cheap — the cache key is the prefix
    text, which doesn't change across iterations within a single run."""
    blocks = build_prompt_blocks(
        current_active_py="def tilt(): pass\n",
        recent_experiments=[],
        perf_context="(test perf)",
    )
    assert len(blocks) == 2
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in blocks[1]
    # Stable block contains constraints + current active.py + perf context.
    assert "Constraints" in blocks[0]["text"]
    assert "def tilt(): pass" in blocks[0]["text"]
    assert "(test perf)" in blocks[0]["text"]
    # Variable block contains the experiment log + the propose instruction.
    assert "Recent experiment log" in blocks[1]["text"]
    assert "Propose ONE focused edit" in blocks[1]["text"]


def test_build_prompt_blocks_stable_part_invariant_across_iterations():
    """Different recent_experiments → variable block changes, but the
    cached stable block stays byte-identical so the cache hits."""
    common = dict(
        current_active_py="def tilt(): pass\n",
        perf_context="(test perf)",
    )
    blocks_iter0 = build_prompt_blocks(recent_experiments=[], **common)
    blocks_iter1 = build_prompt_blocks(
        recent_experiments=[{
            "iter_index": 0, "monotonicity_score": 2, "sharpe_overall": 1.1,
            "proposal_summary": "top-5 only",
        }],
        **common,
    )
    assert blocks_iter0[0]["text"] == blocks_iter1[0]["text"], (
        "stable block must be byte-identical across iterations or cache "
        "invalidates and we pay full input price on every iter"
    )
    assert blocks_iter0[1]["text"] != blocks_iter1[1]["text"]


def test_build_prompt_blocks_handles_failed_experiment_with_null_scores():
    """Regression (2026-06-01): a prior FAILED experiment has
    monotonicity_score=None and sharpe_overall=None. build_prompt_blocks
    formatted mono with f"{mono:>4}", and f"{None:>4}" raises
    `TypeError: unsupported format string passed to NoneType.__format__`.

    Because list_recent() feeds failed experiments back into the next run's
    prompt, ONE failed experiment poisoned every subsequent autoresearch run
    — 0 of 30 experiments ever succeeded. The mono cell must render a
    placeholder for null scores instead of crashing.
    """
    blocks = build_prompt_blocks(
        current_active_py="def tilt(): pass\n",
        recent_experiments=[{
            "iter_index": 0, "monotonicity_score": None, "sharpe_overall": None,
            "proposal_summary": "",
        }],
        perf_context="(test perf)",
    )
    log = blocks[1]["text"]
    assert "Recent experiment log" in log
    # The failed row renders with a placeholder in both score columns.
    assert "| - |" in log or "|    - |" in log or " - " in log


def test_build_prompt_blocks_shows_experiment_status():
    """The proposer must SEE which prior experiments failed vs succeeded —
    otherwise a wall of infra-failures (null scores, empty summaries) looks
    identical to 'nothing tried', giving the agent no signal. Status belongs
    in the prompt so the agent doesn't mistake an infra crash for a bad idea.
    """
    blocks = build_prompt_blocks(
        current_active_py="def tilt(): pass\n",
        recent_experiments=[
            {"iter_index": 0, "monotonicity_score": None, "sharpe_overall": None,
             "proposal_summary": "", "status": "agent_error"},
            {"iter_index": 1, "monotonicity_score": 2.0, "sharpe_overall": 1.1,
             "proposal_summary": "top-5 only", "status": "ok"},
        ],
        perf_context="(test perf)",
    )
    log = blocks[1]["text"]
    assert "agent_error" in log  # failed run is visibly marked
    assert "ok" in log           # successful run distinguishable


def test_prompt_documents_strategydecision_contract():
    """The proposer must be told StrategyDecision's real fields — it is a frozen
    dataclass with exactly (asof_date, ticker, target_weight). Without this the
    agent invented `signal_metadata`, which crashed every eval. The prompt must
    list the fields AND show how to make a modified copy (dataclasses.replace).
    """
    blocks = build_prompt_blocks(
        current_active_py="def tilt(*, asof_date, decisions, ctx): return decisions\n",
        recent_experiments=[],
        perf_context="(x)",
    )
    header = blocks[0]["text"]
    assert "StrategyDecision" in header
    assert "asof_date" in header and "ticker" in header and "target_weight" in header
    assert "dataclasses.replace" in header  # the safe way to change target_weight


def test_build_prompt_legacy_string_still_works():
    """build_prompt is the old single-string form, kept so existing tests
    that pinned the string interface don't have to migrate immediately."""
    out = build_prompt(
        current_active_py="def tilt(): pass\n",
        recent_experiments=[],
        perf_context="(test perf)",
    )
    assert isinstance(out, str)
    assert "def tilt(): pass" in out
    assert "(test perf)" in out
    assert "Propose ONE focused edit" in out


# ---- compute_live_perf_context (end-to-end on a tiny DB) -------------------


def test_load_anthropic_api_key_prefers_env_var(monkeypatch):
    """Process env beats .env when both are set — keeps tests + manual runs
    that export the key directly from being shadowed by a stale .env."""
    from sma.autoresearch.agent import _load_anthropic_api_key
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-from-env")
    assert _load_anthropic_api_key() == "sk-test-from-env"


def test_load_anthropic_api_key_returns_empty_when_missing(monkeypatch, tmp_path):
    """Both env AND .env missing → returns empty (caller raises a helpful error).
    Empirical 5/25 failure mode: launchd job had no env var and no .env in cwd."""
    from sma.autoresearch.agent import _load_anthropic_api_key
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Move cwd to tmp where no .env exists so Secrets() can't find one.
    monkeypatch.chdir(tmp_path)
    # Also clear any inherited .env discovery — pydantic-settings searches cwd.
    assert _load_anthropic_api_key() == ""


def test_compute_live_perf_context_on_empty_db(tmp_path):
    store = Store(path=str(tmp_path / "t.duckdb")).connect()
    try:
        out = compute_live_perf_context(store, asof_today=date(2026, 5, 18))
    finally:
        store.close()
    assert "no live paper-trading snapshots yet" in out


def test_compute_live_perf_context_with_snapshots(tmp_path):
    """End-to-end: seed account_snapshots + paper_fills + SPY prices,
    verify the formatted output includes alpha + position-count sanity."""
    store = Store(path=str(tmp_path / "t.duckdb")).connect()
    try:
        rid = store.allocate_run_id()
        # Seed 2 snapshots: $100k → $102k over 10 days
        for d, equity, cash, pos in [
            (date(2026, 5, 1), 100_000, 50_000, 10),
            (date(2026, 5, 10), 102_000, 50_000, 12),
        ]:
            store.conn.execute(
                "INSERT INTO account_snapshots "
                "(asof_date, equity, cash, buying_power, long_market_value, "
                " position_count, total_unrealized_pnl, run_id) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, ?)",
                [d, equity, cash, equity, equity - cash, pos, rid],
            )
        # Seed SPY prices flat at $500 (0% return → portfolio alpha = +2pp)
        for d in [date(2026, 5, 1), date(2026, 5, 10)]:
            store.conn.execute(
                "INSERT INTO prices "
                "(ticker, date, open, high, low, close, adj_close, volume, "
                " source, run_id) "
                "VALUES ('SPY', ?, 500, 500, 500, 500, 500, 1000000, 'yfinance', ?)",
                [d, rid],
            )
        out = compute_live_perf_context(store, asof_today=date(2026, 5, 18))
    finally:
        store.close()
    assert "+2.00%" in out  # portfolio
    assert "SPY benchmark same period: +0.00%" in out
    assert "+2.00pp" in out
    assert "beating" in out
