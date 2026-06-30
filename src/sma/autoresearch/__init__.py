"""Phase 6 autoresearch loop.

LLM-proposed edits to `src/sma/strategy/active.py:tilt()` → walk-forward
CV evaluation → experiment log → human-reviewed promotion.

Modules:
  - agent.py      : Anthropic API wrapper that proposes new `tilt()` bodies
  - loop.py       : iteration orchestrator (propose → write → eval → log → revert)
  - experiment_log: read/write helpers for the `autoresearch_experiments` table

CLI: `python -m sma.autoresearch run --iterations 10`
     `python -m sma.autoresearch top --k 5`
"""
