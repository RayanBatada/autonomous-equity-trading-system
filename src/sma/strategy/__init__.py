"""Strategy module — agent-editable surfaces for Phase 6 autoresearch.

Per [[2026-04-28-phase-6-autoresearch]], the autoresearch loop edits
`active.py` only. Other modules in this package are infrastructure
(loop runner, agent wrapper, experiment log, monotonicity gate) that
the agent must NOT modify.
"""
