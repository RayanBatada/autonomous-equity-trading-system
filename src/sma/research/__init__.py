"""On-demand single-ticker research: full analysis for any ticker.

CLI: `python -m sma.research ticker NVDA [--asof DATE] [--refresh] [--output FILE]`

Read-only by default. With --refresh, runs the agents thesis pipeline so a
fresh LLM thesis is available even if the nightly run hasn't seen the ticker.
"""
