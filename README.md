# Stock Market Predictor

[![tests](https://github.com/RayanBatada/autonomous-equity-trading-system/actions/workflows/test.yml/badge.svg)](https://github.com/RayanBatada/autonomous-equity-trading-system/actions/workflows/test.yml)

This is a long-only US stock trading system I built and run as research. Every weekday evening it
ranks 264 large and mid-cap stocks by expected 30-day return, builds a book of about ten of them
under risk rails, and sends orders that fill at the next morning's open. It trades an Alpaca
**paper** account (fake money) and has since 30 April 2026; it has never traded real money.

This repository is the public copy of my private working repo, refreshed by hand: one commit per
refresh, my machine's paths replaced with `/Users/youruser`, and internal session notes left out.

## How a night works

Everything runs unattended on my Mac from launchd jobs. Times are US Eastern; the schedule is
defined once in `src/sma/schedule.py` and the plists in `ops/launchd/` are rendered from it.

| Time | Job | What it does |
|---|---|---|
| 18:30 | `sma.ingest run` | Prices (yfinance, Alpaca), news, earnings, filings and politician trades into DuckDB, then quality checks. A blocking failure stops the night. |
| 19:30 | `sma.model predict` | The current XGBoost model scores all 264 names. Refuses if ingest failed. |
| 19:45 | `sma.agents run` | Claude Haiku writes a thesis per held or triggered name (researcher, analyst, strategist). A bearish thesis can veto a buy. Advisory: if it fails, trading goes on without it. |
| 20:00 | `sma.live decide` | Builds the target book and submits DAY market orders for the next open. |
| 16:30 next day | `sma.live reconcile` | Matches fills to intended orders, flags drift, writes the equity snapshot. |
| Mon 04:00 | `sma.model train` | Weekly retrain; the new model serves only if its cross-validated rank IC clears a floor. |
| 10 times a day, mostly evenings | `sma.watchdog` | Re-kicks a job that missed its deadline, including an ingest that failed. |

The book is the model's ranking under these rails: top 15 candidates, at most 10% of
equity per name, at most 35% per sector, a 5% cash floor, a 7-day minimum hold, a held name is
kept until it falls past rank 30, no buys within 3 days of earnings, and buys stop if equity
falls 15% below its peak. The stop-loss sweep exists and is set to 0 (off), because stops lost
money in every backtest window I tried.

## What is true about it today

As of 4 October 2026, from a read-only copy of the live database
(full numbers in my diagnosis note, not in this repo):

- **No edge has been shown.** Rank IC at the model's own 30-day horizon was -0.056 (standard
  error 0.029) over the walk-forward year before going live, and about zero live (-0.02 at 30
  days, standard error 0.05). The model's score runs slightly against plain 60-day momentum.
- **The returns are mostly market exposure.** From 30 April to 1 October the paper account went
  from $100,309 to $116,602 (+16.2%) while SPY gained 6.3%. Beta to SPY was 1.66 and the daily
  alpha was +8 bp with a t-statistic of 0.39, which is noise. One trade (MRNA, $15,649) is most of
  the $20,109 realized; without it the book trailed what its beta alone would have earned.
- **Costs are about four times what the backtests assume.** Paper fills cost about 19 bp per side
  against the day's official open (median and notional-weighted, 361 fills), against the 5 bp the
  backtester charges. The bot trades on 14 to 16 of about 21 nights a month and holds a name a
  median of 6 trading days, on a signal built for 30.
- **It misses nights.** 7 of the 18 trading nights from 9 September to 2 October did not
  rebalance, almost all because the Mac was asleep or offline at 18:30. The gates refuse to trade
  on bad data, which is the right failure, but it is still a missed night.
- **The thesis layer is off right now.** The Anthropic API credit ran out on 30 September; the
  agents job pages every night it fails and the bot trades on the model alone.

So I treat this as a measurement system with a trading bot attached, not as a strategy that works.
Every change to what it decides goes through a pre-registered study first; six candidate
strategies tested in September all failed theirs.

## Tests

```bash
uv run pytest -n 4 -m "not serial"   # the main suite, in parallel
uv run pytest -m serial              # one disk-heavy test that must run alone
```

On 4 October: 2,293 passed and 5 skipped on CI's Ubuntu runner, plus the serial test; 2,295 passed
on my Mac, where two more tests can read the local database.
CI runs `ruff check .` and both commands on every push to `main`.

What they prove: the pipeline's logic does what it says on synthetic and recorded inputs (quality
gates, sentinels and retries, order sizing and the rails, reconcile, the watchdog, backtest
look-ahead rules), and the live decide path produces byte-identical orders to recorded nights.
What they do not prove: that the model makes money, or that a real broker fills like the paper one.
No test calls a real API.

## Running it yourself

You need Python 3.11, [uv](https://docs.astral.sh/uv/), and your own free Alpaca paper account
plus Finnhub, NewsAPI and Anthropic keys. Without Alpaca keys nothing can trade.

```bash
uv sync --all-extras
cp .env.example .env        # then fill in your keys
uv run pytest -n 4 -m "not serial"
```

The tests need no keys: CI sets placeholders (see `.github/workflows/test.yml`). The jobs are
`python -m sma.<job>`, each with `--help`. A fresh clone has no database or trained model, and I
have not tested a from-scratch bootstrap; the history backfills I used are in `scripts/`.
`ops/launchd/README.md` covers installing the schedule on a Mac.

## Layout

- `src/sma/`: the system. `ingest/` data and quality checks, `features/`, `model/` training and
  the promotion gate, `agents/` the thesis pipeline, `live/` decide, reconcile and the broker
  client, `risk/` the rails, `backtest/` the simulator (shares the decide code with live),
  `watchdog.py` and `schedule.py`.
- `dashboard/`: a read-only Streamlit view of the book, the model and the jobs.
- `ops/`: launchd plists and systemd units rendered from the schedule.
- `scripts/`: one-off studies and backfills, kept as the record of what was tried. Nothing
  scheduled runs from here.
- `tests/`: unit and integration tests.
- `docs/REAL_MONEY_CHECKLIST.md`: what would have to be true before any real money, with sources.
- `docs/PUBLISHING.md`: how the public copy is refreshed and what is kept out of it.

## Disclaimer

This is a personal research project on paper money. Nothing here is investment advice, and
nothing here shows that the strategy works.
