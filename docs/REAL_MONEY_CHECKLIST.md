# Going from paper to a $50 real account

Written 2026-08-27. Every regulatory claim below was checked against a primary
source on that date and is cited. Two things people "know" about small trading
accounts changed in 2026 and are now wrong — read §2 before you plan around
anything you remember.

**The paper bot keeps running unchanged.** Nothing in this document alters it.
Every capability here ships behind a config flag whose default reproduces
today's paper behaviour bit for bit (proved in
`tests/unit/live/test_sizing_scale.py::test_real_decide_output_is_bit_identical_at_the_current_config`,
which replays the real 2026-08-27 decide output through both the old and new
code). A real-money account is a *second* account, not a migration.

---

## 1. What $50 is actually for

It is not for returns. At $50, one year at the paper book's pace is a few
dollars, which is inside the noise of a single day's price move. What $50 buys
is the only thing paper cannot give you:

- **Real fills.** Paper does not model market impact, latency slippage, or
  partial fills. Every fill you have ever seen from this bot is a simulation.
- **Real fees.** Paper charges no regulatory fees at all. At $50 they are
  invisible in dollars and enormous in percentage — see §5.
- **Real operational failure modes.** Auth against a different endpoint, keys
  that are not interchangeable, an order type the live venue rejects that paper
  happily accepted.

Judge the experiment on "did every order I intended actually get filled at a
price near the one I sized against, and did the ledger match the broker", not
on P&L. Plan on three months before the answer means anything.

---

## 2. Two rules you probably still believe, that no longer exist

**The pattern-day-trader rule is gone.** FINRA Regulatory Notice 26-10,
effective **2026-06-04**, replaced the day-trading margin regime "in their
entirety", including day-trade counting and the $25,000 pattern-day-trader
minimum equity requirement (SEC accelerated approval SR-FINRA-2025-017,
2026-04-14). Alpaca implemented it the same day. There is no day-trade count and
no $25k floor. The `pattern_day_trader`, `daytrade_count` and
`daytrading_buying_power` fields were removed from Alpaca's API by 2026-07-06 —
this repo references none of them, so nothing here breaks.

It would not have mattered anyway: `live.rails.min_hold_days = 7` blocks any
full exit inside 7 calendar days of entry, and a day trade required buying and
selling the *same security on the same day*. Buying at the open on day D and
selling on D+7 was never a day trade under the old rule either.

**Good-faith violations cannot happen here.** Alpaca opens **no cash accounts**
— "we do not offer cash accounts, all accounts are set up as margin accounts".
Below $2,000 equity your account is a *limited* margin account: 1x buying power,
no shorting, but **trading on unsettled funds is explicitly allowed**. A GFV is
a cash-account concept (buying with unsettled proceeds, then selling before they
settle) and therefore cannot occur. Settlement is T+1 and does not gate you.

Two caveats worth knowing rather than assuming:
- Alpaca publishes **no GFV policy at all**. The widely-repeated "3 GFVs in 12
  months → 90-day settled-cash restriction" is another broker's convention, not
  a FINRA rule and not Alpaca's. Do not code against it.
- Unsettled funds still cannot be **withdrawn** or used to buy **crypto**.

---

## 3. The gates, and what each one is for

Real orders require **three independent affirmative facts**. One typo, one bad
merge, one config copied to another machine cannot get past them.

```yaml
live:
  real_money:
    enabled: true            # route to api.alpaca.markets
    real_money_ack: true     # a human said yes, on purpose
    max_real_equity: 60.0    # refuse to trade real money above this
    dry_run: false           # true = log would-be orders, submit nothing
```

`max_real_equity` is the important one. It is not a per-trade risk limit — it
caps **the size of the experiment**. Set it just above your deposit ($60 for a
$50 account). If the account is ever larger than that, the job refuses to trade
and says so, rather than running your strategy at a size you never authorised.
Raising it is a deliberate act; never raise it to make an error message go away.

Also set, for a $50 account:

```yaml
live:
  sizing:
    fractional_shares: true
    min_order_notional: 1.0
```

Without `fractional_shares`, a $50 account buys **nothing at all**. Measured, not
guessed — the real 2026-08-27 decide output (13 names after rails, weights
5.7%–10.0%) re-translated at three account sizes:

| equity | whole shares | fractional | fractional + $1 floor | gross deployed |
|---|---|---|---|---|
| $50 | **0 of 13** | 10 of 13 | 10 of 13 | 0% → 94.5% |
| $500 | 3 of 13 | 10 of 13 | 10 of 13 | 21.4% → 94.5% |
| $5,000 | 9 of 13 | 10 of 13 | 10 of 13 | 74.2% → 94.5% |

Two things to read off that table.

**Fractional makes the book scale-invariant.** The same 10 names and the same
94.5% gross at $50 as at $5,000 — and as at the live $118k. That is the point:
the strategy stops depending on the account size.

**The effective k at $50 is 10, not 15, and that is the cash floor, not
sizing.** 13 names at an average 8.7% target is ~113% gross, which is more than
the account has; `rails.cash_floor_pct = 0.05` stops funding at 94.5% and the
lowest-conviction slots go unfunded. That happens identically at every account
size, so it is a property of the strategy's gross target, not of being small.

**Every slot at $50 clears Alpaca's $1 minimum.** The smallest order the $50 book
places is **$3.16**, against a smallest slot of $2.86. So `min_order_notional:
1.0` changes nothing at $50 — it is there for the account that drifts down, or
for a larger `k`, where sub-$1 slots would otherwise become one broker rejection
per name per night.

---

## 4. The step-by-step

**Step 0 — leave the paper bot alone.** Do not repoint the existing jobs. The
$50 account is separate: separate keys, separate config, separate DB, separate
launchd labels. If the experiment goes wrong you must be able to kill it without
touching the thing that has been compounding.

**Step 1 — open and fund the live account.** Alpaca live keys are **not
interchangeable** with paper keys; a paper key against `api.alpaca.markets`
fails authentication rather than quietly trading the wrong book. Put the live
keys somewhere separate from `.env`.

**Step 2 — dry run first, for a full week.** Set `enabled: true`,
`real_money_ack: true`, `dry_run: true`. The job authenticates against the live
endpoint, reads the real account, sizes real orders, logs exactly what it would
submit, and submits nothing. Compare that log against what the paper bot did the
same night. They should differ only in quantity.

**Step 3 — verify the sizing warning is quiet.** With `fractional_shares: true`
the preflight sizing check should report no unfillable names. If it names any,
the account is too small for the configured `k` — either raise `k`'s floor
weight or accept a smaller book, and write down which.

**Step 4 — VERIFY THIS ONE THING BEFORE YOU GO LIVE.** ⚠️ **Open question, not
answerable from the docs.** This bot submits at 18:35 ET, after the close. DAY
orders submitted after hours are documented as *queueing* to the next session.
But Alpaca's fractional-trading page also says fractional shares "can only be
bought or sold with market orders during normal market hours". Whether a
fractional DAY market order submitted at 18:35 **queues** or is **rejected** is
not stated anywhere I could find.

Test it in **paper** first: enable `fractional_shares` on the paper account for
one evening and check the next morning whether the fractional orders queued and
filled, or came back rejected. If they are rejected, the options are (a) submit
fractional orders during market hours instead of the evening, or (b) use
fractional **limit** DAY orders with `extended_hours=true`, which the docs do
support. Do not skip this step; it is the single most likely reason a $50 live
account silently does nothing.

**Step 5 — flip `dry_run: false`.** One day. Then read §6 before day two.

---

## 5. Fees, and why they matter at $50 and not at $118k

Alpaca charges no commission on equities and passes these through at cost:

| Fee | Side | Rate (as of 2026-08-27) |
|---|---|---|
| SEC Section 31 | **sells only** | $20.60 per $1,000,000 of principal (eff. 2026-04-04) |
| FINRA TAF | **sells only** | $0.000195 per share, capped at $9.79 per trade (2026) |
| FINRA CAT | **both** | $0.000003 per share |

A $10,000 sell of a $50 stock costs about **$0.25**, roughly a quarter of a
basis point. That is genuinely negligible — at $118k.

At $50 the arithmetic changes character. Fees scale with notional, but the
**spread you cross** does not scale with your account, and a $3 order pays the
same half-spread percentage as a $3,000 one. With 15 names rebalancing on a
30-day signal you are crossing the spread a few dozen times a year on positions
of a few dollars. Expect execution costs to dominate everything the model does.
That is the finding, and it is worth paying $50 to measure precisely.

Two traps:
- **The Section 31 rate is not stable.** $27.80/M in May 2024, **$0.00** for most
  of FY2025, $20.60/M since April 2026, expiring 60 days after the FY2027
  appropriation. Never bake it into a study's conclusion; pass the rate that
  applied on the trade date.
- **There is no $0.01 TAF minimum.** The opposite exists — a de-minimis waiver
  when the execution price is below the per-share rate. Alpaca aggregates each
  fee type per account per **day** before rounding up to the cent, so charging a
  penny per order materially overstates the cost of a many-small-orders
  strategy, which is exactly what a small fractional account is.

The backtest can model all of this: `FeeSchedule.real_money_2026()` in
`sma.backtest.fees`, passed as `simulate(fees=...)`. The default is all-zero
because paper charges nothing and the paper track record must stay comparable.

---

## 6. What to watch in the first week

Every morning, in this order:

1. **Did every intended order fill?** `intended_orders` vs `paper_fills` for the
   date. A fractional order that silently did not queue is the failure mode from
   step 4 and it looks like "nothing happened".
2. **Ledger vs broker book.** The reconcile drift alert compares
   `SUM(paper_fills.filled_shares)` against Alpaca's positions. Under fractional
   sizing this is the number that catches a truncation bug anywhere in the
   chain. It should be silent. If it fires on a tiny fraction, something is
   rounding that should not be.
3. **Fill price vs the price you sized against.** `intended_orders.last_price`
   is the prior close; `paper_fills.fill_price` is the real open. The gap is
   your real slippage — the number this whole experiment exists to measure.
4. **Fees actually charged.** `paper_fills.fees` and `.commission`. Compare
   against `FeeSchedule.real_money_2026()` on the same shares and price. If they
   disagree, the rate table is stale.
5. **Buying power.** Below $2,000 you have 1x buying power and no margin. If
   `buying_power` ever exceeds `cash`, something is wrong with your
   understanding of the account, not with the number.

Do not look at P&L. At $50 it is noise, and looking at it will make you change
something for the wrong reason.

---

## 7. Honest expectations

- **$50 will not make money in any meaningful sense.** A great year is a few
  dollars, and execution costs will plausibly exceed the model's edge outright.
  That is not a failure of the experiment; it *is* the experiment.
- **The paper book's track record does not transfer.** It is gross of fees,
  slippage, market impact, dividends and borrow — Alpaca states plainly that
  paper models none of them. Treat every paper number as an upper bound.
- **The most likely outcome is an operational discovery**, not a financial one:
  an order type that does not queue, a fraction that a column truncates, a
  reconcile mismatch. Those are worth $50. Finding them at $50,000 would not be.
- **Scale up only on evidence.** The thing that should raise `max_real_equity`
  is a month of clean fills and a measured slippage number you can put in a
  sentence — not a good month.

---

## 8. Mirroring signals manually

2026-08-31: every night after decide submits (or decides not to submit)
orders on the paper book, it pushes a summary to your phone via ntfy
(`notify.trade_pushes: true`, the default — set it `false` in `config.yaml`
to silence it without touching anything else). This is a read-only side
channel: it fires after orders are already submitted, and turning it off or
on never changes what the bot itself trades.

**Reading the push.** Each line is percent of equity, never a share count,
so the same message is correct at $50 or $50,000 — but read carefully,
because two different lines mean two different percentages (fixed
2026-09-03; before that date, partial-order lines under-reported an order
this badly: a 1-share trim was pushed as "5.7% of equity" when the actual
order was ~1.0%):

```
SMA trades — Fri 8/28
2026-08-28
SELL MRNA — all (was 14.2% of book)
BUY ENPH — 6.3% of equity
Equity $123,456 | 9 positions
```

- `BUY TICKER — W% of equity`: **this order's own size.** On your own
  account, buy `W% × your equity` worth of TICKER (e.g. 6.3% of a $50
  account is ~$3.15). True whether TICKER is a brand-new position or a
  top-up of one you already hold — `W%` is always what THIS trade moves,
  never the resulting position's total weight.
- `SELL TICKER — all (was W% of book)`: the model dropped TICKER entirely.
  Sell your whole position, not W% of it — `W%` is only telling you how big
  the position *was*, as a sanity check that you're closing the right one.
- `SELL TICKER — W% of equity`: **this order's own size**, same as BUY — a
  partial trim, not a full exit. Sell `W% × your equity` worth of TICKER.
  `W%` is NOT the resulting position's new target weight; it's how much of
  this trim to make.
- Zero orders reads as `No trades tonight — book unchanged (N holds)` — do
  nothing.

**Timing caveat.** The push describes orders queued for the **next** market
open, not fills that already happened — decide runs at 18:35 ET, hours after
the close, sizing everything off the prior close (`intended_orders.last_price`
is a prior-close estimate, not a live quote). By the time you place a mirror
order the next morning, the price has already moved overnight and the target
weight you're sizing against is stale by however much the stock gapped. This
is the same estimate-vs-fill gap decide's own dry-run output has always had;
mirroring by hand just adds a second, human-timed instance of it.

**This is not the bot's execution — it will be worse.** Manually mirroring
adds slippage the bot's own fills don't pay: the bot submits within
milliseconds of computing the target and gets Alpaca's queued DAY-market
fill near the open; you're reading a phone notification, doing arithmetic,
and placing an order whenever you actually get to a broker app, which could
be minutes to hours after the open — an eternity for a volatile name. Rails
like `min_hold_days` and the rebalance dead zone are also enforced by the
bot's own book (this account's actual held shares and entry dates), not by
your account, so a mirrored trade can accidentally recreate the exact
same-day-round-trip churn those rails exist to prevent. Treat the push as a
directional signal to act on with your own judgment and timing, not a fill
you're guaranteed to replicate.

---

## Sources

- FINRA Regulatory Notice 26-10 (PDT retirement, eff. 2026-06-04); SEC
  SR-FINRA-2025-017 accelerated approval 2026-04-14.
- Alpaca, "FINRA retires the PDT rule — introducing Alpaca's new intraday margin
  framework" (implementation + API field removal by 2026-07-06).
- Alpaca support, "Alpaca cash accounts" (no cash accounts offered);
  docs.alpaca.markets "Account plans" (all margin; <$2,000 = limited margin, 1x
  buying power, unsettled-funds trading permitted).
- docs.alpaca.markets "Placing Orders" (fractional TIF matrix, page updated
  2026-08-10): DAY only; GTC/IOC/FOK/OPG/CLS rejected. Note the older
  `postorder` API reference contradicts this and is stale.
- docs.alpaca.markets "Fractional trading": 9 decimal places; `fractionable`
  flag on the Asset model; supported on paper and live; the market-hours
  sentence that step 4 exists to resolve.
- Alpaca support, "Can we submit orders smaller than $1 in notional value":
  $1 minimum, scoped to **buy entry orders**.
- docs.alpaca.markets "Close a Position" (`DELETE /v2/positions/{symbol}`): full
  liquidation without computing a quantity; `percentage` only sells fractional
  if the position was originally fractional.
- Alpaca Brokerage Fee Schedule, rev. 2026-07-20 (SEC 31, TAF, CAT rates;
  per-day per-fee-type aggregation then round up to the cent).
- FINRA By-Laws Schedule A §1 (TAF $0.000195 / $9.79 cap, valid 2026-01-01 to
  2026-12-31, set by SR-FINRA-2024-019).
- docs.alpaca.markets "Paper trading": paper does not model market impact,
  latency slippage, regulatory fees, or dividends.
- docs.alpaca.markets "Authentication": live and paper credentials are not
  interchangeable.

### Deliberately unresolved

- Minimum fractional **quantity**: undocumented. Not assumed anywhere in code.
- Whether Alpaca truncates or rejects beyond 9 decimals: undocumented. This repo
  truncates client-side to 9 rather than find out.
- Whether Alpaca's parser accepts scientific-notation floats (any qty below
  1e-4 serializes as e.g. `1e-09` through `json.dumps`): untested. Another
  reason `min_order_notional` should be ≥ $1.
- **Whether a fractional DAY market order submitted at 18:35 queues or is
  rejected** — step 4. This is the one that decides whether the $50 experiment
  can run on the existing evening schedule at all.


## Resolved 2026-08-27 23:40 ET: after-hours fractional orders DO queue

Tested live on the paper account with the market closed: a fractional DAY market BUY of AAPL, notional $1.00, was ACCEPTED (order 68e1cea5, status accepted, notional 1) and cancelled cleanly two seconds later (status canceled). Evening submission of fractional orders is therefore viable; the open question in step 4 is closed. Rerun this one-line test on the real account before the first live rebalance.
