"""STEP 2 — DECLINER-INCLUSION survivorship test (NO model retrain).

Robust feature-IC method (the one that worked cleanly for the look-ahead arm).
Over data-driven REVERSAL asofs (classify_regimes), at each asof compute the
cross-sectional rank-IC of the core MOMENTUM features vs realized 30d-fwd
returns for TWO cross-sections:

  (i)  SURVIVOR-only: the bot's equities that were ACTUAL S&P members as-of the
       asof (look-ahead-clipped survivor cross-section).
  (ii) WITH-DECLINERS: (i) PLUS the recovered yfinance decliners that were S&P
       members as-of that asof (added only while genuinely in-index =
       look-ahead-free).

DELTA = with-decliners IC - survivor IC. A MORE-NEGATIVE with-decliners
reversal-IC => the survivor universe UNDERSTATES the momentum inversion (the
crash names crashed hardest and are missing) = a hidden momentum-crash tail in
the live ~92%-deployed book.

ALSO: directly characterize the 29 decliners — in their own in-index reversal
windows, did high-past-momentum decliners crash?

Reads the DURABLE de-biased copy (.sma-pit/sma-pit.duckdb). Never prod.
"""
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "/Users/youruser/.sma-pit")
from membership import _norm, members_asof  # noqa: E402
from shared_pit import make_memcon  # noqa: E402

from sma.eval.regime import classify_regimes  # noqa: E402
from sma.features.builder import build_features  # noqa: E402
from sma.ingest.universe import load_universe  # noqa: E402

DB = "/Users/youruser/.sma-pit/sma-pit.duckdb"
SENTINEL_RUN_ID = 9_000_000_000_000_001

ETFS = {"SPY", "NANC", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK",
        "XLP", "XLRE", "XLU", "XLV", "XLY"}

MOM = ["ret_60d", "rel_strength_spy_60d", "vol_adj_mom_60d", "ret_20d", "rsi_14"]

con = make_memcon(db_path=DB)

# obtained decliners = the sentinel-tagged tickers, read from the on-disk copy
import duckdb  # noqa: E402

_disk = duckdb.connect(DB, read_only=True)
DECLINERS = [r[0] for r in _disk.execute(
    "SELECT DISTINCT ticker FROM prices WHERE run_id=? ORDER BY ticker",
    [SENTINEL_RUN_ID]).fetchall()]
_disk.close()
print(f"recovered decliners ({len(DECLINERS)}): {DECLINERS}")

start = date(2018, 6, 1)
end = con.execute("SELECT MAX(date) FROM prices").fetchone()[0]
n_asofs = 48
sessions = [r[0] for r in con.execute(
    "SELECT DISTINCT date FROM prices WHERE date BETWEEN ? AND ? ORDER BY date",
    [start, end - timedelta(days=35)]).fetchall()]
asofs = sessions[::max(1, len(sessions) // n_asofs)][:n_asofs]

universe_all = load_universe(
    "/Users/youruser/code/stock-market-predictor-agents/src/sma/universe.yaml")
equities = [t for t in universe_all if t not in ETFS]
# universe for FEATURE building must include decliners so they get features
feat_universe = equities + [t for t in DECLINERS if t not in equities]
# classify_regimes uses the SURVIVOR equity universe (the bot's regime view)
regime_universe = equities

prices = con.execute(
    "SELECT * FROM prices WHERE date BETWEEN ? AND ?",
    [start - timedelta(days=420), end]).df()
prices["date"] = pd.to_datetime(prices["date"]).dt.date


def realized_30d(asof):
    fwd = con.execute(
        "SELECT DISTINCT date FROM prices WHERE date > ? ORDER BY date LIMIT 31",
        [asof]).fetchall()
    if len(fwd) < 31:
        return {}
    e, f = fwd[0][0], fwd[30][0]
    rows = con.execute("""
        SELECT ticker, MAX(CASE WHEN date=? THEN open*adj_close/NULLIF(close,0) END),
               MAX(CASE WHEN date=? THEN adj_close END)
        FROM prices WHERE source='yfinance' AND date IN (?,?) GROUP BY ticker
    """, [e, f, e, f]).fetchall()
    return {t: (fp / ep - 1) for t, ep, fp in rows if ep and fp and ep > 0}


regimes = classify_regimes(con, asofs, regime_universe)
rev_asofs = [a for a in asofs if regimes.get(a, ("",))[0] == "reversal"]
trend_asofs = [a for a in asofs if regimes.get(a, ("",))[0] == "trend"]
print(f"regime split: {len(trend_asofs)} trend, {len(rev_asofs)} reversal "
      f"(of {len(asofs)} asofs)")

# accumulators: per-feature lists of per-asof IC, for survivor and with-decliners
def fresh():
    return {f: [] for f in MOM}

acc = {"survivor": fresh(), "withdec": fresh()}
per_asof = []  # (asof, n_surv, n_added_decliners)

# decliner self-characterization pools (only at reversal asofs, in-index)
dec_pool_mom = []   # past ret_60d
dec_pool_fwd = []   # realized 30d fwd

for asof in rev_asofs:
    feats = build_features(prices, feat_universe, asof)
    if feats.empty:
        continue
    real = realized_30d(asof)
    members = members_asof(asof)

    surv = [t for t in feats.index
            if t in real and t in equities and _norm(t) in members]
    dec_in = [t for t in feats.index
              if t in real and t in DECLINERS and _norm(t) in members]
    withdec = surv + [t for t in dec_in if t not in surv]
    per_asof.append((asof, len(surv), len(dec_in)))
    if len(surv) < 20:
        continue

    for tag, common in (("survivor", surv), ("withdec", withdec)):
        rv = np.array([real[t] for t in common])
        for f in MOM:
            fv = feats.loc[common, f].to_numpy(dtype=float)
            ok = ~np.isnan(fv)
            if ok.sum() < 20 or np.std(fv[ok]) < 1e-12:
                continue
            ic = spearmanr(fv[ok], rv[ok]).correlation
            if not np.isnan(ic):
                acc[tag][f].append(ic)

    # decliner self-characterization: past momentum vs fwd return for in-index decliners
    for t in dec_in:
        m = feats.loc[t, "ret_60d"]
        if not np.isnan(m):
            dec_pool_mom.append(float(m))
            dec_pool_fwd.append(float(real[t]))


def mean(b, f):
    return float(np.mean(b[f])) if b[f] else float("nan")


print("\n=== per-reversal-asof cross-section sizes ===")
print(f"{'asof':12} {'n_survivor':>10} {'n_decliners_added':>18}")
tot_added = 0
for a, ns, nd in per_asof:
    print(f"{str(a):12} {ns:10d} {nd:18d}")
    tot_added += nd
print(f"total decliner-observations added across asofs: {tot_added}")

print("\n=== MOMENTUM-FEATURE REVERSAL rank-IC: survivor vs with-decliners ===")
print(f"{'feature':24} {'survivor':>10} {'withdec':>10} {'delta':>10} {'n_asof':>7}")
deltas = []
surv_means, with_means = [], []
for f in MOM:
    s = mean(acc["survivor"], f)
    w = mean(acc["withdec"], f)
    d = w - s
    deltas.append(d)
    surv_means.append(s)
    with_means.append(w)
    print(f"{f:24} {s:+10.4f} {w:+10.4f} {d:+10.4f} {len(acc['survivor'][f]):7d}")

surv_avg = float(np.nanmean(surv_means))
with_avg = float(np.nanmean(with_means))
print("\nAVG over 5 momentum features:")
print(f"  survivor reversal-IC      = {surv_avg:+.4f}")
print(f"  with-decliners reversal-IC= {with_avg:+.4f}")
print(f"  DELTA (with - survivor)   = {with_avg - surv_avg:+.4f}")
print("  (negative delta => survivor UNDERSTATES the inversion = hidden tail)")

# decliner self-characterization
print("\n=== DECLINER SELF-CHARACTERIZATION (in-index reversal windows) ===")
print(f"pooled decliner observations: {len(dec_pool_mom)}")
if len(dec_pool_mom) >= 10:
    rho = spearmanr(dec_pool_mom, dec_pool_fwd).correlation
    print(f"Spearman(past ret_60d, realized 30d-fwd) across decliners = {rho:+.4f}")
    print("  (negative => high-past-momentum decliners then CRASHED = adverse names)")
    # high-momentum decliner subset mean fwd return
    arr_m = np.array(dec_pool_mom); arr_f = np.array(dec_pool_fwd)
    hi = arr_m >= np.median(arr_m)
    print(f"  high-momentum decliners mean 30d-fwd = {arr_f[hi].mean():+.4f} "
          f"(n={hi.sum()})")
    print(f"  low-momentum  decliners mean 30d-fwd = {arr_f[~hi].mean():+.4f} "
          f"(n={(~hi).sum()})")

con.close()
