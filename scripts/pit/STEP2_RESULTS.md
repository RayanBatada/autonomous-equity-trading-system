# DECLINER-INCLUSION survivorship test — results (2026-06-26)

Durable artifacts in `/Users/youruser/.sma-pit/` (NOT /tmp — survives sessions):
- `sma-pit.duckdb`     — prod prices COPY + 29 yfinance-backfilled decliners (run_id=9000000000000001)
- `sp500_pit.csv`      — PIT S&P 500 membership (members-per-date, fja05680 snapshot)
- `membership.py`, `shared_pit.py` — staged harness (paths repointed to .sma-pit)
- `fetch_decliners.py` (STEP 1), `step2_decliner_inclusion.py` (STEP 2), logs
Prod DB verified UNCHANGED (sha256 672a6c0b… before == after).

## STEP 1 — backfill
Candidate adverse set = 42 named market-cap-decline + distress omissions from
pit-membership-findings.md §B/C. STI EXCLUDED by design (reused ticker).
yfinance availability filter -> 29 obtainable (2130 rows each, 2018-01-02..2026-06-24):
AAL BBWI COTY CPRI DXC ETSY HOG HP ILMN KSS LNC LUMN M NKTR NOV NWL PCG PENN
PRGO QRVO RRC SEDG SIG UAA UNM VFC WU XRX ZION
Dropped (yfinance empty / <500 rows, ticker retired via later acquisition/rename):
GPS JWN HBI FL CHK HFC CMA DISH SRCL ADS SIVB FRC (empty), SBNY (465 rows).
auto_adjust=False (close != adj_close), source='yfinance'.

## STEP 2 — feature-IC test (NO model retrain)
48 monthly asofs 2018-06+, data-driven classify_regimes -> 23 reversal / 25 trend.
20 reversal asofs had >=20 survivors and 31 fwd sessions => contributed.
Survivor = bot equities that were S&P members as-of asof (look-ahead-CLIPPED).
With-decliners = survivor PLUS the 29 decliners that were members as-of asof
(13-21 added per asof early, tapering to 1 by 2025; 262 decliner-obs total).

MOMENTUM-FEATURE REVERSAL rank-IC (avg over 20 asofs):
  feature                 survivor   withdec    delta
  ret_60d                  -0.2782   -0.2678   +0.0104
  rel_strength_spy_60d     -0.2782   -0.2678   +0.0104
  vol_adj_mom_60d          -0.2775   -0.2673   +0.0102
  ret_20d                  -0.1625   -0.1520   +0.0105
  rsi_14                   -0.2061   -0.1945   +0.0116
  AVG (5 mom features)     -0.2405   -0.2299   +0.0106

DELTA (with - survivor) = +0.0106  (POSITIVE = inclusion WEAKENS the inversion).

## DECLINER self-characterization (their own in-index reversal windows, n=262)
Spearman(past ret_60d, realized 30d-fwd) across decliners = -0.3017.
  high-past-momentum decliners mean 30d-fwd = -0.0161 (n=131)  <- they crashed
  low-past-momentum  decliners mean 30d-fwd = +0.0761 (n=131)
=> the recovered names ARE genuine momentum-crash/adverse names.

## VERDICT
The decliners are unambiguously adverse momentum-crash names (own reversal-IC
-0.30). BUT adding them back does NOT make the survivor momentum reversal-IC
more negative — the delta is small and POSITIVE (+0.0106), robust across all 5
features. The survivor universe does NOT understate the momentum inversion; if
anything decliner inclusion marginally softens it. NO hidden incremental
momentum-crash tail is revealed by the look-ahead-free decliner-inclusion test.
Mechanism: the look-ahead-clipped survivor cross-section ALREADY inverts hard
(-0.24 to -0.28); the crash names slot in at the low-momentum end where that
cross-section already predicts the reversal, so they don't strengthen the IC.
Combined with the prior look-ahead arm (clip moved IC by <0.02), survivorship
in BOTH measurable faces (look-ahead injection + decliner omission) does NOT
manufacture or hide the momentum-reversal inversion — the inversion is real and
already fully expressed on the survivor universe.
