"""yfinance price ingestion.

Bulk-downloads OHLCV for the universe in one call. On bulk failure, falls
back to per-ticker calls so one bad ticker doesn't nuke the whole run.

Price rows are inserted with source='yfinance'. Re-ingestion for the same
(ticker, date) is handled at the runner level via DELETE WHERE run_id, not
in the source.
"""

from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from loguru import logger

from sma.ingest.sources.base import IngestResult
from sma.ingest.store import Store


class YFinancePricesSource:
    name = "yfinance"

    # Lookback must EXCEED the quality gate's 30-day extreme-moves scan:
    # with a 1-day window, a provider-side back-adjustment (KLAC 10:1 split,
    # 2026-06-11) rescales history the nightly fetch never re-syncs, leaving
    # a permanent 10x discontinuity inside the scan that blocks EVERY night
    # for a month. 45 days re-syncs the whole scanned window nightly
    # (same number of yfinance requests; just more upserted rows).
    # 2026-09-18 incident: Yahoo returned the 9/18 bar with NaN closes for ALL
    # 264 tickers (a vendor-side hiccup — the bulk call succeeded and returned
    # a real DataFrame, just with NaN prices), and every row was inserted
    # as-is with status='ok'. If MORE than this fraction of tickers have a
    # NaN close specifically on the ASOF date, the whole source is treated as
    # FAILED for the run (status='error') so enough_sources_succeeded and the
    # run_id-scoped coverage logic see the truth and let the OTHER price
    # source (alpaca) carry the night, instead of quietly trusting worthless
    # data. Rows for tickers that DID come back clean are still inserted
    # (best-effort) -- only the source's overall status/accounting is failed.
    _NAN_ASOF_FAILURE_RATIO = 0.20

    def __init__(self, lookback_days: int = 45):
        self.lookback_days = lookback_days
        self._resync_tickers: set[str] = set()
        # Tickers whose ASOF-date row had a NaN close, tracked across BOTH the
        # bulk and per-ticker-fallback insert paths (both funnel through
        # _insert_single_ticker) -- see _NAN_ASOF_FAILURE_RATIO above.
        self._nan_asof_tickers: set[str] = set()

    def fetch(
        self,
        tickers: list[str],
        asof_date: date,
        store: Store,
        run_id: int,
    ) -> IngestResult:
        start = asof_date - timedelta(days=self.lookback_days)
        end = asof_date + timedelta(days=1)

        rows_inserted = 0
        network_error = False
        self._resync_tickers = set()
        self._nan_asof_tickers = set()
        try:
            df = yf.download(
                tickers,
                start=start.isoformat(),
                end=end.isoformat(),
                progress=False,
                group_by="ticker",
                threads=True,
                auto_adjust=False,
                multi_level_index=False,
            )
            rows_inserted = self._insert_dataframe(tickers, df, store, run_id, asof_date=asof_date)
        except Exception as e:
            logger.warning("bulk yfinance download failed: {}; falling back per-ticker", e)
            per_ticker_failures = 0
            for t in tickers:
                try:
                    sub = yf.download(
                        t,
                        start=start.isoformat(),
                        end=end.isoformat(),
                        progress=False,
                        auto_adjust=False,
                multi_level_index=False,
                    )
                    rows_inserted += self._insert_single_ticker(
                        t, sub, store, run_id, asof_date=asof_date
                    )
                except Exception as inner:
                    per_ticker_failures += 1
                    logger.warning("yfinance per-ticker failed for {}: {}", t, inner)
            # Bulk threw AND every per-ticker fetch also threw → a real network/DNS
            # failure, not just an empty (no-data) response.
            network_error = per_ticker_failures == len(tickers)

        # Split/dividend back-adjustment self-heal: when a provider-side
        # adjustment rescales history (KLAC 10:1 split 2026-06-11 left the
        # stored series 10x off for three weeks, feeding garbage momentum/
        # vol/52w features AND training labels — review 2026-07-01), rows
        # OLDER than the fetch window stay desynced forever. Detection marks
        # drifted tickers and DEFERS their window insert; here (common to BOTH
        # the bulk and per-ticker-fallback paths — review 2026-07-02 H2) each
        # gets a full-history refetch. A refetch failure leaves the old-scale
        # rows in place, so detection re-fires next run, and we PAGE.
        resync_failed = []
        for t in sorted(self._resync_tickers):
            try:
                full = yf.download(
                    t, start="2016-01-01", end=end.isoformat(),
                    progress=False, auto_adjust=False, multi_level_index=False,
                )
                n = self._insert_single_ticker(
                    t, full, store, run_id, detect_drift=False, asof_date=asof_date
                )
                if n == 0:
                    raise RuntimeError("full-history refetch returned no rows")
                rows_inserted += n
                logger.warning(
                    "yfinance adj-drift detected for {}: full-history "
                    "refetch re-synced {} rows", t, n,
                )
            except Exception as inner:
                resync_failed.append(t)
                logger.warning("full-history resync failed for {}: {}", t, inner)
        if resync_failed:
            from sma.ingest.notify import notify_failure
            notify_failure(
                title="SMA ingest: adj-drift resync FAILED",
                message=(
                    f"{', '.join(resync_failed)}: back-adjustment drift detected "
                    f"but full-history refetch failed; their window rows were "
                    f"deferred (stale today) and detection will retry next run."
                ),
            )

        # A network/DNS outage (every fetch threw) previously still returned
        # status="ok", letting enough_sources_succeeded pass on a dead network and
        # masking a total price-ingest failure (the 6/1-6/3 freeze). Report "error"
        # only then — an empty (no-data) response is legitimately "ok".
        network_ok = not (rows_inserted == 0 and network_error)

        # 2026-09-18 incident (see _NAN_ASOF_FAILURE_RATIO docstring): a fetch
        # that "succeeds" but returns NaN closes for most of the universe on
        # the asof date is not tradable data either, even though rows_inserted
        # for the OTHER (clean) tickers may be > 0.
        nan_ratio = (len(self._nan_asof_tickers) / len(tickers)) if tickers else 0.0
        nan_failed = nan_ratio > self._NAN_ASOF_FAILURE_RATIO
        if nan_failed:
            logger.warning(
                "yfinance: NaN close for {} on {}/{} tickers ({:.0%} > {:.0%}); "
                "treating source as FAILED for this run (2026-09-18 incident)",
                asof_date.isoformat(), len(self._nan_asof_tickers), len(tickers),
                nan_ratio, self._NAN_ASOF_FAILURE_RATIO,
            )

        ok = network_ok and not nan_failed
        if ok:
            error = None
        elif not network_ok:
            error = "yfinance inserted 0 rows (all fetches failed — likely network/DNS)"
        else:
            error = (
                f"yfinance: NaN close for {asof_date.isoformat()} on "
                f"{len(self._nan_asof_tickers)}/{len(tickers)} tickers "
                f"({nan_ratio:.0%} > {self._NAN_ASOF_FAILURE_RATIO:.0%} threshold); "
                "source treated as FAILED for this run rather than silently trusting "
                "worthless prices (2026-09-18 incident)"
            )
        return IngestResult(
            source=self.name,
            rows_inserted=rows_inserted,
            status="ok" if ok else "error",
            error=error,
        )

    def _insert_dataframe(
        self,
        tickers: list[str],
        df: pd.DataFrame,
        store: Store,
        run_id: int,
        asof_date: date | None = None,
    ) -> int:
        if df.empty:
            return 0
        if len(tickers) == 1:
            return self._insert_single_ticker(tickers[0], df, store, run_id, asof_date=asof_date)
        total = 0
        for t in tickers:
            try:
                # NOTE: dropna(how="all") only removes a row where EVERY field
                # is NaN. The 2026-09-18 incident's rows had a real open/high/
                # low/volume with just close/adj_close NaN, so they survive
                # this filter and reach _insert_single_ticker below, which is
                # where the per-row NaN-close drop + asof-ratio tracking
                # actually happens.
                sub = df[t].dropna(how="all")
            except KeyError:
                continue
            total += self._insert_single_ticker(t, sub, store, run_id, asof_date=asof_date)
        return total

    # 2026-10-01 (MNST 2:1, 2026-08-11): checking only the OLDEST overlap row's
    # adj_close missed a restatement that Yahoo applied unevenly, leaving the
    # stored series with a fake -51% day at 07-17 -> 07-20 for a month. ANY
    # fetched row whose close differs from the stored yfinance close for the
    # same date by more than this is a split signal: refetch full history.
    # Normal revisions (dividend re-adjustment, late prints) are far below it.
    _SPLIT_SIGNAL_CLOSE_TOL = 0.20

    def _detect_adj_drift(self, ticker: str, df: pd.DataFrame, store: Store, cols: dict) -> None:
        """Mark `ticker` for a full-history refetch when the fetch window shows
        the provider restated history. Two signals, both checked BEFORE the
        INSERT OR REPLACE erases the evidence:

          1. the OLDEST fetched row's adj_close drifts >1% from what's stored
             (KLAC 10:1, 2026-06-11: everything older than the window is
             desynced);
          2. ANY fetched row's close differs >_SPLIT_SIGNAL_CLOSE_TOL from the
             stored close for the same date (MNST 2:1, 2026-08-11)."""
        try:
            adj_col = cols.get("adj_close", "Adj Close")
            if adj_col in df.columns:
                sub = df[df[adj_col].notna()]
                if not sub.empty:
                    ts = sub.index.min()
                    d = ts.date() if hasattr(ts, "date") else ts
                    fetched = float(sub.loc[ts, adj_col])
                    row = store.conn.execute(
                        "SELECT adj_close FROM prices WHERE ticker=? AND date=? AND source=?",
                        [ticker, d, self.name],
                    ).fetchone()
                    if row and row[0] and fetched and abs(fetched / row[0] - 1.0) > 0.01:
                        self._resync_tickers.add(ticker)
                        return
            close_col = cols.get("close", "Close")
            if close_col not in df.columns:
                return
            fetched_close = {}
            for ts, v in df[close_col].items():
                if v is None or pd.isna(v) or float(v) <= 0:
                    continue
                fetched_close[ts.date() if hasattr(ts, "date") else ts] = float(v)
            if not fetched_close:
                return
            stored = store.conn.execute(
                "SELECT date, close FROM prices WHERE ticker=? AND source=? "
                "AND date BETWEEN ? AND ? AND close > 0",
                [ticker, self.name, min(fetched_close), max(fetched_close)],
            ).fetchall()
            tol = self._SPLIT_SIGNAL_CLOSE_TOL
            for d, c in stored:
                f = fetched_close.get(d)
                if f is not None and c and abs(f / c - 1.0) > tol:
                    logger.warning(
                        "yfinance split signal for {}: {} close {} vs stored {} "
                        "(>{:.0%}); full-history refetch", ticker, d, round(f, 4),
                        round(c, 4), tol,
                    )
                    self._resync_tickers.add(ticker)
                    return
        except Exception as e:  # detection must never break the ingest
            logger.debug("adj-drift check failed for {}: {}", ticker, e)

    def _insert_single_ticker(
        self,
        ticker: str,
        df: pd.DataFrame,
        store: Store,
        run_id: int,
        detect_drift: bool = True,
        asof_date: date | None = None,
    ) -> int:
        if df is None or df.empty:
            return 0
        # Defensive flatten in case multi_level_index=False isn't honored
        # (older yfinance, group_by behavior, etc). Pick the level that
        # contains known OHLCV field names.
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            known = {"open", "high", "low", "close", "adj close", "volume"}
            level0_hits = sum(
                1 for v in df.columns.get_level_values(0)
                if str(v).lower() in known
            )
            level1_hits = sum(
                1 for v in df.columns.get_level_values(1)
                if str(v).lower() in known
            )
            field_level = 0 if level0_hits >= level1_hits else 1
            df.columns = df.columns.get_level_values(field_level)
        cols = {c.lower().replace(" ", "_"): c for c in df.columns}
        if detect_drift:
            self._detect_adj_drift(ticker, df, store, cols)
            if ticker in self._resync_tickers:
                # DEFER this ticker's window insert: writing it now would
                # REPLACE the very overlap row the drift was detected against,
                # erasing the evidence — if the full-history refetch then
                # failed, the desync became permanent and undetectable (the
                # original KLAC bug, reintroduced; adversarial review
                # 2026-07-02 H1). By skipping, a failed refetch leaves the
                # stored rows on the OLD scale, so tomorrow's detection fires
                # again — durable and self-retrying by construction.
                return 0
        rows = []
        nan_close_dropped = 0
        close_col = cols.get("close", "Close")
        adj_col = cols.get("adj_close", "Adj Close")
        for ts, r in df.iterrows():
            d = ts.date() if hasattr(ts, "date") else ts
            close_val = r.get(close_col, None)
            # 2026-09-18 incident: Yahoo returned a real row (open/high/low/
            # volume present) with close/adj_close NaN for the whole
            # universe. NEVER insert a row whose close is NaN/None -- a NaN
            # close is a data-availability problem for coverage/enough-
            # sources checks to surface, not a tradable (if degenerate)
            # price. DuckDB compares/orders NaN as the largest value, so a
            # stored NaN silently corrupts every downstream >/</ORDER BY
            # comparison (the extreme-move and divergence gates both got
            # fooled this way).
            if close_val is None or pd.isna(close_val):
                nan_close_dropped += 1
                if asof_date is not None and d == asof_date:
                    self._nan_asof_tickers.add(ticker)
                continue
            adj_val = r.get(adj_col, None)
            # If adj_close ALONE is NaN, store NULL -- matching alpaca's
            # existing NULL-placeholder convention for "no adjusted price" --
            # rather than propagating a NaN into the column.
            adj_close_final = None if adj_val is None or pd.isna(adj_val) else float(adj_val)
            rows.append((
                ticker,
                d,
                float(r[cols.get("open", "Open")]) if cols.get("open", "Open") in r else None,
                float(r[cols.get("high", "High")]) if cols.get("high", "High") in r else None,
                float(r[cols.get("low", "Low")]) if cols.get("low", "Low") in r else None,
                float(close_val),
                adj_close_final,
                int(r[cols.get("volume", "Volume")]) if cols.get("volume", "Volume") in r else None,
                self.name,
                run_id,
            ))
        if nan_close_dropped:
            logger.warning(
                "yfinance: dropped {} row(s) with NaN/None close for {} (run_id={})",
                nan_close_dropped, ticker, run_id,
            )
        if not rows:
            return 0
        rows = self._quarantine_cross_source_outliers(ticker, rows, store)
        if not rows:
            return 0
        store.conn.executemany(
            "INSERT OR REPLACE INTO prices "
            "(ticker, date, open, high, low, close, adj_close, volume, source, run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        return len(rows)

    # A vendor row is a QUARANTINE CANDIDATE when its RAW close disagrees
    # >30% with EVERY other source's stored close for the same (ticker,
    # date) — e.g. Yahoo's phantom half-price MNST 2026-07-31 row (close
    # 48.19 vs alpaca 96.38), which froze trading for a night via
    # no_unjustified_extreme_moves. No other-source row for the date = no
    # evidence = keep the row.
    #
    # That alone isn't the corruption signature, though: Yahoo also
    # split-adjusts historical Close, so after a split, yfinance's
    # re-adjusted pre-split rows legitimately diverge from alpaca's raw
    # (never-adjusted, adj_close NULL) rows by the split factor — e.g. CRWD
    # 2026-06-02, yfinance close 192.24 vs alpaca 768.84, ahead of a 4:1
    # split later that month. A full-history refetch after such a split
    # re-adjusts EVERY pre-split row together, and every one of those rows
    # would trip the cross-source check above. Quarantining them anyway
    # silently failed the drift-detector's resync and reintroduced the
    # KLAC-class mixed-scale-history bug it exists to prevent.
    #
    # So a candidate is only actually quarantined when it's ALSO an
    # ISOLATED SPIKE relative to the SAME vendor's own adjacent rows:
    # |close/prev-1| > 30% AND |close/next-1| > 30%. A split shifts every
    # fetched row together, so each row's own vendor neighbors agree with
    # it (nothing isolated, all kept); a phantom row disagrees with its real
    # neighbors on both sides (isolated, quarantined). Neighbors come from
    # this same fetched batch when available; at a window edge — including a
    # single-row batch, which is an edge on both sides — we fall back to the
    # vendor's own already-stored neighbor row in the DB, one side at a
    # time. When only one side has any evidence at all (window or DB), that
    # one side alone decides (AND-ing against a side with no evidence would
    # wrongly demand proof that can't exist). When NEITHER side has
    # evidence, isolation can't be evaluated either way, so the cross-source
    # signal alone decides — unchanged from before this refinement.
    #
    # Documented edge case: two ADJACENT corrupt rows that happen to agree
    # with each other are each the other's non-diverging same-vendor
    # neighbor on one side — the identical signature a legitimate uniform
    # split shift produces — so both are KEPT. This check cannot tell "two
    # corrupt rows that agree with each other" apart from "two re-adjusted
    # rows that agree with each other" using same-vendor adjacency alone,
    # and conservatively favors not breaking a genuine split resync over
    # catching a rarer double-corruption.
    _CROSS_SOURCE_QUARANTINE_TOL = 0.30

    def _vendor_neighbor_close(
        self, store: Store, ticker: str, d: date, *, before: bool
    ) -> float | None:
        """The vendor's (self.name) own already-stored close for the nearest
        date before/after `d`, used as an isolation-check fallback at fetch-
        window edges (including a single-row window) where no in-window
        neighbor exists."""
        op, direction = ("<", "DESC") if before else (">", "ASC")
        row = store.conn.execute(
            f"SELECT close FROM prices WHERE ticker = ? AND source = ? "
            f"AND close IS NOT NULL AND date {op} ? "
            f"ORDER BY date {direction} LIMIT 1",
            [ticker, self.name, d],
        ).fetchone()
        return row[0] if row else None

    def _quarantine_cross_source_outliers(
        self, ticker: str, rows: list[tuple], store: Store
    ) -> list[tuple]:
        dates = [r[1] for r in rows]
        others = store.conn.execute(
            "SELECT date, close FROM prices WHERE ticker = ? AND source <> ? "
            "AND close IS NOT NULL AND date BETWEEN ? AND ?",
            [ticker, self.name, min(dates), max(dates)],
        ).fetchall()
        ref: dict = {}
        for d, c in others:
            ref.setdefault(d, []).append(c)

        tol = self._CROSS_SOURCE_QUARANTINE_TOL
        # Adjacency is same-vendor and date-order, not fetch-batch order.
        order = sorted(range(len(rows)), key=lambda i: rows[i][1])
        drop: set[int] = set()
        for pos, i in enumerate(order):
            r = rows[i]
            d, close = r[1], r[5]
            refs = ref.get(d)
            cross_source_outlier = (
                close
                and refs
                and all(abs(close / rc - 1.0) > tol for rc in refs)
            )
            if not cross_source_outlier:
                continue

            prev_close = (
                rows[order[pos - 1]][5] if pos > 0
                else self._vendor_neighbor_close(store, ticker, d, before=True)
            )
            next_close = (
                rows[order[pos + 1]][5] if pos < len(order) - 1
                else self._vendor_neighbor_close(store, ticker, d, before=False)
            )
            diverges_prev = prev_close is not None and abs(close / prev_close - 1.0) > tol
            diverges_next = next_close is not None and abs(close / next_close - 1.0) > tol
            if prev_close is not None and next_close is not None:
                isolated = diverges_prev and diverges_next
            elif prev_close is not None or next_close is not None:
                isolated = diverges_prev or diverges_next
            else:
                isolated = True  # no vendor-adjacent evidence -> cross-source signal stands

            if not isolated:
                continue

            logger.warning(
                "{}: quarantined corrupt row {} {} close={} vs other-source "
                "close(s) {} (>{:.0%} divergence); row skipped",
                self.name, ticker, d, round(close, 2),
                [round(x, 2) for x in refs],
                tol,
            )
            drop.add(i)

        return [r for i, r in enumerate(rows) if i not in drop]
