"""DuckDB connection + schema migrations + run_id allocation.

Migrations are a list of (version, sql) tuples. On connect, the store reads
the current schema version from the `_schema_version` table and applies all
pending migrations in order. Idempotent: safe to call connect() repeatedly.

The `path` arg accepts ":memory:" for tests so we never touch the disk.
"""

import os
from datetime import datetime
from pathlib import Path

import duckdb

import sma.locks as _locks
from sma.db_connect import read_only_connect


class WriterLockNotHeld(Exception):  # noqa: N818
    """Raised when Store.connect(read_only=False) is called without the writer_lock held."""


# Schema migrations. Add new ones to the END of this list. Never edit a
# migration that has been applied to a real database. Write a new one.
MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS _schema_version (
            version BIGINT PRIMARY KEY,
            applied_at TIMESTAMP NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ingest_log (
            run_id        BIGINT      NOT NULL,
            source        VARCHAR     NOT NULL,
            started_at    TIMESTAMP   NOT NULL,
            finished_at   TIMESTAMP,
            rows_inserted INTEGER,
            status        VARCHAR     NOT NULL,
            error         VARCHAR,
            PRIMARY KEY (run_id, source)
        );

        CREATE TABLE IF NOT EXISTS prices (
            ticker    VARCHAR NOT NULL,
            date      DATE    NOT NULL,
            open      DOUBLE,
            high      DOUBLE,
            low       DOUBLE,
            close     DOUBLE,
            adj_close DOUBLE,
            volume    BIGINT,
            source    VARCHAR NOT NULL,
            run_id    BIGINT  NOT NULL,
            PRIMARY KEY (ticker, date, source)
        );

        CREATE TABLE IF NOT EXISTS fundamentals (
            ticker             VARCHAR NOT NULL,
            asof_date          DATE    NOT NULL,
            pe                 DOUBLE,
            pb                 DOUBLE,
            roe                DOUBLE,
            debt_to_equity     DOUBLE,
            profit_margin      DOUBLE,
            revenue_growth_yoy DOUBLE,
            market_cap         DOUBLE,
            source             VARCHAR NOT NULL,
            run_id             BIGINT  NOT NULL,
            PRIMARY KEY (ticker, asof_date, source)
        );

        CREATE TABLE IF NOT EXISTS news (
            ticker       VARCHAR   NOT NULL,
            published_at TIMESTAMP NOT NULL,
            source       VARCHAR   NOT NULL,
            headline     VARCHAR   NOT NULL,
            url          VARCHAR   NOT NULL,
            body_excerpt VARCHAR,
            hash         VARCHAR   NOT NULL,
            run_id       BIGINT    NOT NULL,
            PRIMARY KEY (hash, ticker)
        );

        CREATE TABLE IF NOT EXISTS sentiment (
            ticker       VARCHAR NOT NULL,
            date         DATE    NOT NULL,
            score        DOUBLE  NOT NULL,
            source       VARCHAR NOT NULL,
            num_articles INTEGER NOT NULL,
            run_id       BIGINT  NOT NULL,
            PRIMARY KEY (ticker, date, source)
        );

        CREATE TABLE IF NOT EXISTS filings (
            ticker       VARCHAR   NOT NULL,
            filing_type  VARCHAR   NOT NULL,
            filed_at     TIMESTAMP NOT NULL,
            accession_no VARCHAR   NOT NULL,
            url          VARCHAR   NOT NULL,
            summary      VARCHAR,
            run_id       BIGINT    NOT NULL,
            PRIMARY KEY (accession_no)
        );

        CREATE TABLE IF NOT EXISTS earnings (
            ticker           VARCHAR NOT NULL,
            report_date      DATE    NOT NULL,
            eps_estimate     DOUBLE,
            eps_actual       DOUBLE,
            revenue_estimate DOUBLE,
            revenue_actual   DOUBLE,
            source           VARCHAR NOT NULL,
            run_id           BIGINT  NOT NULL,
            PRIMARY KEY (ticker, report_date)
        );

        CREATE TABLE IF NOT EXISTS market_index (
            symbol VARCHAR NOT NULL,
            date   DATE    NOT NULL,
            close  DOUBLE  NOT NULL,
            source VARCHAR NOT NULL,
            run_id BIGINT  NOT NULL,
            PRIMARY KEY (symbol, date, source)
        );
    """,
    ),
    (
        2,
        """
        CREATE TABLE IF NOT EXISTS predictions (
            asof_date       DATE        NOT NULL,
            ticker          VARCHAR     NOT NULL,
            target          VARCHAR     NOT NULL,
            predicted_value DOUBLE      NOT NULL,
            model_id        VARCHAR     NOT NULL,
            computed_at     TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (asof_date, ticker, target, model_id)
        );
    """,
    ),
    (
        3,
        """
        CREATE TABLE IF NOT EXISTS theses (
            ticker            VARCHAR NOT NULL,
            asof_date         DATE    NOT NULL,
            run_id            BIGINT  NOT NULL,
            -- researcher output
            news_summary      TEXT,
            key_developments  JSON,
            notable_filings   JSON,
            -- analyst output
            bull_case         TEXT,
            bear_case         TEXT,
            asymmetric_risks  JSON,
            catalyst_window   VARCHAR,
            -- strategist output
            conviction        VARCHAR,
            score             DOUBLE,
            flags             JSON,
            action_hint       VARCHAR,
            reasoning         TEXT,
            -- meta
            created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, asof_date, run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_theses_asof ON theses(asof_date);

        CREATE TABLE IF NOT EXISTS agent_calls (
            call_id           UUID PRIMARY KEY DEFAULT uuid(),
            run_id            BIGINT  NOT NULL,
            ticker            VARCHAR NOT NULL,
            asof_date         DATE    NOT NULL,
            agent_role        VARCHAR NOT NULL,
            model_id          VARCHAR NOT NULL,
            input_tokens      INTEGER,
            output_tokens     INTEGER,
            cache_read_tokens INTEGER,
            est_cost_usd      DOUBLE,
            latency_ms        INTEGER,
            status            VARCHAR,
            error             TEXT,
            created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_agent_calls_created ON agent_calls(created_at);
    """,
    ),
    (
        4,
        """
        -- Phase 5 paper-trading audit tables.
        CREATE TABLE IF NOT EXISTS intended_orders (
            intended_order_id   UUID    PRIMARY KEY,
            asof_date           DATE    NOT NULL,
            ticker              VARCHAR NOT NULL,
            side                VARCHAR NOT NULL,           -- 'BUY' | 'SELL'
            target_shares       INTEGER NOT NULL,
            target_weight       DOUBLE,                      -- 0.0-1.0 (NULL for stop-loss sells)
            last_price          DOUBLE,                      -- price used for target_shares
            source              VARCHAR NOT NULL,            -- 'decide' | 'stop-loss'
            alpaca_order_id     VARCHAR,                     -- populated after submission
            status              VARCHAR NOT NULL,            -- 'submitted'/'submission_failed'
            error               VARCHAR,                     -- error msg if submission_failed
            run_id              BIGINT  NOT NULL,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (asof_date, ticker, source)
        );

        CREATE INDEX IF NOT EXISTS idx_intended_orders_asof ON intended_orders(asof_date);

        CREATE TABLE IF NOT EXISTS paper_fills (
            alpaca_order_id     VARCHAR PRIMARY KEY,
            intended_order_id   UUID,                          -- FK to intended_orders
            asof_date           DATE    NOT NULL,
            ticker              VARCHAR NOT NULL,
            side                VARCHAR NOT NULL,
            filled_shares       INTEGER NOT NULL,
            fill_price          DOUBLE  NOT NULL,
            commission          DOUBLE  NOT NULL DEFAULT 0,
            fees                DOUBLE  NOT NULL DEFAULT 0,
            status              VARCHAR NOT NULL,            -- 'filled'/'partial'/'rejected'
            submitted_at        TIMESTAMP NOT NULL,
            filled_at           TIMESTAMP,
            run_id              BIGINT  NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_paper_fills_asof ON paper_fills(asof_date);

        CREATE TABLE IF NOT EXISTS account_snapshots (
            asof_date              DATE    PRIMARY KEY,
            equity                 DOUBLE  NOT NULL,
            cash                   DOUBLE  NOT NULL,
            buying_power           DOUBLE  NOT NULL,
            long_market_value      DOUBLE  NOT NULL,
            position_count         INTEGER NOT NULL,
            total_unrealized_pnl   DOUBLE,
            run_id                 BIGINT  NOT NULL,
            created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """,
    ),
    (
        5,
        """
        -- 2026-05-09: politician trade disclosures (PTRs from House
        -- Clerk + Senate EFD). Sourced from public disclosure feeds.
        CREATE TABLE IF NOT EXISTS politician_trades (
            doc_id              VARCHAR NOT NULL,
            chamber             VARCHAR NOT NULL,
            last_name           VARCHAR NOT NULL,
            first_name          VARCHAR NOT NULL,
            state_dst           VARCHAR,
            filing_date         DATE,
            transaction_date    DATE,
            ticker              VARCHAR,
            asset_description   VARCHAR,
            asset_type          VARCHAR,
            transaction_type    VARCHAR,
            amount_min          DOUBLE,
            amount_max          DOUBLE,
            ingested_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            run_id              BIGINT NOT NULL,
            PRIMARY KEY (doc_id, transaction_date, ticker, asset_description, transaction_type)
        );

        CREATE INDEX IF NOT EXISTS idx_politician_trades_ticker
            ON politician_trades(ticker);
        CREATE INDEX IF NOT EXISTS idx_politician_trades_filing_date
            ON politician_trades(filing_date);
        CREATE INDEX IF NOT EXISTS idx_politician_trades_transaction_date
            ON politician_trades(transaction_date);

        CREATE TABLE IF NOT EXISTS politician_disclosure_docs (
            doc_id              VARCHAR NOT NULL,
            chamber             VARCHAR NOT NULL,
            filing_year         INTEGER NOT NULL,
            filing_date         DATE,
            parse_status        VARCHAR NOT NULL,
            error               VARCHAR,
            transactions_parsed INTEGER NOT NULL DEFAULT 0,
            ingested_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (doc_id, chamber)
        );
    """,
    ),
    (
        6,
        """
        -- Phase 6 autoresearch loop: per-iteration experiment log.
        -- Each run of `python -m sma.autoresearch run` writes one row per
        -- iteration with the proposed `active.py` body's hash, the 5
        -- walk-forward CV sub-window Sharpes, the overall val Sharpe, and
        -- the monotonicity score. Promotion criterion (mono >= 3 AND
        -- overall > baseline + 0.1) is evaluated by a separate CLI that
        -- reads this table.
        CREATE TABLE IF NOT EXISTS autoresearch_experiments (
            experiment_id       UUID    PRIMARY KEY,
            run_id              BIGINT  NOT NULL,
            iter_index          INTEGER NOT NULL,
            proposal_sha        VARCHAR NOT NULL,      -- sha256 of proposed active.py
            proposal_summary    VARCHAR,               -- agent's 1-sentence change desc
            active_py_text      TEXT    NOT NULL,      -- full proposed file content
            baseline_overall    DOUBLE,                -- baseline (identity tilt) val Sharpe
            sharpe_w1           DOUBLE,
            sharpe_w2           DOUBLE,
            sharpe_w3           DOUBLE,
            sharpe_w4           DOUBLE,
            sharpe_w5           DOUBLE,
            sharpe_overall      DOUBLE,
            monotonicity_score  INTEGER,               -- sum of (w_i >= baseline + 0.05)
            promoted            BOOLEAN NOT NULL DEFAULT FALSE,
            status              VARCHAR NOT NULL,      -- ok|agent_error|eval_error|parse_error
            error               VARCHAR,
            agent_cost_usd      DOUBLE,
            duration_seconds    DOUBLE,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_autoresearch_run
            ON autoresearch_experiments(run_id, iter_index);
        CREATE INDEX IF NOT EXISTS idx_autoresearch_mono
            ON autoresearch_experiments(monotonicity_score DESC, sharpe_overall DESC);
    """,
    ),
]


def current_schema_version() -> int:
    return MIGRATIONS[-1][0]


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self.conn: duckdb.DuckDBPyConnection | None = None

    def connect(self, *, read_only: bool = False) -> "Store":
        # read_only=True skips migrations (no write access to schema). Use
        # for strategy / dashboard / analysis code paths that must coexist
        # with another reader on the same DB file — DuckDB requires all
        # connections to one file to share configuration.
        if not read_only:
            pid_path = _locks.DEFAULT_LOCK_PATH.with_suffix(".pid")
            if not pid_path.exists():
                raise WriterLockNotHeld(
                    "Store.connect(read_only=False) requires the writer_lock to be held. "
                    "Wrap the call in `with writer_lock(label=...): ...`"
                )
            try:
                holder_line = pid_path.read_text().strip()
                holder_pid = int(holder_line.split()[0])
            except (OSError, ValueError, IndexError) as exc:
                raise WriterLockNotHeld(f"writer_lock pid file is malformed: {exc}") from exc
            if holder_pid != os.getpid():
                raise WriterLockNotHeld(
                    f"writer_lock held by PID {holder_pid}, current PID {os.getpid()}; "
                    f"cannot open Store writable. Use writer_lock(label=...) to acquire first."
                )
        if read_only:
            # Lock-tolerant: a raw read-only open crashes while another process
            # holds the writer lock; read_only_connect retries with backoff.
            self.conn = read_only_connect(self.path)
        else:
            self.conn = duckdb.connect(self.path, read_only=read_only)
            self._apply_migrations()
        return self

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def schema_version(self) -> int:
        assert self.conn is not None
        row = self.conn.execute("SELECT MAX(version) FROM _schema_version").fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def _apply_migrations(self) -> None:
        assert self.conn is not None
        # Bootstrap _schema_version if missing
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS _schema_version (
                version BIGINT PRIMARY KEY,
                applied_at TIMESTAMP NOT NULL
            )
        """)
        applied = {
            r[0] for r in self.conn.execute("SELECT version FROM _schema_version").fetchall()
        }
        for version, sql in MIGRATIONS:
            if version in applied:
                continue
            self.conn.execute(sql)
            self.conn.execute(
                "INSERT INTO _schema_version VALUES (?, ?)",
                [version, datetime.utcnow()],
            )

    def allocate_run_id(self) -> int:
        """Monotonically increasing run id, derived from current epoch microseconds.

        Microseconds give us monotonic ordering even for back-to-back calls in
        tests, and avoids needing a sequence (DuckDB has them, but this is
        simpler).
        """
        assert self.conn is not None
        candidate = int(datetime.utcnow().timestamp() * 1_000_000)
        row = self.conn.execute("SELECT MAX(run_id) FROM ingest_log").fetchone()
        existing_max = int(row[0]) if row and row[0] is not None else 0
        return max(candidate, existing_max + 1)

    def log_run_start(self, run_id: int, source: str) -> None:
        assert self.conn is not None
        self.conn.execute(
            "INSERT INTO ingest_log (run_id, source, started_at, status) VALUES (?, ?, ?, ?)",
            [run_id, source, datetime.utcnow(), "running"],
        )

    def log_run_end(
        self,
        run_id: int,
        source: str,
        rows_inserted: int,
        status: str,
        error: str | None,
    ) -> None:
        assert self.conn is not None
        self.conn.execute(
            "UPDATE ingest_log "
            "SET finished_at = ?, rows_inserted = ?, status = ?, error = ? "
            "WHERE run_id = ? AND source = ?",
            [datetime.utcnow(), rows_inserted, status, error, run_id, source],
        )
