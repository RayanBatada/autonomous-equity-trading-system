"""Self-healing catch-up: backfill trading days missed while offline (6/1-6/3 freeze)."""
from datetime import date, timedelta

from sma.ingest.__main__ import _missing_price_dates


def _weekday_sessions(start: date, end: date) -> list[date]:
    """Weekdays start..end as a stand-in for trading sessions (no holidays)."""
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_returns_gap_oldest_first_excluding_asof_and_weekends():
    sessions = _weekday_sessions(date(2026, 5, 29), date(2026, 6, 4))
    # last priced Fri 5/29; asof Thu 6/4 → backfill Mon/Tue/Wed (6/1-6/3); 6/4 done normally.
    got = _missing_price_dates(last_priced=date(2026, 5, 29), asof=date(2026, 6, 4), sessions=sessions)  # noqa: E501
    assert got == [date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3)]


def test_empty_when_already_current():
    sessions = _weekday_sessions(date(2026, 6, 1), date(2026, 6, 4))
    got = _missing_price_dates(last_priced=date(2026, 6, 3), asof=date(2026, 6, 4), sessions=sessions)  # noqa: E501
    assert got == []


def test_cold_db_does_not_backfill_history():
    assert _missing_price_dates(last_priced=None, asof=date(2026, 6, 4), sessions=[date(2026, 6, 4)]) == []  # noqa: E501


def test_caps_at_max_back_keeping_most_recent_oldest_first():
    sessions = _weekday_sessions(date(2026, 5, 1), date(2026, 6, 4))
    got = _missing_price_dates(last_priced=date(2026, 5, 1), asof=date(2026, 6, 4), sessions=sessions, max_back=3)  # noqa: E501
    assert len(got) == 3
    assert got == sorted(got)
    assert got[-1] == date(2026, 6, 3)


def test_catch_up_calls_ingest_run_for_each_missing_day(monkeypatch):
    """_catch_up_missing_prices backfills each gap day oldest-first via ingest_run."""
    from unittest.mock import MagicMock

    from sma.ingest import __main__ as m

    # last priced date = Fri 5/29
    fake_con = MagicMock()
    fake_con.execute.return_value.fetchone.return_value = (date(2026, 5, 29),)
    monkeypatch.setattr(m, "duckdb", MagicMock(connect=MagicMock(return_value=fake_con)), raising=False)  # noqa: E501
    import duckdb as _real_duckdb  # ensure the name exists to patch the import target
    monkeypatch.setattr(_real_duckdb, "connect", MagicMock(return_value=fake_con))

    fake_alpaca = MagicMock()
    fake_alpaca.sessions_between.return_value = _weekday_sessions(date(2026, 5, 29), date(2026, 6, 4))  # noqa: E501
    import sma.live.alpaca_client as ac
    monkeypatch.setattr(ac.AlpacaClient, "paper_from_env", classmethod(lambda cls, **kw: fake_alpaca))  # noqa: E501

    calls = []
    monkeypatch.setattr(m, "ingest_run", lambda **kw: calls.append(kw["asof"]))

    settings = MagicMock()
    settings.secrets.alpaca_api_key = "k"
    settings.secrets.alpaca_api_secret = "s"

    done = m._catch_up_missing_prices(
        db="data/sma.duckdb", asof=date(2026, 6, 4),
        source_objs=[], universe_list=[], settings=settings,
    )
    assert done == 3
    assert calls == [date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3)]
