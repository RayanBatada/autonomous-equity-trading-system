from datetime import date
from pathlib import Path

import pytest
import yaml

from sma.ingest.universe import (
    load_universe,
    load_universe_membership,
    universe_asof,
)


def _write_map_universe(tmp_path: Path) -> Path:
    """Universe in the dated-map format (ticker -> added date)."""
    p = tmp_path / "universe.yaml"
    p.write_text(
        "universe:\n"
        "  refresh_policy: manual\n"
        "  tickers:\n"
        "    AAPL: 2026-04-25\n"
        "    MSFT: 2026-04-25\n"
        "    NANC: 2026-05-09\n"
    )
    return p


def test_load_universe_handles_dated_map_format(tmp_path: Path):
    # load_universe must still return the plain ticker list for the new map form.
    assert load_universe(_write_map_universe(tmp_path)) == ["AAPL", "MSFT", "NANC"]


def test_load_universe_membership_returns_dates(tmp_path: Path):
    m = load_universe_membership(_write_map_universe(tmp_path))
    assert m["AAPL"] == date(2026, 4, 25)
    assert m["NANC"] == date(2026, 5, 9)


def test_universe_asof_excludes_tickers_added_later(tmp_path: Path):
    p = _write_map_universe(tmp_path)
    # 2026-05-01: NANC (added 5-09) not yet in the universe.
    assert universe_asof(p, date(2026, 5, 1)) == ["AAPL", "MSFT"]
    # 2026-05-09: NANC now present (added on this day, inclusive).
    assert universe_asof(p, date(2026, 5, 9)) == ["AAPL", "MSFT", "NANC"]
    # before birth: empty.
    assert universe_asof(p, date(2026, 1, 1)) == []


def test_universe_asof_list_format_treats_all_as_present(tmp_path: Path):
    # Back-compat: a legacy list-format universe (no dates) → every ticker is
    # present at any asof (added date unknown, assume always-in).
    p = tmp_path / "universe.yaml"
    p.write_text(yaml.safe_dump({"universe": {"tickers": ["AAPL", "MSFT"]}}))
    assert universe_asof(p, date(2020, 1, 1)) == ["AAPL", "MSFT"]


def test_load_universe_returns_ticker_list(tmp_path: Path):
    p = tmp_path / "universe.yaml"
    p.write_text(yaml.safe_dump({
        "universe": {
            "refresh_policy": "manual",
            "tickers": ["AAPL", "MSFT", "TSLA"],
        }
    }))
    tickers = load_universe(p)
    assert tickers == ["AAPL", "MSFT", "TSLA"]


def test_load_universe_dedups_and_sorts_for_determinism(tmp_path: Path):
    p = tmp_path / "universe.yaml"
    p.write_text(yaml.safe_dump({
        "universe": {
            "refresh_policy": "manual",
            "tickers": ["TSLA", "AAPL", "AAPL", "MSFT"],
        }
    }))
    assert load_universe(p) == ["AAPL", "MSFT", "TSLA"]


def test_load_universe_uppercases(tmp_path: Path):
    p = tmp_path / "universe.yaml"
    p.write_text(yaml.safe_dump({
        "universe": {"refresh_policy": "manual", "tickers": ["aapl", "msft"]},
    }))
    assert load_universe(p) == ["AAPL", "MSFT"]


def test_load_universe_raises_on_empty(tmp_path: Path):
    p = tmp_path / "universe.yaml"
    p.write_text(yaml.safe_dump({
        "universe": {"refresh_policy": "manual", "tickers": []},
    }))
    with pytest.raises(ValueError):
        load_universe(p)


def test_seed_universe_has_at_least_50_tickers():
    seed = Path(__file__).resolve().parents[2] / "src" / "sma" / "universe.yaml"
    tickers = load_universe(seed)
    assert len(tickers) >= 50, f"seed universe only has {len(tickers)} tickers"


# --- PIT training membership (survivorship fix, 2026-07-20) -------------------


def _write_history(tmp_path: Path) -> Path:
    p = tmp_path / "universe_history.yaml"
    p.write_text(
        "former_tickers:\n"
        "  BBBY:\n"
        "    added: 2016-01-04\n"
        "    removed: 2023-04-23\n"
        "    note: bankrupt\n"
        "  ATVI:\n"
        "    added: null\n"
        "    removed: 2023-10-13\n"
        "    note: acquired by MSFT\n"
    )
    return p


def test_load_training_membership_parses_intervals(tmp_path: Path):
    from sma.ingest.universe import load_training_membership

    m = load_training_membership(_write_history(tmp_path))
    assert m["BBBY"] == (date(2016, 1, 4), date(2023, 4, 23))
    assert m["ATVI"] == (None, date(2023, 10, 13))  # added unknown = since data start


def test_load_training_membership_empty_file_is_empty(tmp_path: Path):
    from sma.ingest.universe import load_training_membership

    p = tmp_path / "universe_history.yaml"
    p.write_text("former_tickers: {}\n")
    assert load_training_membership(p) == {}


def test_load_training_membership_missing_file_is_empty(tmp_path: Path):
    """A repo without the history file (or a fresh checkout) trains exactly as
    before — PIT is additive, never load-bearing."""
    from sma.ingest.universe import load_training_membership

    assert load_training_membership(tmp_path / "nope.yaml") == {}


# --- PIT training membership for CURRENT members (activated 2026-08-17) -------
# universe.yaml's `added` dates are operational April-2026 file dates, so
# without a membership map training treats today's universe as having always
# existed. universe_history.yaml's `members:` section carries the real
# point-in-time S&P 500 windows; `former_tickers:` still carries names that
# left. Both sections are read and merged.


def test_load_training_membership_reads_members_section(tmp_path: Path):
    from sma.ingest.universe import load_training_membership

    p = tmp_path / "universe_history.yaml"
    p.write_text(
        "members:\n"
        "  PLTR: {added: 2024-09-30}\n"
        "  AAPL: {added: null}\n"
    )
    m = load_training_membership(p)
    assert m["PLTR"] == (date(2024, 9, 30), None)
    assert m["AAPL"] == (None, None)


def test_load_training_membership_merges_both_sections(tmp_path: Path):
    from sma.ingest.universe import load_training_membership

    p = tmp_path / "universe_history.yaml"
    p.write_text(
        "members:\n"
        "  PLTR: {added: 2024-09-30}\n"
        "former_tickers:\n"
        "  BBBY:\n"
        "    added: 2016-01-04\n"
        "    removed: 2023-04-23\n"
    )
    m = load_training_membership(p)
    assert m == {
        "PLTR": (date(2024, 9, 30), None),
        "BBBY": (date(2016, 1, 4), date(2023, 4, 23)),
    }


def test_load_training_membership_reads_removed_from_members_section(tmp_path: Path):
    """A `members:` entry can carry BOTH added and removed -- for a name that
    was a real universe member and then genuinely left (delisted), not just a
    `former_tickers:` name. The two sections share one merged map, so this
    must work identically to former_tickers (2026-09-01, AVB/EA delisting
    fix -- see test_shipped_history_has_sane_windows for the real file)."""
    from sma.ingest.universe import load_training_membership

    p = tmp_path / "universe_history.yaml"
    p.write_text(
        "members:\n"
        "  AVB: {added: null, removed: 2026-08-15}\n"
        "  AAPL: {added: null}\n"
    )
    m = load_training_membership(p)
    assert m["AVB"] == (None, date(2026, 8, 15))
    assert m["AAPL"] == (None, None)


def _shipped_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2] / "src" / "sma"
    return root / "universe.yaml", root / "universe_history.yaml"


def test_shipped_history_covers_every_universe_ticker():
    """Every tradable name needs a window, or it silently keeps its
    survivorship-biased full history while its neighbours are restricted."""
    from sma.ingest.universe import load_training_membership, load_universe

    uni_path, hist_path = _shipped_paths()
    universe = load_universe(uni_path)
    membership = load_training_membership(hist_path)
    missing = [t for t in universe if t not in membership]
    assert not missing, f"universe_history.yaml is missing {missing}"


# Delisted `members:` entries that legitimately carry a `removed` date
# (2026-09-01): both stopped trading for real corporate-action reasons and
# were pulled from the active universe.yaml in 92474dc. Dates are the day
# after each one's real last trading day, per
# scripts/verify_avb_ea_delisting.py.
_DELISTED_REMOVED = {
    "AVB": date(2026, 8, 15),  # last trade 2026-08-14, NYSE halt/delist 2026-08-17
    "EA": date(2026, 8, 5),    # last trade 2026-08-04, Nasdaq delisting 2026-08-04
}


def test_shipped_history_has_sane_windows():
    """Bounds stay inside the snapshot span, and no `members:` entry carries a
    `removed` date except the two known real delistings (AVB, EA) -- every
    other entry is a CURRENT universe member and none of them has left OUR
    universe (ENPH left the S&P 500 in 2025-08 but we still trade it)."""
    from sma.ingest.universe import load_training_membership, load_universe

    uni_path, hist_path = _shipped_paths()
    universe = set(load_universe(uni_path))
    membership = load_training_membership(hist_path)
    snap_start, snap_end = date(2017, 12, 31), date(2026, 7, 31)
    for sym, (added, removed) in membership.items():
        if sym in _DELISTED_REMOVED:
            assert removed == _DELISTED_REMOVED[sym], f"{sym} removed date drifted"
            assert sym not in universe, f"{sym} delisted but still in universe.yaml"
            continue
        assert removed is None, f"{sym} has an unexpected removed={removed}"
        if sym not in universe:
            continue
        if added is not None:
            assert snap_start <= added <= snap_end, f"{sym} added={added} out of span"


def test_shipped_history_spot_checks():
    """Five hand-verified windows: a long-standing member, two post-2020
    additions, a 2023+ addition, and the symbol-change case. Wrong dates here
    silently delete (or invent) years of training rows."""
    from sma.ingest.universe import load_training_membership

    m = load_training_membership(_shipped_paths()[1])
    # Long-standing: in the S&P 500 since long before the first snapshot.
    assert m["AAPL"] == (None, None)
    # Tesla joined 2020-12-21 -> first monthly snapshot that sees it.
    assert m["TSLA"] == (date(2020, 12, 31), None)
    # Palantir joined 2024-09-23.
    assert m["PLTR"] == (date(2024, 9, 30), None)
    # Uber joined 2023-12-18 — a 2023+ addition, where the study measured the
    # survivorship premium at +2.73pp.
    assert m["UBER"] == (date(2023, 12, 31), None)
    # Meta: FB -> META in 2022-06 was a SYMBOL change, not an index addition.
    # The naive first-seen date would have deleted 4.5 years of real rows.
    assert m["META"] == (None, None)
    # Enphase joined 2021-01, left the S&P 500 in 2025-08, still traded by us.
    assert m["ENPH"] == (date(2021, 1, 31), None)


def test_shipped_history_undatable_tickers_fall_back_to_project_start():
    """Names that were never S&P members (and the benchmark/sector ETFs) can't
    be dated from the snapshots. They fall back to the training start, which is
    a no-op at the default 2018-01-01 rather than a silent full exclusion."""
    from sma.ingest.universe import load_training_membership

    m = load_training_membership(_shipped_paths()[1])
    for sym in ("SPY", "XLK", "ARM", "RIVN", "SNOW"):
        assert m[sym] == (date(2018, 1, 1), None), f"{sym} fallback changed"
