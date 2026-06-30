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
