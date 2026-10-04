"""scripts/check_public_tree.py refuses real data in a public tree."""

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

from check_public_tree import check  # noqa: E402


def _tree(tmp_path):
    root = tmp_path / "t"
    (root / "tests" / "fixtures").mkdir(parents=True)
    shutil.copy2(REPO / "tests/fixtures/replay_synthetic.duckdb", root / "tests/fixtures")
    (root / "README.md").write_text("hi\n")
    return root


def test_clean_tree_with_the_synthetic_fixture_passes(tmp_path):
    assert check(_tree(tmp_path)) == []


def test_a_real_replay_fixture_fails(tmp_path):
    root = _tree(tmp_path)
    shutil.copy2(REPO / "tests/fixtures/replay_synthetic.duckdb",
                 root / "tests/fixtures/replay_2026_08_28.duckdb")
    problems = check(root)
    assert any("replay_2026_08_28" in p for p in problems)


def test_any_unlisted_duckdb_fails(tmp_path):
    root = _tree(tmp_path)
    (root / "data").mkdir()
    (root / "data" / "copy.duckdb").write_bytes(b"x")
    assert any("not on the allowlist" in p for p in check(root))


def test_a_fake_synthetic_fixture_fails(tmp_path):
    """The allowlisted name alone is not enough: a real DB renamed to
    replay_synthetic.duckdb is caught by its content."""
    import duckdb

    root = _tree(tmp_path)
    f = root / "tests/fixtures/replay_synthetic.duckdb"
    f.unlink()
    con = duckdb.connect(str(f))
    con.execute("CREATE TABLE paper_fills (alpaca_order_id VARCHAR)")
    con.execute("INSERT INTO paper_fills VALUES ('cef674f5-e0f0-4538-966f-665c1f1c8dac')")
    con.execute("CREATE TABLE predictions (model_id VARCHAR)")
    con.execute("INSERT INTO predictions VALUES ('synthetic_model_for_tests')")
    con.close()
    assert any("non-synthetic order ids" in p for p in check(root))


def test_env_values_and_personal_paths_fail(tmp_path):
    root = _tree(tmp_path)
    # Built at runtime so the publish scrub cannot rewrite the test's own input.
    home = "/Users/" + "rayan" + "batada"
    (root / "a.py").write_text(f'KEY = "abcdefgh12345678"\nP = "{home}/x"\n')
    env = tmp_path / ".env"
    env.write_text("FINNHUB_API_KEY=abcdefgh12345678\n")
    problems = check(root, env)
    assert any("FINNHUB_API_KEY" in p for p in problems)
    assert any("personal string" in p for p in problems)
    assert not any("abcdefgh12345678" in p for p in problems)  # never prints the value



def test_venv_and_caches_are_not_scanned(tmp_path):
    root = _tree(tmp_path)
    (root / ".venv" / "lib").mkdir(parents=True)
    (root / ".venv" / "lib" / "x.duckdb").write_bytes(b"x")
    assert check(root) == []
