

def test_startup_recovery_restores_stranded_active_py(tmp_path, monkeypatch):
    """SIGKILL between _SwappedActivePy.__enter__/__exit__ leaves the LLM
    proposal live. recover_stale_active_py() must restore the backup at
    autoresearch startup."""
    from sma.autoresearch import loop as ar_loop

    active = tmp_path / "active.py"
    active.write_text("PROPOSAL = True\n")
    bak = tmp_path / "active.py.autoresearch_bak"
    bak.write_text("ORIGINAL = True\n")
    monkeypatch.setattr(ar_loop, "ACTIVE_PY_PATH", active)

    restored = ar_loop.recover_stale_active_py()
    assert restored is True
    assert "ORIGINAL" in active.read_text()
    assert not bak.exists()

    # idempotent no-op when no backup is stranded
    assert ar_loop.recover_stale_active_py() is False
