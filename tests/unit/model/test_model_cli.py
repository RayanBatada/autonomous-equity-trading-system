

def test_train_default_start_env_override(monkeypatch):
    """SMA_TRAIN_START overrides the 2023 default (multi-regime experiment)."""
    from datetime import date

    from sma.model.__main__ import _train_default_start

    monkeypatch.delenv("SMA_TRAIN_START", raising=False)
    assert _train_default_start() == date(2018, 1, 1)
    monkeypatch.setenv("SMA_TRAIN_START", "2018-01-01")
    assert _train_default_start() == date(2018, 1, 1)
