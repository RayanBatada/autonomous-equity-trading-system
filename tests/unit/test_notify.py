import os
from unittest.mock import MagicMock, patch

from sma.ingest.notify import notify_failure


def test_notify_failure_invokes_osascript_on_macos():
    fake_run = MagicMock()
    with patch("sma.ingest.notify.platform.system", return_value="Darwin"), \
         patch("sma.ingest.notify.subprocess.run", fake_run):
        notify_failure(title="SMA Ingest", message="2 quality checks failed")
    assert fake_run.called
    args = fake_run.call_args[0][0]
    assert "osascript" in args[0]
    assert any("SMA Ingest" in str(a) for a in args)


def test_notify_failure_no_op_on_non_macos():
    fake_run = MagicMock()
    with patch("sma.ingest.notify.platform.system", return_value="Linux"), \
         patch("sma.ingest.notify.subprocess.run", fake_run):
        notify_failure(title="t", message="m")
    fake_run.assert_not_called()


# ---------------------------------------------------------------------------
# 2026-07-30: notify_failure ALSO attempts Telegram, so infra failures reach a
# phone, not just a macOS notification center nobody's looking at. Tokens are
# unset today (pre-wiring only), so send_telegram itself still no-ops — these
# tests cover notify_failure's new call into it.
# ---------------------------------------------------------------------------


def test_notify_failure_attempts_telegram_when_configured():
    """When Telegram creds are set (monkeypatched here via send_telegram
    itself), notify_failure must call send_telegram with the composed
    "{title}\\n{message}" text."""
    calls = []

    def _fake_send(text):
        calls.append(text)
        return True

    with patch("sma.ingest.notify.platform.system", return_value="Darwin"), \
         patch("sma.ingest.notify.subprocess.run"), \
         patch("sma.ingest.notify.send_telegram", side_effect=_fake_send):
        notify_failure(title="sma: backup FAILED", message="verification failed")
    assert calls == ["sma: backup FAILED\nverification failed"]


def test_notify_failure_survives_telegram_raising():
    """A Telegram failure (e.g. a bug bypassing send_telegram's own
    try/except) must NEVER break the caller — notify_failure must still
    complete, and the osascript path must still have run."""
    def _boom(text):
        raise RuntimeError("telegram down")

    fake_run = MagicMock()
    with patch("sma.ingest.notify.platform.system", return_value="Darwin"), \
         patch("sma.ingest.notify.subprocess.run", fake_run), \
         patch("sma.ingest.notify.send_telegram", side_effect=_boom):
        notify_failure(title="t", message="m")  # must not raise
    assert fake_run.called


def test_notify_failure_telegram_silent_noop_when_tokens_absent(monkeypatch):
    """Tokens are unset today — send_telegram's own no-op fires, and
    notify_failure must complete without exception (real, unmocked
    send_telegram)."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    fake_run = MagicMock()
    with patch("sma.ingest.notify.platform.system", return_value="Darwin"), \
         patch("sma.ingest.notify.subprocess.run", fake_run):
        notify_failure(title="t", message="m")  # must not raise
    assert fake_run.called


# ---------------------------------------------------------------------------
# 2026-08-25: notify_failure ALSO attempts ntfy.sh -- closes the gap Telegram
# was meant to close but never did (tokens are still unset a month later).
# SMA_NTFY_TOPIC is unset in the tests above unless a test sets it, so
# send_ntfy itself still no-ops there too — these cover notify_failure's call
# into it specifically.
# ---------------------------------------------------------------------------


def test_notify_failure_attempts_ntfy_when_configured():
    """When SMA_NTFY_TOPIC is set (monkeypatched here via send_ntfy itself),
    notify_failure must call send_ntfy with the message, the title, and
    priority='high' (failures are always high priority)."""
    calls = []

    def _fake_send(text, *, title=None, priority="default"):
        calls.append((text, title, priority))
        return True

    with patch("sma.ingest.notify.platform.system", return_value="Darwin"), \
         patch("sma.ingest.notify.subprocess.run"), \
         patch("sma.ingest.notify.send_telegram", return_value=False), \
         patch("sma.ingest.notify.send_ntfy", side_effect=_fake_send):
        notify_failure(title="sma: backup FAILED", message="verification failed")
    assert calls == [("verification failed", "sma: backup FAILED", "high")]


def test_notify_failure_survives_ntfy_raising():
    """An ntfy failure (e.g. a bug bypassing send_ntfy's own try/except)
    must NEVER break the caller — notify_failure must still complete, and
    the osascript + Telegram paths must still have run."""
    def _boom(text, *, title=None, priority="default"):
        raise RuntimeError("ntfy down")

    fake_run = MagicMock()
    telegram_calls = []
    with patch("sma.ingest.notify.platform.system", return_value="Darwin"), \
         patch("sma.ingest.notify.subprocess.run", fake_run), \
         patch("sma.ingest.notify.send_telegram", side_effect=telegram_calls.append), \
         patch("sma.ingest.notify.send_ntfy", side_effect=_boom):
        notify_failure(title="t", message="m")  # must not raise
    assert fake_run.called
    assert telegram_calls == ["t\nm"]


def test_notify_failure_ntfy_silent_noop_when_topic_absent(monkeypatch):
    """Topic is unset — send_ntfy's own no-op fires, and notify_failure
    must complete without exception (real, unmocked send_ntfy)."""
    monkeypatch.delenv("SMA_NTFY_TOPIC", raising=False)
    fake_run = MagicMock()
    with patch("sma.ingest.notify.platform.system", return_value="Darwin"), \
         patch("sma.ingest.notify.subprocess.run", fake_run), \
         patch("sma.ingest.notify.send_telegram", return_value=False):
        notify_failure(title="t", message="m")  # must not raise
    assert fake_run.called


# ---------------------------------------------------------------------------
# 2026-08-25: _ensure_dotenv_loaded -- pydantic-settings' Secrets model reads
# .env WITHOUT mutating os.environ (verified: FINNHUB_API_KEY, a REQUIRED
# secret always present in .env, is absent from os.environ after
# load_settings() runs), and no launchd plist bakes secrets into
# EnvironmentVariables (they're git-tracked). This closes that gap for
# send_telegram/send_ntfy's raw os.environ.get() lookups.
# ---------------------------------------------------------------------------


def test_ensure_dotenv_loaded_populates_missing_vars_without_overriding_set_ones(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "SMA_TEST_ONLY_MISSING=from-dotenv\nSMA_TEST_ONLY_ALREADY_SET=from-dotenv\n"
    )
    monkeypatch.delenv("SMA_TEST_ONLY_MISSING", raising=False)
    monkeypatch.setenv("SMA_TEST_ONLY_ALREADY_SET", "from-shell")

    import sma.ingest.notify as notify

    notify._ensure_dotenv_loaded()

    assert os.environ["SMA_TEST_ONLY_MISSING"] == "from-dotenv"
    assert os.environ["SMA_TEST_ONLY_ALREADY_SET"] == "from-shell"  # not overridden


def test_ensure_dotenv_loaded_never_raises_without_a_dotenv_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env here

    import sma.ingest.notify as notify

    notify._ensure_dotenv_loaded()  # must not raise
