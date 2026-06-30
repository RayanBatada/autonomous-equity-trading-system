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
