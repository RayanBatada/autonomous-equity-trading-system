"""send_ntfy: env-gated, posts correctly, and never raises.

Mirrors tests/unit/ingest/test_notify_telegram.py's structure exactly (same
env-gate / mocked-HTTP / swallow-errors contract, different transport).
"""
import requests

import sma.ingest.notify as notify


def test_noop_without_topic(monkeypatch):
    monkeypatch.delenv("SMA_NTFY_TOPIC", raising=False)
    assert notify.send_ntfy("hi") is False


def test_posts_when_topic_set(monkeypatch):
    monkeypatch.setenv("SMA_NTFY_TOPIC", "sma-testtopic123")
    captured = {}

    class _Resp:
        ok = True

    def _fake_post(url, data, headers, timeout):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(requests, "post", _fake_post)
    assert notify.send_ntfy("hello", title="t", priority="high") is True
    assert captured["url"] == "https://ntfy.sh/sma-testtopic123"
    assert captured["data"] == b"hello"
    assert captured["headers"]["Title"] == "t"
    assert captured["headers"]["Priority"] == "high"


def test_default_priority_and_no_title_header_when_unspecified(monkeypatch):
    monkeypatch.setenv("SMA_NTFY_TOPIC", "sma-testtopic123")
    captured = {}

    class _Resp:
        ok = True

    def _fake_post(url, data, headers, timeout):
        captured["headers"] = headers
        return _Resp()

    monkeypatch.setattr(requests, "post", _fake_post)
    notify.send_ntfy("hello")
    assert captured["headers"]["Priority"] == "default"
    assert "Title" not in captured["headers"]


def test_swallows_send_errors(monkeypatch):
    monkeypatch.setenv("SMA_NTFY_TOPIC", "sma-testtopic123")

    def _boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(requests, "post", _boom)
    assert notify.send_ntfy("x") is False


def test_returns_false_on_non_ok_response(monkeypatch):
    monkeypatch.setenv("SMA_NTFY_TOPIC", "sma-testtopic123")

    class _Resp:
        ok = False

    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    assert notify.send_ntfy("x") is False
