"""send_telegram: env-gated, posts correctly, and never raises."""
import requests

import sma.ingest.notify as notify


def test_noop_without_creds(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert notify.send_telegram("hi") is False


def test_posts_when_creds_set(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    captured = {}

    class _Resp:
        ok = True

    def _fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return _Resp()

    monkeypatch.setattr(requests, "post", _fake_post)
    assert notify.send_telegram("hello") is True
    assert "bottok/sendMessage" in captured["url"]
    assert captured["json"]["chat_id"] == "123"
    assert captured["json"]["text"] == "hello"


def test_swallows_send_errors(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")

    def _boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(requests, "post", _boom)
    assert notify.send_telegram("x") is False
