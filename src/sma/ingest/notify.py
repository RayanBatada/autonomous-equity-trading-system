"""macOS notification wrapper.

A no-op on non-macOS so the same code can run in CI / Linux dev
environments without crashing.
"""

import platform
import subprocess


def notify_failure(title: str, message: str) -> None:
    if platform.system() != "Darwin":
        return
    safe_title = title.replace('"', "'")
    safe_message = message.replace('"', "'")
    script = f'display notification "{safe_message}" with title "{safe_title}"'
    subprocess.run(["/usr/bin/osascript", "-e", script], check=False)


def send_telegram(text: str) -> bool:
    """Send a Telegram message via TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID env vars.

    Returns True on success. No-op returning False if creds are unset, and
    swallows any send error — a notification must NEVER break the caller (a
    failed digest send cannot be allowed to crash a scheduled job).
    """
    import os

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    try:
        import requests

        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=15,
        )
        return bool(resp.ok)
    except Exception:
        return False
