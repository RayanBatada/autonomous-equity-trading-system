"""macOS notification wrapper.

A no-op on non-macOS so the same code can run in CI / Linux dev
environments without crashing.
"""

import contextlib
import platform
import subprocess


def _ensure_dotenv_loaded() -> None:
    """Best-effort: populate os.environ from .env (searched from CWD, same
    convention as data/sentinels, data/sma.duckdb etc. -- every scheduled
    job's launchd plist sets WorkingDirectory to the repo root) so the raw
    os.environ.get() lookups in send_telegram/send_ntfy below actually see a
    value a human added to .env.

    2026-08-25 finding: sma.config's Secrets(BaseSettings) reads .env via
    pydantic-settings' OWN internal parsing (SettingsConfigDict(env_file=
    ".env")) WITHOUT mutating process os.environ -- verified empirically:
    FINNHUB_API_KEY, a REQUIRED secret always present in .env, is absent
    from os.environ even after sma.config.load_settings() runs. Scheduled
    (launchd) jobs' EnvironmentVariables blocks only carry PATH/TZ; secrets
    never go there because ops/launchd/*.plist is git-tracked and a secret
    baked into a plist would leak into the repo. Net effect: TELEGRAM_
    BOT_TOKEN/CHAT_ID has silently had this same gap since 2026-07-30 --
    adding tokens to .env alone would still never have reached a real
    scheduled run. load_dotenv() closes it for every os.environ.get()
    caller in the process, not just this module.

    Safe to call repeatedly / at import time: python-dotenv's override=False
    default never clobbers an already-set env var, and any failure (missing
    .env, permissions, a malformed line) is swallowed -- this must never be
    allowed to break every job that imports this module for notify_failure.
    """
    with contextlib.suppress(Exception):
        from dotenv import find_dotenv, load_dotenv

        # find_dotenv(usecwd=True) explicitly, then load_dotenv(dotenv_path=...):
        # load_dotenv() itself takes no usecwd kwarg. Without resolving the
        # path this way, load_dotenv()'s own default walks up from the
        # CALLING FILE's on-disk location (this module's, since it's the
        # direct caller) rather than CWD -- which happens to also resolve
        # correctly in production (notify.py always lives under the repo
        # root) but makes CWD-based test isolation (monkeypatch.chdir + a
        # tmp .env) impossible, since it would still find the real repo
        # .env. Explicit usecwd=True keeps this predictable and consistent
        # with every other CWD-relative path in this codebase
        # (data/sentinels, data/sma.duckdb).
        load_dotenv(dotenv_path=find_dotenv(usecwd=True))


_ensure_dotenv_loaded()


def notify_failure(title: str, message: str) -> None:
    if platform.system() == "Darwin":
        safe_title = title.replace('"', "'")
        safe_message = message.replace('"', "'")
        script = f'display notification "{safe_message}" with title "{safe_title}"'
        subprocess.run(["/usr/bin/osascript", "-e", script], check=False)

    # 2026-07-30: also route to Telegram so infra failures reach a phone, not
    # just a macOS notification center nobody's looking at. Best-effort and
    # wrapped: send_telegram already no-ops when TELEGRAM_BOT_TOKEN/
    # TELEGRAM_CHAT_ID are unset (they currently are — this pre-wires phone
    # alerting so adding tokens to .env is the only remaining step) and
    # swallows its own send errors, but a Telegram failure must NEVER be
    # allowed to break the caller, which runs inside scheduled jobs.
    with contextlib.suppress(Exception):
        send_telegram(f"{title}\n{message}")

    # 2026-08-25: ntfy.sh push -- closes the operator-blindness gap Telegram
    # was meant to close but never did (its tokens are still unset a month
    # later). ntfy needs no account/token, just a topic name: POSTing to
    # https://ntfy.sh/<topic> pushes to anyone subscribed to that topic in
    # the ntfy app/web client. Same best-effort/never-raise contract as
    # Telegram: send_ntfy no-ops when SMA_NTFY_TOPIC is unset and swallows
    # its own send errors. Always high priority here -- notify_failure is
    # reserved for actual failures.
    with contextlib.suppress(Exception):
        send_ntfy(message, title=title, priority="high")


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


def send_ntfy(text: str, *, title: str | None = None, priority: str = "default") -> bool:
    """Push `text` to https://ntfy.sh/$SMA_NTFY_TOPIC.

    Returns True on an HTTP-OK response. No-op returning False if
    SMA_NTFY_TOPIC is unset, and swallows any send error — same contract as
    send_telegram: a notification must NEVER break the caller.

    SECURITY NOTE: ntfy.sh's public instance has no account/auth model for a
    topic — the topic NAME is the only thing gating who can read (and post
    to) it. Anyone who learns/guesses SMA_NTFY_TOPIC can subscribe to it or
    spoof messages into it. This is why SMA_NTFY_TOPIC must be a long random
    slug (sma-<24 random alphanumeric chars>, ~142 bits of entropy) rather
    than something guessable like "sma-alerts" — the random slug IS the
    secret; treat it like one (never log it, never commit it — it lives
    only in the gitignored .env).
    """
    import os

    topic = os.environ.get("SMA_NTFY_TOPIC")
    if not topic:
        return False
    try:
        import requests

        headers: dict[str, str] = {"Priority": priority}
        if title:
            # HTTP headers are latin-1; a unicode title (an em dash, say) made
            # requests raise and this function return False SILENTLY — the
            # first live trade push (2026-09-01) was lost exactly this way.
            # Degrade unencodable characters instead of losing the message.
            headers["Title"] = title.encode("latin-1", "replace").decode("latin-1")
        resp = requests.post(
            f"https://ntfy.sh/{topic}",
            data=text.encode("utf-8"),
            headers=headers,
            timeout=15,
        )
        return bool(resp.ok)
    except Exception:
        return False
