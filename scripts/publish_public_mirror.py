#!/usr/bin/env python
"""Refresh the public copy (RayanBatada/autonomous-equity-trading-system).

    uv run python scripts/publish_public_mirror.py --mirror ~/path/to/mirror-clone [--ref main]

Exports --ref of this repo with `git archive` (tracked files only, so .env,
data/, logs/ and models_artifacts/ can never come along), applies the rules
in docs/PUBLISHING.md, runs scripts/check_public_tree.py on the result
(with this checkout's .env, so no key value can slip through), and syncs it
into the mirror clone. It does not commit or push: review `git status` and
`git diff --stat` in the clone, run the suite there, then commit and push.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from check_public_tree import DENIED_PATHS, check  # noqa: E402

PRIVATE_SLUG = "RayanBatada/Stock-Market-Predictor-Agents"
PUBLIC_SLUG = "RayanBatada/autonomous-equity-trading-system"
REPLACEMENTS = (
    (b"/Users/youruser", b"/Users/youruser"),
    (b"-Users-youruser-", b"-Users-youruser-"),
    (b"your-email@example.com", b"your-email@example.com"),
    (b"user youruser.", b"user youruser."),
)
README_PUBLIC_NOTE_OLD = (
    "A scrubbed public copy lives at\n"
    "[autonomous-equity-trading-system](https://github.com/RayanBatada/autonomous-equity-trading-system)."
)
README_PUBLIC_NOTE_NEW = (
    "This repository is the public copy of my private working repo, refreshed by hand: one commit per\n"
    "refresh, my machine's paths replaced with `/Users/youruser`, and internal session notes left out."
)


def export(ref: str, dest: Path) -> None:
    arc = subprocess.run(["git", "-C", str(REPO), "archive", ref], check=True, capture_output=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=arc.stdout, check=True)


def scrub(root: Path) -> None:
    for d in DENIED_PATHS:
        p = root / d
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix in (".duckdb",) or p.name == "check_public_tree.py":
            continue
        b = p.read_bytes()
        nb = b
        for old, new in REPLACEMENTS:
            nb = nb.replace(old, new)
        if nb != b:
            p.write_bytes(nb)
    readme = root / "README.md"
    text = readme.read_text()
    text = text.replace(f"github.com/{PRIVATE_SLUG}/actions", f"github.com/{PUBLIC_SLUG}/actions")
    text = text.replace(README_PUBLIC_NOTE_OLD, README_PUBLIC_NOTE_NEW)
    readme.write_text(text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mirror", type=Path, required=True, help="a git clone of the public repo")
    ap.add_argument("--ref", default="main")
    a = ap.parse_args(argv)
    mirror = a.mirror.expanduser().resolve()
    if not (mirror / ".git").exists():
        print(f"{mirror} is not a git clone of {PUBLIC_SLUG}")
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "tree"
        out.mkdir()
        export(a.ref, out)
        scrub(out)
        license_src = mirror / "LICENSE"
        if license_src.exists():
            shutil.copy2(license_src, out / "LICENSE")
        problems = check(out, REPO / ".env")
        if problems:
            for p in problems:
                print(f"FAIL {p}")
            print("not synced: fix the problems above first")
            return 1
        subprocess.run(
            ["rsync", "-a", "--delete", "--exclude", ".git", f"{out}/", f"{mirror}/"], check=True
        )
    print(f"synced {a.ref} into {mirror}; review, test, commit and push there by hand")
    return 0


if __name__ == "__main__":
    sys.exit(main())
