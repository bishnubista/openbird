#!/usr/bin/env python3
"""Update Formula/openbird.rb for a tagged source release."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def replace_once(text: str, pattern: str, replacement: str) -> str:
    regex = re.compile(pattern, flags=re.MULTILINE)
    match_count = sum(1 for _ in regex.finditer(text))
    if match_count != 1:
        raise SystemExit(f"expected exactly one match for {pattern!r}, found {match_count}")
    return regex.sub(replacement, text, count=1)


def update_formula(path: Path, version: str, sha256: str) -> None:
    version = version.removeprefix("v")
    if not re.match(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$", version):
        raise SystemExit(f"invalid version: {version}")
    if not SHA256_RE.match(sha256):
        raise SystemExit("sha256 must be 64 lowercase hex characters")

    text = path.read_text()
    # OpenBird is a public repo, so the formula downloads the source tarball from
    # the standard GitHub release-download URL — no token required. Bump the URL,
    # version, and sha256 — each appears exactly once.
    url = (
        "https://github.com/bishnubista/openbird/releases/download/"
        f"v{version}/openbird-{version}.tar.gz"
    )
    text = replace_once(text, r'^  url ".+"$', f'  url "{url}"')
    text = replace_once(text, r'^  version ".+"$', f'  version "{version}"')
    text = replace_once(text, r'^  sha256 ".+"$', f'  sha256 "{sha256}"')
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--formula", default=Path("Formula/openbird.rb"), type=Path)
    args = parser.parse_args()
    update_formula(args.formula, args.version, args.sha256)


if __name__ == "__main__":
    main()
