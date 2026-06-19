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


def update_formula(path: Path, version: str, sha256: str, asset_id: int) -> None:
    version = version.removeprefix("v")
    if not re.match(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$", version):
        raise SystemExit(f"invalid version: {version}")
    if not SHA256_RE.match(sha256):
        raise SystemExit("sha256 must be 64 lowercase hex characters")
    if asset_id <= 0:
        raise SystemExit("asset-id must be a positive integer")

    text = path.read_text()
    # OpenBird is a private repo, so the formula fetches the release tarball from
    # the GitHub API asset endpoint (authenticated via HOMEBREW_GITHUB_API_TOKEN).
    # Bump the asset id in that URL, the version, and the sha256 — each appears
    # exactly once.
    text = replace_once(
        text,
        r"(?<=/releases/assets/)\d+",
        str(asset_id),
    )
    text = replace_once(text, r'^  version ".+"$', f'  version "{version}"')
    text = replace_once(text, r'^  sha256 ".+"$', f'  sha256 "{sha256}"')
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument(
        "--asset-id",
        required=True,
        type=int,
        help="GitHub release asset id for the source tarball (private-repo download).",
    )
    parser.add_argument("--formula", default=Path("Formula/openbird.rb"), type=Path)
    args = parser.parse_args()
    update_formula(args.formula, args.version, args.sha256, args.asset_id)


if __name__ == "__main__":
    main()
