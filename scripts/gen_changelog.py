#!/usr/bin/env python3
"""Generate a categorized Markdown changelog from conventional-commit history.

OpenBird's commit subjects follow Conventional Commits (``type(scope): summary
(#PR)``), so a changelog can be derived mechanically instead of hand-assembled
for every release (the 0.3.0 source release shipped with an auto-stub because no
generator existed). This parses ``git log <from>..<to>`` and groups commits into
Features / Fixes / Performance / Maintenance sections.

Usage:
    scripts/gen_changelog.py --from v0.2.0 --to v0.3.0
    scripts/gen_changelog.py --from beta-dmg-0.2.0          # --to defaults to HEAD

Output goes to stdout as Markdown, ready for ``gh release create --notes-file``.
Stdlib only — no third-party dependencies, so it runs unchanged on a CI runner.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import OrderedDict

# Conventional-commit subject: `type(scope)!: description`, with an optional
# trailing `(#123)` PR reference we lift out into its own link.
_SUBJECT_RE = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]*)\))?(?P<bang>!)?: (?P<desc>.+?)"
    r"(?:\s+\(#(?P<pr>\d+)\))?$"
)

# Section heading order. Anything in MAINT_TYPES collapses into "Maintenance";
# anything that does not parse falls under "Other" so nothing is silently lost.
_SECTIONS: list[tuple[str, str]] = [
    ("feat", "Features"),
    ("fix", "Fixes"),
    ("perf", "Performance"),
]
_MAINT_TYPES = {"refactor", "docs", "test", "build", "ci", "style", "chore"}

# Pure release plumbing — version bumps and formula/cask pin updates — is noise in
# a user-facing changelog. Drop these (type, scope) pairs.
_SKIP_TYPE_SCOPES = {("chore", "release"), ("chore", "homebrew")}


def _git_subjects(rev_range: str) -> list[str]:
    """Return commit subjects in ``rev_range`` (newest first), merges excluded."""
    result = subprocess.run(
        ["git", "log", "--no-merges", "--pretty=%s", rev_range],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _format_entry(desc: str, pr: str | None) -> str:
    return f"- {desc} (#{pr})" if pr else f"- {desc}"


def build_changelog(subjects: list[str]) -> str:
    """Group parsed subjects into Markdown sections."""
    buckets: "OrderedDict[str, list[str]]" = OrderedDict(
        (heading, []) for _, heading in _SECTIONS
    )
    buckets["Maintenance"] = []
    buckets["Other"] = []

    heading_for_type = {ctype: heading for ctype, heading in _SECTIONS}

    for subject in subjects:
        match = _SUBJECT_RE.match(subject)
        if not match:
            buckets["Other"].append(_format_entry(subject, None))
            continue
        ctype = match["type"]
        scope = match["scope"]
        if (ctype, scope) in _SKIP_TYPE_SCOPES:
            continue
        entry = _format_entry(match["desc"], match["pr"])
        if ctype in heading_for_type:
            buckets[heading_for_type[ctype]].append(entry)
        elif ctype in _MAINT_TYPES:
            buckets["Maintenance"].append(entry)
        else:
            buckets["Other"].append(entry)

    parts: list[str] = []
    for heading, entries in buckets.items():
        if not entries:
            continue
        parts.append(f"### {heading}")
        parts.append("\n".join(entries))
        parts.append("")  # blank line between sections

    return "\n".join(parts).strip() + "\n" if parts else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from",
        dest="from_ref",
        required=True,
        help="exclusive lower bound (e.g. the previous release tag)",
    )
    parser.add_argument(
        "--to",
        dest="to_ref",
        default="HEAD",
        help="upper bound (default: HEAD)",
    )
    args = parser.parse_args(argv)

    rev_range = f"{args.from_ref}..{args.to_ref}"
    try:
        subjects = _git_subjects(rev_range)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise SystemExit(f"git log {rev_range} failed: {stderr}") from exc

    changelog = build_changelog(subjects)
    if not changelog:
        raise SystemExit(f"no changelog-worthy commits in {rev_range}")

    sys.stdout.write(changelog)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
