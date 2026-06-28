#!/usr/bin/env python3
"""Preflight the local machine for an OpenBird notarized beta DMG release."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_REPO = "bishnubista/openbird"
DEFAULT_NOTARY_PROFILE = "openbird-notary"


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    missing: bool = False


@dataclass(frozen=True)
class Row:
    status: str
    name: str
    detail: str


def _run(cmd: list[str], *, timeout: float = 10.0, env: dict[str, str] | None = None) -> CommandResult:
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(returncode=124, timed_out=True)
    except FileNotFoundError:
        return CommandResult(returncode=127, missing=True)
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _read_version() -> str:
    for line in (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        match = re.match(r'^version\s*=\s*"([^"]+)"\s*$', line)
        if match:
            return match.group(1)
    raise RuntimeError("missing version in pyproject.toml")


def _resolve_release_env() -> tuple[dict[str, str], Row | None]:
    script = r'''
set -a
if [ -f script/release.env ]; then
  . script/release.env >/dev/null 2>&1 || exit 7
fi
printf '%s\0%s\0' "${OPENBIRD_SIGN_IDENTITY-}" "${OPENBIRD_NOTARY_PROFILE-}"
'''
    env = os.environ.copy()
    result = _run(["bash", "-c", script], timeout=5.0, env=env)
    if result.missing:
        return {}, Row("BLOCKED", "release env", "reason=bash_unavailable")
    if result.timed_out:
        return {}, Row("BLOCKED", "release env", "reason=timeout")
    if result.returncode != 0:
        return {}, Row("BLOCKED", "release env", "reason=source_failed")
    parts = result.stdout.split("\0")
    if len(parts) < 2:
        return {}, Row("BLOCKED", "release env", "reason=unparseable")
    return {
        "OPENBIRD_SIGN_IDENTITY": parts[0],
        "OPENBIRD_NOTARY_PROFILE": parts[1],
    }, None


def _release_exists(tag: str) -> bool:
    result = _run(["gh", "release", "view", tag, "--repo", CANONICAL_REPO], timeout=10.0)
    return result.returncode == 0


def _classify_notary(result: CommandResult) -> str:
    if result.timed_out:
        return "timeout"
    if result.missing:
        return "xcrun_unavailable"
    combined = f"{result.stdout}\n{result.stderr}"
    if "No Keychain password item found" in combined:
        return "profile_missing"
    return "auth_failed"


def _developer_id_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if "Developer ID Application" in line]


def _check_remote_sync() -> Row:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    fetch = _run(["git", "fetch", "origin", "main"], timeout=30.0, env=env)
    if fetch.missing:
        return Row("BLOCKED", "remote sync", "reason=git_unavailable")
    if fetch.timed_out:
        return Row("BLOCKED", "remote sync", "reason=timeout")
    if fetch.returncode != 0:
        return Row("BLOCKED", "remote sync", "reason=fetch_failed")

    head = _run(["git", "rev-parse", "--verify", "HEAD"], timeout=5.0)
    fetched = _run(["git", "rev-parse", "--verify", "FETCH_HEAD"], timeout=5.0)
    if head.returncode != 0 or fetched.returncode != 0 or head.missing or fetched.missing:
        return Row("BLOCKED", "remote sync", "reason=unreadable")

    head_sha = head.stdout.strip()
    fetch_sha = fetched.stdout.strip()
    if head_sha and head_sha == fetch_sha:
        return Row("PASS", "remote sync", "head_matches_fetch_head=1")

    # A release must come from the reviewed commit on origin/main: behind, ahead,
    # and diverged local main states are all blockers.
    behind = _run(["git", "merge-base", "--is-ancestor", "HEAD", "FETCH_HEAD"], timeout=5.0)
    if behind.returncode > 1 or behind.missing or behind.timed_out:
        return Row("BLOCKED", "remote sync", "reason=not_synced")
    if behind.returncode == 0:
        return Row("BLOCKED", "remote sync", "reason=behind")

    ahead = _run(["git", "merge-base", "--is-ancestor", "FETCH_HEAD", "HEAD"], timeout=5.0)
    if ahead.returncode > 1 or ahead.missing or ahead.timed_out:
        return Row("BLOCKED", "remote sync", "reason=not_synced")
    if ahead.returncode == 0:
        return Row("BLOCKED", "remote sync", "reason=ahead")
    return Row("BLOCKED", "remote sync", "reason=diverged")


def _check_git() -> list[Row]:
    rows: list[Row] = []
    on_main = False
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout=5.0)
    if branch.missing:
        rows.append(Row("BLOCKED", "git branch", "reason=git_unavailable"))
    elif branch.returncode != 0:
        rows.append(Row("BLOCKED", "git branch", "reason=unreadable"))
    else:
        current = branch.stdout.strip()
        if current == "main":
            on_main = True
            rows.append(Row("PASS", "git branch", "main"))
        else:
            rows.append(Row("BLOCKED", "git branch", f"expected=main actual={current or 'unknown'}"))

    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], timeout=5.0)
    if status.missing:
        rows.append(Row("BLOCKED", "tracked tree", "reason=git_unavailable"))
    elif status.returncode != 0:
        rows.append(Row("BLOCKED", "tracked tree", "reason=unreadable"))
    elif status.stdout.strip():
        rows.append(Row("BLOCKED", "tracked tree", "dirty_tracked_files=1"))
    else:
        rows.append(Row("PASS", "tracked tree", "clean_tracked_files=1"))
    if on_main:
        rows.append(_check_remote_sync())
    return rows


def _check_github(version: str, *, allow_existing_version: bool) -> list[Row]:
    rows: list[Row] = []
    auth = _run(["gh", "auth", "status"], timeout=10.0)
    if auth.missing:
        return [Row("BLOCKED", "gh auth", "reason=gh_unavailable")]
    if auth.returncode != 0:
        return [Row("BLOCKED", "gh auth", "reason=unauthenticated")]
    rows.append(Row("PASS", "gh auth", "authenticated=1"))

    beta_tag = f"beta-dmg-{version}"
    if _release_exists(beta_tag):
        if allow_existing_version:
            rows.append(Row("ADVISORY", "beta dmg tag", f"{beta_tag} exists allow_existing_version=1"))
        else:
            rows.append(Row("BLOCKED", "beta dmg tag", f"{beta_tag} exists"))
    else:
        rows.append(Row("PASS", "beta dmg tag", f"{beta_tag} available"))

    source_tag = f"v{version}"
    if _release_exists(source_tag):
        rows.append(Row("ADVISORY", "source tag", f"{source_tag} exists source-channel-only=1"))
    else:
        rows.append(Row("PASS", "source tag", f"{source_tag} absent_for_dmg_run=1"))
    return rows


def _check_identity(sign_identity: str) -> Row:
    result = _run(["security", "find-identity", "-v", "-p", "codesigning"], timeout=10.0)
    if result.missing:
        return Row("BLOCKED", "Developer ID identity", "reason=security_unavailable")
    if result.returncode != 0:
        return Row("BLOCKED", "Developer ID identity", "reason=security_failed")

    lines = _developer_id_lines(result.stdout)
    if sign_identity:
        matches = [line for line in lines if sign_identity in line]
        if len(matches) == 1:
            return Row("PASS", "Developer ID identity", "override_present=1 match_count=1")
        return Row(
            "BLOCKED",
            "Developer ID identity",
            f"override_present=1 match_count={len(matches)} reason=override_not_unique",
        )
    if len(lines) == 1:
        return Row("PASS", "Developer ID identity", "override_present=0 developer_id_count=1")
    return Row("BLOCKED", "Developer ID identity", f"override_present=0 developer_id_count={len(lines)}")


def _check_notary(profile: str, *, timeout: float) -> Row:
    result = _run(["xcrun", "notarytool", "history", "--keychain-profile", profile], timeout=timeout)
    if result.returncode == 0 and not result.timed_out and not result.missing:
        return Row("PASS", "notary profile", f"profile={profile} authenticated=1")
    reason = _classify_notary(result)
    return Row("BLOCKED", "notary profile", f"profile={profile} reason={reason}")


def _print(rows: list[Row], version: str) -> None:
    print(f"OpenBird beta DMG release prerequisites (pyproject.toml = {version})")
    print()
    print(f"  {'status':<9} {'check':<23} detail")
    print(f"  {'------':<9} {'-----':<23} ------")
    for row in rows:
        print(f"  {row.status:<9} {row.name:<23} {row.detail}")
    print()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight local signing/notary/GitHub prerequisites before script/package_dmg.sh.",
    )
    parser.add_argument(
        "--allow-existing-version",
        action="store_true",
        help="Do not block when beta-dmg-<version> already exists.",
    )
    parser.add_argument(
        "--notary-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for notarytool profile validation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.notary_timeout <= 0:
        print("error: --notary-timeout must be positive", file=sys.stderr)
        return 2

    rows: list[Row] = []
    try:
        version = _read_version()
    except Exception:
        print("error: could not read pyproject.toml version", file=sys.stderr)
        return 2
    rows.append(Row("PASS", "version", version))

    rows.extend(_check_git())

    release_env, release_env_row = _resolve_release_env()
    if release_env_row is not None:
        rows.append(release_env_row)
        profile = DEFAULT_NOTARY_PROFILE
        sign_identity = ""
    else:
        sign_identity = release_env.get("OPENBIRD_SIGN_IDENTITY", "")
        profile = release_env.get("OPENBIRD_NOTARY_PROFILE", "") or DEFAULT_NOTARY_PROFILE
        rows.append(
            Row(
                "PASS",
                "release env",
                f"profile={profile} identity_override={1 if sign_identity else 0}",
            )
        )

    rows.extend(_check_github(version, allow_existing_version=args.allow_existing_version))
    rows.append(_check_identity(sign_identity))
    rows.append(_check_notary(profile, timeout=args.notary_timeout))

    _print(rows, version)
    if any(row.status == "BLOCKED" for row in rows):
        print("=> blocked: fix the prerequisites above before cutting a beta DMG.")
        return 1
    print("=> ready: local prerequisites for the beta DMG release are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
