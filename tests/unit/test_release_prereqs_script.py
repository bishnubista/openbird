from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _make_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    (repo / "script").mkdir(parents=True)
    shutil.copy(ROOT / "script" / "release_prereqs.py", repo / "script" / "release_prereqs.py")
    (repo / "script" / "release_prereqs.py").chmod(0o755)
    (repo / "pyproject.toml").write_text('[project]\nversion = "0.6.1"\n', encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "git",
        """#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  rev-parse)
    [ "${2:-}" = "--abbrev-ref" ]
    [ "${3:-}" = "HEAD" ]
    printf '%s\\n' "${FAKE_GIT_BRANCH:-main}"
    exit 0
    ;;
  status)
    [ "${2:-}" = "--porcelain" ]
    [ "${3:-}" = "--untracked-files=no" ]
    printf '%s' "${FAKE_GIT_STATUS:-}"
    exit 0
    ;;
esac
exit 2
""",
    )
    _write_executable(
        fake_bin / "gh",
        """#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  auth)
    [ "${2:-}" = "status" ]
    [ "${FAKE_GH_AUTH:-ok}" = "ok" ] || exit 1
    exit 0
    ;;
  release)
    [ "${2:-}" = "view" ]
    case "${3:-}" in
      beta-dmg-0.6.1) [ "${FAKE_BETA_TAG_EXISTS:-no}" = "yes" ] || exit 1 ;;
      v0.6.1) [ "${FAKE_SOURCE_TAG_EXISTS:-no}" = "yes" ] || exit 1 ;;
      *) exit 1 ;;
    esac
    exit 0
    ;;
esac
exit 2
""",
    )
    _write_executable(
        fake_bin / "security",
        """#!/usr/bin/env bash
set -euo pipefail
[ "${1:-}" = "find-identity" ]
[ "${2:-}" = "-v" ]
[ "${3:-}" = "-p" ]
[ "${4:-}" = "codesigning" ]
case "${FAKE_IDENTITIES:-one}" in
  none)
    printf '     0 valid identities found\\n'
    ;;
  one)
    printf '  1) HASH "Developer ID Application: Example One (TEAMONE)"\\n'
    printf '     1 valid identities found\\n'
    ;;
  multiple)
    printf '  1) HASH "Developer ID Application: Example One (TEAMONE)"\\n'
    printf '  2) HASH "Developer ID Application: Example Two (TEAMTWO)"\\n'
    printf '     2 valid identities found\\n'
    ;;
  app_and_installer)
    printf '  1) HASH "Developer ID Application: Example Two (TEAMTWO)"\\n'
    printf '  2) HASH "Developer ID Installer: Example Two (TEAMTWO)"\\n'
    printf '     2 valid identities found\\n'
    ;;
esac
""",
    )
    _write_executable(
        fake_bin / "xcrun",
        """#!/usr/bin/env bash
set -euo pipefail
[ "${1:-}" = "notarytool" ]
[ "${2:-}" = "history" ]
[ "${3:-}" = "--keychain-profile" ]
profile="${4:-}"
case "${FAKE_NOTARY:-ok}" in
  ok)
    printf 'history ok for %s\\n' "$profile"
    exit 0
    ;;
  missing)
    printf 'No Keychain password item found for profile: %s SECRET_NOTARY_PASSWORD\\n' "$profile" >&2
    exit 1
    ;;
  invalid)
    printf 'Authentication failed for Apple ID SECRET_NOTARY_PASSWORD\\n' >&2
    exit 1
    ;;
  slow)
    sleep 2
    exit 0
    ;;
esac
exit 2
""",
    )
    return repo, fake_bin


def _run_prereqs(
    repo: Path,
    fake_bin: Path,
    *args: str,
    branch: str = "main",
    git_status: str = "",
    beta_tag_exists: str = "no",
    source_tag_exists: str = "no",
    identities: str = "one",
    notary: str = "ok",
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "FAKE_GIT_BRANCH": branch,
            "FAKE_GIT_STATUS": git_status,
            "FAKE_BETA_TAG_EXISTS": beta_tag_exists,
            "FAKE_SOURCE_TAG_EXISTS": source_tag_exists,
            "FAKE_IDENTITIES": identities,
            "FAKE_NOTARY": notary,
        }
    )
    return subprocess.run(
        [str(repo / "script" / "release_prereqs.py"), *args],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_release_prereqs_pass_with_single_identity_and_notary_profile(tmp_path: Path) -> None:
    repo, fake_bin = _make_repo(tmp_path)

    result = _run_prereqs(repo, fake_bin)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS      git branch" in result.stdout
    assert "PASS      Developer ID identity" in result.stdout
    assert "developer_id_count=1" in result.stdout
    assert "PASS      notary profile" in result.stdout
    assert "=> ready:" in result.stdout


def test_release_env_override_selects_one_identity_without_printing_cn(tmp_path: Path) -> None:
    repo, fake_bin = _make_repo(tmp_path)
    (repo / "script" / "release.env").write_text(
        'OPENBIRD_SIGN_IDENTITY="Developer ID Application: Example Two"\n'
        'OPENBIRD_NOTARY_PROFILE="custom-profile"\n',
        encoding="utf-8",
    )

    result = _run_prereqs(repo, fake_bin, identities="multiple")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "profile=custom-profile identity_override=1" in result.stdout
    assert "override_present=1 match_count=1" in result.stdout
    assert "Example Two" not in result.stdout
    assert "TEAMTWO" not in result.stdout


def test_missing_identity_override_match_blocks_without_printing_cn(tmp_path: Path) -> None:
    repo, fake_bin = _make_repo(tmp_path)
    (repo / "script" / "release.env").write_text(
        'OPENBIRD_SIGN_IDENTITY="Developer ID Application: Missing"\n',
        encoding="utf-8",
    )

    result = _run_prereqs(repo, fake_bin, identities="multiple")

    assert result.returncode == 1
    assert "BLOCKED   Developer ID identity" in result.stdout
    assert "override_present=1 match_count=0" in result.stdout
    assert "Example One" not in result.stdout
    assert "Example Two" not in result.stdout
    assert "TEAMONE" not in result.stdout
    assert "TEAMTWO" not in result.stdout


def test_identity_override_ignores_matching_installer_certificate(tmp_path: Path) -> None:
    repo, fake_bin = _make_repo(tmp_path)
    (repo / "script" / "release.env").write_text(
        'OPENBIRD_SIGN_IDENTITY="Example Two"\n',
        encoding="utf-8",
    )

    result = _run_prereqs(repo, fake_bin, identities="app_and_installer")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "override_present=1 match_count=1" in result.stdout


def test_existing_beta_tag_blocks_unless_explicitly_allowed(tmp_path: Path) -> None:
    repo, fake_bin = _make_repo(tmp_path)

    blocked = _run_prereqs(repo, fake_bin, beta_tag_exists="yes")
    allowed = _run_prereqs(repo, fake_bin, "--allow-existing-version", beta_tag_exists="yes")

    assert blocked.returncode == 1
    assert "BLOCKED   beta dmg tag" in blocked.stdout
    assert "beta-dmg-0.6.1 exists" in blocked.stdout
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr
    assert "ADVISORY  beta dmg tag" in allowed.stdout
    assert "allow_existing_version=1" in allowed.stdout


def test_source_tag_is_advisory_for_dmg_preflight(tmp_path: Path) -> None:
    repo, fake_bin = _make_repo(tmp_path)

    result = _run_prereqs(repo, fake_bin, source_tag_exists="yes")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ADVISORY  source tag" in result.stdout
    assert "source-channel-only=1" in result.stdout


def test_untracked_files_are_ignored_but_tracked_dirty_tree_blocks(tmp_path: Path) -> None:
    repo, fake_bin = _make_repo(tmp_path)

    result = _run_prereqs(repo, fake_bin, git_status=" M pyproject.toml\n")

    assert result.returncode == 1
    assert "BLOCKED   tracked tree" in result.stdout
    assert "dirty_tracked_files=1" in result.stdout


def test_non_main_branch_blocks_release_preflight(tmp_path: Path) -> None:
    repo, fake_bin = _make_repo(tmp_path)

    result = _run_prereqs(repo, fake_bin, branch="fix/release-prereq-diagnostic")

    assert result.returncode == 1
    assert "BLOCKED   git branch" in result.stdout
    assert "expected=main actual=fix/release-prereq-diagnostic" in result.stdout


def test_notary_missing_profile_reason_is_safe(tmp_path: Path) -> None:
    repo, fake_bin = _make_repo(tmp_path)

    result = _run_prereqs(repo, fake_bin, notary="missing")

    assert result.returncode == 1
    assert "BLOCKED   notary profile" in result.stdout
    assert "reason=profile_missing" in result.stdout
    assert "SECRET_NOTARY_PASSWORD" not in result.stdout
    assert "SECRET_NOTARY_PASSWORD" not in result.stderr


def test_notary_invalid_credentials_reason_is_safe(tmp_path: Path) -> None:
    repo, fake_bin = _make_repo(tmp_path)

    result = _run_prereqs(repo, fake_bin, notary="invalid")

    assert result.returncode == 1
    assert "BLOCKED   notary profile" in result.stdout
    assert "reason=auth_failed" in result.stdout
    assert "SECRET_NOTARY_PASSWORD" not in result.stdout
    assert "SECRET_NOTARY_PASSWORD" not in result.stderr


def test_notary_timeout_reason_is_safe(tmp_path: Path) -> None:
    repo, fake_bin = _make_repo(tmp_path)

    result = _run_prereqs(repo, fake_bin, "--notary-timeout", "0.1", notary="slow")

    assert result.returncode == 1
    assert "BLOCKED   notary profile" in result.stdout
    assert "reason=timeout" in result.stdout


def test_raw_security_and_notary_outputs_never_leak_on_multiple_failures(tmp_path: Path) -> None:
    repo, fake_bin = _make_repo(tmp_path)

    result = _run_prereqs(repo, fake_bin, identities="multiple", notary="invalid")

    assert result.returncode == 1
    assert "developer_id_count=2" in result.stdout
    assert "reason=auth_failed" in result.stdout
    assert "Example One" not in result.stdout
    assert "Example Two" not in result.stdout
    assert "TEAMONE" not in result.stdout
    assert "TEAMTWO" not in result.stdout
    assert "SECRET_NOTARY_PASSWORD" not in result.stdout
    assert "SECRET_NOTARY_PASSWORD" not in result.stderr
