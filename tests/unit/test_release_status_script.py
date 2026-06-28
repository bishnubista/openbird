from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _make_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    (repo / "script").mkdir(parents=True)
    (repo / "Casks").mkdir()
    (repo / "Formula").mkdir()
    shutil.copy(ROOT / "script" / "release_status.sh", repo / "script" / "release_status.sh")
    (repo / "script" / "release_status.sh").chmod(0o755)

    (repo / "pyproject.toml").write_text('version = "0.6.1"\n', encoding="utf-8")
    (repo / "uv.lock").write_text('name = "openbird"\nversion = "0.6.1"\n', encoding="utf-8")
    (repo / "Casks" / "openbird.rb").write_text('version "0.6.1"\n', encoding="utf-8")
    (repo / "Formula" / "openbird.rb").write_text(
        'url "https://github.com/bishnubista/openbird/releases/download/v0.6.1/openbird-0.6.1.tar.gz"\n',
        encoding="utf-8",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "gh",
        """#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  auth)
    [ "${2:-}" = "status" ]
    exit 0
    ;;
  release)
    [ "${2:-}" = "view" ]
    case "${3:-}" in
      beta-dmg-*) [ "${FAKE_DMG_PUBLISHED:-yes}" = "yes" ] || exit 1 ;;
      v*) [ "${FAKE_SRC_PUBLISHED:-yes}" = "yes" ] || exit 1 ;;
    esac
    exit 0
    ;;
  repo)
    [ "${2:-}" = "view" ]
    printf '%s\\n' "${FAKE_REPO_VISIBILITY:-PRIVATE}"
    exit 0
    ;;
esac
exit 2
""",
    )
    _write_executable(
        fake_bin / "codesign",
        """#!/usr/bin/env bash
set -euo pipefail
case "${FAKE_CODESIGN_MODE:-developer_id}" in
  developer_id)
    {
      printf 'Executable=/tmp/OpenBird.app/Contents/MacOS/OpenBird\\n'
      printf 'Authority=Developer ID Application: bishnu bista (SB26BAMXJM)\\n'
      printf 'Authority=Developer ID Certification Authority\\n'
      printf 'TeamIdentifier=SB26BAMXJM\\n'
    } >&2
    exit 0
    ;;
  adhoc)
    {
      printf 'Executable=/tmp/OpenBird.app/Contents/MacOS/OpenBird\\n'
      printf 'Signature=adhoc\\n'
      printf 'TeamIdentifier=not set\\n'
    } >&2
    exit 0
    ;;
  apple_development)
    {
      printf 'Executable=/tmp/OpenBird.app/Contents/MacOS/OpenBird\\n'
      printf 'Authority=Apple Development: Example Developer (ABCDE12345)\\n'
      printf 'TeamIdentifier=ABCDE12345\\n'
    } >&2
    exit 0
    ;;
  unsigned)
    printf 'code object is not signed at all\\n' >&2
    exit 1
    ;;
esac
exit 2
""",
    )
    _write_executable(
        fake_bin / "xcrun",
        """#!/usr/bin/env bash
set -euo pipefail
if [ "${1:-}" = "stapler" ] && [ "${2:-}" = "validate" ]; then
  case "${FAKE_STAPLER_MODE:-stapled}" in
    stapled)
      printf 'The validate action worked!\\n' >&2
      exit 0
      ;;
    absent)
      printf 'The validate action failed!\\n' >&2
      exit 65
      ;;
  esac
fi
exit 2
""",
    )
    _write_executable(
        fake_bin / "curl",
        """#!/usr/bin/env bash
set -euo pipefail
case "${FAKE_CURL_MODE:-http}" in
  timeout)
    printf '000'
    exit 28
    ;;
  http)
    printf '%s' "${FAKE_CURL_CODE:-404}"
    exit 0
    ;;
esac
exit 2
""",
    )
    _write_executable(
        fake_bin / "brew",
        """#!/usr/bin/env bash
set -euo pipefail
if [ "${FAKE_BREW_MODE:-installed}" = "not-installed" ]; then
  exit 1
fi
if [ "${1:-}" = "list" ] && [ "${2:-}" = "--cask" ]; then
  printf 'openbird %s\\n' "${FAKE_BREW_CASK_VERSION:-0.6.0}"
  exit 0
fi
if [ "${1:-}" = "list" ]; then
  printf 'openbird %s\\n' "${FAKE_BREW_FORMULA_VERSION:-0.6.0}"
  exit 0
fi
exit 2
""",
    )

    app_bundle = tmp_path / "Applications" / "OpenBird.app"
    cli = app_bundle / "Contents" / "MacOS" / "openbird-cli"
    cli.parent.mkdir(parents=True)
    cli.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    cli.chmod(0o755)
    (app_bundle / "Contents" / "Info.plist").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleShortVersionString</key>
  <string>0.6.1</string>
</dict>
</plist>
""",
        encoding="utf-8",
    )
    (fake_bin / "openbird").symlink_to(cli)
    return repo, fake_bin, app_bundle


def _run_release_status(
    repo: Path,
    fake_bin: Path,
    app_bundle: Path,
    *args: str,
    repo_visibility: str = "PRIVATE",
    curl_code: str = "404",
    curl_mode: str = "http",
    brew_mode: str = "installed",
    codesign_mode: str = "developer_id",
    stapler_mode: str = "stapled",
    dmg_published: str = "yes",
    src_published: str = "yes",
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "FAKE_REPO_VISIBILITY": repo_visibility,
            "FAKE_CURL_CODE": curl_code,
            "FAKE_CURL_MODE": curl_mode,
            "FAKE_BREW_MODE": brew_mode,
            "FAKE_CODESIGN_MODE": codesign_mode,
            "FAKE_STAPLER_MODE": stapler_mode,
            "FAKE_DMG_PUBLISHED": dmg_published,
            "FAKE_SRC_PUBLISHED": src_published,
            "OPENBIRD_RELEASE_STATUS_APP_BUNDLE": str(app_bundle),
        }
    )
    return subprocess.run(
        [str(repo / "script" / "release_status.sh"), *args],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _set_app_version(app_bundle: Path, version: str) -> None:
    (app_bundle / "Contents" / "Info.plist").write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleShortVersionString</key>
  <string>{version}</string>
</dict>
</plist>
""",
        encoding="utf-8",
    )


def test_default_mode_keeps_release_alignment_contract(tmp_path: Path) -> None:
    repo, fake_bin, app_bundle = _make_repo(tmp_path)

    result = _run_release_status(repo, fake_bin, app_bundle)

    assert result.returncode == 0
    assert "=> aligned: every channel is on 0.6.1." in result.stdout
    assert "Beta rehearsal" not in result.stdout


def test_private_repo_404_is_expected_in_beta_rehearsal(tmp_path: Path) -> None:
    repo, fake_bin, app_bundle = _make_repo(tmp_path)

    result = _run_release_status(repo, fake_bin, app_bundle, "--beta-rehearsal")

    assert result.returncode == 0
    assert "repo visibility" in result.stdout
    assert "PRIVATE" in result.stdout
    assert "private_expected (HTTP 404; token-gated release)" in result.stdout
    assert "brew cask receipt" in result.stdout
    assert "0.6.0" in result.stdout
    assert "app signature" in result.stdout
    assert "developer_id (ok)" in result.stdout
    assert "app notarization" in result.stdout
    assert "stapled" in result.stdout
    assert "PATH status" in result.stdout
    assert "bundled app CLI" in result.stdout


def test_public_repo_404_is_beta_blocker(tmp_path: Path) -> None:
    repo, fake_bin, app_bundle = _make_repo(tmp_path)

    result = _run_release_status(
        repo,
        fake_bin,
        app_bundle,
        "--beta-rehearsal",
        repo_visibility="PUBLIC",
    )

    assert result.returncode == 1
    assert "DMG public download" in result.stdout
    assert "blocked (HTTP 404)" in result.stdout
    assert "=> beta blocker:" in result.stdout


def test_adhoc_selected_app_is_beta_blocker(tmp_path: Path) -> None:
    repo, fake_bin, app_bundle = _make_repo(tmp_path)

    result = _run_release_status(
        repo,
        fake_bin,
        app_bundle,
        "--beta-rehearsal",
        codesign_mode="adhoc",
    )

    assert result.returncode == 1
    assert "app signature" in result.stdout
    assert "adhoc (adhoc identity)" in result.stdout
    assert str(app_bundle) in result.stdout
    assert "=> beta blocker:" in result.stdout
    assert "installed app signing blocker" in result.stdout


def test_apple_development_signature_is_beta_blocker_even_with_team_id(tmp_path: Path) -> None:
    repo, fake_bin, app_bundle = _make_repo(tmp_path)

    result = _run_release_status(
        repo,
        fake_bin,
        app_bundle,
        "--beta-rehearsal",
        codesign_mode="apple_development",
    )

    assert result.returncode == 1
    assert "app signature" in result.stdout
    assert "other_signed (non-Developer ID signature)" in result.stdout
    assert "app team id" in result.stdout
    assert "ABCDE12345" in result.stdout
    assert "installed app signing blocker" in result.stdout


def test_unsigned_selected_app_is_beta_blocker(tmp_path: Path) -> None:
    repo, fake_bin, app_bundle = _make_repo(tmp_path)

    result = _run_release_status(
        repo,
        fake_bin,
        app_bundle,
        "--beta-rehearsal",
        codesign_mode="unsigned",
    )

    assert result.returncode == 1
    assert "app signature" in result.stdout
    assert "unsigned (unsigned app bundle)" in result.stdout
    assert "installed app signing blocker" in result.stdout


def test_developer_id_without_stapled_ticket_is_beta_blocker(tmp_path: Path) -> None:
    repo, fake_bin, app_bundle = _make_repo(tmp_path)

    result = _run_release_status(
        repo,
        fake_bin,
        app_bundle,
        "--beta-rehearsal",
        stapler_mode="absent",
    )

    assert result.returncode == 1
    assert "app signature" in result.stdout
    assert "developer_id (notarization not stapled)" in result.stdout
    assert "app notarization" in result.stdout
    assert "absent" in result.stdout
    assert "installed app signing blocker" in result.stdout


def test_published_dmg_blocks_stale_installed_app_version(tmp_path: Path) -> None:
    repo, fake_bin, app_bundle = _make_repo(tmp_path)
    _set_app_version(app_bundle, "0.6.0")

    result = _run_release_status(repo, fake_bin, app_bundle, "--beta-rehearsal")

    assert result.returncode == 1
    assert "installed app" in result.stdout
    assert "0.6.0" in result.stdout
    assert "installed app version blocker" in result.stdout
    assert "published target is 0.6.1" in result.stdout


def test_unpublished_dmg_keeps_stale_installed_app_version_advisory(tmp_path: Path) -> None:
    repo, fake_bin, app_bundle = _make_repo(tmp_path)
    _set_app_version(app_bundle, "0.6.0")

    result = _run_release_status(
        repo,
        fake_bin,
        app_bundle,
        "--beta-rehearsal",
        dmg_published="no",
    )

    assert result.returncode == 0
    assert "installed app" in result.stdout
    assert "0.6.0" in result.stdout
    assert "installed app version blocker" not in result.stdout
    assert "=> aligned: every channel is on 0.6.1." in result.stdout


def test_curl_timeout_is_unknown_and_nonblocking(tmp_path: Path) -> None:
    repo, fake_bin, app_bundle = _make_repo(tmp_path)

    result = _run_release_status(
        repo,
        fake_bin,
        app_bundle,
        "--beta-rehearsal",
        repo_visibility="PUBLIC",
        curl_mode="timeout",
    )

    assert result.returncode == 0
    assert "DMG public download" in result.stdout
    assert "source public download" in result.stdout
    assert "unknown (http-000)" in result.stdout


def test_missing_homebrew_receipts_are_advisory(tmp_path: Path) -> None:
    repo, fake_bin, app_bundle = _make_repo(tmp_path)

    result = _run_release_status(
        repo,
        fake_bin,
        app_bundle,
        "--beta-rehearsal",
        brew_mode="not-installed",
    )

    assert result.returncode == 0
    assert "brew cask receipt" in result.stdout
    assert "brew formula receipt" in result.stdout
    assert "not-installed" in result.stdout
