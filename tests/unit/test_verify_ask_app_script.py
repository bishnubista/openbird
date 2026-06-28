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


def _make_app_bundle(root: Path, relative: str) -> Path:
    bundle = root / relative
    _write_executable(
        bundle / "Contents" / "MacOS" / "OpenBird",
        "#!/usr/bin/env bash\nexit 0\n",
    )
    return bundle


def _make_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    (repo / "script").mkdir(parents=True)
    shutil.copy(ROOT / "script" / "verify_ask_app.sh", repo / "script" / "verify_ask_app.sh")
    (repo / "script" / "verify_ask_app.sh").chmod(0o755)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    pkill_log = tmp_path / "pkill.log"
    _write_executable(
        fake_bin / "pkill",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "${FAKE_PKILL_LOG}"
exit 1
""",
    )
    _write_executable(
        fake_bin / "gtimeout",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --signal=* ]]; then
  shift
fi
shift
"$@"
""",
    )
    return repo, fake_bin, pkill_log


def _run_verify(
    repo: Path,
    fake_bin: Path,
    pkill_log: Path,
    *args: str,
    app_env: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "FAKE_PKILL_LOG": str(pkill_log),
            "OPENBIRD_VERIFY_WAIT": "1",
            "OPENBIRD_VERIFY_OUT": str(repo / ".verify-out"),
        }
    )
    if app_env is not None:
        env["OPENBIRD_VERIFY_APP_PATH"] = str(app_env)
    else:
        env.pop("OPENBIRD_VERIFY_APP_PATH", None)
    return subprocess.run(
        [str(repo / "script" / "verify_ask_app.sh"), *args],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_default_target_remains_dist_and_allows_dist_cleanup(tmp_path: Path) -> None:
    repo, fake_bin, pkill_log = _make_repo(tmp_path)
    default_app = _make_app_bundle(repo, "dist/OpenBird.app")

    result = _run_verify(repo, fake_bin, pkill_log)

    assert result.returncode == 2
    assert f"target app: {default_app}" in result.stderr
    assert "VERDICT: BLOCKED" in result.stdout
    assert "dist/OpenBird.app/Contents/MacOS/OpenBird" in pkill_log.read_text(encoding="utf-8")


def test_app_flag_targets_installed_bundle_without_cleanup(tmp_path: Path) -> None:
    repo, fake_bin, pkill_log = _make_repo(tmp_path)
    custom_app = _make_app_bundle(tmp_path, "Applications/OpenBird.app")

    result = _run_verify(repo, fake_bin, pkill_log, "--app", str(custom_app))

    assert result.returncode == 2
    assert f"target app: {custom_app}" in result.stderr
    assert "VERDICT: BLOCKED" in result.stdout
    assert not pkill_log.exists()


def test_app_env_targets_installed_bundle_without_cleanup(tmp_path: Path) -> None:
    repo, fake_bin, pkill_log = _make_repo(tmp_path)
    custom_app = _make_app_bundle(tmp_path, "Applications/OpenBird.app")

    result = _run_verify(repo, fake_bin, pkill_log, app_env=custom_app)

    assert result.returncode == 2
    assert f"target app: {custom_app}" in result.stderr
    assert not pkill_log.exists()


def test_app_flag_overrides_app_env(tmp_path: Path) -> None:
    repo, fake_bin, pkill_log = _make_repo(tmp_path)
    env_app = _make_app_bundle(tmp_path, "Env/OpenBird.app")
    flag_app = _make_app_bundle(tmp_path, "Flag/OpenBird.app")

    result = _run_verify(repo, fake_bin, pkill_log, "--app", str(flag_app), app_env=env_app)

    assert result.returncode == 2
    assert f"target app: {flag_app}" in result.stderr
    assert str(env_app) not in result.stderr
    assert not pkill_log.exists()


def test_build_with_custom_app_is_rejected_before_cleanup(tmp_path: Path) -> None:
    repo, fake_bin, pkill_log = _make_repo(tmp_path)
    custom_app = _make_app_bundle(tmp_path, "Applications/OpenBird.app")

    result = _run_verify(repo, fake_bin, pkill_log, "--app", str(custom_app), "--build")

    assert result.returncode == 2
    assert "--build only applies to the default dist app target" in result.stderr
    assert "VERDICT: BLOCKED" in result.stdout
    assert not pkill_log.exists()


def test_build_with_custom_app_env_is_rejected_before_cleanup(tmp_path: Path) -> None:
    repo, fake_bin, pkill_log = _make_repo(tmp_path)
    custom_app = _make_app_bundle(tmp_path, "Applications/OpenBird.app")

    result = _run_verify(repo, fake_bin, pkill_log, "--build", app_env=custom_app)

    assert result.returncode == 2
    assert "--build only applies to the default dist app target" in result.stderr
    assert not pkill_log.exists()
