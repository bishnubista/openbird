#!/usr/bin/env python3
"""Install a completed OpenBird release and audit the real local capture."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import os
import plistlib
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
APP = Path("/Applications/OpenBird.app")
CLI = APP / "Contents" / "MacOS" / "openbird-cli"
LIVENESS = Path.home() / ".openbird" / "capture.liveness.json"
REPORT_DIR = Path.home() / ".openbird" / "audits"


class Blocked(RuntimeError):
    """A release/install proof could not be completed safely."""


def _run(args: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            args,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise Blocked(f"command_timeout:{Path(args[0]).name}") from exc
    except OSError as exc:
        raise Blocked(f"command_unavailable:{Path(args[0]).name}") from exc
    if result.returncode != 0:
        raise Blocked(f"command_failed:{Path(args[0]).name}:rc_{result.returncode}")
    return result


def _json_command(args: list[str], *, timeout: float = 60.0) -> dict[str, Any]:
    result = _run(args, timeout=timeout)
    try:
        payload = json.loads(result.stdout)
    except (TypeError, ValueError) as exc:
        raise Blocked(f"unparseable_json:{Path(args[0]).name}") from exc
    if not isinstance(payload, dict):
        raise Blocked(f"unparseable_json:{Path(args[0]).name}")
    return payload


def _source_version() -> str:
    in_project = False
    version_pattern = re.compile(r'^version\s*=\s*"([^"]+)"$')
    for raw_line in (ROOT / "pyproject.toml").read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            continue
        if in_project and (match := version_pattern.fullmatch(line)):
            return match.group(1)
    raise Blocked("source_version_unavailable")


def _bundle_version() -> str:
    try:
        with (APP / "Contents" / "Info.plist").open("rb") as handle:
            return str(plistlib.load(handle)["CFBundleShortVersionString"])
    except (KeyError, OSError, plistlib.InvalidFileException) as exc:
        raise Blocked("installed_bundle_version_unavailable") from exc


def _read_old_instance() -> str | None:
    try:
        raw = json.loads(LIVENESS.read_text())
        return str(uuid.UUID(raw.get("instance_uuid")))
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _pid_executable(pid: int) -> Path | None:
    """Read a PID's kernel executable path using macOS libproc."""
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
        buffer = ctypes.create_string_buffer(4096)
        size = libproc.proc_pidpath(pid, buffer, len(buffer))
        if size <= 0:
            return None
        return Path(os.path.realpath(os.fsdecode(buffer.value)))
    except (OSError, ValueError):
        return None


def _is_app_process(pid: int) -> bool:
    # Packaged capture runs through the bundled interpreter. A different
    # launcher image intentionally fails closed even if its argv names OpenBird.
    executable = _pid_executable(pid)
    if executable is None:
        return False
    try:
        executable.relative_to(APP.resolve())
        return True
    except ValueError:
        return False


def _app_processes() -> set[int]:
    try:
        result = subprocess.run(
            ["pgrep", "-f", str(APP)],
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise Blocked("process_discovery_unavailable") from exc
    if result.returncode not in (0, 1):
        raise Blocked(f"process_discovery_failed:rc_{result.returncode}")
    candidates = {int(value) for value in result.stdout.split() if value.isdigit()}
    return {pid for pid in candidates if _is_app_process(pid)}


def _wait_for_exit(pids: set[int], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = {pid for pid in pids if _is_app_process(pid)}
        if not remaining:
            return
        time.sleep(0.5)
    raise Blocked("old_app_process_still_running")


def _quit_current_app() -> str | None:
    old_instance = _read_old_instance()
    old_pids = _app_processes()
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                'tell application id "ai.openbird.OpenBird" to quit',
            ],
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Blocked("app_quit_unavailable") from exc
    _wait_for_exit(old_pids, 30.0)
    # Remove only the metadata sidecar after every prior app-rooted process is
    # gone. The next file must therefore be written by the new daemon instance.
    LIVENESS.unlink(missing_ok=True)
    return old_instance


def _install_release() -> str | None:
    old_instance = _quit_current_app()
    _run(["brew", "update"], timeout=300.0)
    _run(
        ["brew", "reinstall", "--cask", "--force", "openbird"],
        timeout=600.0,
    )
    _run(["open", str(APP)], timeout=30.0)
    return old_instance


def _validate_daemon(
    audit: dict[str, Any], *, expected_version: str, old_instance: str | None
) -> None:
    daemon = audit.get("daemon")
    if not isinstance(daemon, dict) or daemon.get("state") != "ok":
        raise Blocked("capture_daemon_not_fresh")
    instance = daemon.get("instance_uuid")
    try:
        instance = str(uuid.UUID(instance))
    except (AttributeError, TypeError, ValueError) as exc:
        raise Blocked("capture_daemon_identity_missing") from exc
    if old_instance is not None and instance == old_instance:
        raise Blocked("capture_daemon_identity_not_rotated")
    pid = daemon.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or not _is_app_process(pid):
        raise Blocked("capture_daemon_not_rooted_in_installed_app")
    if daemon.get("runtime_version") != expected_version:
        raise Blocked("capture_daemon_version_mismatch")


def _wait_for_audit(
    *,
    expected_version: str,
    old_instance: str | None,
    recent_window_seconds: float,
    minimum_samples: int,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_reason = "capture_audit_unavailable"
    while time.monotonic() < deadline:
        try:
            remaining = max(0.1, deadline - time.monotonic())
            audit = _json_command(
                [
                    str(CLI),
                    "data",
                    "capture-audit",
                    "--json",
                    "--recent-window-seconds",
                    str(recent_window_seconds),
                    "--minimum-samples",
                    str(minimum_samples),
                ],
                timeout=min(15.0, remaining),
            )
            _validate_daemon(
                audit,
                expected_version=expected_version,
                old_instance=old_instance,
            )
            return audit
        except Blocked as exc:
            last_reason = str(exc)
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    raise Blocked(last_reason)


_QUALITY_RANK = {
    "low_context": 0,
    "inconsistent_context": 1,
    "usable_context": 2,
    "rich_context": 3,
}
_CONTEXT_QUALITIES = frozenset(
    {
        "unavailable",
        "insufficient_data",
        "low_context",
        "inconsistent_context",
        "usable_context",
        "rich_context",
    }
)


def _validated_app_rows(audit: Any, *, reason: str) -> dict[str, dict[str, Any]]:
    if not isinstance(audit, dict) or not isinstance(audit.get("apps"), list):
        raise Blocked(reason)
    rows: dict[str, dict[str, Any]] = {}
    for row in audit["apps"]:
        if not isinstance(row, dict):
            raise Blocked(reason)
        bundle_id = row.get("bundle_id")
        samples = row.get("sample_count")
        quality = row.get("context_quality")
        if (
            not isinstance(bundle_id, str)
            or not bundle_id
            or not isinstance(samples, int)
            or isinstance(samples, bool)
            or samples < 0
            or quality not in _CONTEXT_QUALITIES
        ):
            raise Blocked(reason)
        rows[bundle_id] = row
    return rows


def compare_audits(
    current: dict[str, Any], previous: dict[str, Any] | None, *, minimum_samples: int
) -> list[dict[str, Any]]:
    """Compare stable quality buckets; repetition remains advisory only."""
    if previous is None:
        return []
    current_apps = _validated_app_rows(current, reason="current_audit_malformed")
    if not isinstance(previous, dict):
        raise Blocked("prior_report_malformed")
    previous_apps = _validated_app_rows(
        previous.get("capture_audit"), reason="prior_report_malformed"
    )
    comparisons: list[dict[str, Any]] = []
    for bundle_id in sorted(current_apps.keys() & previous_apps.keys()):
        new = current_apps[bundle_id]
        old = previous_apps[bundle_id]
        if (
            min(new.get("sample_count", 0), old.get("sample_count", 0))
            < minimum_samples
        ):
            continue
        new_rank = _QUALITY_RANK.get(new.get("context_quality"))
        old_rank = _QUALITY_RANK.get(old.get("context_quality"))
        if new_rank is None or old_rank is None:
            continue
        if new_rank > old_rank:
            change = "improved"
        elif new_rank < old_rank:
            change = "regressed"
        else:
            change = "unchanged"
        comparisons.append(
            {
                "bundle_id": bundle_id,
                "change": change,
                "previous_quality": old["context_quality"],
                "current_quality": new["context_quality"],
            }
        )
    return comparisons


def _latest_report() -> dict[str, Any] | None:
    candidates: list[tuple[int, Path]] = []
    for path in REPORT_DIR.glob("post-release-*.json"):
        try:
            candidates.append((path.stat().st_mtime_ns, path))
        except OSError:
            continue
    malformed_found = False
    for _mtime, path in sorted(candidates, reverse=True):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            malformed_found = True
            continue
        try:
            if not isinstance(payload, dict) or payload.get("schema_version") != 1:
                raise Blocked("prior_report_malformed")
            _validated_app_rows(
                payload.get("capture_audit"), reason="prior_report_malformed"
            )
        except Blocked:
            malformed_found = True
            continue
        return payload
    if malformed_found:
        raise Blocked("prior_report_malformed")
    return None


def _write_report(report: dict[str, Any], version: str) -> Path:
    REPORT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(REPORT_DIR, 0o700)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = REPORT_DIR / f"post-release-{version}-{stamp}.json"
    fd, temp_name = tempfile.mkstemp(prefix=".post-release-", dir=REPORT_DIR)
    temp_path = Path(temp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    except Exception:
        if fd >= 0:
            os.close(fd)
        temp_path.unlink(missing_ok=True)
        raise
    return destination


def _assistant_summary() -> dict[str, bool]:
    payload = _json_command([str(CLI), "assistant", "status", "--json"])
    command = payload.get("command")
    uses_installed_cli = False
    if isinstance(command, str):
        uses_installed_cli = Path(command).resolve() == CLI.resolve()
    return {
        "configured": bool(payload.get("configured")),
        "uses_installed_cli": uses_installed_cli,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install and privacy-safely audit a completed OpenBird release."
    )
    parser.add_argument(
        "--install", action="store_true", help="Reinstall the cask before auditing."
    )
    parser.add_argument(
        "--expected-version",
        help="Version that must be installed; defaults to pyproject.",
    )
    parser.add_argument("--recent-window-seconds", type=float, default=86400.0)
    parser.add_argument("--minimum-samples", type=int, default=5)
    parser.add_argument("--wait-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)

    try:
        if args.minimum_samples < 1:
            raise Blocked("minimum_samples_must_be_positive")
        if args.recent_window_seconds <= 0 or args.wait_seconds <= 0:
            raise Blocked("time_window_must_be_positive")
        expected = args.expected_version or _source_version()
        if expected != _source_version():
            raise Blocked("expected_version_differs_from_source")
        _run([str(ROOT / "script" / "release_status.sh")], timeout=90.0)
        old_instance = _install_release() if args.install else None
        if not CLI.is_file() or not os.access(CLI, os.X_OK):
            raise Blocked("installed_cli_missing")
        if _bundle_version() != expected:
            raise Blocked("installed_bundle_version_mismatch")
        _run(
            [str(ROOT / "script" / "release_status.sh"), "--beta-rehearsal"],
            timeout=120.0,
        )
        previous = _latest_report()
        audit = _wait_for_audit(
            expected_version=expected,
            old_instance=old_instance,
            recent_window_seconds=args.recent_window_seconds,
            minimum_samples=args.minimum_samples,
            timeout=args.wait_seconds,
        )
        if audit.get("overall_state") == "blocked":
            raise Blocked("capture_audit_blocked")
        comparison = compare_audits(
            audit, previous, minimum_samples=args.minimum_samples
        )
        report = {
            "schema_version": 1,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "version": expected,
            "installed_app": str(APP),
            "capture_audit": audit,
            "comparison": comparison,
            "assistant": _assistant_summary(),
        }
        report_path = _write_report(report, expected)
    except Blocked as exc:
        print(f"BLOCKED post-release audit: {exc}")
        return 2
    except OSError:
        print("BLOCKED post-release audit: local_io_error")
        return 2

    print(f"PASS    installed release: {expected}")
    print("PASS    capture daemon: fresh installed-app process")
    print(f"{audit['overall_state'].upper():7} capture context: {audit['summary']}")
    for row in audit.get("apps", []):
        print(
            f"        {row['bundle_id']}: {row['context_quality']} "
            f"samples={row['sample_count']} chars={row['chars_p50']}/{row['chars_p90']}"
        )
    for row in comparison:
        print(f"        {row['bundle_id']}: {row['change']} since prior audit")
    print(f"PASS    private report: {report_path} (mode 0600)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
