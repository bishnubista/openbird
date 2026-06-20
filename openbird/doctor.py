"""``openbird doctor`` — a content-safe, shareable diagnostic for beta support.

Built as a redaction/projection layer over :func:`openbird.preflight.run_preflight`
plus a few macOS-only probes (code-signing identity, quarantine state) that
preflight does not cover. Two hard contracts:

* **Content-safe.** The report carries only versions, paths (home-redacted),
  signing identities, counts, and booleans — never captured text, window titles,
  URLs, allow/block-list *values*, or secrets. Every string is scrubbed.
* **Never crashes.** Any probe (and the whole build) degrades to ``"unknown"`` /
  an error *type* rather than raising, so a struggling tester can always produce a
  report.

All external seams (preflight runner, system info, helper resolver, command
runner, home dir) are injectable for tests.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from collections.abc import Callable, Sequence
from typing import Any

from openbird.capture.daemon import DEFAULT_SIGNED_HELPER_PATH, HELPER_PATH_ENV
from openbird.config import Settings, get_settings
from openbird.preflight import run_preflight

# Token-shaped secrets to redact defensively from any emitted string. This is a
# small local set (not the full capture redaction engine) to avoid coupling the
# diagnostic to the capture pipeline; the report should never contain secrets in
# the first place, so this is defense-in-depth.
_TOKEN_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"ghp_[A-Za-z0-9]{8,}"),
    re.compile(r"AKIA[0-9A-Z]{12,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{8,}"),
    re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),  # JWT
)

CommandRunner = Callable[[Sequence[str], float], "tuple[int, str, str] | None"]


def _default_command_runner(argv: Sequence[str], timeout: float) -> tuple[int, str, str] | None:
    """Run ``argv`` safely. Returns ``(rc, stdout, stderr)`` or ``None`` on any
    failure (missing binary, timeout, OS error). Never raises."""
    try:
        proc = subprocess.run(  # noqa: S603 - argv list, no shell, fixed commands
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except (OSError, subprocess.SubprocessError):
        return None


def _redact_text(value: str, home: str) -> str:
    """Home-redact and scrub secrets (token-shaped, URL userinfo, credential
    assignments) from a single string."""
    out = value
    if home and home != "/" and home in out:
        out = out.replace(home, "~")
    # URL userinfo: scheme://user:pass@host -> scheme://<redacted>@host
    out = re.sub(r"://[^/\s:@]+:[^/\s@]+@", "://<redacted>@", out)
    # generic credential assignments: password=..., token: ..., api_key=...
    out = re.sub(
        r"(?i)(password|passwd|pwd|secret|token|api[_-]?key)(\s*[=:]\s*)\S+",
        r"\1\2<redacted>",
        out,
    )
    for pat in _TOKEN_PATTERNS:
        out = pat.sub("<redacted>", out)
    return out


def _scrub(obj: Any, home: str) -> Any:
    """Recursively home-redact + secret-scrub every string — KEYS and values —
    in a JSON-ish value (preflight keys some maps by model/host strings)."""
    if isinstance(obj, str):
        return _redact_text(obj, home)
    if isinstance(obj, dict):
        return {
            (_redact_text(k, home) if isinstance(k, str) else k): _scrub(v, home)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_scrub(v, home) for v in obj]
    return obj


def _project_preflight(report: dict[str, Any]) -> dict[str, Any]:
    """Replace allow/block-list *values* with counts before the report is shared.

    Bundle identifiers a user captures are identifying; the diagnostic only needs
    how many are configured, not which.
    """
    privacy = report.get("privacy")
    if isinstance(privacy, dict):
        allow = privacy.get("allowlist") or []
        block = privacy.get("blocklist") or []
        report = {**report}
        report["privacy"] = {
            **{k: v for k, v in privacy.items() if k not in ("allowlist", "blocklist")},
            "allowlist_count": len(allow) if isinstance(allow, (list, tuple)) else 0,
            "blocklist_count": len(block) if isinstance(block, (list, tuple)) else 0,
            "allowlist_empty": not allow,
        }
    return report


def _resolve_helper_path() -> str:
    return os.environ.get(HELPER_PATH_ENV) or DEFAULT_SIGNED_HELPER_PATH


def _app_bundle_for(helper_path: str) -> str | None:
    """Derive the ``.app`` bundle path from a bundled helper at
    ``…/OpenBird.app/Contents/MacOS/<helper>``."""
    marker = "/Contents/MacOS/"
    idx = helper_path.find(marker)
    return helper_path[:idx] if idx != -1 else None


def _signing_info(helper_path: str, runner: CommandRunner, timeout: float) -> dict[str, Any]:
    """Best-effort code-signing identity of the bundled helper. Parses only known
    fields from ``codesign`` output (which goes to *stderr* on success); never
    emits raw output."""
    if not os.path.exists(helper_path):
        return {"signed": "unknown", "reason": "helper-not-found"}
    res = runner(["codesign", "-dvv", helper_path], timeout)
    if res is None:
        return {"signed": "unknown", "reason": "codesign-unavailable"}
    rc, out, err = res
    blob = f"{out}\n{err}"  # codesign -dvv prints to stderr
    if rc != 0:
        return {"signed": False, "reason": "not-signed"}
    info: dict[str, Any] = {"signed": True}
    for line in blob.splitlines():
        line = line.strip()
        if line.startswith("Authority=") and "authority" not in info:
            info["authority"] = line.split("=", 1)[1]
        elif line.startswith("TeamIdentifier="):
            info["team_identifier"] = line.split("=", 1)[1]
    dr = runner(["codesign", "-dr", "-", helper_path], timeout)
    if dr is not None and dr[0] == 0:
        # `designated => ...` is emitted to stdout
        for line in (dr[1] + "\n" + dr[2]).splitlines():
            if "designated =>" in line:
                info["designated_requirement"] = line.split("designated =>", 1)[1].strip()
                break
    return info


def _quarantine_state(helper_path: str, runner: CommandRunner, timeout: float) -> str:
    """present / absent / unknown — never the attribute value."""
    app = _app_bundle_for(helper_path)
    if not app or not os.path.exists(app):
        return "unknown"
    res = runner(["xattr", "-p", "com.apple.quarantine", app], timeout)
    if res is None:
        return "unknown"
    rc, _out, err = res
    if rc == 0:
        return "present"
    # Only the known "attribute not set" case is a confident "absent"; permission
    # or other failures must degrade to "unknown" rather than imply not-quarantined.
    low = (err or "").lower()
    if "no such xattr" in low or "no such attr" in low or "could not be found" in low:
        return "absent"
    return "unknown"


def build_doctor_report(
    settings: Settings | None = None,
    *,
    preflight_runner: Callable[..., dict[str, Any]] = run_preflight,
    probe_ollama: bool = True,
    system_info: Callable[[], dict[str, str]] | None = None,
    helper_path: str | None = None,
    command_runner: CommandRunner = _default_command_runner,
    home: str | None = None,
    probe_timeout: float = 5.0,
    is_macos: bool | None = None,
) -> dict[str, Any]:
    """Assemble the content-safe diagnostic. Never raises."""
    home = home if home is not None else os.path.expanduser("~")
    if is_macos is None:
        is_macos = platform.system() == "Darwin"

    def _sys() -> dict[str, str]:
        try:
            from importlib.metadata import version

            ob_version = version("openbird")
        except Exception:  # noqa: BLE001 - version is best-effort
            ob_version = "unknown"
        return {
            "openbird_version": ob_version,
            "os": platform.system(),
            "os_version": platform.mac_ver()[0] or platform.release(),
            "arch": platform.machine(),
            "python": platform.python_version(),
        }

    report: dict[str, Any] = {}
    try:
        report["system"] = (system_info or _sys)()

        try:
            pf = preflight_runner(settings or get_settings(), probe_ollama=probe_ollama)
            report["preflight"] = _project_preflight(pf)
            report["runtime_ok"] = bool(pf.get("runtime_ok"))
        except Exception as exc:  # noqa: BLE001 - doctor's contract is stronger than preflight's
            report["preflight"] = {"error": type(exc).__name__}
            report["runtime_ok"] = False

        hp = helper_path if helper_path is not None else _resolve_helper_path()
        if is_macos:
            report["signing"] = _signing_info(hp, command_runner, probe_timeout)
            report["quarantine"] = _quarantine_state(hp, command_runner, probe_timeout)
        else:
            report["signing"] = {"signed": "n/a"}
            report["quarantine"] = "n/a"
        report["helper_path"] = hp
    except Exception as exc:  # noqa: BLE001 - never crash
        report.setdefault("runtime_ok", False)
        report["error"] = type(exc).__name__

    return _scrub(report, home)


def render(report: dict[str, Any]) -> str:
    """Human-readable, copy-pasteable summary."""
    sys_i = report.get("system", {})
    pf = report.get("preflight", {})
    lines = [
        "OpenBird doctor",
        "===============",
        f"version : {sys_i.get('openbird_version', '?')}",
        f"system  : {sys_i.get('os', '?')} {sys_i.get('os_version', '')} ({sys_i.get('arch', '?')}), py {sys_i.get('python', '?')}",
        f"runtime : {'OK' if report.get('runtime_ok') else 'NOT READY'}",
    ]
    sign = report.get("signing", {})
    lines.append(
        f"signing : {sign.get('authority', sign.get('signed', '?'))}"
        + (f"  team={sign['team_identifier']}" if sign.get("team_identifier") else "")
    )
    lines.append(f"quarant.: {report.get('quarantine', '?')}")
    if isinstance(pf, dict):
        priv = pf.get("privacy", {})
        if priv.get("allowlist_empty"):
            lines.append("allowlist: EMPTY — capture records nothing until you add apps in Setup")
        else:
            lines.append(f"allowlist: {priv.get('allowlist_count', '?')} app(s)")
        ollama = pf.get("ollama", {})
        if isinstance(ollama, dict) and "reachable" in ollama:
            reachable = ollama.get("reachable")
            if reachable is True:
                label = "reachable"
            elif reachable is False:
                label = "unreachable"
            else:  # preserve tri-state (e.g. "unknown") instead of coercing to truthy
                label = str(reachable) if reachable is not None else "unknown"
            lines.append(f"ollama  : {label}")
    if report.get("error"):
        lines.append(f"error   : {report['error']}")
    return "\n".join(lines)
