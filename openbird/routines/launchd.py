"""launchd LaunchAgent generation for the routine daemon [B2].

The routine scheduler is only durable if something actually runs
``openbird routine start`` and keeps it alive. On macOS the right OS-level
supervisor is a per-user **LaunchAgent** (``~/Library/LaunchAgents``), which
starts the daemon at login and restarts it if it crashes.

This module builds the plist deterministically (``plistlib``) and resolves the
agent path; it does NOT load/unload launchd itself — that is an explicit,
user-driven step in the CLI (``--load`` / ``--unload``), since modifying the
running launchd domain is a system-state change the user should opt into.

Privacy [R4/R5]: the daemon runs with the content-safe ``null_deliverer``, so
its stdout has no summary bodies. We still route stderr to a ``0600`` log under
the data dir (metadata-only scheduler logs) rather than the default system log.
"""

from __future__ import annotations

import plistlib
from pathlib import Path

# Reverse-DNS LaunchAgent label; also the plist filename stem.
AGENT_LABEL = "ai.openbird.routines"


def agent_plist_path(*, home: Path | None = None) -> Path:
    """Return the per-user LaunchAgent plist path for the routine daemon."""
    base = (home or Path.home()) / "Library" / "LaunchAgents"
    return base / f"{AGENT_LABEL}.plist"


def build_agent_plist(
    *,
    program_args: list[str],
    stderr_path: str,
    throttle_seconds: int = 60,
    label: str = AGENT_LABEL,
) -> bytes:
    """Build the LaunchAgent plist bytes for ``program_args``.

    Args:
        program_args: The argv to exec, e.g. ``["/usr/local/bin/openbird",
            "routine", "start"]``. The caller resolves the absolute openbird
            path at install time (launchd does not consult ``PATH``).
        stderr_path: Absolute path for the daemon's stderr (metadata-only logs).
        throttle_seconds: Minimum seconds between (re)launches, so a crash-loop
            cannot spin hot.
        label: The launchd label / plist stem.

    Returns:
        The plist serialized as XML bytes (write directly to the agent path).
    """
    if not program_args:
        raise ValueError("program_args must be non-empty")
    if int(throttle_seconds) < 1:
        raise ValueError("throttle_seconds must be >= 1")
    if not stderr_path or not stderr_path.strip():
        raise ValueError("stderr_path must be a non-empty path")
    plist: dict[str, object] = {
        "Label": label,
        "ProgramArguments": list(program_args),
        # Start at login and keep alive — but do NOT restart after a clean exit
        # (e.g. an explicit `launchctl unload`/SIGTERM), only after a crash.
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        # Don't relaunch faster than this; prevents a crash-loop from spinning.
        "ThrottleInterval": int(throttle_seconds),
        # Background QoS — this is a low-priority always-on helper.
        "ProcessType": "Background",
        # Metadata-only scheduler logs (the daemon delivers no content to stdout).
        "StandardErrorPath": stderr_path,
    }
    return plistlib.dumps(plist)
