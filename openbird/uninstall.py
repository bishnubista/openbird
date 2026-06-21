"""System-state cleanup for ``openbird uninstall``.

Removes the OpenBird state that lingers after the app is trashed — the routines
launchd job + LaunchAgent, stale Launch Services registrations, pause/lock files,
and (only when no encrypted DB depends on it) the Keychain DB key. Captured data
in the data dir is PRESERVED unless ``purge_data`` is set.

Design + Codex consensus: ``docs/design/cleanup-tooling.md``. Content-safe: results
carry only paths and reason codes, never captured text.
"""

from __future__ import annotations

import plistlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from openbird.config import data_dir_path, db_file_path

# The OpenBird app bundle identifier. A Launch Services registration is only
# unregistered after its bundle is confirmed to carry THIS id (Codex #5).
BUNDLE_ID = "ai.openbird.OpenBird"

# launchd label for the routines agent (mirrors routines.launchd.AGENT_LABEL).
ROUTINES_LABEL = "ai.openbird.routines"

# The Launch Services registration tool (fixed system path).
_LSREGISTER = (
    "/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/"
    "LaunchServices.framework/Versions/A/Support/lsregister"
)

# Pause/lock sidecars the app/daemon leave in the data dir.
_SIDECAR_NAMES = ("capture.paused",)


@dataclass(frozen=True)
class StepResult:
    """One cleanup step's outcome (content-safe — paths/reason codes only)."""

    action: str
    status: str  # "done" | "skip" | "retained" | "would" | "error"
    detail: str


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _run(cmd: list[str]) -> tuple[int, str]:
    """Run a command, returning (rc, combined-output). Never raises."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=20
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, type(exc).__name__
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _bundle_id_of(app_path: Path) -> str | None:
    """Read CFBundleIdentifier from an app bundle's Info.plist, or None."""
    info = app_path / "Contents" / "Info.plist"
    try:
        with info.open("rb") as fh:
            data = plistlib.load(fh)
    except (OSError, plistlib.InvalidFileException, ValueError):
        return None
    value = data.get("CFBundleIdentifier")
    return value if isinstance(value, str) else None


def remove_routines_job(*, dry_run: bool) -> list[StepResult]:
    """Clear the routines launchd job AND its LaunchAgent plist.

    Boots out / removes the job by LABEL first so an orphaned, still-loaded job
    (plist already gone — the observed zombie) is cleared too, then unlinks the
    plist if present. macOS-only; a no-op elsewhere.
    """
    if not _is_macos():
        return [StepResult("routines launchd job", "skip", "not macOS")]

    from openbird.routines.launchd import agent_plist_path

    results: list[StepResult] = []
    uid = _current_uid()
    label_target = f"gui/{uid}/{ROUTINES_LABEL}" if uid is not None else ROUTINES_LABEL

    if dry_run:
        results.append(
            StepResult("routines launchd job", "would", f"bootout {label_target}")
        )
    else:
        results.append(_boot_out_routines(label_target))

    plist = agent_plist_path()
    if plist.exists():
        if dry_run:
            results.append(StepResult("routines LaunchAgent", "would", str(plist)))
        else:
            try:
                plist.unlink()
                results.append(StepResult("routines LaunchAgent", "done", str(plist)))
            except OSError as exc:
                results.append(
                    StepResult("routines LaunchAgent", "error", type(exc).__name__)
                )
    else:
        results.append(StepResult("routines LaunchAgent", "skip", "no plist"))
    return results


# launchctl messages that mean "the job simply wasn't loaded" — benign, not a
# failure (a fresh machine, or already unloaded). Anything else is a real error.
_NOT_LOADED_MARKERS = (
    "no such process",
    "could not find",
    "not find",
    "no such file",
    "not loaded",
    "disabled",
)


def _is_not_loaded(output: str) -> bool:
    low = output.lower()
    return any(marker in low for marker in _NOT_LOADED_MARKERS)


def _boot_out_routines(label_target: str) -> StepResult:
    """bootout (then legacy remove) the routines job; classify the outcome.

    A nonzero rc whose output means "not loaded" is benign (done — nothing was
    loaded). Any other failure of BOTH attempts is surfaced as an error so the
    caller exits nonzero rather than reporting a clean uninstall over a live job.
    """
    rc, out = _run(["launchctl", "bootout", label_target])
    if rc == 0:
        return StepResult("routines launchd job", "done", f"booted out {ROUTINES_LABEL}")
    if _is_not_loaded(out):
        return StepResult("routines launchd job", "skip", "not loaded")
    # bootout failed for some other reason — try the legacy API before giving up.
    rc2, out2 = _run(["launchctl", "remove", ROUTINES_LABEL])
    if rc2 == 0:
        return StepResult("routines launchd job", "done", f"removed {ROUTINES_LABEL}")
    if _is_not_loaded(out2):
        return StepResult("routines launchd job", "skip", "not loaded")
    return StepResult(
        "routines launchd job",
        "error",
        f"could not bootout/remove {ROUTINES_LABEL} (job may still be loaded)",
    )


def _current_uid() -> int | None:
    try:
        import os

        return os.getuid()
    except (AttributeError, OSError):
        return None


def _registered_app_paths() -> list[Path]:
    """Parse ``lsregister -dump`` for registered ``OpenBird.app`` bundle paths."""
    if not Path(_LSREGISTER).exists():
        return []
    rc, out = _run([_LSREGISTER, "-dump"])
    if rc != 0 and not out:
        return []
    seen: dict[str, Path] = {}
    for line in out.splitlines():
        stripped = line.strip()
        if not stripped.startswith("path:"):
            continue
        # "path:        /some/OpenBird.app (0x...)" -> "/some/OpenBird.app"
        value = stripped[len("path:"):].strip()
        if " (0x" in value:
            value = value[: value.rindex(" (0x")]
        if value.endswith("OpenBird.app"):
            seen.setdefault(value, Path(value))
    return list(seen.values())


def unregister_launch_services(*, dry_run: bool) -> list[StepResult]:
    """Unregister OpenBird Launch Services entries (bundle-id validated).

    A ghost path (bundle no longer on disk) is unregistered directly; an existing
    bundle is unregistered only when its CFBundleIdentifier == BUNDLE_ID so an
    unrelated ``OpenBird.app`` is never touched (Codex #5). macOS-only.
    """
    if not _is_macos():
        return [StepResult("launch services", "skip", "not macOS")]
    if not Path(_LSREGISTER).exists():
        return [StepResult("launch services", "skip", "lsregister absent")]

    results: list[StepResult] = []
    paths = _registered_app_paths()
    if not paths:
        return [StepResult("launch services", "skip", "no registrations")]

    for path in paths:
        exists = path.exists()
        if exists:
            bid = _bundle_id_of(path)
            if bid != BUNDLE_ID:
                results.append(
                    StepResult("launch services", "skip", f"{path} (id={bid})")
                )
                continue
            reason = str(path)
        else:
            reason = f"{path} (ghost)"

        if dry_run:
            results.append(StepResult("launch services", "would", f"unregister {reason}"))
            continue
        rc, _ = _run([_LSREGISTER, "-u", str(path)])
        status = "done" if rc == 0 else "error"
        results.append(StepResult("launch services", status, f"unregister {reason}"))
    return results


def remove_sidecars(data_dir: Path, *, dry_run: bool) -> list[StepResult]:
    """Remove pause/lock sidecars in the data dir (data files untouched)."""
    results: list[StepResult] = []
    for name in _SIDECAR_NAMES:
        target = data_dir / name
        if not target.exists():
            continue
        if dry_run:
            results.append(StepResult("sidecar", "would", str(target)))
            continue
        try:
            target.unlink()
            results.append(StepResult("sidecar", "done", str(target)))
        except OSError as exc:
            results.append(StepResult("sidecar", "error", type(exc).__name__))
    if not results:
        results.append(StepResult("sidecar", "skip", "none present"))
    return results


def _purge_data_dir(data_dir: Path, *, dry_run: bool) -> StepResult:
    if dry_run:
        return StepResult("purge data", "would", str(data_dir))
    if not data_dir.exists():
        return StepResult("purge data", "skip", "already absent")
    try:
        shutil.rmtree(data_dir)
    except OSError as exc:
        return StepResult("purge data", "error", type(exc).__name__)
    if data_dir.exists():
        return StepResult("purge data", "error", "partial removal")
    return StepResult("purge data", "done", str(data_dir))


def _key_action(*, dry_run: bool) -> StepResult:
    """Delete the Keychain key only if no encrypted DB depends on it.

    Re-resolves the DB path AFTER any data removal and applies the single
    key-safety rule: delete iff the DB is absent/empty or has the plaintext SQLite
    header; otherwise RETAIN (a non-empty DB lacking the magic looks encrypted).
    """
    from openbird.storage.crypto import db_is_plaintext_or_absent, delete_key

    db_path = db_file_path()
    if not db_is_plaintext_or_absent(db_path):
        return StepResult(
            "keychain key",
            "retained",
            f"encrypted DB still present at {db_path}; key kept to avoid stranding",
        )
    if dry_run:
        return StepResult("keychain key", "would", "delete db-encryption-key")
    removed = delete_key()
    return StepResult(
        "keychain key",
        "done" if removed else "skip",
        "deleted" if removed else "already absent / unavailable",
    )


def run_uninstall(*, purge_data: bool, dry_run: bool) -> list[StepResult]:
    """Orchestrate the full cleanup, returning a content-safe result per step.

    Order matters for the key-safety rule (Codex #1): all data removal happens
    BEFORE the key decision, and the key is re-evaluated against the resolved DB
    path so a custom ``OPENBIRD_DB_PATH`` outside the data dir cannot be stranded.
    """
    data_dir = data_dir_path()
    results: list[StepResult] = []
    results.extend(remove_routines_job(dry_run=dry_run))
    results.extend(unregister_launch_services(dry_run=dry_run))
    results.extend(remove_sidecars(data_dir, dry_run=dry_run))
    if purge_data:
        results.append(_purge_data_dir(data_dir, dry_run=dry_run))
    # Key decision LAST — after data removal, evaluated against the resolved DB path.
    results.append(_key_action(dry_run=dry_run))
    return results
