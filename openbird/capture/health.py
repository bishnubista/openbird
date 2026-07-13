"""Metadata-only capture health for user-facing trust surfaces."""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from pathlib import Path
from typing import Any

from openbird.capture import adapters, redact
from openbird.config import Settings, get_settings

DEFAULT_RECENT_WINDOW_SECONDS = 24 * 60 * 60

# Daemon-liveness staleness bound: the sidecar is written at most every
# ~10s by an eventing daemon (see daemon._LIVENESS_WRITE_INTERVAL), so a
# sidecar older than 3x that gap means the daemon is gone or wedged. Per the
# design budget: NEVER report "ok" off a stale timestamp; an absent/unreadable
# sidecar is "unknown", never ok. Public (SHARED, single definition): the
# battery/idle gate in openbird.summaries imports it so both consumers agree on
# what "fresh" means — never duplicate the number.
DAEMON_STALE_AFTER_SECONDS = 30.0
# Back-compat alias for the pre-Phase-D private name.
_DAEMON_STALE_AFTER_SECONDS = DAEMON_STALE_AFTER_SECONDS
_RUNTIME_VERSION_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z.+!-]{0,63}$")


def _policy_for_app(bundle_id: str, settings: Settings) -> dict[str, Any]:
    """Return the effective capture decision for an app without reading content."""
    decision = redact.decide(
        app=bundle_id,
        window=None,
        text="probe",
        incognito=False,
        settings=settings,
    )
    return {"capture": decision.capture, "reason": decision.reason}


def _is_pattern_entry(entry: str) -> bool:
    return entry.startswith("glob:") or entry.startswith("re:")


def _health_row_app_ids(
    *, settings: Settings, activity_by_app: dict[str, dict[str, float | int | None]]
) -> list[str]:
    """Return concrete app ids to render; resolve pattern allowlists via activity."""
    seen: set[str] = set()
    app_ids: list[str] = []
    for entry in settings.allowlist:
        if _is_pattern_entry(entry):
            continue
        if entry not in seen:
            seen.add(entry)
            app_ids.append(entry)
    for bundle_id in sorted(activity_by_app):
        if bundle_id in seen:
            continue
        policy = _policy_for_app(bundle_id, settings)
        if policy["reason"] == "not_allowlisted":
            continue
        seen.add(bundle_id)
        app_ids.append(bundle_id)
    return app_ids


def _quality(
    *, policy_capture: bool, coverage: str, total_observations: int, recent_observations: int
) -> str:
    """Small user-facing quality bucket from metadata and known compatibility."""
    if not policy_capture:
        return "blocked"
    if total_observations <= 0:
        return "no_recent"
    if recent_observations <= 0:
        return "no_recent"
    if coverage == "degraded":
        return "low_signal"
    if coverage == "partial":
        return "partial"
    if coverage == "full":
        return "good"
    return "good" if recent_observations > 0 else "partial"


def _effective_state(*, policy: dict[str, Any], total_observations: int, recent_observations: int) -> str:
    """Metadata-only state; app runtime overlays pause/permissions/running state."""
    if not policy["capture"]:
        return "blocked"
    if recent_observations > 0:
        return "allowed_recent"
    if total_observations > 0:
        return "allowed_stale"
    return "allowed_no_recent"


def _daemon_liveness(*, settings: Settings, now: float) -> dict[str, Any]:
    """Read the daemon's liveness sidecar into a metadata-only health block.

    States: ``ok`` (fresh sidecar), ``stale`` (sidecar exists but is older than
    the staleness bound — daemon gone or wedged), ``unknown`` (no sidecar /
    unreadable / malformed — e.g. the daemon predates stream mode or hasn't
    started). Only writer identity, timestamps, mode, AFK, and seq are surfaced
    — no captured content.
    """
    path = Path(settings.data_dir, "capture.liveness.json")
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"state": "unknown"}
    if not isinstance(raw, dict):
        return {"state": "unknown"}

    def _finite(value: object) -> float | None:
        try:
            parsed = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    updated_at = _finite(raw.get("updated_at"))
    if updated_at is None:
        return {"state": "unknown"}
    age = now - updated_at
    state = "ok" if 0 <= age <= _DAEMON_STALE_AFTER_SECONDS else "stale"
    seq = raw.get("heartbeat_seq")
    instance_uuid = raw.get("instance_uuid")
    try:
        instance_uuid = str(uuid.UUID(instance_uuid))
    except (AttributeError, TypeError, ValueError):
        instance_uuid = None
    pid = raw.get("pid")
    pid = pid if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0 else None
    runtime_version = raw.get("runtime_version")
    if not isinstance(runtime_version, str) or not _RUNTIME_VERSION_RE.fullmatch(
        runtime_version
    ):
        runtime_version = None
    # OCR fallback availability (Phase C2): sanitized against the closed pair
    # AND gated on daemon FRESHNESS — a stale sidecar's "available" is a dead
    # daemon's old claim, and the design budget says never report ok off a
    # stale timestamp. None = never reported / OCR off / old daemon / stale.
    ocr_state = raw.get("ocr_state") if state == "ok" else None
    return {
        "state": state,
        "instance_uuid": instance_uuid,
        "pid": pid,
        "runtime_version": runtime_version,
        "updated_at": updated_at,
        "age_seconds": age,
        "mode": raw.get("mode") if raw.get("mode") in ("stream", "oneshot") else None,
        "afk": bool(raw.get("afk", False)),
        "last_event_ts": _finite(raw.get("last_event_ts")),
        "last_capture_ts": _finite(raw.get("last_capture_ts")),
        "heartbeat_seq": seq if isinstance(seq, int) else None,
        "ocr_state": ocr_state if ocr_state in ("available", "unavailable") else None,
    }


def build_capture_health(
    *,
    settings: Settings | None = None,
    activity_by_app: dict[str, dict[str, float | int | None]] | None = None,
    generated_at: float | None = None,
    recent_window_seconds: float = DEFAULT_RECENT_WINDOW_SECONDS,
    paused: bool | None = None,
) -> dict[str, Any]:
    """Build a content-free capture health payload.

    ``activity_by_app`` must contain only app ids, counts, and timestamps, usually
    from :meth:`MemoryStore.capture_app_activity`.
    """
    settings = settings or get_settings()
    now = time.time() if generated_at is None else float(generated_at)
    activity_by_app = activity_by_app or {}
    if paused is None:
        paused = Path(settings.data_dir, "capture.paused").exists()

    # OCR opt-in set (Phase C2); getattr-guarded for injected test Settings
    # that predate the field. Matching uses the same allow/block grammar.
    ocr_apps = list(getattr(settings, "capture_ocr_apps", None) or [])

    apps: list[dict[str, Any]] = []
    for bundle_id in _health_row_app_ids(settings=settings, activity_by_app=activity_by_app):
        activity = activity_by_app.get(bundle_id, {})
        total = int(activity.get("total_observations") or 0)
        recent = int(activity.get("recent_observations") or 0)
        last_ts = activity.get("last_captured_ts")
        coverage = adapters.coverage_for(bundle_id)
        policy = _policy_for_app(bundle_id, settings)
        row: dict[str, Any] = {
            "bundle_id": bundle_id,
            "policy": policy,
            "effective_state": _effective_state(
                policy=policy,
                total_observations=total,
                recent_observations=recent,
            ),
            "quality": _quality(
                policy_capture=bool(policy["capture"]),
                coverage=coverage,
                total_observations=total,
                recent_observations=recent,
            ),
            "coverage": coverage,
            "total_observations": total,
            "recent_observations": recent,
            "last_captured_ts": last_ts,
        }
        # Additive key on OPTED-IN rows only (Phase C2). The daemon-level
        # availability (ocr_state above) is combined with this at render
        # time — see the capture-health CLI table.
        if ocr_apps and redact._bundle_matches_any(bundle_id, ocr_apps):
            row["ocr"] = "opted_in"
        if redact.is_detailed_capture_eligible(bundle_id):
            row["detailed_capture"] = (
                "enabled"
                if redact._is_detailed_capture_enabled(
                    bundle_id, settings.detailed_capture_apps
                )
                else "available"
            )
        apps.append(row)

    return {
        "generated_at": now,
        "recent_window_seconds": recent_window_seconds,
        "paused": bool(paused),
        "allowlist_count": len(settings.allowlist),
        "blocklist_count": len(settings.blocklist),
        "detailed_capture_apps_count": len(settings.detailed_capture_apps),
        "ocr_apps_count": len(ocr_apps),
        # Daemon liveness from the metadata-only sidecar (additive block;
        # Swift's JSONDecoder ignores unknown keys, and the Python consumers
        # key into it explicitly).
        "daemon": _daemon_liveness(settings=settings, now=now),
        "apps": apps,
    }


__all__ = [
    "DAEMON_STALE_AFTER_SECONDS",
    "DEFAULT_RECENT_WINDOW_SECONDS",
    "build_capture_health",
]
