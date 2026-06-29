"""Metadata-only capture health for user-facing trust surfaces."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from openbird.capture import adapters, redact
from openbird.config import Settings, get_settings

DEFAULT_RECENT_WINDOW_SECONDS = 24 * 60 * 60


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

    apps: list[dict[str, Any]] = []
    for bundle_id in _health_row_app_ids(settings=settings, activity_by_app=activity_by_app):
        activity = activity_by_app.get(bundle_id, {})
        total = int(activity.get("total_observations") or 0)
        recent = int(activity.get("recent_observations") or 0)
        last_ts = activity.get("last_captured_ts")
        coverage = adapters.coverage_for(bundle_id)
        policy = _policy_for_app(bundle_id, settings)
        apps.append(
            {
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
        )

    return {
        "generated_at": now,
        "recent_window_seconds": recent_window_seconds,
        "paused": bool(paused),
        "allowlist_count": len(settings.allowlist),
        "blocklist_count": len(settings.blocklist),
        "apps": apps,
    }


__all__ = ["DEFAULT_RECENT_WINDOW_SECONDS", "build_capture_health"]
