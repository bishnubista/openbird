"""Local-only Deep Brain packet preview.

The preview is a consent surface, not a sender. It shows the compact,
exclusion-filtered packet a future cloud reasoning route may use, while keeping
egress at zero and avoiding any provider construction.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from openbird.capture.redact import _bundle_matches_any
from openbird.config import Settings
from openbird.day_memory import build_day_memory, local_date_for_window
from openbird.routines.templates import DEFAULT_BRIEFING_SOURCES, select_briefing_sources
from openbird.types import Observation


Rows = Sequence[tuple[Observation, str]]


def build_deep_brain_preview(
    rows: Rows,
    *,
    start_ts: float,
    end_ts: float,
    day_offset: int,
    source_scope: str,
    settings: Settings,
    sources_limit: int = DEFAULT_BRIEFING_SOURCES,
) -> dict[str, Any]:
    """Build the local-only preview packet for a future Deep Brain route.

    Exclusions are applied before day-memory distillation. The returned packet is
    content-bearing and intended for stdout/UI preview only; this function never
    logs, constructs an LLM provider, or opens a network path.
    """
    filtered_rows, exclusion_meta = filter_rows_for_deep_brain(rows, settings=settings)
    built = build_day_memory(
        list(filtered_rows),
        start_ts=start_ts,
        end_ts=end_ts,
        day_offset=day_offset,
        source_scope=source_scope,
        gap_seconds=settings.session_gap_seconds,
    )
    selected_sources, total_sources = select_briefing_sources(
        list(filtered_rows), limit=sources_limit
    )
    blocked_reasons = deep_brain_blocked_reasons(settings)
    return {
        "route": "deep_brain.preview",
        "egress": "none_preview",
        "cloud_ready": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
        "local_date": local_date_for_window(start_ts),
        "day_offset": day_offset,
        "source_scope": source_scope,
        "memory_summary": _memory_summary(built.payload),
        "selected_sources": selected_sources,
        "sources_total": total_sources,
        "exclusions": exclusion_meta,
    }


def deep_brain_blocked_reasons(settings: Settings) -> list[str]:
    """Return opt-in gates still missing for future cloud reasoning."""
    reasons: list[str] = []
    if not settings.allow_cloud:
        reasons.append("OPENBIRD_ALLOW_CLOUD is not enabled")
    if not settings.deep_brain_enabled:
        reasons.append("OPENBIRD_DEEP_BRAIN_ENABLED is not enabled")
    return reasons


def filter_rows_for_deep_brain(
    rows: Rows, *, settings: Settings
) -> tuple[list[tuple[Observation, str]], dict[str, Any]]:
    """Apply app/source/id exclusions before any many-to-one distillation."""
    kept: list[tuple[Observation, str]] = []
    excluded: Counter[str] = Counter()
    unknown_app_kept = 0
    source_exclusions = {
        item.casefold() for item in settings.deep_brain_excluded_sources if item
    }
    id_exclusions = {
        item.casefold() for item in settings.deep_brain_excluded_observation_ids if item
    }

    for obs, text in rows:
        reason = _exclusion_reason(
            obs,
            app_exclusions=settings.deep_brain_excluded_apps,
            source_exclusions=source_exclusions,
            id_exclusions=id_exclusions,
        )
        if reason is not None:
            excluded[reason] += 1
            continue
        if obs.app is None and settings.deep_brain_excluded_apps:
            unknown_app_kept += 1
        kept.append((obs, text))

    return kept, {
        "input_observations": len(rows),
        "kept_observations": len(kept),
        "excluded_observations": sum(excluded.values()),
        "excluded_by": dict(sorted(excluded.items())),
        "unknown_app_kept": unknown_app_kept,
        "excluded_apps_configured": list(settings.deep_brain_excluded_apps),
        "excluded_sources_configured": list(settings.deep_brain_excluded_sources),
        "excluded_observation_ids_configured": len(settings.deep_brain_excluded_observation_ids),
    }


def _exclusion_reason(
    obs: Observation,
    *,
    app_exclusions: list[str],
    source_exclusions: set[str],
    id_exclusions: set[str],
) -> str | None:
    if obs.id.casefold() in id_exclusions:
        return "observation_id"
    if _bundle_matches_any(obs.app, app_exclusions):
        return "app"
    if (obs.source or "").casefold() in source_exclusions:
        return "source"
    return None


def _memory_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Construct a positive-allowlist summary; never copy source_ids/fingerprints."""
    coverage = payload.get("coverage") or {}
    metrics = payload.get("metrics") or {}
    return {
        "schema": payload.get("schema"),
        "extractor_version": payload.get("extractor_version"),
        "narrative_status": payload.get("narrative_status"),
        "local_date": payload.get("local_date"),
        "source_scope": payload.get("source_scope"),
        "as_of": payload.get("as_of"),
        "day_offset": payload.get("day_offset"),
        "window": _copy_mapping(payload.get("window"), ("start", "end")),
        "coverage": {
            "observations": coverage.get("observations", 0),
            "sessions": coverage.get("sessions", 0),
            "apps": coverage.get("apps", 0),
        },
        "sessions": [
            _copy_mapping(
                item,
                ("app", "category", "category_confidence", "start", "end", "count", "title"),
            )
            for item in payload.get("sessions") or []
        ],
        "workstreams": [
            _copy_mapping(
                item,
                (
                    "kind",
                    "label",
                    "category",
                    "session_count",
                    "active_seconds",
                    "source_count",
                ),
            )
            for item in payload.get("workstreams") or []
        ],
        "open_loops": [
            _copy_mapping(item, ("kind", "title", "cue", "source_count"))
            for item in payload.get("open_loops") or []
        ],
        "metrics": _copy_mapping(
            metrics,
            (
                "active_seconds",
                "time_by_hour",
                "time_by_app",
                "time_by_category",
                "context_switch_count",
                "longest_same_category_streak",
                "first_seen",
                "last_seen",
                "unknown_category_count",
            ),
        ),
        "entities": _entities_summary(payload.get("entities") or {}),
    }


def _entities_summary(entities: dict[str, Any]) -> dict[str, Any]:
    return {
        "domains": [
            _copy_mapping(item, ("value", "count"))
            for item in entities.get("domains") or []
        ],
        "repos": [
            _copy_mapping(item, ("value", "count")) for item in entities.get("repos") or []
        ],
        "title_tokens": [
            _copy_mapping(item, ("token", "count"))
            for item in entities.get("title_tokens") or []
        ],
    }


def _copy_mapping(value: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: value.get(key) for key in keys if key in value}
