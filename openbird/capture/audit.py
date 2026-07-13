"""Privacy-safe evaluation of whether capture contains useful context."""

from __future__ import annotations

from typing import Any


def _context_quality(
    *,
    effective_state: str,
    samples: int,
    chars_p50: int,
    chars_p90: int,
    lines_p90: int,
    substantive_ratio: float,
    rich_ratio: float,
    min_samples: int,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if effective_state == "blocked":
        return "unavailable", ["capture_blocked"]
    if samples == 0:
        return "unavailable", ["no_recent_capture"]
    if samples < min_samples:
        return "insufficient_data", ["below_sample_floor"]
    if chars_p50 < 120 and (chars_p90 >= 400 or lines_p90 >= 8):
        return "inconsistent_context", ["bimodal_context_depth"]
    if rich_ratio >= 0.7 and substantive_ratio >= 0.7:
        return "rich_context", ["consistently_rich"]
    if substantive_ratio >= 0.5 or rich_ratio >= 0.3:
        return "usable_context", ["mostly_substantive"]
    reasons.append("shallow_context")
    return "low_context", reasons


def build_capture_audit(
    *,
    health: dict[str, Any],
    content_quality: dict[str, dict[str, float | int]],
    min_samples: int = 5,
) -> dict[str, Any]:
    """Combine capture activity and in-database richness aggregates.

    The input and output are content-free. ``content_quality`` must come from
    ``MemoryStore.capture_content_quality`` so no captured text or per-row hash
    crosses the SQLite boundary.
    """
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")

    apps: list[dict[str, Any]] = []
    recommendations: list[dict[str, str]] = []
    for health_row in health.get("apps", []):
        bundle_id = health_row["bundle_id"]
        metrics = content_quality.get(bundle_id, {})
        samples = int(metrics.get("sample_count", 0))
        distinct_contexts = int(metrics.get("distinct_contexts", 0))
        chars_p50 = int(metrics.get("chars_p50", 0))
        chars_p90 = int(metrics.get("chars_p90", 0))
        lines_p50 = int(metrics.get("lines_p50", 0))
        lines_p90 = int(metrics.get("lines_p90", 0))
        substantive_ratio = float(metrics.get("substantive_ratio", 0.0))
        rich_ratio = float(metrics.get("rich_ratio", 0.0))
        context_quality, reasons = _context_quality(
            effective_state=health_row["effective_state"],
            samples=samples,
            chars_p50=chars_p50,
            chars_p90=chars_p90,
            lines_p90=lines_p90,
            substantive_ratio=substantive_ratio,
            rich_ratio=rich_ratio,
            min_samples=min_samples,
        )
        coverage = health_row.get("coverage", "unknown")
        if coverage == "partial":
            reasons.append("partial_capture_coverage")
        distinct_ratio = distinct_contexts / samples if samples else 0.0
        if samples >= min_samples and distinct_ratio < 0.25:
            reasons.append("repeated_context_advisory")

        action_code = {
            "low_context": "improve_capture_depth",
            "inconsistent_context": "stabilize_capture_depth",
            "insufficient_data": "collect_more_samples",
        }.get(context_quality)
        if (
            context_quality == "unavailable"
            and health_row["effective_state"] != "blocked"
        ):
            action_code = "investigate_missing_capture"
        if action_code:
            recommendations.append({"bundle_id": bundle_id, "code": action_code})
        if coverage == "partial":
            recommendations.append(
                {"bundle_id": bundle_id, "code": "improve_capture_coverage"}
            )

        apps.append(
            {
                "bundle_id": bundle_id,
                "effective_state": health_row["effective_state"],
                "coverage": coverage,
                "context_quality": context_quality,
                "reason_codes": reasons,
                "sample_count": samples,
                "distinct_contexts": distinct_contexts,
                "distinct_ratio": round(distinct_ratio, 4),
                "chars_p50": chars_p50,
                "chars_p90": chars_p90,
                "lines_p50": lines_p50,
                "lines_p90": lines_p90,
                "substantive_ratio": round(substantive_ratio, 4),
                "rich_ratio": round(rich_ratio, 4),
            }
        )

    daemon = health.get("daemon", {"state": "unknown"})
    identity_complete = all(
        daemon.get(field) is not None
        for field in ("instance_uuid", "pid", "runtime_version")
    )
    blocked = daemon.get("state") != "ok" or not identity_complete
    warning_qualities = {
        "low_context",
        "inconsistent_context",
        "insufficient_data",
    }
    warning = not apps or any(
        row["context_quality"] in warning_qualities
        or (
            row["context_quality"] == "unavailable"
            and row["effective_state"] != "blocked"
        )
        for row in apps
    )
    overall_state = "blocked" if blocked else "warn" if warning else "pass"
    return {
        "generated_at": health.get("generated_at"),
        "recent_window_seconds": health.get("recent_window_seconds"),
        "minimum_samples": min_samples,
        "overall_state": overall_state,
        "daemon": daemon,
        "summary": {
            "apps_audited": len(apps),
            "rich_context": sum(
                row["context_quality"] == "rich_context" for row in apps
            ),
            "usable_context": sum(
                row["context_quality"] == "usable_context" for row in apps
            ),
            "needs_improvement": sum(
                row["context_quality"] in warning_qualities
                or (
                    row["context_quality"] == "unavailable"
                    and row["effective_state"] != "blocked"
                )
                for row in apps
            ),
        },
        "apps": apps,
        "recommendations": recommendations,
    }
