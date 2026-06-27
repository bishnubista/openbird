"""Privacy-safe helpers for remote reasoning send-attempt audit metadata."""

from __future__ import annotations

import hashlib
from typing import Any

from openbird.config import Settings, is_loopback_host, is_ollama_model, resolved_ollama_host

ALLOWED_EXCLUSION_REASONS = frozenset({"app", "source", "observation_id"})


def packet_payload_audit(
    packet_json: str,
    *,
    selected_source_count: int = 0,
    exclusions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return non-content metadata for the exact packet JSON sent to a model."""
    payload = packet_json.encode("utf-8")
    excluded_observations, excluded_by = safe_exclusion_counts(exclusions)
    return {
        "packet_hash": hashlib.sha256(payload).hexdigest(),
        "packet_bytes": len(payload),
        "selected_source_count": int(selected_source_count),
        "excluded_observations": excluded_observations,
        "excluded_by": excluded_by,
    }


def safe_exclusion_counts(
    exclusions: dict[str, Any] | None,
) -> tuple[int, dict[str, int]]:
    """Extract only reason-code counts from an exclusions block.

    Never copy configured app/source names or observation IDs from the packet.
    """
    if not isinstance(exclusions, dict):
        return 0, {}
    raw_by = exclusions.get("excluded_by") or {}
    safe_by: dict[str, int] = {}
    if isinstance(raw_by, dict):
        for key, value in raw_by.items():
            reason = str(key)
            if reason not in ALLOWED_EXCLUSION_REASONS:
                continue
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            if count < 0:
                continue
            safe_by[reason] = count
    try:
        total = int(exclusions.get("excluded_observations") or 0)
    except (TypeError, ValueError):
        total = sum(safe_by.values())
    if total < 0:
        total = sum(safe_by.values())
    return total, safe_by


def provider_family(model: str | None) -> str:
    """Best-effort provider family label; advisory, not an egress gate."""
    name = (model or "").strip().lower()
    if not name:
        return "unknown"
    if name.startswith(("ollama/", "ollama_chat/")):
        return "ollama"
    if name.startswith(("mlx/", "mlx-community/", "mlx_community/")):
        return "mlx"
    if name.startswith(("anthropic/", "claude-")):
        return "anthropic"
    if name.startswith(("gemini/", "google/", "models/gemini", "gemini-")):
        return "gemini"
    if name.startswith(("openai/", "gpt-", "o1", "o3", "o4")):
        return "openai"
    return "unknown"


def advisory_route_class(model: str | None, settings: Settings) -> str:
    """Return advisory route class for an already-known remote LLM route."""
    if provider_family(model) == "mlx":
        return "local"
    if model and is_ollama_model(model):
        host = resolved_ollama_host(settings)
        if not is_loopback_host(host):
            return "self-hosted-remote"
    return "third-party-cloud"
