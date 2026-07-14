"""Closed, content-free vocabulary for capture-attempt telemetry."""

from __future__ import annotations

import re

CAPTURE_TRIGGERS = frozenset(
    {
        "app_activated",
        "window_changed",
        "title_changed",
        "focus_changed",
        "typing_pause",
        "idle_tick",
        "force_ceiling",
        "return_from_afk",
        "startup",
    }
)
CAPTURE_ATTEMPT_STATUSES = frozenset({"started", "finished"})
CAPTURE_OUTCOMES = frozenset(
    {
        "captured_full",
        "captured_partial",
        "captured_shallow",
        "captured_unchanged",
        "coalesced_inflight",
        "skipped_policy",
        "skipped_afk",
        "skipped_paused",
        "unsupported",
        "failed_bounded",
    }
)
CAPTURE_COMPLETENESS = frozenset({"full", "partial", "shallow", "none"})
CAPTURE_REASON_CODES = frozenset(
    {
        "paused",
        "self_capture",
        "not_allowlisted",
        "dangerous_app",
        "private_window",
        "no_frontmost_app",
        "no_window",
        "ax_timeout",
        "budget_exhausted",
        "empty_text",
        "unchanged",
        "normalized_empty",
        "policy_rejected",
        "ingest_failed",
    }
)
CAPTURE_ADAPTERS = frozenset({"generic_ax"})
CAPTURE_EXTRACTOR_VERSIONS = frozenset({"generic_ax_v1"})
CAPTURE_BUNDLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,254}$")
