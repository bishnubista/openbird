"""Deterministic helpers for the founder-context recap Ask route.

The route itself lives in :mod:`openbird.chat.rag`; this module keeps intent
matching, bounded source selection, and structured-response validation pure so
they can be exercised without a database or model.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

FOUNDER_CONTEXT_QUERY = "Bring me back up to speed on what I was working on."
FOUNDER_CONTEXT_DAYS = 5
FOUNDER_CONTEXT_MAX_SOURCES = 8
FOUNDER_CONTEXT_PAGE_SIZE = 120
FOUNDER_CONTEXT_PER_DAY_DISTINCT_CAP = 240
FOUNDER_CONTEXT_PER_DAY_SCAN_CAP = 1200
FOUNDER_CONTEXT_TOTAL_DISTINCT_CAP = 1200
FOUNDER_CONTEXT_TOTAL_SCAN_CAP = 6000
FOUNDER_CONTEXT_COMPLETION_ATTEMPTS = 2
FOUNDER_CONTEXT_COMPLETION_TIMEOUT_SECONDS = 20.0

_FOUNDER_CONTEXT_RE = re.compile(
    r"^\s*(?:"
    r"bring\s+me\s+back\s+up\s+to\s+speed"
    r"(?:\s+on\s+what\s+i\s+was\s+working\s+on)?"
    r"|where\s+did\s+i\s+leave\s+off"
    r"|catch\s+me\s+back\s+up(?:\s+on\s+my\s+work)?"
    r"|get\s+me\s+back\s+up\s+to\s+speed(?:\s+on\s+my\s+work)?"
    r")\s*[?.!]*\s*$",
    re.IGNORECASE,
)

_CUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "decision",
        re.compile(
            r"\b(decid(?:e|ed|ing)|decision|chose|chosen|agreed|"
            r"trade[- ]?off|will\s+use|going\s+with)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "progress",
        re.compile(
            r"\b(shipp(?:ed|ing)|merg(?:ed|ing)|implement(?:ed|ing)|"
            r"finish(?:ed|ing)|complet(?:ed|ing)|fixed|landed|passed|"
            r"in\s+progress|working\s+on)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "open_loop",
        re.compile(
            r"\b(todo|to[- ]do|follow[- ]?up|next\s+step|open\s+loop|"
            r"blocked|blocker|waiting\s+on|need(?:s|ed)?\s+to|"
            r"still\s+need|remaining)\b",
            re.IGNORECASE,
        ),
    ),
)

FOUNDER_CONTEXT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "likely_focus": {
            "type": ["object", "null"],
            "properties": {
                "text": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["text", "citations"],
        },
        "recent_activity": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "citations"],
            },
        },
        "decisions_progress": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "citations"],
            },
        },
        "open_loops": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "citations"],
            },
        },
    },
    "required": [
        "likely_focus",
        "recent_activity",
        "decisions_progress",
        "open_loops",
    ],
}


def is_founder_context_query(query: str) -> bool:
    """Return whether ``query`` is one of the deliberately narrow recap forms."""
    return _FOUNDER_CONTEXT_RE.fullmatch(query or "") is not None


def cue_kinds(text: str) -> tuple[str, ...]:
    """Return the closed cue categories present in text."""
    return tuple(name for name, pattern in _CUE_PATTERNS if pattern.search(text or ""))


def select_founder_context_rows(
    rows: list[tuple[Any, str]],
    *,
    now: float,
    signal_score: Callable[[Any, str], int],
    max_sources: int = FOUNDER_CONTEXT_MAX_SOURCES,
) -> list[tuple[Any, str]]:
    """Select recent, information-rich rows with bounded cue reservation.

    At most one decision, progress, and open-loop row is reserved, and reserved
    rows must come from distinct ``(app, source)`` pairs. This keeps keyword
    noise from dominating the likely active thread while still ensuring a real
    decision or loose end is not crowded out by a single high-volume app.
    """
    if max_sources <= 0:
        return []

    def rank(row: tuple[Any, str]) -> tuple[float, float, str]:
        obs, text = row
        age_days = max(0.0, (float(now) - float(obs.ts)) / 86_400.0)
        recency = max(0.0, 300.0 - age_days * 60.0)
        return (
            float(signal_score(obs, text)) + recency,
            float(obs.ts),
            str(obs.id),
        )

    ranked = sorted(rows, key=rank, reverse=True)
    chosen: list[tuple[Any, str]] = []
    chosen_ids: set[str] = set()
    reserved_origins: set[tuple[str | None, str]] = set()

    for kind, _pattern in _CUE_PATTERNS:
        for row in ranked:
            obs, text = row
            if obs.id in chosen_ids or kind not in cue_kinds(text):
                continue
            origin = (obs.app, obs.source)
            # A cue must carry real content signal; recency alone cannot reserve
            # an otherwise shallow row.
            if signal_score(obs, text) < 80 or origin in reserved_origins:
                continue
            chosen.append(row)
            chosen_ids.add(obs.id)
            reserved_origins.add(origin)
            break
        if len(chosen) >= min(3, max_sources):
            break

    for row in ranked:
        if len(chosen) >= max_sources:
            break
        if row[0].id in chosen_ids:
            continue
        chosen.append(row)
        chosen_ids.add(row[0].id)

    # Prompt order is chronological; source ids then read as a compact work
    # sequence while selection itself remains weighted toward the recent tail.
    chosen.sort(key=lambda row: (float(row[0].ts), str(row[0].id)))
    return chosen


def parse_founder_context_response(
    raw: str | dict[str, Any],
    *,
    valid_source_ids: set[str],
) -> tuple[str, list[str]]:
    """Validate claim-level grounding and render a compact founder recap.

    A likely-focus claim needs two distinct in-context sources; every other
    surfaced claim needs one. Invalid, hallucinated, or uncited claims are
    dropped individually. The returned citation list is the ordered union of
    citations for claims that survived.
    """
    if not isinstance(raw, dict):
        return "", []

    claimed_ids: list[str] = []

    def valid_claim(value: Any, *, minimum: int) -> tuple[str, list[str]] | None:
        if not isinstance(value, dict):
            return None
        text = value.get("text")
        citations = value.get("citations")
        if not isinstance(text, str) or not text.strip() or not isinstance(citations, list):
            return None
        ids: list[str] = []
        for item in citations:
            source_id = str(item)
            if source_id in valid_source_ids and source_id not in ids:
                ids.append(source_id)
        if len(ids) < minimum:
            return None
        return text.strip(), ids

    focus = valid_claim(raw.get("likely_focus"), minimum=2)
    sections: list[tuple[str, list[tuple[str, list[str]]]]] = []
    for key, label in (
        ("recent_activity", "Recent activity"),
        ("decisions_progress", "Decisions and progress"),
        ("open_loops", "Open loops"),
    ):
        values = raw.get(key)
        claims: list[tuple[str, list[str]]] = []
        if isinstance(values, list):
            for value in values[:3]:
                claim = valid_claim(value, minimum=1)
                if claim is not None:
                    claims.append(claim)
        if claims:
            sections.append((label, claims))

    lines: list[str] = []
    if focus is not None:
        lines.append(f"Likely focus: {focus[0]}")
        claimed_ids.extend(focus[1])
    for label, claims in sections:
        if lines:
            lines.append("")
        lines.append(f"{label}:")
        for text, ids in claims:
            lines.append(f"- {text}")
            claimed_ids.extend(ids)

    ordered_ids = list(dict.fromkeys(claimed_ids))
    return "\n".join(lines).strip(), ordered_ids


__all__ = [
    "FOUNDER_CONTEXT_COMPLETION_ATTEMPTS",
    "FOUNDER_CONTEXT_COMPLETION_TIMEOUT_SECONDS",
    "FOUNDER_CONTEXT_DAYS",
    "FOUNDER_CONTEXT_MAX_SOURCES",
    "FOUNDER_CONTEXT_PAGE_SIZE",
    "FOUNDER_CONTEXT_PER_DAY_DISTINCT_CAP",
    "FOUNDER_CONTEXT_PER_DAY_SCAN_CAP",
    "FOUNDER_CONTEXT_QUERY",
    "FOUNDER_CONTEXT_RESPONSE_SCHEMA",
    "FOUNDER_CONTEXT_TOTAL_DISTINCT_CAP",
    "FOUNDER_CONTEXT_TOTAL_SCAN_CAP",
    "cue_kinds",
    "is_founder_context_query",
    "parse_founder_context_response",
    "select_founder_context_rows",
]
