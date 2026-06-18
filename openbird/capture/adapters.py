"""Per-app capture normalization strategies + an app-compatibility matrix.

The Swift capture helper emits raw active-window text from the macOS
Accessibility (AX) tree. AX trees differ wildly per app: some expose clean
document text (Notes, Mail), some bury content in nested web areas (Slack,
Notion, Electron), and some virtualize or withhold most text (Terminal,
canvas/GPU apps). This module:

  * Records *measured* coverage per app in :data:`COMPATIBILITY_MATRIX` so the
    product can honestly report which apps are well-supported vs degraded
    (Accepted residual risk [R5]: AX coverage is uneven; no universal-text
    guarantee).
  * Provides light per-app text normalization (:func:`normalize_for_app`) to
    strip recurring chrome/boilerplate before the memory store chunks and
    dedups, so static UI text doesn't dominate BM25/embeddings.

Normalization here is deliberately conservative: heavy semantic parsing belongs
in the Swift helper / future adapters. The goal is to make capture text from
different apps comparably clean for the chunk-level dedup pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Compatibility matrix.
#
# ``coverage`` is a coarse, measured rating of how much useful text AX yields:
#   "full"     — reliable document/message text
#   "partial"  — usable but lossy (web areas, virtualization, scroll windows)
#   "degraded" — little/no text via AX; OCR (flag-gated) is the escape hatch
#
# ``fields`` lists which event fields the helper can usually populate for the
# app. ``notes`` documents the known failure mode. These are honest, scoped
# claims, not promises — see PLAN.md "Accepted residual risks".
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppProfile:
    """A single app's measured AX-capture profile."""

    bundle_id: str
    name: str
    coverage: str  # "full" | "partial" | "degraded"
    fields: tuple[str, ...]
    notes: str


_PROFILES: tuple[AppProfile, ...] = (
    AppProfile(
        "com.apple.Safari",
        "Safari",
        "partial",
        ("app", "window", "url", "text"),
        "Rendered page text via web area; private windows excluded by redaction.",
    ),
    AppProfile(
        "com.google.Chrome",
        "Chrome",
        "partial",
        ("app", "window", "url", "text"),
        "Web area text varies by site; incognito detected via title heuristic.",
    ),
    AppProfile(
        "com.tinyspeck.slackmacgap",
        "Slack",
        "partial",
        ("app", "window", "text"),
        "Electron; messages live in nested web area, heavy repeated chrome.",
    ),
    AppProfile(
        "com.microsoft.teams2",
        "Teams",
        "degraded",
        ("app", "window"),
        "Electron with virtualized lists; AX text often sparse, OCR fallback.",
    ),
    AppProfile(
        "us.zoom.xos",
        "Zoom",
        "degraded",
        ("app", "window"),
        "Mostly GPU-drawn UI; little AX text. Meetings subsystem handles audio.",
    ),
    AppProfile(
        "com.microsoft.VSCode",
        "VS Code",
        "partial",
        ("app", "window", "text"),
        "Editor; blocked by default until user enables (code is sensitive).",
    ),
    AppProfile(
        "notion.id",
        "Notion",
        "partial",
        ("app", "window", "text"),
        "Electron; block-structured web area, recurring sidebar boilerplate.",
    ),
    AppProfile(
        "com.apple.Terminal",
        "Terminal",
        "degraded",
        ("app", "window"),
        "Blocked by default; scrollback virtualized, high secret-leak risk.",
    ),
    AppProfile(
        "com.apple.mail",
        "Mail",
        "full",
        ("app", "window", "text"),
        "Clean message body text via AX.",
    ),
    AppProfile(
        "com.apple.Notes",
        "Notes",
        "full",
        ("app", "window", "text"),
        "Clean document text via AX text area.",
    ),
)

#: Public compatibility matrix keyed by bundle id.
COMPATIBILITY_MATRIX: dict[str, AppProfile] = {p.bundle_id: p for p in _PROFILES}


def get_profile(bundle_id: str | None) -> AppProfile | None:
    """Return the :class:`AppProfile` for ``bundle_id``, or ``None`` if unknown."""
    if not bundle_id:
        return None
    return COMPATIBILITY_MATRIX.get(bundle_id)


def coverage_for(bundle_id: str | None) -> str:
    """Return the coverage rating for an app, or ``"unknown"`` if unprofiled."""
    profile = get_profile(bundle_id)
    return profile.coverage if profile is not None else "unknown"


# ---------------------------------------------------------------------------
# Normalization strategies.
# ---------------------------------------------------------------------------

# Lines that are pure UI chrome and add no memory value. Matched whole-line,
# case-insensitively, after stripping. Per-app extras are appended below.
_GENERIC_CHROME: tuple[str, ...] = (
    "new message",
    "search",
    "settings",
    "home",
    "back",
    "forward",
    "reload",
)

_PER_APP_CHROME: dict[str, tuple[str, ...]] = {
    "com.tinyspeck.slackmacgap": (
        "threads",
        "huddles",
        "drafts & sent",
        "direct messages",
        "add a bookmark",
        "you're viewing",
    ),
    "notion.id": (
        "add a page",
        "quick find",
        "updates",
        "click + to add a page",
    ),
    "com.google.Chrome": ("address and search bar", "bookmark this tab"),
    "com.apple.Safari": ("show sidebar", "share"),
}

_WS_RUN = re.compile(r"[ \t]+")


def _chrome_set(bundle_id: str | None) -> frozenset[str]:
    """Build the lowercased chrome line-set for an app (generic + per-app)."""
    extra = _PER_APP_CHROME.get(bundle_id or "", ())
    return frozenset(line.lower() for line in (*_GENERIC_CHROME, *extra))


def normalize_for_app(text: str, bundle_id: str | None = None) -> str:
    """Apply light, per-app normalization to raw AX text.

    Collapses intra-line whitespace runs, drops blank and pure-chrome lines, and
    de-duplicates immediately-repeated lines (AX trees frequently emit the same
    label twice). This runs *before* the memory store's own normalization and
    chunking, so it only removes obvious app-specific boilerplate and never
    rewrites meaningful content.

    Args:
        text: Raw active-window text from the capture helper.
        bundle_id: The app's bundle id, used to select chrome filters.

    Returns:
        Cleaned text with trailing newline stripped. May be empty if every line
        was chrome/blank.
    """
    chrome = _chrome_set(bundle_id)
    out_lines: list[str] = []
    prev: str | None = None
    for raw_line in text.splitlines():
        line = _WS_RUN.sub(" ", raw_line).strip()
        if not line:
            continue
        if line.lower() in chrome:
            continue
        if line == prev:  # collapse immediately-repeated AX labels
            continue
        out_lines.append(line)
        prev = line
    return "\n".join(out_lines)


__all__ = [
    "AppProfile",
    "COMPATIBILITY_MATRIX",
    "get_profile",
    "coverage_for",
    "normalize_for_app",
]
