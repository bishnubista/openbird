"""Text normalization, chunking, and content hashing for the memory store.

Dedup and hashing operate on *normalized chunks* (whitespace-collapsed,
boilerplate-stripped) rather than full windows, so a one-character edit
does not fork an entire window and static chrome does not dominate ranking.
"""

from __future__ import annotations

import hashlib
import re

# Target chunk size (characters) and overlap between consecutive chunks. These
# operate on normalized text; chunking prefers paragraph/sentence boundaries.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

_WHITESPACE_RE = re.compile(r"[^\S\n]+")  # runs of spaces/tabs, not newlines
_BLANKLINES_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")


def normalize(text: str) -> str:
    """Normalize captured text for stable hashing and ranking.

    Collapses runs of intra-line whitespace to a single space, strips trailing
    whitespace on each line, collapses 3+ blank lines to a single blank line,
    and trims leading/trailing whitespace. Idempotent.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _TRAILING_WS_RE.sub("\n", text)
    text = _BLANKLINES_RE.sub("\n\n", text)
    return text.strip()


def content_hash(text: str) -> str:
    """Return a stable SHA-256 hex digest of the *normalized* text.

    Normalization is applied here so callers that hash directly and callers that
    chunk-then-hash agree on identity.
    """
    norm = normalize(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _split_points(text: str) -> list[int]:
    """Candidate soft-break offsets (paragraph then sentence boundaries)."""
    points: set[int] = set()
    for m in re.finditer(r"\n\n", text):
        points.add(m.end())
    for m in re.finditer(r"(?<=[.!?])\s+", text):
        points.add(m.end())
    return sorted(points)


def chunk(text: str) -> list[tuple[tuple[int, int], str]]:
    """Split *normalized* text into overlapping chunks.

    Returns a list of ``((start, end), chunk_text)`` where ``start``/``end`` are
    character offsets into the normalized text. Chunks prefer to end on a
    paragraph or sentence boundary near :data:`CHUNK_SIZE`; consecutive chunks
    overlap by up to :data:`CHUNK_OVERLAP` characters. Short text yields a single
    chunk. Returns ``[]`` for empty/whitespace-only input.
    """
    norm = normalize(text)
    if not norm:
        return []
    if len(norm) <= CHUNK_SIZE:
        return [((0, len(norm)), norm)]

    breaks = _split_points(norm)
    chunks: list[tuple[tuple[int, int], str]] = []
    start = 0
    n = len(norm)

    while start < n:
        target_end = min(start + CHUNK_SIZE, n)
        if target_end >= n:
            end = n
        else:
            # Prefer the latest soft break within [start + CHUNK_SIZE//2, target_end].
            window_lo = start + CHUNK_SIZE // 2
            candidates = [b for b in breaks if window_lo < b <= target_end]
            end = max(candidates) if candidates else target_end

        raw = norm[start:end]
        piece = raw.strip()
        if piece:
            # Recompute a precise span so norm[span_start:span_end] == piece,
            # accounting for whitespace stripped from both ends.
            lead = len(raw) - len(raw.lstrip())
            trail = len(raw) - len(raw.rstrip())
            span_start = start + lead
            span_end = end - trail
            chunks.append(((span_start, span_end), piece))

        if end >= n:
            break
        # Advance with overlap, but always make forward progress.
        next_start = end - CHUNK_OVERLAP
        start = next_start if next_start > start else end

    return chunks


__all__ = ["normalize", "chunk", "content_hash", "CHUNK_SIZE", "CHUNK_OVERLAP"]
