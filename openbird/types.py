"""Frozen pydantic v2 record/event schemas shared across all OpenBird subsystems.

These models are *contracts*: their field names and types are depended on by the
capture, meetings, chat, routines, and integrations subsystems. Do not change
signatures without a serialized foundation pass.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """One timestamped occurrence of captured content.

    An observation records *when and where* something was seen. The actual text
    lives in a deduped :class:`ContentBlob` referenced by ``content_hash``. The
    same window seen 50 times produces 50 observations but a single blob, so
    dedup never collapses the timeline.
    """

    id: str
    content_hash: str
    ts: float
    app: str | None = None
    window: str | None = None
    url: str | None = None
    session_id: str | None = None
    source: str
    # Activity span this occurrence was captured within (v4; event-scoped
    # assignment by the capture daemon). Nullable: non-capture sources and
    # span-store failures ingest with no span link.
    span_id: str | None = None


class ContentBlob(BaseModel):
    """Deduped canonical text, embedded once and addressed by content hash."""

    content_hash: str
    text: str


class Chunk(BaseModel):
    """A retrievable span of text within a blob.

    ``span`` is the (start, end) character offset of the chunk inside the blob's
    normalized text. ``content_hash`` maps the chunk back to its blob (and from
    there to every observation that produced it).
    """

    id: str
    content_hash: str
    span: tuple[int, int]
    text: str


class SearchHit(BaseModel):
    """A single ranked search result, resolved back to an observation."""

    chunk_id: str
    content_hash: str
    text: str
    score: float
    observation: Observation | None = None


class Citation(BaseModel):
    """An occurrence-level citation: where a piece of context actually came from.

    ``chunk_id`` records the specific retrieved chunk the answer drew on, so a
    citation is auditable down to the chunk (not just the observation).
    """

    observation_id: str
    chunk_id: str | None = None
    app: str | None = None
    window: str | None = None
    ts: float
    snippet: str


class DerivedCitation(BaseModel):
    """A citation for a deterministic derived fact backed by observations.

    This is intentionally separate from :class:`Citation`: aggregate facts such
    as "42 minutes coding" do not live inside one occurrence, so they must carry
    an explicit source set instead of pretending to be an occurrence excerpt.
    """

    index: int
    source_id: str
    type: str = "day_memory"
    label: str
    snippet: str
    derived_from: list[str]
    derived_from_total: int


class RoutineRun(BaseModel):
    """A durable record of a single scheduled-routine execution."""

    id: str
    routine: str
    scheduled_ts: float
    started_ts: float | None = None
    finished_ts: float | None = None
    status: str
    output: str | None = None
    idempotency_key: str


__all__ = [
    "Observation",
    "ContentBlob",
    "Chunk",
    "SearchHit",
    "Citation",
    "DerivedCitation",
    "RoutineRun",
    "Field",
]
