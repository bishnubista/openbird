"""Opt-in, privacy-tiered diagnostics for the RAG grounding path.

Enabled via the ``OPENBIRD_DEBUG_RAG`` environment variable. The goal is to make
the most common production failure — an answer blanked by the grounding gate
because no citation validated — *diagnosable from one real run*, without
violating OpenBird's hard privacy rule (never log captured text, window titles,
or URLs).

Two tiers:

- ``meta`` (``1`` / ``on`` / ``meta`` / ``true`` / ``yes``): structure only —
  counts, signal-score stats, lengths, OUR source ids (``S1`` …), timing, a
  refusal heuristic, and a *format-drift probe* that reports (without changing
  behavior) how many dropped citations WOULD have matched after normalization.
  The model's raw citation tokens are NEVER echoed here: a model can place a
  captured window title / URL / snippet in the ``citations`` array, so only the
  parts whose shape we minted (``S\\d+``) are logged; everything else is counted.
- ``full`` (``2`` / ``full``): everything in ``meta`` PLUS captured content
  (answer text, window titles, snippets, raw citation tokens). Gated behind a
  one-time warning so a user cannot enable it without being told it prints
  captured data to stderr.

Isolation: when active, the module attaches its OWN stderr handler to its own
logger with ``propagate=False`` and never touches the root logger — so it cannot
un-gag other libraries (e.g. LiteLLM, which handles the captured prompt) and
cannot corrupt the CLI's ``--json`` stdout. When the flag is unset it does no
work beyond a single cheap env read.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from openbird.chat.rag import _ContextItem

logger = logging.getLogger(__name__)

DebugLevel = Literal["meta", "full"]

_ENV = "OPENBIRD_DEBUG_RAG"
_META_VALUES = frozenset({"1", "on", "meta", "true", "yes"})
_FULL_VALUES = frozenset({"2", "full"})

# Cap raw tokens echoed in the FULL tier only: collapse controls so a
# captured-text token can't forge extra log lines, and cap length.
_TOKEN_MAXLEN = 60
_TOKEN_MAXCOUNT = 20
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
# The source-id shape OpenBird mints: ``S`` followed by a positional integer.
_SOURCE_ID_RE = re.compile(r"S\d+")

# A signal score at or below this is "low signal" — a chosen source the model is
# likely to refuse to cite. Mirrors RAG._signal_score (a bare generic window
# nets ~-300; a rich row scores in the hundreds), so this flags non-positive net.
_LOW_SIGNAL_THRESHOLD = 0

# Heuristic: does the model's answer read as a refusal/abstain rather than a real
# summary? Lets meta separate "model wrote a summary but cited nothing" (H1) from
# "model honestly refused over thin sources" (H3) WITHOUT logging the answer text.
_REFUSAL_RE = re.compile(
    r"don'?t have|do not have|no recorded activity|couldn'?t find|"
    r"could not find|nothing (?:notable|in memory)|not in memory",
    re.IGNORECASE,
)

_full_banner_emitted = False
_handler_attached = False


def debug_level() -> DebugLevel | None:
    """Return the active debug tier from ``OPENBIRD_DEBUG_RAG``, or ``None``.

    Unset / ``0`` / ``off`` / ``false`` / ``no`` (and anything unrecognized)
    disable instrumentation. Case-insensitive; surrounding whitespace ignored.
    """
    raw = os.environ.get(_ENV, "").strip().lower()
    if raw in _FULL_VALUES:
        return "full"
    if raw in _META_VALUES:
        return "meta"
    return None


def _ensure_handler() -> None:
    """Attach an isolated stderr handler to this module's logger (idempotent).

    Keeps debug output visible regardless of the app's root-logger config while
    never reconfiguring root (which would un-gag other libraries that hold the
    captured prompt) and never writing to stdout (which carries ``--json``).
    """
    global _handler_attached
    if _handler_attached:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _handler_attached = True


def _median(values: list[int]) -> int:
    """Integer median of a non-empty list (caller guarantees non-empty)."""
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) // 2


def _classify_citations(
    claimed_ids: list[str], valid_source_ids: set[str]
) -> tuple[int, int]:
    """Return ``(unknown, drift)`` for citation tokens not already valid.

    ``unknown`` = tokens that don't exactly match a minted source id.
    ``drift`` = of those, how many CONTAIN an ``S\\d+`` that IS a real source id
    (i.e. the model cited ``[source_id: S1]`` / ``Source 1`` and the exact-match
    validator silently dropped it). A pure report — changes no behavior.
    """
    unknown = 0
    drift = 0
    for cid in claimed_ids:
        if cid in valid_source_ids:
            continue
        unknown += 1
        m = _SOURCE_ID_RE.search(cid)
        if m is not None and m.group(0) in valid_source_ids:
            drift += 1
    return unknown, drift


def emit_grounding_trace(
    *,
    route: str,
    raw: Any,
    answer_text: str,
    claimed_ids: list[str],
    citations: list[Any],
    context: list[_ContextItem],
    retrieval: dict[str, Any] | None = None,
    latency_s: float | None = None,
    model: str | None = None,
) -> None:
    """Emit one structured grounding-trace line (no-op unless the flag is set).

    Call this with the model's ORIGINAL parsed answer (before the gate swaps in
    the ungrounded message), so ``answer_len`` / ``refusal`` describe what the
    model actually produced.

    ``retrieval`` is an optional dict of path-specific counts the caller already
    holds — recognized keys: ``rows``, ``deduped``, ``dropped_self``,
    ``signal_scores`` (ints; the temporal path passes the chosen rows' scores).
    Missing keys are simply omitted.
    """
    level = debug_level()
    if level is None:
        return
    _ensure_handler()

    valid_source_ids = {
        item.source_id for item in context if getattr(item, "source_id", "")
    }
    parse_mode = "dict" if isinstance(raw, dict) else "string_fallback"
    unknown, drift = _classify_citations(claimed_ids, valid_source_ids)
    refusal = int(bool(answer_text) and _REFUSAL_RE.search(answer_text) is not None)

    # Time span actually covered by the chosen sources (morning->evening check).
    # Union-aware (Phase E1): summary items carry no observation and are
    # counted separately; getattr(None, ...) keeps hit=None items filtered.
    span_s = 0.0
    chosen_ts = [
        item.hit.observation.ts
        for item in context
        if getattr(item.hit, "observation", None) is not None
    ]
    if len(chosen_ts) >= 2:
        span_s = max(chosen_ts) - min(chosen_ts)
    summary_items = sum(
        1 for item in context if getattr(item, "summary", None) is not None
    )

    fields = [
        f"route={route}",
        f"chosen={len(context)}",
        f"parse={parse_mode}",
        f"answer_len={len(answer_text)}",
        f"answer_empty={int(not answer_text.strip())}",
        f"refusal={refusal}",
        f"claimed_n={len(claimed_ids)}",
        f"valid={len(citations)}",
        f"unknown={unknown}",
        f"format_drift={drift}",
        f"grounded={int(bool(citations))}",
        f"replaced={int(not citations)}",
        f"span_s={span_s:.0f}",
        f"summary_items={summary_items}",
    ]
    if model:
        fields.append(f"model={model}")
    if latency_s is not None:
        fields.append(f"latency_s={latency_s:.2f}")

    retrieval = retrieval or {}
    for key in ("rows", "deduped", "dropped_self"):
        if key in retrieval:
            fields.append(f"{key}={retrieval[key]}")
    scores = retrieval.get("signal_scores") or []
    fields.append(f"signal_count={len(scores)}")
    if scores:
        lowsig = sum(1 for s in scores if s <= _LOW_SIGNAL_THRESHOLD)
        fields.append(
            f"lowsig={lowsig} signal_min={min(scores)} "
            f"signal_med={_median(scores)} signal_max={max(scores)}"
        )

    logger.info("rag.grounding %s", " ".join(fields))

    if level == "full":
        _emit_full(answer_text, claimed_ids, context)


def emit_retrieval_empty(
    *, route: str, reason: str, retrieval: dict[str, Any] | None = None
) -> None:
    """Trace a pre-model early return (no rows / all-self / all-dupe).

    Distinct from the gate trace: the model was never called, so this records
    only the route, the reason, and whatever counts the caller had. Helps confirm
    a thin/over-filtered day (H3) when the user-facing message is the "no recorded
    activity" path rather than the ungrounded-gate path.
    """
    if debug_level() is None:
        return
    _ensure_handler()
    fields = [f"route={route}", f"reason={reason}", "grounded=0"]
    retrieval = retrieval or {}
    for key in ("rows", "deduped", "dropped_self"):
        if key in retrieval:
            fields.append(f"{key}={retrieval[key]}")
    logger.info("rag.grounding %s", " ".join(fields))


def _sanitize_token(token: str) -> str:
    """Make a raw model-emitted token safe to embed in one FULL-tier log line."""
    collapsed = _CONTROL_RE.sub(" ", str(token)).strip()
    if len(collapsed) > _TOKEN_MAXLEN:
        collapsed = collapsed[: _TOKEN_MAXLEN - 1] + "…"
    return collapsed


def _emit_full(
    answer_text: str, claimed_ids: list[str], context: list[_ContextItem]
) -> None:
    """Emit the captured-content tier (answer + raw tokens + windows + snippets).

    Guarded by a one-time warning so enabling ``full`` always announces that it
    prints captured content to stderr.
    """
    global _full_banner_emitted
    if not _full_banner_emitted:
        logger.warning(
            "%s=full prints CAPTURED CONTENT (answer text, window titles, "
            "snippets, raw citation tokens) to stderr — do not share these logs",
            _ENV,
        )
        _full_banner_emitted = True

    tokens = [_sanitize_token(c) for c in claimed_ids[:_TOKEN_MAXCOUNT]]
    logger.info("rag.grounding.full answer=%r claimed_raw=%r", answer_text, tokens)
    for item in context:
        summary = getattr(item, "summary", None)
        if summary is not None:
            # Summary-backed item (Phase E1): kind + id metadata, plus the
            # derived text (full tier already prints captured content).
            logger.info(
                "rag.grounding.full source=%s summary_kind=%s summary_id=%s "
                "snippet=%r",
                getattr(item, "source_id", ""),
                summary.get("summary_kind"),
                summary.get("summary_id"),
                str(summary.get("text") or "")[:200],
            )
            continue
        obs = getattr(item.hit, "observation", None)
        logger.info(
            "rag.grounding.full source=%s app=%r window=%r snippet=%r",
            getattr(item, "source_id", ""),
            getattr(obs, "app", None) if obs is not None else None,
            getattr(obs, "window", None) if obs is not None else None,
            item.hit.text[:200],
        )
