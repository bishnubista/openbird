"""Deterministic entity-ledger aggregation (Phase E2).

Mines the durable entity ledger (``entities`` + ``entity_evidence``, schema
v7) from STORED sources — observations, activity spans, block summaries, and
day-memory open-loop payloads. Runs only inside the gated routines pass
(:func:`openbird.summaries.run_block_summaries` calls it right after the
summary-index step) or via the on-demand ``openbird summaries build``.

STRUCTURALLY NO LLM: :func:`run_entity_aggregation` takes no provider
argument at all. Every evidence row is a regex match over a stored, citable
source, so any row can be re-verified against its citation; an LLM judge
cannot be, and "looks done" vibes are exactly the failure mode this ledger
exists to prevent (LLM extraction is E3+ territory, if ever).

Non-goals (binding for E2):
  * NO LLM evidence extraction (no provider in this signature).
  * NO fuzzy entity merging — identity is the EXACT casefolded name within a
    kind; the only automatic alias is a repo's bare name, added only while it
    is globally unique across all entities and removed on collision.
  * NO mark-done UI — the ``user_marked_done`` status value + CHECK are
    schema-ready only (this pass never touches that status).
  * NO ``document``/``topic`` entities — window titles churn per page and
    cannot be conservatively keyed; the v7 CHECK admits all four kinds
    (schema-ready) but no E2 code mints those rows.
  * NO OCR, NO cross-machine state.

Privacy: entity names, aliases, and evidence details are DERIVED SENSITIVE —
they live only in the encrypted DB and are never logged; this module logs and
returns counts and reason codes exclusively.
"""

from __future__ import annotations

import datetime as _dt
import logging
import re
from collections import Counter
from typing import Any

from openbird.day_memory import GITHUB_ITEM_RE, REPO_RE, domain_from_url

logger = logging.getLogger("openbird.entities")

# Watermark keys (embedding_meta KV; a FULL purge wipes the whole
# ``entity_aggregation.`` prefix). Per-source-class cursors:
#   * observations/spans advance the ACTIVITY-TIME composite (ts, id) cursor;
#   * block summaries are mined by GENERATION time — ``save_block_summary``
#     regeneration replaces a historical row in place with a fresh
#     ``generated_at``, so an activity-time watermark would strand a
#     regenerated old block forever behind the cursor.
_OBS_TS_KEY = "entity_aggregation.obs_ts"
_OBS_ID_KEY = "entity_aggregation.obs_id"
_SPAN_TS_KEY = "entity_aggregation.span_ts"
_SPAN_ID_KEY = "entity_aggregation.span_id"
_SUMMARY_GENERATED_AT_KEY = "entity_aggregation.summary_generated_at"
_SUMMARY_ID_KEY = "entity_aggregation.summary_id"

# Idempotent re-scan overlap behind the observation cursor. The overlap is for
# IDEMPOTENCY (late-arriving rows with backdated timestamps get a second look;
# UNIQUE evidence dedup makes re-mining harmless) — it is never the batching
# guarantee, and it is a SEPARATE bounded scan so it can never stall the
# forward cursor (no livelock when the overlap window holds more rows than the
# batch cap).
_OVERLAP_SECONDS = 48 * 3600.0

# Item-anchored completion mining: the status word must sit within this many
# characters of the SPECIFIC item match (an AX list page can hold one PR URL
# and an UNRELATED "Merged" badge — blob-global matching is forbidden).
_STATUS_WINDOW_CHARS = 120
# shipped-language proximity window around the status word in summary prose.
_SHIPPED_WINDOW_CHARS = 160

_MERGED_RE = re.compile(r"\bmerged\b", re.IGNORECASE)
_CLOSED_RE = re.compile(r"\bclosed\b", re.IGNORECASE)
_SHIPPED_RE = re.compile(
    r"\b(shipped|released|deployed|landed|finished|completed)\b", re.IGNORECASE
)
# Linear/Jira ticket completion: host + key + word CONJUNCTION (all three).
_TICKET_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")
_TICKET_DONE_RE = re.compile(r"\b(done|closed|resolved)\b", re.IGNORECASE)
_TICKET_HOSTS = frozenset({"linear.app"})
_TICKET_HOST_SUFFIX = ".atlassian.net"

# Only item-keyed loops promote; generic `cue` loops stay day-scoped
# (uncontrolled fuzzy matching; E3).
_PROMOTABLE_LOOP_KINDS = frozenset({"github_pr", "github_issue"})


def _github_detail(owner: str, repo: str, number: str) -> str:
    """Normalized item identity: casefolded so resolution matching is exact."""
    return f"github:{owner.casefold()}/{repo.casefold()}#{number}"


def _obs_blob(obs: Any, text: str) -> str:
    """The exact blob shape day_memory mines (window + url + text[:500])."""
    return " ".join(
        part for part in (obs.window or "", obs.url or "", text[:500]) if part
    )


class _Run:
    """Mutable per-run state: counts + the set of touched entity ids."""

    def __init__(self, store: Any, now: float) -> None:
        self.store = store
        self.now = float(now)
        self.touched: set[str] = set()
        self.counts: dict[str, Any] = {
            "entities": 0,
            "evidence": 0,
            "loops_promoted": 0,
            "loops_resolved": 0,
            "loops_skipped": 0,
            "dormant": 0,
        }

    def upsert(
        self,
        kind: str,
        name: str,
        *,
        seen_ts: float,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> dict:
        entity = self.store.upsert_entity(
            kind, name, seen_ts=seen_ts, source_kind=source_kind,
            source_id=source_id,
        )
        self.touched.add(entity["id"])
        return entity

    def evidence(self, entity_id: str, **kwargs: Any) -> bool:
        inserted = self.store.add_entity_evidence(entity_id, **kwargs)
        if inserted:
            self.counts["evidence"] += 1
        return inserted


def run_entity_aggregation(store: Any, *, now: float, settings: Any) -> dict:
    """One bounded deterministic aggregation pass. Counts-only result.

    Called from ``run_block_summaries`` AFTER the battery/idle/meeting gate has
    already passed — this function performs no gating of its own beyond the
    ``entity_ledger_enabled`` setting (checked by the caller). The returned
    dict carries counts and reason codes exclusively; entity names and details
    never leave the store.
    """
    run = _Run(store, now)
    limit = max(0, int(getattr(settings, "entity_evidence_batch_limit", 2000)))
    lookback_s = (
        float(getattr(settings, "entity_aggregation_lookback_days", 14.0)) * 86400.0
    )
    if limit == 0:
        return run.counts

    scan_floor = _scan_observations(run, limit=limit, lookback_s=lookback_s)
    _scan_spans(run, limit=limit, lookback_s=lookback_s)
    _refresh_repo_aliases(store)
    _mine_summaries(run, limit=limit, lookback_s=lookback_s)
    _promote_open_loops(run, scan_floor=scan_floor)
    _resolve_open_loops(run)

    dormant_days = float(getattr(settings, "entity_dormant_after_days", 21.0))
    run.counts["dormant"] = store.mark_dormant_entities(
        run.now - dormant_days * 86400.0
    )
    run.counts["entities"] = len(run.touched)
    logger.info(
        "entity aggregation run: entities=%d evidence=%d loops_promoted=%d "
        "loops_resolved=%d loops_skipped=%d dormant=%d",
        run.counts["entities"],
        run.counts["evidence"],
        run.counts["loops_promoted"],
        run.counts["loops_resolved"],
        run.counts["loops_skipped"],
        run.counts["dormant"],
    )
    return run.counts


# -- observation scan (activity-time composite cursor) ---------------------------


def _scan_observations(run: _Run, *, limit: int, lookback_s: float) -> float:
    """Forward scan + bounded overlap re-scan. Returns the scan floor ts.

    Forward scan: strictly after the stored ``(obs_ts, obs_id)`` composite
    cursor (first run: ``now - lookback``), at most ``limit`` rows, cursor
    advanced through the LAST ACTUALLY-PROCESSED row — this is the real
    batching bound and the only thing that moves the cursor. Overlap re-scan:
    a SEPARATE, equally row-capped pass over the trailing 48h behind the
    cursor for idempotency (UNIQUE dedup absorbs the re-mines); it never moves
    the cursor, so a dense overlap window can never stall forward progress.
    """
    store = run.store
    wm_ts_raw = store.get_kv(_OBS_TS_KEY)
    wm_id = str(store.get_kv(_OBS_ID_KEY) or "")
    first_run = wm_ts_raw is None
    if first_run:
        cursor_ts, cursor_id = run.now - lookback_s, ""
    else:
        cursor_ts, cursor_id = float(wm_ts_raw), wm_id

    rows = store.observations_text_page(cursor_ts, cursor_id, limit=limit)
    for obs, text in rows:
        _mine_observation(run, obs, text)
    if rows:
        last = rows[-1][0]
        store.set_kv(_OBS_TS_KEY, repr(float(last.ts)))
        store.set_kv(_OBS_ID_KEY, str(last.id))

    overlap_floor = cursor_ts - _OVERLAP_SECONDS
    if not first_run:
        overlap_rows = store.observations_text_page(
            overlap_floor, "", limit=limit
        )
        for obs, text in overlap_rows:
            if (float(obs.ts), str(obs.id)) > (cursor_ts, cursor_id):
                break  # past the old cursor: the forward scan owns these rows
            _mine_observation(run, obs, text)

    return overlap_floor if not first_run else cursor_ts


def _mine_observation(run: _Run, obs: Any, text: str) -> None:
    """Entities + item-anchored completion evidence from ONE observation."""
    blob = _obs_blob(obs, text)
    ts = float(obs.ts)

    # Repo entities (exact day_memory extractor, exported not duplicated).
    repo_entities: dict[str, dict] = {}
    for owner, repo in REPO_RE.findall(blob):
        name = f"{owner}/{repo}".rstrip("/")
        entity = run.upsert(
            "repo", name, seen_ts=ts, source_kind="observation", source_id=obs.id
        )
        repo_entities[name.casefold()] = entity

    # Domain entity from the observation URL host (same extractor as
    # day-memory domains).
    host = domain_from_url(obs.url or "")
    if host:
        run.upsert(
            "domain", host, seen_ts=ts, source_kind="observation",
            source_id=obs.id,
        )

    _mine_github_completions(run, obs, text, ts)
    _mine_ticket_completions(run, obs, text, ts, host)


def _mine_github_completions(run: _Run, obs: Any, text: str, ts: float) -> None:
    """pr_merged / ticket_closed (GitHub): item-anchored, never blob-global.

    Rule (a): when the OBSERVATION URL path itself is the item (``/pull/N`` or
    ``/issues/N``), the status word may match anywhere in window+text — the
    whole page is about that item. Rule (b): otherwise, for each item match
    inside window+text, the status word must occur within ±120 chars of THAT
    match — a list page holding PR #1 (open) and PR #2 (merged) must mint
    evidence for #2 only.
    """
    hay = " ".join(part for part in (obs.window or "", text) if part)

    url_item = GITHUB_ITEM_RE.search(obs.url or "")
    if url_item is not None:
        owner, repo, item_kind, number = url_item.groups()
        status_re = _MERGED_RE if item_kind.lower() == "pull" else _CLOSED_RE
        if status_re.search(hay):
            _add_github_completion(run, obs, ts, owner, repo, item_kind, number)
        return

    for match in GITHUB_ITEM_RE.finditer(hay):
        owner, repo, item_kind, number = match.groups()
        lo = max(0, match.start() - _STATUS_WINDOW_CHARS)
        hi = match.end() + _STATUS_WINDOW_CHARS
        window_slice = hay[lo:hi]
        status_re = _MERGED_RE if item_kind.lower() == "pull" else _CLOSED_RE
        if status_re.search(window_slice):
            _add_github_completion(run, obs, ts, owner, repo, item_kind, number)


def _add_github_completion(
    run: _Run, obs: Any, ts: float, owner: str, repo: str, item_kind: str,
    number: str,
) -> None:
    name = f"{owner}/{repo}".rstrip("/")
    entity = run.upsert(
        "repo", name, seen_ts=ts, source_kind="observation", source_id=obs.id
    )
    kind = "pr_merged" if item_kind.lower() == "pull" else "ticket_closed"
    run.evidence(
        entity["id"], ts=ts, kind=kind, source_kind="observation",
        source_id=obs.id, detail=_github_detail(owner, repo, number),
    )


def _mine_ticket_completions(
    run: _Run, obs: Any, text: str, ts: float, host: str | None
) -> None:
    """Linear/Jira ticket completion: host + key + status word CONJUNCTION."""
    if not host:
        return
    if host not in _TICKET_HOSTS and not host.endswith(_TICKET_HOST_SUFFIX):
        return
    hay = " ".join(part for part in (obs.window or "", text) if part)
    if not _TICKET_DONE_RE.search(hay):
        return
    keys = sorted(set(_TICKET_KEY_RE.findall(hay)))
    if not keys:
        return
    entity = run.upsert(
        "domain", host, seen_ts=ts, source_kind="observation", source_id=obs.id
    )
    for key in keys:
        run.evidence(
            entity["id"], ts=ts, kind="ticket_closed",
            source_kind="observation", source_id=obs.id, detail=key,
        )


# -- span scan (domain extent) ----------------------------------------------------


def _scan_spans(run: _Run, *, limit: int, lookback_s: float) -> None:
    """Domain entities from tier-1 span url_hosts (span-backed extent).

    Spans carry host-only metadata (no paths), so they can support domain
    entities and their activity extent — nothing finer. INDEPENDENT
    row-capped composite ``(start_ts, span_id)`` cursor (its own
    ``entity_aggregation.span_ts``/``.span_id`` watermark pair), mirroring
    the observation cursor semantics: the forward page advances the cursor
    only through the last actually-processed row, and a SEPARATE, equally
    row-capped 48h overlap re-scan (idempotent upserts absorb it; catches
    still-extending spans behind the cursor) never moves it — a dense span
    history therefore pages forward run over run instead of re-slicing the
    earliest overlapping spans forever.
    """
    store = run.store
    wm_ts_raw = store.get_kv(_SPAN_TS_KEY)
    wm_id = str(store.get_kv(_SPAN_ID_KEY) or "")
    first_run = wm_ts_raw is None
    if first_run:
        cursor_ts, cursor_id = run.now - lookback_s, ""
    else:
        cursor_ts, cursor_id = float(wm_ts_raw), wm_id

    spans = store.spans_page(cursor_ts, cursor_id, limit=limit)
    for span in spans:
        _mine_span(run, span)
    if spans:
        last = spans[-1]
        store.set_kv(_SPAN_TS_KEY, repr(float(last["start_ts"])))
        store.set_kv(_SPAN_ID_KEY, str(last["span_id"]))

    if not first_run:
        overlap = store.spans_page(
            cursor_ts - _OVERLAP_SECONDS, "", limit=limit
        )
        for span in overlap:
            if (float(span["start_ts"]), str(span["span_id"])) > (
                cursor_ts,
                cursor_id,
            ):
                break  # past the old cursor: the forward page owns these rows
            _mine_span(run, span)


def _mine_span(run: _Run, span: dict) -> None:
    """Upsert the domain entity for ONE tier-1 span (host-only metadata)."""
    host = span.get("url_host")
    if not host:
        return
    end_ts = min(float(span.get("end_ts") or 0.0), run.now)
    run.upsert(
        "domain", str(host), seen_ts=end_ts, source_kind="span",
        source_id=str(span.get("span_id")),
    )


# -- aliases -----------------------------------------------------------------------


def _refresh_repo_aliases(store: Any) -> None:
    """Conservative aliasing: bare repo name, only while GLOBALLY unique.

    Re-checked every run: ``openbird`` aliases ``bbista/openbird`` only while
    no other entity (any kind) claims that casefolded bare name; on collision
    the alias is REMOVED from every claimant. The bare repo name is the ONLY
    automatic alias — no fuzzy merging of any kind.
    """
    entities = store.list_entities()
    repos = [e for e in entities if e.get("kind") == "repo" and "/" in e["name"]]
    name_keys = {str(e["name"]).casefold() for e in entities}
    bare_counts: Counter[str] = Counter(
        e["name"].split("/", 1)[1].casefold() for e in repos
    )
    for entity in repos:
        bare_raw = entity["name"].split("/", 1)[1]
        bare = bare_raw.casefold()
        unique = bool(bare) and bare_counts[bare] == 1 and bare not in name_keys
        current = [str(a).casefold() for a in entity.get("aliases") or []]
        if unique and current != [bare]:
            store.set_entity_aliases(entity["id"], [bare_raw])
        elif not unique and current:
            store.set_entity_aliases(entity["id"], [])


# -- summary mining (generation-time cursor) ----------------------------------------


def _mine_summaries(run: _Run, *, limit: int, lookback_s: float) -> None:
    """shipped_language over block-summary prose, citing the summary id.

    Cursor is ``generated_at`` (composite with the row id), NOT activity time:
    regeneration replaces a historical row in place with a fresh
    ``generated_at``, its trigger cascade deletes the old evidence, and this
    cursor re-mines the replacement — an activity-time watermark would strand
    it permanently. UNIQUE dedup keeps re-mines idempotent.
    """
    store = run.store
    wm_raw = store.get_kv(_SUMMARY_GENERATED_AT_KEY)
    wm_id = str(store.get_kv(_SUMMARY_ID_KEY) or "")
    if wm_raw is None:
        after, after_id = run.now - lookback_s, ""
    else:
        after, after_id = float(wm_raw), wm_id

    summaries = store.block_summaries_generated_since(after, after_id, limit=limit)
    if not summaries:
        return

    # Match against the CURRENT ledger (names + aliases, casefolded) — the
    # observation/span phases above already upserted this run's sightings.
    # TOKEN-BOUNDARY regexes, never bare substrings: a short alias like
    # ``ai``/``ui``/``go`` inside an unrelated word ("maintained") near a
    # shipped/completed verb must NOT mint completion evidence. ``(?<!\w)`` /
    # ``(?!\w)`` lookarounds (rather than ``\b``) keep slash-containing repo
    # names matching as a full owner/repo token even inside path/URL contexts
    # ("github.com/owner/repo/pull/3": the surrounding "/" is a non-word
    # char, so the lookarounds pass; "owner/repository" does not match).
    matchers: list[tuple[re.Pattern[str], str, str]] = []  # (pattern, kind, name)
    for entity in store.list_entities():
        keys = {str(entity["name"]).casefold()}
        keys.update(str(a).casefold() for a in entity.get("aliases") or [])
        for key in keys:
            if key:
                pattern = re.compile(r"(?<!\w)" + re.escape(key) + r"(?!\w)")
                matchers.append((pattern, entity["kind"], entity["name"]))

    for summary in summaries:
        text = str(summary.get("summary_text") or "")
        folded = text.casefold()
        ts = float(summary.get("end_ts") or summary.get("generated_at") or run.now)
        for match in _SHIPPED_RE.finditer(folded):
            lo = max(0, match.start() - _SHIPPED_WINDOW_CHARS)
            hi = match.end() + _SHIPPED_WINDOW_CHARS
            window_slice = folded[lo:hi]
            for pattern, kind, name in matchers:
                if pattern.search(window_slice) is None:
                    continue
                entity = run.upsert(
                    kind, name, seen_ts=ts, source_kind="summary",
                    source_id=str(summary["id"]),
                )
                run.evidence(
                    entity["id"], ts=ts, kind="shipped_language",
                    source_kind="summary", source_id=str(summary["id"]),
                    detail=str(match.group(1)),
                )

    last = summaries[-1]
    store.set_kv(_SUMMARY_GENERATED_AT_KEY, repr(float(last["generated_at"])))
    store.set_kv(_SUMMARY_ID_KEY, str(last["id"]))


# -- open-loop promotion (rehydrated, never payload-trusted) -------------------------


def _promote_open_loops(run: _Run, *, scan_floor: float) -> None:
    """Promote item-keyed day-memory open loops into durable ledger evidence.

    The payload's loop shape carries ``source_ids`` that are SORTED AND CAPPED
    (never chronological) and no owner/repo/number — so this REHYDRATES: it
    re-reads the cited source observations, re-parses the item id from the
    original rows, and cites the TRUE EARLIEST row. Loops whose rehydration
    yields no single stable item id are skipped (counted). Generic ``cue``
    loops stay day-scoped (uncontrolled fuzzy matching; E3).
    """
    store = run.store
    reader = getattr(store, "get_day_memory", None)
    if not callable(reader):
        return
    for local_date in _dates_between(scan_floor, run.now):
        saved = reader(local_date=local_date)
        if not saved:
            continue
        payload = saved.get("payload") or {}
        for loop in payload.get("open_loops") or []:
            if str(loop.get("kind") or "") not in _PROMOTABLE_LOOP_KINDS:
                continue
            rows = store.observations_text_for_ids(
                [str(i) for i in loop.get("source_ids") or []]
            )
            item, earliest = _stable_loop_item(rows)
            if item is None or earliest is None:
                run.counts["loops_skipped"] += 1
                logger.info(
                    "open loop skipped: reason=unstable_item_id date=%s",
                    local_date,
                )
                continue
            owner, repo, _item_kind, number = item
            obs = earliest[0]
            entity = run.upsert(
                "repo", f"{owner}/{repo}".rstrip("/"), seen_ts=float(obs.ts),
                source_kind="observation", source_id=obs.id,
            )
            inserted = run.evidence(
                entity["id"], ts=float(obs.ts), kind="open_loop",
                source_kind="observation", source_id=obs.id,
                detail=_github_detail(owner, repo, number),
            )
            if inserted:
                run.counts["loops_promoted"] += 1


def _stable_loop_item(
    rows: list[tuple[Any, str]]
) -> tuple[tuple[str, str, str, str] | None, tuple[Any, str] | None]:
    """The single item id all rehydrated rows agree on, plus its earliest row.

    Items are keyed casefolded (``pull``/``issues`` kept as matched, lowered);
    if the rows collectively reference more than one distinct item — or none —
    there is no stable id and the loop is skipped.
    """
    items: dict[tuple[str, str, str, str], tuple[Any, str]] = {}
    keys: set[tuple[str, str, str, str]] = set()
    for obs, text in rows:  # rows arrive ordered by (ts, id) ascending
        blob = _obs_blob(obs, text)
        for owner, repo, item_kind, number in GITHUB_ITEM_RE.findall(blob):
            key = (owner.casefold(), repo.casefold(), item_kind.lower(), number)
            keys.add(key)
            if key not in items:
                items[key] = (obs, text)
    if len(keys) != 1:
        return None, None
    key = next(iter(keys))
    return key, items[key]


# -- resolution matching (exact detail key only) -------------------------------------


def _resolve_open_loops(run: _Run) -> None:
    """Insert ``open_loop_resolved`` for every exact-detail completion.

    Precise rule: an ``open_loop`` row is resolved iff a ``pr_merged`` /
    ``ticket_closed`` row exists on the SAME entity with the IDENTICAL
    ``detail`` key and a LATER ``ts`` — never loop-text similarity. The
    resolution row cites the RESOLVING source (earliest such completion).
    """
    seen: set[tuple[str, str]] = set()
    for cand in run.store.entity_open_loop_candidates():
        pair = (str(cand["entity_id"]), str(cand["detail"]))
        if pair in seen:
            continue  # keep the earliest resolving row per loop
        seen.add(pair)
        inserted = run.evidence(
            str(cand["entity_id"]), ts=float(cand["resolved_ts"]),
            kind="open_loop_resolved", source_kind=str(cand["source_kind"]),
            source_id=str(cand["source_id"]), detail=str(cand["detail"]),
        )
        if inserted:
            run.counts["loops_resolved"] += 1


def _dates_between(start_ts: float, end_ts: float) -> list[str]:
    """Local YYYY-MM-DD dates covering [start_ts, end_ts], oldest first."""
    if end_ts < start_ts:
        return []
    cursor = _dt.datetime.fromtimestamp(start_ts).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end_date = _dt.datetime.fromtimestamp(end_ts).date()
    out: list[str] = []
    while cursor.date() <= end_date:
        out.append(cursor.strftime("%Y-%m-%d"))
        cursor += _dt.timedelta(days=1)
    return out


__all__ = ["run_entity_aggregation"]
