-- OpenBird memory schema.
--
-- Data model: observations (every timestamped occurrence) are separated
-- from deduped content (content_blobs), so dedup never destroys timeline
-- semantics. Indexing/ranking is CHUNK-LEVEL: FTS5 over chunks, one vector per
-- chunk. Each chunk maps content_hash -> blob -> observations.
--
-- The vec_chunks virtual table is created in Python (store.py) because its
-- dimension is taken from Settings.embed_dim; the rest lives here.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Deduped canonical text, addressed by SHA-256 of its normalized form.
CREATE TABLE IF NOT EXISTS content_blobs (
    content_hash TEXT PRIMARY KEY,
    text         TEXT NOT NULL
);

-- One row per occurrence (never deduped). The same window seen 50 times = 50
-- observations referencing 1 blob. ``span_id`` (v4) links an observation to the
-- activity span it was captured within (event-scoped assignment; nullable —
-- spans are ground-truth time, observations are content occurrences).
--
-- NOTE: idx_observations_span is created ONLY by the v4 migration, never here:
-- schema.sql runs BEFORE the migration ladder, and on a pre-v4 DB this table
-- already exists without span_id, so an index definition here would raise
-- "no such column" before the ladder could add it.
CREATE TABLE IF NOT EXISTS observations (
    id           TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL REFERENCES content_blobs(content_hash) ON DELETE CASCADE,
    ts           REAL NOT NULL,
    app          TEXT,
    window       TEXT,
    url          TEXT,
    session_id   TEXT,
    source       TEXT NOT NULL,
    span_id      TEXT REFERENCES activity_spans(span_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_observations_ts      ON observations(ts);
CREATE INDEX IF NOT EXISTS idx_observations_hash    ON observations(content_hash);
CREATE INDEX IF NOT EXISTS idx_observations_session ON observations(session_id);
-- v8: covers the assistant keyset page scan (WHERE source = 'capture' AND the
-- (ts, id) descent) so equal-timestamp groups never force a temp-B-tree re-sort
-- per page. Safe here (all three columns exist in every released shape) AND in
-- the v8 migration (IF NOT EXISTS keeps the two in lockstep).
CREATE INDEX IF NOT EXISTS idx_observations_source_ts_id ON observations(source, ts, id);
-- v11: covers the all-supported-source founder recap keyset walk. The source
-- filter is intentionally an IN-list, so a leading source column would group
-- rather than globally order rows; this index serves ORDER BY ts DESC, id DESC.
CREATE INDEX IF NOT EXISTS idx_observations_ts_id ON observations(ts DESC, id DESC);
-- v10: meeting UUIDs are idempotency keys. Only meeting observations participate,
-- so existing capture/import session ids retain their occurrence semantics.
CREATE UNIQUE INDEX IF NOT EXISTS idx_observations_meeting_session
    ON observations(session_id)
    WHERE source = 'meeting' AND session_id IS NOT NULL;

-- Encrypted-at-rest meeting transcript checkpoints (v10). This table lives in
-- the same SQLCipher-gated database as observations; raw PCM is never stored.
-- The controller updates this row as transcript windows complete, then deletes
-- it only after the corresponding meeting observation and indexes commit.
CREATE TABLE IF NOT EXISTS pending_meetings (
    meeting_id       TEXT PRIMARY KEY,
    version          INTEGER NOT NULL DEFAULT 1 CHECK (version = 1),
    started_ts       REAL NOT NULL,
    ended_ts         REAL,
    transcript       TEXT NOT NULL DEFAULT '',
    segments_json    TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(segments_json)),
    backend          TEXT,
    partial_reason   TEXT,
    dropped_windows  INTEGER NOT NULL DEFAULT 0 CHECK (dropped_windows >= 0),
    failed_windows   INTEGER NOT NULL DEFAULT 0 CHECK (failed_windows >= 0),
    truncated_bytes  INTEGER NOT NULL DEFAULT 0 CHECK (truncated_bytes >= 0),
    observation_id   TEXT REFERENCES observations(id) ON DELETE SET NULL,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pending_meetings_started_ts
    ON pending_meetings(started_ts);

-- Content-free capture accountability (v9). This table deliberately excludes
-- captured text, titles, URLs, content hashes, and free-form reasons. Started
-- rows make interrupted attempts visible; finished events upsert the outcome.
CREATE TABLE IF NOT EXISTS capture_attempts (
    attempt_id                TEXT PRIMARY KEY,
    helper_epoch              TEXT NOT NULL,
    trigger_seq               INTEGER NOT NULL CHECK (trigger_seq >= 0),
    trigger_ts                REAL NOT NULL,
    started_ts                REAL,
    finished_ts               REAL,
    status                    TEXT NOT NULL CHECK (status IN ('started','finished')),
    bundle_id                 TEXT,
    trigger                   TEXT NOT NULL CHECK (trigger IN (
        'app_activated','window_changed','title_changed','focus_changed',
        'typing_pause','idle_tick','force_ceiling','return_from_afk','startup'
    )),
    adapter_id                TEXT CHECK (adapter_id IS NULL OR adapter_id IN ('generic_ax')),
    extractor_version         TEXT CHECK (
        extractor_version IS NULL OR extractor_version IN ('generic_ax_v1')
    ),
    policy_tier               INTEGER CHECK (policy_tier IS NULL OR policy_tier IN (0, 1)),
    outcome                   TEXT CHECK (outcome IS NULL OR outcome IN (
        'captured_full','captured_partial','captured_shallow','captured_unchanged',
        'coalesced_inflight','skipped_policy','skipped_afk','skipped_paused',
        'unsupported','failed_bounded'
    )),
    nodes_visited             INTEGER NOT NULL DEFAULT 0 CHECK (nodes_visited >= 0),
    bytes_emitted             INTEGER NOT NULL DEFAULT 0 CHECK (bytes_emitted >= 0),
    elapsed_ms                INTEGER NOT NULL DEFAULT 0 CHECK (elapsed_ms >= 0),
    completeness              TEXT CHECK (
        completeness IS NULL OR completeness IN ('full','partial','shallow','none')
    ),
    reason_codes_json         TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(reason_codes_json)),
    coalesced_trigger_count   INTEGER NOT NULL DEFAULT 0 CHECK (coalesced_trigger_count >= 0),
    earliest_coalesced_ts     REAL,
    successor_attempt_id      TEXT REFERENCES capture_attempts(attempt_id) ON DELETE SET NULL,
    observation_id            TEXT REFERENCES observations(id) ON DELETE SET NULL,
    UNIQUE(helper_epoch, trigger_seq),
    CHECK ((status = 'started' AND outcome IS NULL AND finished_ts IS NULL)
        OR (status = 'finished' AND outcome IS NOT NULL AND finished_ts IS NOT NULL)),
    CHECK ((outcome = 'coalesced_inflight'
            AND coalesced_trigger_count >= 1
            AND earliest_coalesced_ts IS NOT NULL)
        OR (outcome IS NULL
            AND coalesced_trigger_count = 0
            AND earliest_coalesced_ts IS NULL)
        OR (outcome != 'coalesced_inflight'
            AND coalesced_trigger_count = 0
            AND earliest_coalesced_ts IS NULL)),
    CHECK (successor_attempt_id IS NULL OR outcome IS 'coalesced_inflight')
);

CREATE INDEX IF NOT EXISTS idx_capture_attempts_trigger_ts
    ON capture_attempts(trigger_ts);
CREATE INDEX IF NOT EXISTS idx_capture_attempts_outcome
    ON capture_attempts(outcome, trigger_ts);

-- Heartbeat-merged activity spans (v4): ground-truth "app X was frontmost from
-- t1 to t2" rows, two-tier by the STRUCTURAL policy classification
-- (redact.classify_policy). Tier 0 (coarse) carries NO window/url_host/
-- identity_key — enforced here by CHECK and again in Python (open_span).
-- Wall-clock timestamps are storage only; merge deadlines are monotonic and
-- never persisted. epoch_id scopes merging to one process lifetime.
CREATE TABLE IF NOT EXISTS activity_spans (
    span_id      TEXT PRIMARY KEY,
    epoch_id     TEXT NOT NULL,
    start_ts     REAL NOT NULL,
    end_ts       REAL NOT NULL,
    bundle_id    TEXT,               -- NULL only for reason='paused' spans
    app          TEXT,               -- reserved (localized name); NULL for now
    detail_tier  INTEGER NOT NULL CHECK (detail_tier IN (0, 1)),
    window       TEXT,               -- tier 1 only (scrubbed); NULL if untitled
    url_host     TEXT,               -- tier 1 only; host only; opt-in
    identity_key TEXT,               -- tier 1 only; per-app identity (file path/
                                     -- repo/document); extraction deferred
    afk          INTEGER NOT NULL DEFAULT 0,
    meeting      INTEGER NOT NULL DEFAULT 0,  -- Phase C
    reason       TEXT CHECK (reason IN ('not_allowlisted','blocklisted','dangerous',
                                        'private','paused','self_capture')),
    CHECK (end_ts >= start_ts),
    CHECK ((detail_tier = 0 AND window IS NULL AND url_host IS NULL
            AND identity_key IS NULL AND reason IS NOT NULL)
        OR (detail_tier = 1 AND reason IS NULL))
);

CREATE INDEX IF NOT EXISTS idx_activity_spans_start  ON activity_spans(start_ts);
CREATE INDEX IF NOT EXISTS idx_activity_spans_end    ON activity_spans(end_ts);
CREATE INDEX IF NOT EXISTS idx_activity_spans_bundle ON activity_spans(bundle_id, start_ts);

-- Globally-deduped retrievable chunks, addressed by SHA-256 of their *normalized
-- chunk text* (not the parent window). A unique chunk is stored,
-- embedded, and indexed exactly once even when it recurs across many different
-- windows/blobs — so a one-character edit elsewhere in a window does not re-embed
-- the unchanged chunks.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_hash   TEXT PRIMARY KEY,
    text         TEXT NOT NULL,
    rowid_int    INTEGER UNIQUE  -- stable integer key mirrored into fts/vec
);

-- Occurrence mapping: which chunks appear in which blob, with the chunk's span
-- within that blob. Deleting a blob cascades its mappings; a chunk is reclaimed
-- once no blob references it.
CREATE TABLE IF NOT EXISTS blob_chunks (
    content_hash TEXT NOT NULL REFERENCES content_blobs(content_hash) ON DELETE CASCADE,
    chunk_hash   TEXT NOT NULL REFERENCES chunks(chunk_hash) ON DELETE CASCADE,
    span_start   INTEGER NOT NULL,
    span_end     INTEGER NOT NULL,
    PRIMARY KEY (content_hash, chunk_hash)
);

CREATE INDEX IF NOT EXISTS idx_blob_chunks_chunk ON blob_chunks(chunk_hash);
CREATE INDEX IF NOT EXISTS idx_blob_chunks_blob  ON blob_chunks(content_hash);

-- FTS5 over chunk text. Rows are keyed by an explicit rowid that mirrors
-- chunks.rowid_int, so FTS matches resolve straight back to a chunk. We store
-- the text (not contentless) so rows can be DELETEd directly during cascade.
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    text
);

-- Embedding cohort metadata: refuse search across incompatible cohorts.
CREATE TABLE IF NOT EXISTS embedding_meta (
    key        TEXT PRIMARY KEY,
    value      TEXT
);

-- Durable deterministic day memories. These are derived from captured activity,
-- so they live in the same encrypted DB as observations and are invalidated when
-- any source observation is deleted.
CREATE TABLE IF NOT EXISTS day_memories (
    id                TEXT PRIMARY KEY,
    local_date        TEXT NOT NULL,
    source_scope      TEXT NOT NULL DEFAULT 'capture',
    extractor_version TEXT NOT NULL,
    generated_at      REAL NOT NULL,
    payload_json      TEXT NOT NULL,
    source_count      INTEGER NOT NULL,
    UNIQUE(local_date, source_scope)
);

CREATE INDEX IF NOT EXISTS idx_day_memories_date_scope
    ON day_memories(local_date, source_scope);

-- Typed derived-artifact citations (v4; replaces the observation-only
-- day_memory_sources, which this file deliberately no longer defines — the v4
-- migration backfills and drops it on upgraded DBs; NO DROP statements here).
-- source_kind types the citation so day memories can cite spans as well as
-- observations; integrity is enforced by the BEFORE INSERT triggers below
-- (typed refs cannot use a single FK).
-- ``source_kind`` gained 'summary' in v5 (a day memory may cite a block
-- summary). SQLite cannot ALTER a CHECK, so the v5 migration REBUILDS this
-- table on upgrading DBs; here it carries the final shape for fresh DBs (the
-- migration's sqlite_master probe then no-ops). NO DROP statements here.
CREATE TABLE IF NOT EXISTS day_memory_source_refs (
    day_memory_id TEXT NOT NULL REFERENCES day_memories(id) ON DELETE CASCADE,
    source_kind   TEXT NOT NULL CHECK (source_kind IN ('observation','span','summary')),
    source_id     TEXT NOT NULL,
    PRIMARY KEY (day_memory_id, source_kind, source_id)
);

CREATE INDEX IF NOT EXISTS idx_day_memory_source_refs_source
    ON day_memory_source_refs(source_kind, source_id);

-- Insert-time integrity per source_kind: a derived row must never cite a
-- missing source (typo/stale IDs fail loudly, in the same transaction).
CREATE TRIGGER IF NOT EXISTS trg_day_memory_source_refs_obs_exists
BEFORE INSERT ON day_memory_source_refs
WHEN NEW.source_kind = 'observation'
    AND NOT EXISTS (SELECT 1 FROM observations WHERE id = NEW.source_id)
BEGIN
    SELECT RAISE(ABORT, 'day_memory_source_refs: unknown observation ref');
END;

CREATE TRIGGER IF NOT EXISTS trg_day_memory_source_refs_span_exists
BEFORE INSERT ON day_memory_source_refs
WHEN NEW.source_kind = 'span'
    AND NOT EXISTS (SELECT 1 FROM activity_spans WHERE span_id = NEW.source_id)
BEGIN
    SELECT RAISE(ABORT, 'day_memory_source_refs: unknown span ref');
END;

-- Use BEFORE DELETE so the junction row is still present when the trigger reads
-- it; SQLite FK cascades would remove the refs before an AFTER trigger.
CREATE TRIGGER IF NOT EXISTS trg_day_memory_source_observation_delete
BEFORE DELETE ON observations
BEGIN
    DELETE FROM day_memories
    WHERE id IN (
        SELECT day_memory_id
        FROM day_memory_source_refs
        WHERE source_kind = 'observation' AND source_id = OLD.id
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_day_memory_source_span_delete
BEFORE DELETE ON activity_spans
BEGIN
    DELETE FROM day_memories
    WHERE id IN (
        SELECT day_memory_id
        FROM day_memory_source_refs
        WHERE source_kind = 'span' AND source_id = OLD.span_id
    );
END;

-- Idle-time block summaries (v5, Phase D): one local-model prose summary per
-- settled focus block. ``summary_text`` is DERIVED SENSITIVE data (distilled
-- from captured content) — it lives only in this encrypted DB and is never
-- logged; loggers emit counts and reason codes only.
CREATE TABLE IF NOT EXISTS block_summaries (
    id                TEXT PRIMARY KEY,
    local_date        TEXT NOT NULL,
    block_key         TEXT NOT NULL UNIQUE,   -- sha256 over the block's sorted span_ids
    block_fingerprint TEXT NOT NULL,          -- sha256 over sorted (span_id, end_ts): staleness probe
    start_ts          REAL NOT NULL,
    end_ts            REAL NOT NULL,
    dominant_bundle   TEXT,
    level             TEXT CHECK (level IS NULL OR level IN
                      ('focus_work','other_work','neutral','personal','distracting')),
    summary_text      TEXT NOT NULL,          -- DERIVED SENSITIVE: never logged
    model             TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    generated_at      REAL NOT NULL,
    source_count      INTEGER NOT NULL,
    CHECK (end_ts >= start_ts)
);

CREATE INDEX IF NOT EXISTS idx_block_summaries_date  ON block_summaries(local_date);
CREATE INDEX IF NOT EXISTS idx_block_summaries_start ON block_summaries(start_ts);

-- Typed source refs for block summaries (mirrors day_memory_source_refs):
-- integrity is enforced by the BEFORE INSERT triggers below, deletion cascade
-- by the BEFORE DELETE triggers on the source tables.
CREATE TABLE IF NOT EXISTS block_summary_source_refs (
    summary_id  TEXT NOT NULL REFERENCES block_summaries(id) ON DELETE CASCADE,
    source_kind TEXT NOT NULL CHECK (source_kind IN ('observation','span')),
    source_id   TEXT NOT NULL,
    PRIMARY KEY (summary_id, source_kind, source_id)
);

CREATE INDEX IF NOT EXISTS idx_block_summary_source_refs_source
    ON block_summary_source_refs(source_kind, source_id);

-- LLM-fallback taxonomy cache ONLY (rules and user overrides are config, never
-- cached here). Stores identity key + level + provenance metadata — no captured
-- content. LLM-derived from captured context, so a full purge wipes it.
CREATE TABLE IF NOT EXISTS category_assignments (
    identity_key TEXT PRIMARY KEY,              -- 'bundle:<id>' | 'host:<host>'
    level        TEXT NOT NULL CHECK (level IN
                 ('focus_work','other_work','neutral','personal','distracting')),
    model        TEXT NOT NULL,
    generated_at REAL NOT NULL
);

-- Insert-time integrity per source_kind (mirrors the day-memory triggers): a
-- block summary must never cite a missing source.
CREATE TRIGGER IF NOT EXISTS trg_block_summary_source_refs_obs_exists
BEFORE INSERT ON block_summary_source_refs
WHEN NEW.source_kind = 'observation'
    AND NOT EXISTS (SELECT 1 FROM observations WHERE id = NEW.source_id)
BEGIN
    SELECT RAISE(ABORT, 'block_summary_source_refs: unknown observation ref');
END;

CREATE TRIGGER IF NOT EXISTS trg_block_summary_source_refs_span_exists
BEFORE INSERT ON block_summary_source_refs
WHEN NEW.source_kind = 'span'
    AND NOT EXISTS (SELECT 1 FROM activity_spans WHERE span_id = NEW.source_id)
BEGIN
    SELECT RAISE(ABORT, 'block_summary_source_refs: unknown span ref');
END;

-- Deleting a cited source invalidates the derived block summary (BEFORE DELETE
-- so the junction row is still readable, exactly like the day-memory pair).
CREATE TRIGGER IF NOT EXISTS trg_block_summary_source_observation_delete
BEFORE DELETE ON observations
BEGIN
    DELETE FROM block_summaries
    WHERE id IN (
        SELECT summary_id
        FROM block_summary_source_refs
        WHERE source_kind = 'observation' AND source_id = OLD.id
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_block_summary_source_span_delete
BEFORE DELETE ON activity_spans
BEGIN
    DELETE FROM block_summaries
    WHERE id IN (
        SELECT summary_id
        FROM block_summary_source_refs
        WHERE source_kind = 'span' AND source_id = OLD.span_id
    );
END;

-- Summary-kind integrity + invalidation for day memories. The summary-delete
-- trigger fires when a block summary is removed DIRECTLY or by the cascade
-- triggers above — the latter chain (span delete -> block summary delete ->
-- day memory delete) requires PRAGMA recursive_triggers = ON, which the store
-- sets and VERIFIES per connection before applying this schema.
CREATE TRIGGER IF NOT EXISTS trg_day_memory_source_refs_summary_exists
BEFORE INSERT ON day_memory_source_refs
WHEN NEW.source_kind = 'summary'
    AND NOT EXISTS (SELECT 1 FROM block_summaries WHERE id = NEW.source_id)
BEGIN
    SELECT RAISE(ABORT, 'day_memory_source_refs: unknown summary ref');
END;

CREATE TRIGGER IF NOT EXISTS trg_day_memory_source_summary_delete
BEFORE DELETE ON block_summaries
BEGIN
    DELETE FROM day_memories
    WHERE id IN (
        SELECT day_memory_id
        FROM day_memory_source_refs
        WHERE source_kind = 'summary' AND source_id = OLD.id
    );
END;

-- Parallel summary index (v6, Phase E1): retrievable derived-narrative entries
-- for block summaries and week digests. DELIBERATELY separate from the
-- occurrence model (chunks/observations) — summary prose must never
-- double-count as captured activity in time-range scans, and chunk_hash is
-- content-addressed/globally deduped so summary text cannot share that table.
-- ``text`` is DERIVED SENSITIVE (distilled from captured content): it lives
-- only in this encrypted DB and is never logged.
--
-- DELETION CONTRACT (API-enforced): the two BEFORE DELETE triggers below clean
-- the PLAIN table only — SQLite triggers cannot write virtual tables, so the
-- paired fts_summaries/vec_summaries rows are removed by
-- MemoryStore._sweep_summary_index_for_blocks IN THE SAME TRANSACTION, BEFORE
-- the source delete fires these triggers. Raw SQL deletes against
-- block_summaries or week-scope day_memories rows outside the MemoryStore
-- APIs (delete()/prune()/save_block_summary()/save_week_memory()) are
-- FORBIDDEN: they would strand fts/vec orphans. `openbird data integrity`
-- carries an orphan-count probe as the safety net.
CREATE TABLE IF NOT EXISTS summary_index_entries (
    entry_rowid  INTEGER PRIMARY KEY,          -- mirrored into fts/vec rowids
    summary_kind TEXT NOT NULL CHECK (summary_kind IN ('block','week')),
    summary_id   TEXT NOT NULL,
    seq          INTEGER NOT NULL DEFAULT 0,   -- ingest.chunk() piece order
    text         TEXT NOT NULL,                -- DERIVED SENSITIVE: never logged
    fingerprint  TEXT NOT NULL,                -- staleness probe (block_fingerprint /
                                               -- member_fingerprint)
    UNIQUE(summary_kind, summary_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_summary_index_entries_summary
    ON summary_index_entries(summary_kind, summary_id);

-- FTS5 over summary-entry text; rowid mirrors summary_index_entries.entry_rowid.
-- vec_summaries (vec0) is created in Python (store._apply_schema) because its
-- dimension comes from Settings.embed_dim, exactly like vec_chunks.
CREATE VIRTUAL TABLE IF NOT EXISTS fts_summaries USING fts5(
    text
);

-- Entry cleanup on source deletion (plain table only; see the contract above).
CREATE TRIGGER IF NOT EXISTS trg_summary_index_block_delete
BEFORE DELETE ON block_summaries
BEGIN
    DELETE FROM summary_index_entries
    WHERE summary_kind = 'block' AND summary_id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_summary_index_week_delete
BEFORE DELETE ON day_memories
WHEN OLD.source_scope = 'week'
BEGIN
    DELETE FROM summary_index_entries
    WHERE summary_kind = 'week' AND summary_id = OLD.id;
END;

-- Entity ledger (v7, Phase E2): durable per-project/domain state derived
-- DETERMINISTICALLY from stored sources (observations, spans, block summaries)
-- by the nightly aggregation pass — no LLM ever writes these tables.
-- ``name``/``aliases`` are DERIVED SENSITIVE (distilled from captured content):
-- encrypted DB only, never logged; loggers emit counts and reason codes only.
-- ``id`` is sha256("kind:casefold(name)") so upserts are deterministic and
-- idempotent. ``last_seen_source_kind``/``last_seen_source_id`` is a TYPED
-- last-activity ref (span-derived domain entities have no observation to
-- cite); validity is enforced by the BEFORE INSERT/UPDATE triggers below and
-- the per-kind BEFORE DELETE triggers NULL the pair when the source dies.
--
-- NO fts/vec indexing for entities or evidence — DELIBERATE: the ledger is
-- queried by exact casefolded name/alias match and answers are composed
-- deterministically from rows; there is no semantic-retrieval consumer.
-- Indexing name/detail strings would extend the API-enforced sweep contract
-- (_sweep_summary_index_*) and every deletion path for zero recall benefit,
-- and would risk the double-counting E1 explicitly rejected. Consequence:
-- delete()/prune() need NO new pre-select sweep — plain-table triggers fully
-- cover entity evidence.
CREATE TABLE IF NOT EXISTS entities (
    id               TEXT PRIMARY KEY,
    kind             TEXT NOT NULL CHECK (kind IN ('repo','domain','document','topic')),
    name             TEXT NOT NULL,               -- DERIVED SENSITIVE: never logged
    aliases          TEXT NOT NULL DEFAULT '[]',  -- JSON array; DERIVED SENSITIVE
    first_ts         REAL,
    last_ts          REAL,
    status           TEXT NOT NULL DEFAULT 'active'
                     CHECK (status IN ('active','dormant','user_marked_done')),
    last_seen_source_kind TEXT CHECK (last_seen_source_kind IN
                     ('observation','span','summary')),
    last_seen_source_id  TEXT,
    CHECK ((last_seen_source_kind IS NULL) = (last_seen_source_id IS NULL)),
    UNIQUE(kind, name)
);

-- One row per completion/open-loop signal, always citing a live typed source.
-- ``detail`` is the normalized matched cue (e.g. 'github:owner/repo#123') —
-- load-bearing for open-loop -> resolution matching and the honest answer;
-- NOT NULL DEFAULT '' so the UNIQUE constraint dedupes re-runs (SQLite treats
-- NULLs as distinct in UNIQUE). DERIVED SENSITIVE: never logged.
CREATE TABLE IF NOT EXISTS entity_evidence (
    id          TEXT PRIMARY KEY,
    entity_id   TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    ts          REAL NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('pr_merged','ticket_closed',
                'shipped_language','open_loop','open_loop_resolved')),
    source_kind TEXT NOT NULL CHECK (source_kind IN ('observation','span','summary')),
    source_id   TEXT NOT NULL,
    detail      TEXT NOT NULL DEFAULT '',
    UNIQUE(entity_id, kind, source_kind, source_id, detail)
);

CREATE INDEX IF NOT EXISTS idx_entity_evidence_entity ON entity_evidence(entity_id, ts);
CREATE INDEX IF NOT EXISTS idx_entity_evidence_source ON entity_evidence(source_kind, source_id);

-- Insert-time integrity per source_kind (mirrors the day-memory/block-summary
-- trigger pairs): evidence must never cite a missing source.
CREATE TRIGGER IF NOT EXISTS trg_entity_evidence_obs_exists
BEFORE INSERT ON entity_evidence
WHEN NEW.source_kind = 'observation'
    AND NOT EXISTS (SELECT 1 FROM observations WHERE id = NEW.source_id)
BEGIN
    SELECT RAISE(ABORT, 'entity_evidence: unknown observation ref');
END;

CREATE TRIGGER IF NOT EXISTS trg_entity_evidence_span_exists
BEFORE INSERT ON entity_evidence
WHEN NEW.source_kind = 'span'
    AND NOT EXISTS (SELECT 1 FROM activity_spans WHERE span_id = NEW.source_id)
BEGIN
    SELECT RAISE(ABORT, 'entity_evidence: unknown span ref');
END;

CREATE TRIGGER IF NOT EXISTS trg_entity_evidence_summary_exists
BEFORE INSERT ON entity_evidence
WHEN NEW.source_kind = 'summary'
    AND NOT EXISTS (SELECT 1 FROM block_summaries WHERE id = NEW.source_id)
BEGIN
    SELECT RAISE(ABORT, 'entity_evidence: unknown summary ref');
END;

-- Evidence dies with its source (BEFORE DELETE, like every other derived
-- artifact); the ENTITY row survives — it goes dormant synchronously in the
-- delete()/prune() transaction when its evidence set empties (never inside a
-- trigger), and by inactivity at aggregation time.
CREATE TRIGGER IF NOT EXISTS trg_entity_evidence_observation_delete
BEFORE DELETE ON observations
BEGIN
    DELETE FROM entity_evidence
    WHERE source_kind = 'observation' AND source_id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_entity_evidence_span_delete
BEFORE DELETE ON activity_spans
BEGIN
    DELETE FROM entity_evidence
    WHERE source_kind = 'span' AND source_id = OLD.span_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_entity_evidence_summary_delete
BEFORE DELETE ON block_summaries
BEGIN
    DELETE FROM entity_evidence
    WHERE source_kind = 'summary' AND source_id = OLD.id;
END;

-- Typed last-seen validity: an entity row must never point at a missing
-- source (insert AND update), and the per-kind BEFORE DELETE triggers NULL
-- the pair when the referenced source dies (the completion answer then
-- degrades to the honest date-only line).
CREATE TRIGGER IF NOT EXISTS trg_entities_last_seen_exists_insert
BEFORE INSERT ON entities
WHEN NEW.last_seen_source_kind IS NOT NULL AND (
    (NEW.last_seen_source_kind = 'observation'
        AND NOT EXISTS (SELECT 1 FROM observations WHERE id = NEW.last_seen_source_id))
    OR (NEW.last_seen_source_kind = 'span'
        AND NOT EXISTS (SELECT 1 FROM activity_spans WHERE span_id = NEW.last_seen_source_id))
    OR (NEW.last_seen_source_kind = 'summary'
        AND NOT EXISTS (SELECT 1 FROM block_summaries WHERE id = NEW.last_seen_source_id))
)
BEGIN
    SELECT RAISE(ABORT, 'entities: unknown last_seen source ref');
END;

CREATE TRIGGER IF NOT EXISTS trg_entities_last_seen_exists_update
BEFORE UPDATE OF last_seen_source_kind, last_seen_source_id ON entities
WHEN NEW.last_seen_source_kind IS NOT NULL AND (
    (NEW.last_seen_source_kind = 'observation'
        AND NOT EXISTS (SELECT 1 FROM observations WHERE id = NEW.last_seen_source_id))
    OR (NEW.last_seen_source_kind = 'span'
        AND NOT EXISTS (SELECT 1 FROM activity_spans WHERE span_id = NEW.last_seen_source_id))
    OR (NEW.last_seen_source_kind = 'summary'
        AND NOT EXISTS (SELECT 1 FROM block_summaries WHERE id = NEW.last_seen_source_id))
)
BEGIN
    SELECT RAISE(ABORT, 'entities: unknown last_seen source ref');
END;

CREATE TRIGGER IF NOT EXISTS trg_entities_last_seen_observation_delete
BEFORE DELETE ON observations
BEGIN
    UPDATE entities SET last_seen_source_kind = NULL, last_seen_source_id = NULL
    WHERE last_seen_source_kind = 'observation' AND last_seen_source_id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_entities_last_seen_span_delete
BEFORE DELETE ON activity_spans
BEGIN
    UPDATE entities SET last_seen_source_kind = NULL, last_seen_source_id = NULL
    WHERE last_seen_source_kind = 'span' AND last_seen_source_id = OLD.span_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_entities_last_seen_summary_delete
BEFORE DELETE ON block_summaries
BEGIN
    UPDATE entities SET last_seen_source_kind = NULL, last_seen_source_id = NULL
    WHERE last_seen_source_kind = 'summary' AND last_seen_source_id = OLD.id;
END;

-- Redacted audit metadata for remote reasoning packet send attempts. This table
-- intentionally stores counts and a packet-content hash only — never raw
-- question text, answer text, packet JSON, snippets, window titles, URLs,
-- observation IDs, citation IDs, or configured exclusion names.
CREATE TABLE IF NOT EXISTS reasoning_send_ledger (
    id                    TEXT PRIMARY KEY,
    created_at            REAL NOT NULL,
    feature               TEXT NOT NULL,
    packet_route          TEXT,
    reasoning_route       TEXT,
    egress                TEXT NOT NULL,
    route_class           TEXT NOT NULL,
    provider_family       TEXT NOT NULL,
    model                 TEXT,
    packet_hash           TEXT,
    packet_bytes          INTEGER,
    selected_source_count INTEGER NOT NULL DEFAULT 0,
    citation_count        INTEGER NOT NULL DEFAULT 0,
    excluded_observations INTEGER NOT NULL DEFAULT 0,
    excluded_by_json      TEXT NOT NULL DEFAULT '{}',
    outcome               TEXT NOT NULL,
    error_kind            TEXT,
    deletion_caveat       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reasoning_send_ledger_created_at
    ON reasoning_send_ledger(created_at);
CREATE INDEX IF NOT EXISTS idx_reasoning_send_ledger_feature
    ON reasoning_send_ledger(feature, created_at);
