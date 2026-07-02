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
CREATE TABLE IF NOT EXISTS day_memory_source_refs (
    day_memory_id TEXT NOT NULL REFERENCES day_memories(id) ON DELETE CASCADE,
    source_kind   TEXT NOT NULL CHECK (source_kind IN ('observation','span')),
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
