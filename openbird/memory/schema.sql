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
-- observations referencing 1 blob.
CREATE TABLE IF NOT EXISTS observations (
    id           TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL REFERENCES content_blobs(content_hash) ON DELETE CASCADE,
    ts           REAL NOT NULL,
    app          TEXT,
    window       TEXT,
    url          TEXT,
    session_id   TEXT,
    source       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_observations_ts   ON observations(ts);
CREATE INDEX IF NOT EXISTS idx_observations_hash ON observations(content_hash);

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
