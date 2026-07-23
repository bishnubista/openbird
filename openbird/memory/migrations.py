"""Schema versioning and a forward-only migration ladder.

Before this module, ``schema.sql`` used only ``CREATE TABLE IF NOT EXISTS`` and
there was no ``PRAGMA user_version``: a future schema change had no way to
upgrade an existing on-disk DB and would surface as a raw ``no such column``
error at query time. This introduces an explicit version contract:

  * :data:`SCHEMA_VERSION` is the schema version this build understands.
  * :data:`MIGRATIONS` is an ordered ladder; each :class:`Migration` upgrades a
    DB from ``version - 1`` to ``version``.
  * :func:`ensure_schema_version` is the single gate run at startup. It stamps
    fresh/legacy DBs, runs any pending upgrade migrations (each in its own
    transaction so a crash cannot leave a half-applied schema), and REFUSES to
    open a DB stamped NEWER than this build (rather than silently corrupting it).

Version 1 is the original baseline schema (``content_blobs`` / ``observations`` /
``chunks`` / ``blob_chunks`` / ``fts_chunks`` / ``embedding_meta`` tables;
``vec_chunks`` is created in Python because its dimension is dynamic). Later
versions append forward migrations for additive durable features.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

# The schema version this build of OpenBird understands. Bump this and append a
# Migration to MIGRATIONS whenever schema.sql changes shape.
SCHEMA_VERSION = 10


@dataclass(frozen=True)
class Migration:
    """One forward step in the schema ladder.

    ``apply`` receives the live connection and performs the DDL/DML needed to
    move the DB from ``version - 1`` to ``version``. It must NOT commit or change
    ``user_version`` itself — :func:`ensure_schema_version` owns the transaction
    and the version stamp so each step is atomic.
    """

    version: int
    description: str
    apply: Callable[[sqlite3.Connection], None]


def _apply_v2_day_memories(conn: sqlite3.Connection) -> None:
    """Add purge-safe deterministic day-memory tables."""
    statements = [
        """
        CREATE TABLE IF NOT EXISTS day_memories (
            id                TEXT PRIMARY KEY,
            local_date        TEXT NOT NULL,
            source_scope      TEXT NOT NULL DEFAULT 'capture',
            extractor_version TEXT NOT NULL,
            generated_at      REAL NOT NULL,
            payload_json      TEXT NOT NULL,
            source_count      INTEGER NOT NULL,
            UNIQUE(local_date, source_scope)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS day_memory_sources (
            day_memory_id TEXT NOT NULL REFERENCES day_memories(id) ON DELETE CASCADE,
            observation_id TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
            PRIMARY KEY(day_memory_id, observation_id)
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_day_memories_date_scope
            ON day_memories(local_date, source_scope)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_day_memory_sources_observation
            ON day_memory_sources(observation_id)
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_day_memory_observation_delete
        BEFORE DELETE ON observations
        BEGIN
            DELETE FROM day_memories
            WHERE id IN (
                SELECT day_memory_id
                FROM day_memory_sources
                WHERE observation_id = OLD.id
            );
        END
        """,
    ]
    for statement in statements:
        conn.execute(statement)


def _apply_v3_reasoning_send_ledger(conn: sqlite3.Connection) -> None:
    """Add redacted remote-reasoning send-attempt ledger metadata."""
    statements = [
        """
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
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_reasoning_send_ledger_created_at
            ON reasoning_send_ledger(created_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_reasoning_send_ledger_feature
            ON reasoning_send_ledger(feature, created_at)
        """,
    ]
    for statement in statements:
        conn.execute(statement)


def _apply_v4_activity_spans(conn: sqlite3.Connection) -> None:
    """Add activity spans + typed day-memory source refs (Phase B).

    IDEMPOTENCY IS LOAD-BEARING: ``schema.sql`` executes BEFORE the ladder and
    a fresh DB is stamped v1 (its tables exist by then) and walks v2->v3->v4 —
    so most objects here were usually ALREADY created by schema.sql (their
    definitions must stay textually in lockstep), and the v2 migration will
    have just re-created the OLD day_memory_sources table this step drops.
    Every statement is therefore IF-NOT-EXISTS / guarded / IF-EXISTS.
    """
    statements = [
        """
        CREATE TABLE IF NOT EXISTS activity_spans (
            span_id      TEXT PRIMARY KEY,
            epoch_id     TEXT NOT NULL,
            start_ts     REAL NOT NULL,
            end_ts       REAL NOT NULL,
            bundle_id    TEXT,
            app          TEXT,
            detail_tier  INTEGER NOT NULL CHECK (detail_tier IN (0, 1)),
            window       TEXT,
            url_host     TEXT,
            identity_key TEXT,
            afk          INTEGER NOT NULL DEFAULT 0,
            meeting      INTEGER NOT NULL DEFAULT 0,
            reason       TEXT CHECK (reason IN ('not_allowlisted','blocklisted',
                                                'dangerous','private','paused',
                                                'self_capture')),
            CHECK (end_ts >= start_ts),
            CHECK ((detail_tier = 0 AND window IS NULL AND url_host IS NULL
                    AND identity_key IS NULL AND reason IS NOT NULL)
                OR (detail_tier = 1 AND reason IS NULL))
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_activity_spans_start ON activity_spans(start_ts)",
        "CREATE INDEX IF NOT EXISTS idx_activity_spans_end ON activity_spans(end_ts)",
        """
        CREATE INDEX IF NOT EXISTS idx_activity_spans_bundle
            ON activity_spans(bundle_id, start_ts)
        """,
        """
        CREATE TABLE IF NOT EXISTS day_memory_source_refs (
            day_memory_id TEXT NOT NULL REFERENCES day_memories(id) ON DELETE CASCADE,
            source_kind   TEXT NOT NULL CHECK (source_kind IN ('observation','span')),
            source_id     TEXT NOT NULL,
            PRIMARY KEY (day_memory_id, source_kind, source_id)
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_day_memory_source_refs_source
            ON day_memory_source_refs(source_kind, source_id)
        """,
    ]
    for statement in statements:
        conn.execute(statement)

    # observations.span_id: ALTER has no IF NOT EXISTS, so guard via table_info.
    # (On a fresh DB schema.sql already created the column; on an upgrading DB
    # this ALTER adds it. ADD COLUMN with a REFERENCES clause is legal because
    # the column's default is NULL.)
    if "span_id" not in _table_columns(conn, "observations"):
        conn.execute(
            "ALTER TABLE observations ADD COLUMN span_id TEXT "
            "REFERENCES activity_spans(span_id) ON DELETE SET NULL"
        )
    # This index lives ONLY here (never in schema.sql): schema.sql runs before
    # the ladder, where a pre-v4 observations table has no span_id column yet.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_observations_span ON observations(span_id)"
    )

    # Backfill typed refs from the legacy observation-only junction, then drop
    # it (guarded: a DB whose v2 ran under an OLD build has it; a fresh DB's v2
    # just re-created it empty; either way it is gone after this step).
    legacy = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'day_memory_sources'"
    ).fetchone()
    if legacy is not None:
        conn.execute(
            """
            INSERT OR IGNORE INTO day_memory_source_refs
                (day_memory_id, source_kind, source_id)
            SELECT day_memory_id, 'observation', observation_id
            FROM day_memory_sources
            """
        )
    conn.execute("DROP TRIGGER IF EXISTS trg_day_memory_observation_delete")
    conn.execute("DROP TABLE IF EXISTS day_memory_sources")

    # Integrity + invalidation triggers (textually in lockstep with schema.sql).
    triggers = [
        """
        CREATE TRIGGER IF NOT EXISTS trg_day_memory_source_refs_obs_exists
        BEFORE INSERT ON day_memory_source_refs
        WHEN NEW.source_kind = 'observation'
            AND NOT EXISTS (SELECT 1 FROM observations WHERE id = NEW.source_id)
        BEGIN
            SELECT RAISE(ABORT, 'day_memory_source_refs: unknown observation ref');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_day_memory_source_refs_span_exists
        BEFORE INSERT ON day_memory_source_refs
        WHEN NEW.source_kind = 'span'
            AND NOT EXISTS (SELECT 1 FROM activity_spans WHERE span_id = NEW.source_id)
        BEGIN
            SELECT RAISE(ABORT, 'day_memory_source_refs: unknown span ref');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_day_memory_source_observation_delete
        BEFORE DELETE ON observations
        BEGIN
            DELETE FROM day_memories
            WHERE id IN (
                SELECT day_memory_id
                FROM day_memory_source_refs
                WHERE source_kind = 'observation' AND source_id = OLD.id
            );
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_day_memory_source_span_delete
        BEFORE DELETE ON activity_spans
        BEGIN
            DELETE FROM day_memories
            WHERE id IN (
                SELECT day_memory_id
                FROM day_memory_source_refs
                WHERE source_kind = 'span' AND source_id = OLD.span_id
            );
        END
        """,
    ]
    for statement in triggers:
        conn.execute(statement)


# The six day-memory-source-refs triggers the v5 rebuild replaces. Two of them
# live on observations/activity_spans (NOT on the rebuilt table), so a DROP
# TABLE would leave them behind with stale semantics; the rebuild therefore
# drops ALL SIX explicitly and recreates every one (never bare IF NOT EXISTS
# during the rebuild). Keyed by trigger name -> the exact CREATE statement,
# textually in lockstep with schema.sql (tests assert the sqlite_master SQL).
_V5_DAY_MEMORY_TRIGGERS: dict[str, str] = {
    "trg_day_memory_source_refs_obs_exists": """
        CREATE TRIGGER trg_day_memory_source_refs_obs_exists
        BEFORE INSERT ON day_memory_source_refs
        WHEN NEW.source_kind = 'observation'
            AND NOT EXISTS (SELECT 1 FROM observations WHERE id = NEW.source_id)
        BEGIN
            SELECT RAISE(ABORT, 'day_memory_source_refs: unknown observation ref');
        END
    """,
    "trg_day_memory_source_refs_span_exists": """
        CREATE TRIGGER trg_day_memory_source_refs_span_exists
        BEFORE INSERT ON day_memory_source_refs
        WHEN NEW.source_kind = 'span'
            AND NOT EXISTS (SELECT 1 FROM activity_spans WHERE span_id = NEW.source_id)
        BEGIN
            SELECT RAISE(ABORT, 'day_memory_source_refs: unknown span ref');
        END
    """,
    "trg_day_memory_source_refs_summary_exists": """
        CREATE TRIGGER trg_day_memory_source_refs_summary_exists
        BEFORE INSERT ON day_memory_source_refs
        WHEN NEW.source_kind = 'summary'
            AND NOT EXISTS (SELECT 1 FROM block_summaries WHERE id = NEW.source_id)
        BEGIN
            SELECT RAISE(ABORT, 'day_memory_source_refs: unknown summary ref');
        END
    """,
    "trg_day_memory_source_observation_delete": """
        CREATE TRIGGER trg_day_memory_source_observation_delete
        BEFORE DELETE ON observations
        BEGIN
            DELETE FROM day_memories
            WHERE id IN (
                SELECT day_memory_id
                FROM day_memory_source_refs
                WHERE source_kind = 'observation' AND source_id = OLD.id
            );
        END
    """,
    "trg_day_memory_source_span_delete": """
        CREATE TRIGGER trg_day_memory_source_span_delete
        BEFORE DELETE ON activity_spans
        BEGIN
            DELETE FROM day_memories
            WHERE id IN (
                SELECT day_memory_id
                FROM day_memory_source_refs
                WHERE source_kind = 'span' AND source_id = OLD.span_id
            );
        END
    """,
    "trg_day_memory_source_summary_delete": """
        CREATE TRIGGER trg_day_memory_source_summary_delete
        BEFORE DELETE ON block_summaries
        BEGIN
            DELETE FROM day_memories
            WHERE id IN (
                SELECT day_memory_id
                FROM day_memory_source_refs
                WHERE source_kind = 'summary' AND source_id = OLD.id
            );
        END
    """,
}


def _apply_v5_block_summaries(conn: sqlite3.Connection) -> None:
    """Add block summaries, the taxonomy cache, and the 'summary' source kind (Phase D).

    Same idempotency contract as v4: ``schema.sql`` (with the FINAL shape) has
    already run on this connection, so the new tables/indexes/triggers usually
    exist — every statement is IF-NOT-EXISTS / guarded.

    The one non-additive change is ``day_memory_source_refs``'s CHECK gaining
    'summary'. SQLite cannot ALTER a CHECK, so an upgrading DB (whose table
    predates v5 and was silently kept by schema.sql's CREATE IF NOT EXISTS) is
    REBUILT: detect the old shape via the table's sqlite_master SQL lacking
    'summary', copy rows into a new-shape table, drop, and rename. Trigger
    replacement around the rebuild is EXPLICIT: all six affected triggers are
    dropped by name FIRST (two live on observations/activity_spans and would
    survive the table drop with pre-rebuild text) and every one is recreated
    after — never bare IF NOT EXISTS during the rebuild.
    """
    statements = [
        """
        CREATE TABLE IF NOT EXISTS block_summaries (
            id                TEXT PRIMARY KEY,
            local_date        TEXT NOT NULL,
            block_key         TEXT NOT NULL UNIQUE,
            block_fingerprint TEXT NOT NULL,
            start_ts          REAL NOT NULL,
            end_ts            REAL NOT NULL,
            dominant_bundle   TEXT,
            level             TEXT CHECK (level IS NULL OR level IN
                              ('focus_work','other_work','neutral','personal',
                               'distracting')),
            summary_text      TEXT NOT NULL,
            model             TEXT NOT NULL,
            extractor_version TEXT NOT NULL,
            generated_at      REAL NOT NULL,
            source_count      INTEGER NOT NULL,
            CHECK (end_ts >= start_ts)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_block_summaries_date ON block_summaries(local_date)",
        "CREATE INDEX IF NOT EXISTS idx_block_summaries_start ON block_summaries(start_ts)",
        """
        CREATE TABLE IF NOT EXISTS block_summary_source_refs (
            summary_id  TEXT NOT NULL REFERENCES block_summaries(id) ON DELETE CASCADE,
            source_kind TEXT NOT NULL CHECK (source_kind IN ('observation','span')),
            source_id   TEXT NOT NULL,
            PRIMARY KEY (summary_id, source_kind, source_id)
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_block_summary_source_refs_source
            ON block_summary_source_refs(source_kind, source_id)
        """,
        """
        CREATE TABLE IF NOT EXISTS category_assignments (
            identity_key TEXT PRIMARY KEY,
            level        TEXT NOT NULL CHECK (level IN
                         ('focus_work','other_work','neutral','personal','distracting')),
            model        TEXT NOT NULL,
            generated_at REAL NOT NULL
        )
        """,
        # Block-summary integrity/cascade triggers (textually in lockstep with
        # schema.sql; not part of the six-trigger rebuild set below because none
        # of them reference day_memory_source_refs).
        """
        CREATE TRIGGER IF NOT EXISTS trg_block_summary_source_refs_obs_exists
        BEFORE INSERT ON block_summary_source_refs
        WHEN NEW.source_kind = 'observation'
            AND NOT EXISTS (SELECT 1 FROM observations WHERE id = NEW.source_id)
        BEGIN
            SELECT RAISE(ABORT, 'block_summary_source_refs: unknown observation ref');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_block_summary_source_refs_span_exists
        BEFORE INSERT ON block_summary_source_refs
        WHEN NEW.source_kind = 'span'
            AND NOT EXISTS (SELECT 1 FROM activity_spans WHERE span_id = NEW.source_id)
        BEGIN
            SELECT RAISE(ABORT, 'block_summary_source_refs: unknown span ref');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_block_summary_source_observation_delete
        BEFORE DELETE ON observations
        BEGIN
            DELETE FROM block_summaries
            WHERE id IN (
                SELECT summary_id
                FROM block_summary_source_refs
                WHERE source_kind = 'observation' AND source_id = OLD.id
            );
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_block_summary_source_span_delete
        BEFORE DELETE ON activity_spans
        BEGIN
            DELETE FROM block_summaries
            WHERE id IN (
                SELECT summary_id
                FROM block_summary_source_refs
                WHERE source_kind = 'span' AND source_id = OLD.span_id
            );
        END
        """,
    ]
    for statement in statements:
        conn.execute(statement)

    # Drop the six affected triggers BEFORE the rebuild so the DROP/RENAME never
    # reparses a trigger body against a transiently missing table, then recreate
    # all six after — replacement is explicit either way (rebuild or not).
    for name in _V5_DAY_MEMORY_TRIGGERS:
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")

    # Rebuild day_memory_source_refs only when its stored SQL still carries the
    # pre-v5 CHECK (fresh DBs got the final shape from schema.sql and no-op).
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' "
        "AND name = 'day_memory_source_refs'"
    ).fetchone()
    table_sql = str(_scalar(row) or "")
    if table_sql and "'summary'" not in table_sql:
        conn.execute(
            """
            CREATE TABLE day_memory_source_refs_v5 (
                day_memory_id TEXT NOT NULL REFERENCES day_memories(id) ON DELETE CASCADE,
                source_kind   TEXT NOT NULL CHECK (source_kind IN
                              ('observation','span','summary')),
                source_id     TEXT NOT NULL,
                PRIMARY KEY (day_memory_id, source_kind, source_id)
            )
            """
        )
        conn.execute(
            "INSERT INTO day_memory_source_refs_v5 (day_memory_id, source_kind, source_id) "
            "SELECT day_memory_id, source_kind, source_id FROM day_memory_source_refs"
        )
        conn.execute("DROP TABLE day_memory_source_refs")
        conn.execute(
            "ALTER TABLE day_memory_source_refs_v5 RENAME TO day_memory_source_refs"
        )
        # The index was dropped with the old table; recreate it.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_day_memory_source_refs_source "
            "ON day_memory_source_refs(source_kind, source_id)"
        )

    for statement in _V5_DAY_MEMORY_TRIGGERS.values():
        conn.execute(statement)


def _apply_v6_summary_index(conn: sqlite3.Connection) -> None:
    """Add the parallel summary index (Phase E1): entries + FTS + cleanup triggers.

    Same idempotency contract as v4/v5: ``schema.sql`` (with the FINAL shape)
    has already run on this connection under the schema.sql-first startup order,
    so every object here usually exists — every statement is IF-NOT-EXISTS and
    stays textually in lockstep with schema.sql. ``vec_summaries`` is
    DELIBERATELY absent: it is a vec0 virtual table whose dimension comes from
    ``Settings.embed_dim``, so it is created in ``MemoryStore._apply_schema``
    (exactly like ``vec_chunks``), never in SQL files or this ladder.

    Week rows need no DDL — they reuse ``day_memories`` with
    ``source_scope='week'`` and the existing v5 'summary' source-ref kind.
    """
    statements = [
        """
        CREATE TABLE IF NOT EXISTS summary_index_entries (
            entry_rowid  INTEGER PRIMARY KEY,
            summary_kind TEXT NOT NULL CHECK (summary_kind IN ('block','week')),
            summary_id   TEXT NOT NULL,
            seq          INTEGER NOT NULL DEFAULT 0,
            text         TEXT NOT NULL,
            fingerprint  TEXT NOT NULL,
            UNIQUE(summary_kind, summary_id, seq)
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_summary_index_entries_summary
            ON summary_index_entries(summary_kind, summary_id)
        """,
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_summaries USING fts5(
            text
        )
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_summary_index_block_delete
        BEFORE DELETE ON block_summaries
        BEGIN
            DELETE FROM summary_index_entries
            WHERE summary_kind = 'block' AND summary_id = OLD.id;
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_summary_index_week_delete
        BEFORE DELETE ON day_memories
        WHEN OLD.source_scope = 'week'
        BEGIN
            DELETE FROM summary_index_entries
            WHERE summary_kind = 'week' AND summary_id = OLD.id;
        END
        """,
    ]
    for statement in statements:
        conn.execute(statement)


def _apply_v7_entity_ledger(conn: sqlite3.Connection) -> None:
    """Add the entity ledger + completion evidence (Phase E2).

    Same idempotency contract as v4/v5/v6: ``schema.sql`` (with the FINAL
    shape) has already run on this connection under the schema.sql-first
    startup order, so every object here usually exists — every statement is
    IF-NOT-EXISTS and stays textually in lockstep with schema.sql. Purely
    additive: no table rebuilds, no drops.

    Deliberately NO fts/vec indexing for entities or evidence: the ledger is
    queried by exact casefolded name/alias match and answered deterministically
    from rows — indexing would extend the API-enforced sweep contract for zero
    recall benefit (see the schema.sql comment). Plain-table triggers therefore
    fully cover evidence deletion.
    """
    statements = [
        """
        CREATE TABLE IF NOT EXISTS entities (
            id               TEXT PRIMARY KEY,
            kind             TEXT NOT NULL CHECK (kind IN ('repo','domain','document','topic')),
            name             TEXT NOT NULL,
            aliases          TEXT NOT NULL DEFAULT '[]',
            first_ts         REAL,
            last_ts          REAL,
            status           TEXT NOT NULL DEFAULT 'active'
                             CHECK (status IN ('active','dormant','user_marked_done')),
            last_seen_source_kind TEXT CHECK (last_seen_source_kind IN
                             ('observation','span','summary')),
            last_seen_source_id  TEXT,
            CHECK ((last_seen_source_kind IS NULL) = (last_seen_source_id IS NULL)),
            UNIQUE(kind, name)
        )
        """,
        """
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
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_entity_evidence_entity
            ON entity_evidence(entity_id, ts)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_entity_evidence_source
            ON entity_evidence(source_kind, source_id)
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_entity_evidence_obs_exists
        BEFORE INSERT ON entity_evidence
        WHEN NEW.source_kind = 'observation'
            AND NOT EXISTS (SELECT 1 FROM observations WHERE id = NEW.source_id)
        BEGIN
            SELECT RAISE(ABORT, 'entity_evidence: unknown observation ref');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_entity_evidence_span_exists
        BEFORE INSERT ON entity_evidence
        WHEN NEW.source_kind = 'span'
            AND NOT EXISTS (SELECT 1 FROM activity_spans WHERE span_id = NEW.source_id)
        BEGIN
            SELECT RAISE(ABORT, 'entity_evidence: unknown span ref');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_entity_evidence_summary_exists
        BEFORE INSERT ON entity_evidence
        WHEN NEW.source_kind = 'summary'
            AND NOT EXISTS (SELECT 1 FROM block_summaries WHERE id = NEW.source_id)
        BEGIN
            SELECT RAISE(ABORT, 'entity_evidence: unknown summary ref');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_entity_evidence_observation_delete
        BEFORE DELETE ON observations
        BEGIN
            DELETE FROM entity_evidence
            WHERE source_kind = 'observation' AND source_id = OLD.id;
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_entity_evidence_span_delete
        BEFORE DELETE ON activity_spans
        BEGIN
            DELETE FROM entity_evidence
            WHERE source_kind = 'span' AND source_id = OLD.span_id;
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_entity_evidence_summary_delete
        BEFORE DELETE ON block_summaries
        BEGIN
            DELETE FROM entity_evidence
            WHERE source_kind = 'summary' AND source_id = OLD.id;
        END
        """,
        """
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
        END
        """,
        """
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
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_entities_last_seen_observation_delete
        BEFORE DELETE ON observations
        BEGIN
            UPDATE entities SET last_seen_source_kind = NULL, last_seen_source_id = NULL
            WHERE last_seen_source_kind = 'observation' AND last_seen_source_id = OLD.id;
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_entities_last_seen_span_delete
        BEFORE DELETE ON activity_spans
        BEGIN
            UPDATE entities SET last_seen_source_kind = NULL, last_seen_source_id = NULL
            WHERE last_seen_source_kind = 'span' AND last_seen_source_id = OLD.span_id;
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_entities_last_seen_summary_delete
        BEFORE DELETE ON block_summaries
        BEGIN
            UPDATE entities SET last_seen_source_kind = NULL, last_seen_source_id = NULL
            WHERE last_seen_source_kind = 'summary' AND last_seen_source_id = OLD.id;
        END
        """,
    ]
    for statement in statements:
        conn.execute(statement)


def _apply_v8_observation_keyset_index(conn: sqlite3.Connection) -> None:
    """Add the assistant keyset-pagination index (source, ts, id).

    Same idempotency contract as v4-v7: ``schema.sql`` (with the FINAL shape)
    has already run on this connection, so the index usually exists — the
    statement is IF-NOT-EXISTS and stays textually in lockstep with schema.sql.
    Purely additive (index-only): no table rebuilds, no data rewrite. All three
    columns are part of the frozen v1 shape, so this applies cleanly to every
    released DB.
    """
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_observations_source_ts_id "
        "ON observations(source, ts, id)"
    )


def _apply_v9_capture_attempts(conn: sqlite3.Connection) -> None:
    """Add the content-free capture-attempt accountability ledger."""
    statements = [
        """
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
            reason_codes_json         TEXT NOT NULL DEFAULT '[]'
                CHECK (json_valid(reason_codes_json)),
            coalesced_trigger_count   INTEGER NOT NULL DEFAULT 0
                CHECK (coalesced_trigger_count >= 0),
            earliest_coalesced_ts     REAL,
            successor_attempt_id      TEXT REFERENCES capture_attempts(attempt_id)
                ON DELETE SET NULL,
            observation_id            TEXT REFERENCES observations(id) ON DELETE SET NULL,
            UNIQUE(helper_epoch, trigger_seq),
            CHECK ((status = 'started' AND outcome IS NULL AND finished_ts IS NULL)
                OR (status = 'finished' AND outcome IS NOT NULL
                    AND finished_ts IS NOT NULL)),
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
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_capture_attempts_trigger_ts
            ON capture_attempts(trigger_ts)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_capture_attempts_outcome
            ON capture_attempts(outcome, trigger_ts)
        """,
    ]
    for statement in statements:
        conn.execute(statement)


def _apply_v10_pending_meetings(conn: sqlite3.Connection) -> None:
    """Add encrypted meeting checkpoints and meeting-session idempotency."""
    statements = [
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_observations_meeting_session
            ON observations(session_id)
            WHERE source = 'meeting' AND session_id IS NOT NULL
        """,
        """
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
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_pending_meetings_started_ts
            ON pending_meetings(started_ts)
        """,
    ]
    for statement in statements:
        conn.execute(statement)


# Forward-only ladder. Version 1 IS the baseline schema (applied by schema.sql),
# so migrations here only ever upgrade an existing DB from one version to the
# next. Append future steps (version 3, 4, ...) in order; never edit or reorder a
# released migration.
MIGRATIONS: list[Migration] = [
    Migration(
        version=2,
        description="add purge-safe deterministic day memories",
        apply=_apply_v2_day_memories,
    ),
    Migration(
        version=3,
        description="add redacted reasoning send ledger",
        apply=_apply_v3_reasoning_send_ledger,
    ),
    Migration(
        version=4,
        description="add activity spans + typed day-memory source refs",
        apply=_apply_v4_activity_spans,
    ),
    Migration(
        version=5,
        description="add block summaries, taxonomy cache, summary source kind",
        apply=_apply_v5_block_summaries,
    ),
    Migration(
        version=6,
        description="add the parallel summary index (entries, FTS, cleanup triggers)",
        apply=_apply_v6_summary_index,
    ),
    Migration(
        version=7,
        description="add the entity ledger + completion evidence",
        apply=_apply_v7_entity_ledger,
    ),
    Migration(
        version=8,
        description="add the assistant keyset-pagination index (source, ts, id)",
        apply=_apply_v8_observation_keyset_index,
    ),
    Migration(
        version=9,
        description="add the content-free capture-attempt ledger",
        apply=_apply_v9_capture_attempts,
    ),
    Migration(
        version=10,
        description="add SQLCipher-gated pending meeting transcripts",
        apply=_apply_v10_pending_meetings,
    ),
]


def _scalar(row: object) -> object:
    """Return the single value from a sqlite row regardless of row_factory.

    The store sets a dict row_factory (``mapping_row_factory``), so ``row[0]``
    raises ``KeyError``; a default connection yields tuples. Handle both.
    """
    if row is None:
        return None
    if isinstance(row, dict):
        return next(iter(row.values()))
    return row[0]


def _user_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    value = _scalar(row)
    return int(value) if value is not None else 0


def _set_user_version(conn: sqlite3.Connection, version: int) -> None:
    # PRAGMA user_version does not accept a bound parameter; the value is an int
    # we fully control, so interpolation is safe.
    conn.execute(f"PRAGMA user_version = {int(version)}")


# The on-disk shape that schema version 1 REQUIRES, expressed as
# table -> set of required columns. Version 1 is a frozen, released schema (it is
# never edited — future shape changes append a migration and bump SCHEMA_VERSION),
# so this contract can be pinned here. It is deliberately a *required-subset*
# check, not an exact-equality check: additive, migration-free extras (e.g. a new
# CREATE INDEX IF NOT EXISTS, or a column a forward migration will add) must not
# make a legitimate v1 DB look invalid. We only assert that every table/column the
# v1 query layer depends on is actually present (this is the `ts` column the
# "no such column: ts" repro hit). Keep in sync with schema.sql if v1 ever needed
# correcting — but it should not, by the freeze rule above.
_V1_REQUIRED_SHAPE: dict[str, frozenset[str]] = {
    "content_blobs": frozenset({"content_hash", "text"}),
    "observations": frozenset(
        {"id", "content_hash", "ts", "app", "window", "url", "session_id", "source"}
    ),
    "chunks": frozenset({"chunk_hash", "text", "rowid_int"}),
    "blob_chunks": frozenset(
        {"content_hash", "chunk_hash", "span_start", "span_end"}
    ),
    "fts_chunks": frozenset({"text"}),
    "embedding_meta": frozenset({"key", "value"}),
}


def _db_has_existing_tables(conn: sqlite3.Connection) -> bool:
    """True if the DB already holds OpenBird's core tables (a legacy, pre-version DB)."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'observations' LIMIT 1"
    ).fetchone()
    return row is not None


def _column_name(row: object) -> object:
    """Pull the column name from a ``PRAGMA table_info`` row (dict or tuple).

    table_info columns are: cid, name, type, notnull, dflt_value, pk. We want
    ``name``, robust to either row_factory (see :func:`_scalar`).
    """
    if isinstance(row, dict):
        return row["name"]
    return row[1]


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Return the set of column names for ``table`` (empty if it does not exist).

    Uses ``PRAGMA table_info`` — the canonical SQLite introspection for columns.
    PRAGMA does not accept a bound parameter for the table name, so this is only
    ever called with the hardcoded names from :data:`_V1_REQUIRED_SHAPE` (no
    caller-/data-controlled value reaches the SQL), avoiding any injection risk.
    """
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(_column_name(row)) for row in rows}


def _v1_shape_mismatch(conn: sqlite3.Connection) -> str | None:
    """Return why a non-empty DB is NOT the v1 shape, or ``None`` if it matches.

    Validates that every table version 1 requires exists AND carries each of its
    required columns. Returns the first concrete mismatch (a missing table or a
    missing column) so the caller can raise an actionable error instead of
    stamping a partial / pre-v1 DB to version 1 and then failing cryptically later
    at query time (e.g. ``no such column: ts``).
    """
    for table, required_columns in _V1_REQUIRED_SHAPE.items():
        present = _table_columns(conn, table)
        if not present:
            return f"required table {table!r} is missing"
        missing = required_columns - present
        if missing:
            cols = ", ".join(sorted(missing))
            return f"table {table!r} is missing required column(s): {cols}"
    return None


def _raise_partial_shape(mismatch: str) -> None:
    """Raise the canonical "tables present but not the v1 shape" error."""
    raise RuntimeError(
        "Database has tables but does not match the OpenBird schema "
        f"version 1 shape ({mismatch}). It looks like a partial, "
        "pre-v1, or unsupported database. Refusing to use it to avoid "
        "failing later at query time. Start from a fresh database, or "
        "restore a known-good backup."
    )


def _guard_existing_shape(conn: sqlite3.Connection) -> None:
    """Raise the partial-shape error if existing tables miss any required column.

    The version-1 required columns are a permanent FLOOR: the forward-only ladder
    only ever ADDS tables/columns and never drops a released one, so this check is
    valid for a DB stamped at ANY version (>= 1) — the floor must always hold. It is
    run both when adopting a legacy unstamped DB AND when re-opening an
    already-stamped DB, so a wrong-shaped-but-stamped DB fails with the clear,
    actionable migration error instead of a cryptic ``OperationalError`` at first
    query/write (which the capture daemon swallows, making capture silently store
    nothing). A DB with no core tables yet (fresh) is a no-op.
    """
    if not _db_has_existing_tables(conn):
        return
    mismatch = _v1_shape_mismatch(conn)
    if mismatch is not None:
        _raise_partial_shape(mismatch)


def preflight_legacy_shape_guard(conn: sqlite3.Connection) -> None:
    """Reject a partial/foreign legacy DB BEFORE the baseline schema is applied.

    This MUST run before ``schema.sql`` is (re)applied. ``schema.sql`` uses
    ``CREATE TABLE IF NOT EXISTS`` (which silently skips an already-present but
    wrong-shaped table) and then ``CREATE INDEX ... ON observations(ts)`` — so an
    existing ``observations`` table missing the ``ts`` column makes the *index*
    creation raise a raw ``OperationalError: no such column: ts`` before
    :func:`ensure_schema_version` ever runs. Catching the mismatch here turns that
    cryptic, mid-``executescript`` failure into the clear, actionable migration
    error.

    Only a DB that is unstamped (``user_version == 0``) AND already has the core
    tables is inspected: a fresh/empty DB (no tables yet) and an already-versioned
    DB are both left for :func:`ensure_schema_version` to handle.
    """
    if _user_version(conn) != 0:
        return
    if not _db_has_existing_tables(conn):
        return
    mismatch = _v1_shape_mismatch(conn)
    if mismatch is not None:
        _raise_partial_shape(mismatch)


def ensure_schema_version(conn: sqlite3.Connection) -> int:
    """Reconcile ``PRAGMA user_version`` with :data:`SCHEMA_VERSION`.

    Must be called AFTER the baseline schema (``schema.sql``) has been applied so
    a fresh DB already has the version-1 tables present.

    Behavior:
      * version 0, tables present  -> legacy pre-versioning DB. Only adopted as
        version 1 if its on-disk shape actually MATCHES version 1 (required tables
        AND columns). If the tables exist but the shape does not match (a partial,
        pre-v1, or foreign DB) this RAISES rather than stamping it as 1 and failing
        cryptically later at query time.
      * version 0, no tables       -> fresh DB; stamp directly to SCHEMA_VERSION.
      * 0 < version < SCHEMA_VERSION -> run each pending migration (version >
        current) in ascending order, each in its own transaction + version stamp.
      * version == SCHEMA_VERSION  -> nothing to do.
      * version  > SCHEMA_VERSION  -> RAISE: the DB was written by a newer build;
        refuse to open it rather than silently corrupt it.

    Returns the final (current-build) schema version.
    """
    current = _user_version(conn)

    if current > SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema version {current} is newer than this build "
            f"supports ({SCHEMA_VERSION}). Refusing to open it to avoid "
            f"corruption — upgrade OpenBird to a build that understands "
            f"version {current}."
        )

    # Floor-shape guard for ANY already-stamped DB (current >= 1), evaluated BEFORE
    # the version branches below so it covers every stamped path — a DB at the
    # current version (trusted as-is) AND a stamped-but-older DB about to run the
    # migration ladder. A DB can carry a stamp yet miss a required column:
    # schema.sql's ``CREATE TABLE IF NOT EXISTS`` silently keeps an already-present,
    # wrong-shaped table, and the ladder has no step to rebuild it. Without this a
    # stranded DB is either trusted (``current == SCHEMA_VERSION``) or re-stamped by
    # the ladder while still wrong-shaped (``0 < current < SCHEMA_VERSION``); both
    # then fail with a cryptic ``OperationalError`` at first query/write, which the
    # capture daemon swallows (capture silently stores nothing). The version-1
    # required columns are a permanent floor — the forward-only ladder only adds,
    # never drops, a released column — so the check is valid at any stamped version.
    # (``current == 0`` is handled in its own branch below: it is not yet stamped and
    # may be a partial/foreign DB that must be refused before adoption.)
    if current > 0:
        _guard_existing_shape(conn)

    if current == SCHEMA_VERSION:
        return current

    if current == 0:
        # Distinguish a legacy DB (created before versioning, but already at the
        # v1 shape) from a brand-new empty DB.
        #
        #   * No tables  -> brand-new DB; stamp directly to SCHEMA_VERSION.
        #   * Tables that match the v1 shape -> genuine legacy v1 DB; stamp to 1
        #     and let the ladder upgrade it from there.
        #   * Tables that do NOT match the v1 shape -> a partial / pre-v1 /
        #     foreign DB. A bare presence check used to treat this as v1 and
        #     stamp+continue, which "succeeded" here but then failed cryptically
        #     at query time (e.g. ``no such column: ts``). Refuse loudly and
        #     early instead — consistent with the newer-than-supported guard
        #     above — rather than corrupting the version contract.
        # Defense in depth: normally preflight_legacy_shape_guard already caught a
        # wrong-shaped DB before schema.sql ran, but ensure_schema_version is also
        # called directly (e.g. in tests) so it must stay self-safe.
        _guard_existing_shape(conn)
        baseline = 1 if _db_has_existing_tables(conn) else SCHEMA_VERSION
        _stamp_in_txn(conn, baseline)
        current = baseline
        if current == SCHEMA_VERSION:
            return current

    for migration in sorted(MIGRATIONS, key=lambda m: m.version):
        if migration.version <= current:
            continue
        if migration.version > SCHEMA_VERSION:
            break
        _run_migration(conn, migration)
        current = migration.version

    if current != SCHEMA_VERSION:
        # The ladder is missing a step needed to reach the build's version.
        raise RuntimeError(
            f"No migration path from schema version {current} to "
            f"{SCHEMA_VERSION}; the migration ladder is incomplete."
        )
    return current


def _stamp_in_txn(conn: sqlite3.Connection, version: int) -> None:
    """Stamp ``user_version`` atomically (rollback on failure)."""
    try:
        conn.execute("BEGIN IMMEDIATE")
        _set_user_version(conn, version)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _run_migration(conn: sqlite3.Connection, migration: Migration) -> None:
    """Apply one migration and bump ``user_version`` in a single transaction.

    PRAGMA user_version is set inside the same transaction as the migration's
    DDL/DML so a crash mid-migration rolls BOTH back together — the DB is never
    left at a version that doesn't match its actual shape.
    """
    try:
        conn.execute("BEGIN IMMEDIATE")
        migration.apply(conn)
        _set_user_version(conn, migration.version)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


__all__ = [
    "Migration",
    "MIGRATIONS",
    "SCHEMA_VERSION",
    "ensure_schema_version",
    "preflight_legacy_shape_guard",
]
