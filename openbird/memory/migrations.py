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
SCHEMA_VERSION = 4


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
