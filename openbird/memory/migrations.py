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

Version 1 is the current schema as defined by ``schema.sql`` (the
``content_blobs`` / ``observations`` / ``chunks`` / ``blob_chunks`` /
``fts_chunks`` / ``embedding_meta`` tables; ``vec_chunks`` is created in Python
because its dimension is dynamic).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

# The schema version this build of OpenBird understands. Bump this and append a
# Migration to MIGRATIONS whenever schema.sql changes shape.
SCHEMA_VERSION = 1


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


# Forward-only ladder. Version 1 IS the baseline schema (applied by schema.sql),
# so there is no migration that *creates* it — migrations here only ever upgrade
# an existing DB from one version to the next. Append future steps (version 2,
# 3, ...) in order; never edit or reorder a released migration.
MIGRATIONS: list[Migration] = []


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
