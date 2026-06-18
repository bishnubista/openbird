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


def _db_has_existing_tables(conn: sqlite3.Connection) -> bool:
    """True if the DB already holds OpenBird's core tables (a legacy, pre-version DB)."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'observations' LIMIT 1"
    ).fetchone()
    return row is not None


def ensure_schema_version(conn: sqlite3.Connection) -> int:
    """Reconcile ``PRAGMA user_version`` with :data:`SCHEMA_VERSION`.

    Must be called AFTER the baseline schema (``schema.sql``) has been applied so
    a fresh DB already has the version-1 tables present.

    Behavior:
      * version 0, tables present  -> legacy pre-versioning DB whose shape equals
        version 1; stamp to 1 without running migrations.
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

    if current == SCHEMA_VERSION:
        return current

    if current == 0:
        # Distinguish a legacy DB (created before versioning, but already at the
        # v1 shape) from a brand-new empty DB. Either way the on-disk shape is
        # version 1, so we stamp and then continue the ladder from there.
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
]
