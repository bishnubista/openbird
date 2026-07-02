"""Regression tests for DB concurrency, migrations, strict encryption,
delete atomicity, and retention/vacuum.

These complement test_memory.py / test_crypto.py and exercise the on-disk
behaviors that must stay reliable in production.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
import sys
import threading
import time

import pytest

from openbird.config import Settings
from openbird.memory import migrations
from openbird.memory.migrations import (
    Migration,
    SCHEMA_VERSION,
    ensure_schema_version,
)
from openbird.memory.store import MemoryStore
from openbird.storage import crypto
from openbird.storage.crypto import EncryptionUnavailableError, open_db_verified

from tests.unit.conftest import FakeProvider


# --------------------------------------------------------------------------- #
# embed runs OUTSIDE the write txn; busy_timeout queues, no lock-during-IO #
# --------------------------------------------------------------------------- #


class _SlowProvider(FakeProvider):
    """Embeds with a deliberate sleep to simulate a slow/wedged Ollama."""

    def __init__(self, embed_dim: int = 768, delay: float = 1.5) -> None:
        super().__init__(embed_dim=embed_dim)
        self.delay = delay
        self.embedding_started = threading.Event()

    def embed(self, texts):
        self.embedding_started.set()
        time.sleep(self.delay)
        return super().embed(texts)


def test_concurrent_reader_not_blocked_by_slow_embed(tmp_path):
    """A slow embed must NOT hold the write lock; a concurrent reader stays live.

    Before the fix, provider.embed() ran inside the write transaction, so a
    concurrent reader/writer hit `database is locked` immediately and the writer
    held the single WAL slot across the whole network round-trip. We prove:
      * the writer is mid-embed (lock not yet taken), and
      * a reader on a SEPARATE connection completes promptly with no
        OperationalError while the embed sleeps.
    """
    db = str(tmp_path / "concurrent.db")
    settings = Settings(data_dir=tmp_path, embed_dim=64)
    slow = _SlowProvider(embed_dim=64, delay=1.5)

    # Seed the DB (and create the file) with a fast provider first.
    seed = MemoryStore(db_path=db, settings=settings, provider=FakeProvider(embed_dim=64))
    seed.add_observation("seed content for the reader", source="capture", ts=1.0)
    seed.close()

    errors: list[Exception] = []
    reader_elapsed: list[float] = []

    def writer():
        # Build the writer store INSIDE its own thread: SQLite connections are
        # not shareable across threads (this mirrors the routines store's
        # one-connection-per-thread model).
        try:
            ws = MemoryStore(db_path=db, settings=settings, provider=slow)
            try:
                ws.add_observation(
                    "a brand new observation that requires embedding",
                    source="capture",
                    ts=2.0,
                )
            finally:
                ws.close()
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    t = threading.Thread(target=writer)
    t.start()
    try:
        # Wait until the writer is INSIDE provider.embed (i.e. the slow window).
        assert slow.embedding_started.wait(timeout=5.0)

        # Open an independent reader connection (mirrors routines-store usage:
        # a different thread/connection to the same on-disk DB).
        reader = MemoryStore(db_path=db, settings=settings, provider=FakeProvider(embed_dim=64))
        try:
            started = time.perf_counter()
            rows = reader.time_range(0.0, 10.0)  # plain read
            reader_elapsed.append(time.perf_counter() - started)
            assert rows, "reader should see the seeded observation"
        finally:
            reader.close()
    finally:
        t.join(timeout=10.0)

    # A timed-out join would silently leave the writer hung; fail loudly instead.
    assert not t.is_alive(), "writer thread did not finish within the join timeout"
    assert not errors, f"writer raised: {errors}"
    # The read happened DURING the 1.5s embed and returned fast => the writer was
    # not holding the lock across the network round-trip.
    assert reader_elapsed and reader_elapsed[0] < 1.0, reader_elapsed


def test_embed_failure_writes_no_partial_state(tmp_path):
    """If provider.embed() raises, add_observation leaves NO partial rows.

    Embedding now happens BEFORE the write transaction opens, so a provider
    failure must abort with zero observations/blobs/chunks written.
    """
    db = str(tmp_path / "embedfail.db")
    settings = Settings(data_dir=tmp_path, embed_dim=64)

    class BoomProvider(FakeProvider):
        def __init__(self):
            super().__init__(embed_dim=64)
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            # First call is the cohort/setup-free path; fail on the real ingest.
            raise RuntimeError("ollama exploded")

    s = MemoryStore(db_path=db, settings=settings, provider=BoomProvider())
    try:
        with pytest.raises(RuntimeError, match="ollama exploded"):
            s.add_observation("content that needs an embedding", source="t", ts=1.0)
        stats = s.stats()
        assert stats["observations"] == 0
        assert stats["blobs"] == 0
        assert stats["chunks"] == 0
        assert stats["vectors"] == 0
    finally:
        s.close()


def test_busy_timeout_set_on_plaintext_and_sqlcipher_paths(monkeypatch, tmp_path):
    """Every on-disk connection sets PRAGMA busy_timeout."""
    db = str(tmp_path / "bt.db")
    monkeypatch.setattr(crypto, "_get_or_create_key", lambda: None)  # force plaintext
    settings = Settings(data_dir=tmp_path, db_path=db)
    handle = open_db_verified(settings=settings)
    try:
        row = handle.conn.execute("PRAGMA busy_timeout").fetchone()
        assert int(row[0]) == crypto._BUSY_TIMEOUT_MS
        assert handle.wal_enabled is True  # plaintext now enables WAL
    finally:
        handle.conn.close()


# --------------------------------------------------------------------------- #
# schema versioning / migration ladder                                   #
# --------------------------------------------------------------------------- #


# The canonical v1 schema lives in schema.sql. Tests that need a "real v1 shape"
# DB apply it directly so they validate against the actual on-disk contract (and
# stay correct if v1 ever needs the unlikely correction the freeze rule allows).
_SCHEMA_SQL = (
    pathlib.Path(__file__).resolve().parents[2]
    / "openbird"
    / "memory"
    / "schema.sql"
).read_text(encoding="utf-8")


def _make_v1_shaped_db(path) -> sqlite3.Connection:
    """Create a DB carrying the real v1 shape but left at user_version=0.

    Mirrors a genuine legacy pre-versioning DB: schema.sql was applied (so every
    v1 table/column is present) but PRAGMA user_version was never stamped.
    """
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA_SQL)
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    return conn


def test_fresh_db_is_stamped_to_current_version(tmp_path):
    db = str(tmp_path / "fresh.db")
    s = MemoryStore(db_path=db, settings=Settings(data_dir=tmp_path, embed_dim=64),
                    provider=FakeProvider(embed_dim=64))
    try:
        ver = s.conn.execute("PRAGMA user_version").fetchone()
        # mapping_row_factory -> dict; value is the single column.
        assert int(next(iter(ver.values()))) == SCHEMA_VERSION
    finally:
        s.close()


def test_fresh_store_ships_session_index(tmp_path):
    """A fresh store carries idx_observations_session (added in schema.sql, no
    migration — _apply_schema runs schema.sql on every open, so it lands on fresh
    AND pre-existing DBs). SCHEMA_VERSION stays 1: an additive CREATE INDEX IF NOT
    EXISTS needs no migration ladder step."""
    db = str(tmp_path / "idx.db")
    s = MemoryStore(db_path=db, settings=Settings(data_dir=tmp_path, embed_dim=64),
                    provider=FakeProvider(embed_dim=64))
    try:
        row = s.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name='idx_observations_session'"
        ).fetchone()
        assert row is not None
        assert int(next(iter(
            s.conn.execute("PRAGMA user_version").fetchone().values()
        ))) == SCHEMA_VERSION  # unchanged: still 1
    finally:
        s.close()


def test_fresh_store_ships_day_memory_and_reasoning_ledger_tables(tmp_path):
    db = str(tmp_path / "daymem-schema.db")
    s = MemoryStore(db_path=db, settings=Settings(data_dir=tmp_path, embed_dim=64),
                    provider=FakeProvider(embed_dim=64))
    try:
        tables = {
            r["name"]
            for r in s.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {
            "day_memories",
            "day_memory_source_refs",
            "activity_spans",
            "reasoning_send_ledger",
        } <= tables
        # The legacy observation-only junction must NOT survive on a fresh DB
        # (v4 drops it after the v2 ladder step transiently re-creates it).
        assert "day_memory_sources" not in tables
        trigger = s.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' "
            "AND name='trg_day_memory_source_observation_delete'"
        ).fetchone()
        assert trigger is not None
        assert "BEFORE DELETE ON observations" in trigger["sql"]
    finally:
        s.close()


def test_legacy_unversioned_db_with_real_v1_shape_is_adopted_as_v1(tmp_path):
    """A DB with the REAL v1 shape but user_version=0 is stamped to 1, not migrated.

    This is the genuine legacy path: schema.sql was applied long ago, but
    PRAGMA user_version was never set. It must be adopted as version 1.
    """
    conn = _make_v1_shaped_db(tmp_path / "legacy.db")
    try:
        assert ensure_schema_version(conn) == SCHEMA_VERSION
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        # And the adopted DB is actually usable: the `ts` column the original bug
        # tripped over is present, so a real query against it succeeds.
        conn.execute(
            "INSERT INTO content_blobs (content_hash, text) VALUES ('h', 't')"
        )
        conn.execute(
            "INSERT INTO observations (id, content_hash, ts, source) "
            "VALUES ('o', 'h', 1.0, 'capture')"
        )
        rows = conn.execute(
            "SELECT id FROM observations WHERE ts >= 0 ORDER BY ts"
        ).fetchall()
        assert [r[0] for r in rows] == ["o"]
    finally:
        conn.close()


def test_v0_db_with_partial_shape_raises_instead_of_failing_later(tmp_path):
    """A v0 DB with tables but the WRONG shape RAISES at migrate() time.

    Repro of the original bug: a pre-v1/partial DB whose `observations` table is
    missing the `ts` column used to be stamped to v1 and "succeed", then blow up
    with `no such column: ts` at query time. The presence check is now a SHAPE
    check, so this fails loudly and early — and the DB is left UNSTAMPED.
    """
    # Start from the real v1 shape, then break exactly the `ts` column on
    # observations — isolating the precise defect the original repro hit.
    conn = _make_v1_shaped_db(tmp_path / "partial.db")
    try:
        conn.execute("DROP TABLE observations")
        conn.execute(
            "CREATE TABLE observations ("
            "id TEXT PRIMARY KEY, content_hash TEXT, app TEXT, "
            "window TEXT, url TEXT, session_id TEXT, source TEXT)"  # no `ts`
        )
        conn.execute("PRAGMA user_version = 0")
        conn.commit()

        with pytest.raises(RuntimeError) as exc:
            ensure_schema_version(conn)
        msg = str(exc.value)
        assert "does not match" in msg
        assert "ts" in msg  # names the concrete missing column

        # The DB was NOT silently stamped — it stays at version 0.
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
    finally:
        conn.close()


def _break_chunks_to_pre_redesign_shape(conn: sqlite3.Connection) -> None:
    """Replace `chunks` with its pre-chunk-dedup shape (keyed by content_hash, NO
    `chunk_hash`) — the exact stranded shape behind issue #143. foreign_keys is OFF
    on these bare test connections, so the dependent blob_chunks FK does not block
    the swap; the resulting schema is deliberately inconsistent."""
    conn.execute("DROP TABLE chunks")
    conn.execute(
        "CREATE TABLE chunks ("
        "id TEXT PRIMARY KEY, content_hash TEXT NOT NULL, "
        "span_start INTEGER, span_end INTEGER, text TEXT NOT NULL, "
        "rowid_int INTEGER UNIQUE)"  # note: no chunk_hash
    )


def test_stamped_current_db_with_wrong_chunks_shape_raises(tmp_path):
    """A DB stamped at SCHEMA_VERSION but missing `chunks.chunk_hash` must RAISE.

    Regression for issue #143. The stamp used to be trusted blindly (the shape
    guard only inspected user_version=0 DBs), so a `chunks` table created before an
    in-place change to the v1 shape — silently kept by schema.sql's CREATE TABLE IF
    NOT EXISTS — sailed through and then threw a raw `OperationalError: no such
    column: chunk_hash` at the first add_observation (which the capture daemon
    swallows). The floor-shape check now revalidates an already-stamped DB.
    """
    conn = _make_v1_shaped_db(tmp_path / "stamped_wrong.db")
    try:
        _break_chunks_to_pre_redesign_shape(conn)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()

        with pytest.raises(RuntimeError) as exc:
            ensure_schema_version(conn)
        msg = str(exc.value)
        assert "does not match" in msg
        assert "chunk_hash" in msg  # names the concrete missing column
    finally:
        conn.close()


def test_store_open_on_stamped_wrong_shape_db_raises_not_silent(tmp_path):
    """End-to-end (issue #143): opening MemoryStore on a DB stamped at the current
    version but missing `chunks.chunk_hash` raises the CLEAR migration error — NOT a
    silent open that then throws a raw OperationalError on first write (which the
    capture daemon swallows, so capture stores nothing while looking healthy)."""
    db = str(tmp_path / "stamped_wrong_store.db")
    seed = _make_v1_shaped_db(db)
    _break_chunks_to_pre_redesign_shape(seed)
    seed.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    seed.commit()
    seed.close()

    settings = Settings(data_dir=tmp_path, embed_dim=64)
    with pytest.raises(RuntimeError) as exc:
        MemoryStore(db_path=db, settings=settings, provider=FakeProvider(embed_dim=64))
    msg = str(exc.value)
    assert "does not match" in msg
    assert "chunk_hash" in msg
    assert not isinstance(exc.value, sqlite3.OperationalError)


def test_stamped_old_db_with_wrong_shape_blocks_migration_ladder(tmp_path, monkeypatch):
    """A stamped-but-OLDER wrong-shaped DB must RAISE before the ladder runs.

    Forward-looking guard (issue #143): with a synthetic SCHEMA_VERSION=2, a
    user_version=1 DB whose `chunks` lacks `chunk_hash` must NOT be quietly migrated
    and re-stamped to 2 while still wrong-shaped (it would then fail at first write,
    swallowed by the capture daemon). The floor guard runs for any stamped DB
    BEFORE the migration ladder, so it fails loudly and stays unmigrated.
    """
    monkeypatch.setattr(migrations, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(
        migrations,
        "MIGRATIONS",
        [Migration(version=2, description="noop", apply=lambda conn: None)],
    )
    conn = _make_v1_shaped_db(tmp_path / "stamped_old_wrong.db")
    try:
        _break_chunks_to_pre_redesign_shape(conn)
        conn.execute("PRAGMA user_version = 1")
        conn.commit()

        with pytest.raises(RuntimeError) as exc:
            ensure_schema_version(conn)
        msg = str(exc.value)
        assert "does not match" in msg
        assert "chunk_hash" in msg
        # NOT migrated or re-stamped: the bad DB stays at version 1.
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    finally:
        conn.close()


def test_v0_db_missing_a_required_table_raises(tmp_path):
    """A v0 DB with `observations` present but other required tables missing RAISES."""
    conn = sqlite3.connect(tmp_path / "missing_table.db")
    try:
        # Full observations shape, but the rest of the v1 tables are absent.
        conn.execute(
            "CREATE TABLE observations ("
            "id TEXT PRIMARY KEY, content_hash TEXT, ts REAL, app TEXT, "
            "window TEXT, url TEXT, session_id TEXT, source TEXT)"
        )
        conn.execute("PRAGMA user_version = 0")
        conn.commit()

        with pytest.raises(RuntimeError, match="does not match"):
            ensure_schema_version(conn)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
    finally:
        conn.close()


def test_fresh_empty_db_still_stamps_to_current_version(tmp_path):
    """A brand-new empty DB (no tables) is stamped directly to SCHEMA_VERSION."""
    conn = sqlite3.connect(tmp_path / "empty.db")
    try:
        conn.execute("PRAGMA user_version = 0")
        conn.commit()
        assert ensure_schema_version(conn) == SCHEMA_VERSION
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        conn.close()


def test_store_open_on_partial_db_raises_clear_error_not_raw_operationalerror(tmp_path):
    """End-to-end repro: opening a MemoryStore on a partial v0 DB raises the CLEAR
    migration error, NOT a raw ``OperationalError: no such column: ts``.

    This is the actual production path. ``_apply_schema`` runs ``schema.sql``
    first, and schema.sql creates ``idx_observations_ts ON observations(ts)``.
    Without the preflight guard, a pre-existing ``observations`` table missing
    ``ts`` makes that CREATE INDEX raise a cryptic OperationalError mid-script —
    exactly the failure mode this fix exists to prevent.
    """
    db = str(tmp_path / "partial_store.db")
    # Build the partial DB out-of-band so MemoryStore opens an EXISTING wrong DB.
    seed = sqlite3.connect(db)
    seed.execute("CREATE TABLE observations (id TEXT PRIMARY KEY, app TEXT)")  # no ts
    seed.execute("PRAGMA user_version = 0")
    seed.commit()
    seed.close()

    settings = Settings(data_dir=tmp_path, embed_dim=64)
    with pytest.raises(RuntimeError) as exc:
        MemoryStore(db_path=db, settings=settings, provider=FakeProvider(embed_dim=64))
    msg = str(exc.value)
    assert "does not match" in msg
    assert not isinstance(exc.value, sqlite3.OperationalError)
    # The partial DB was never stamped.
    check = sqlite3.connect(db)
    try:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 0
    finally:
        check.close()


def test_store_open_on_real_v1_legacy_db_succeeds(tmp_path):
    """The genuine legacy path still works through the real store-open code path:
    an unstamped DB with the real v1 shape opens and is stamped to SCHEMA_VERSION."""
    db = str(tmp_path / "legacy_store.db")
    _make_v1_shaped_db(db).close()

    settings = Settings(data_dir=tmp_path, embed_dim=64)
    s = MemoryStore(db_path=db, settings=settings, provider=FakeProvider(embed_dim=64))
    try:
        ver = s.conn.execute("PRAGMA user_version").fetchone()
        assert int(next(iter(ver.values()))) == SCHEMA_VERSION
        # And it is fully usable end-to-end.
        s.add_observation("legacy db still works", source="capture", ts=1.0)
        assert s.time_range(0.0, 10.0)
    finally:
        s.close()


def test_migration_ladder_upgrades_old_db(tmp_path, monkeypatch):
    """A DB at version 1 is upgraded by a (synthetic) version-2 migration."""
    applied: list[int] = []

    def _v2(conn):
        conn.execute("ALTER TABLE observations ADD COLUMN note TEXT")
        applied.append(2)

    monkeypatch.setattr(migrations, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(
        migrations,
        "MIGRATIONS",
        [Migration(version=2, description="add note", apply=_v2)],
    )

    # A realistic stamped-v1 DB: the floor guard now runs for any stamped DB
    # before the ladder, so a stamped DB must carry the real v1 shape (not a bare
    # `observations(id)` stub, which would now — correctly — be refused).
    conn = _make_v1_shaped_db(tmp_path / "upgrade.db")
    conn.execute("PRAGMA user_version = 1")
    conn.commit()

    assert ensure_schema_version(conn) == 2
    assert applied == [2]
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    cols = [r[1] for r in conn.execute("PRAGMA table_info(observations)").fetchall()]
    assert "note" in cols
    conn.close()


def test_real_v1_db_migrates_to_v2_day_memory_shape(tmp_path):
    """A released v1-stamped DB gets the v2 day-memory tables/triggers."""
    conn = _make_v1_shaped_db(tmp_path / "real_v1_to_v2.db")
    try:
        # _make_v1_shaped_db applies the current idempotent schema.sql. Strip the
        # additive v2 objects so this models a real DB created by the released v1
        # build, then stamp it as v1 and run the real ladder.
        conn.execute("DROP TRIGGER IF EXISTS trg_day_memory_observation_delete")
        conn.execute("DROP TABLE IF EXISTS day_memory_sources")
        conn.execute("DROP TABLE IF EXISTS day_memories")
        conn.execute("PRAGMA user_version = 1")
        conn.commit()

        assert ensure_schema_version(conn) == SCHEMA_VERSION
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='day_memories'"
        ).fetchone()
        # After the full ladder (v4), the OLD observation-only trigger is
        # replaced by the typed-refs pair; the legacy junction table is gone.
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='trigger' "
            "AND name='trg_day_memory_observation_delete'"
        ).fetchone() is None
        trigger = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' "
            "AND name='trg_day_memory_source_observation_delete'"
        ).fetchone()
        assert trigger is not None
        assert "BEFORE DELETE ON observations" in trigger[0]
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='day_memory_sources'"
        ).fetchone() is None
    finally:
        conn.close()


def test_real_v2_db_migrates_to_v3_reasoning_send_ledger_shape(tmp_path):
    """A v2-stamped DB gets the v3 redacted reasoning-send ledger."""
    conn = _make_v1_shaped_db(tmp_path / "real_v2_to_v3.db")
    try:
        conn.execute("DROP INDEX IF EXISTS idx_reasoning_send_ledger_feature")
        conn.execute("DROP INDEX IF EXISTS idx_reasoning_send_ledger_created_at")
        conn.execute("DROP TABLE IF EXISTS reasoning_send_ledger")
        conn.execute("PRAGMA user_version = 2")
        conn.commit()

        assert ensure_schema_version(conn) == SCHEMA_VERSION
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='reasoning_send_ledger'"
        ).fetchone()
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND tbl_name='reasoning_send_ledger'"
            ).fetchall()
        }
        assert {
            "idx_reasoning_send_ledger_feature",
            "idx_reasoning_send_ledger_created_at",
        }.issubset(indexes)
        cols = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(reasoning_send_ledger)"
            ).fetchall()
        }
        assert {
            "packet_hash",
            "packet_bytes",
            "citation_count",
            "excluded_by_json",
            "outcome",
            "error_kind",
        }.issubset(cols)
    finally:
        conn.close()


def test_newer_than_supported_db_is_refused(tmp_path):
    """A DB stamped NEWER than this build must RAISE, not silently corrupt."""
    conn = sqlite3.connect(tmp_path / "future.db")
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
    conn.commit()
    with pytest.raises(RuntimeError, match="newer than this build"):
        ensure_schema_version(conn)
    conn.close()


def test_migration_rolls_back_on_failure(tmp_path, monkeypatch):
    """A failing migration leaves user_version unchanged (atomic step)."""
    def _bad(conn):
        conn.execute("ALTER TABLE observations ADD COLUMN ok TEXT")
        raise RuntimeError("boom mid-migration")

    monkeypatch.setattr(migrations, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(
        migrations,
        "MIGRATIONS",
        [Migration(version=2, description="bad", apply=_bad)],
    )
    # Realistic stamped-v1 DB (the floor guard runs before the ladder now).
    conn = _make_v1_shaped_db(tmp_path / "rollback.db")
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    with pytest.raises(RuntimeError, match="boom"):
        ensure_schema_version(conn)
    # Rolled back: still version 1, column not added.
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    cols = [r[1] for r in conn.execute("PRAGMA table_info(observations)").fetchall()]
    assert "ok" not in cols
    conn.close()


# --------------------------------------------------------------------------- #
# strict encryption mode                                                 #
# --------------------------------------------------------------------------- #


def test_require_encryption_raises_when_key_unavailable(monkeypatch, tmp_path):
    """OPENBIRD_REQUIRE_ENCRYPTION + no key => RAISE, no plaintext file written."""
    db = str(tmp_path / "strict.db")
    monkeypatch.setattr(crypto, "_get_or_create_key", lambda: None)
    settings = Settings(data_dir=tmp_path, db_path=db, require_encryption=True)
    with pytest.raises(EncryptionUnavailableError) as exc:
        open_db_verified(settings=settings)
    assert "REQUIRE_ENCRYPTION" in str(exc.value)
    # Crucially: no plaintext DB file was created.
    import os
    assert not os.path.exists(db)


def test_require_encryption_raises_when_sqlcipher_missing(monkeypatch, tmp_path):
    """With sqlcipher3 import forced absent + strict mode, open RAISES."""
    db = str(tmp_path / "strict2.db")
    # Force the sqlcipher3 import to fail and no key available.
    monkeypatch.setitem(sys.modules, "sqlcipher3", None)  # import => ImportError
    monkeypatch.setattr(crypto, "_get_or_create_key", lambda: None)
    settings = Settings(data_dir=tmp_path, db_path=db, require_encryption=True)
    with pytest.raises(EncryptionUnavailableError) as exc:
        open_db_verified(settings=settings)
    assert "sqlcipher3 is not installed" in str(exc.value)


def test_require_encryption_env_parsing(monkeypatch):
    """OPENBIRD_REQUIRE_ENCRYPTION / OPENBIRD_RETENTION_DAYS coerce correctly."""
    from openbird.config import _settings_from_env

    monkeypatch.setenv("OPENBIRD_REQUIRE_ENCRYPTION", "1")
    monkeypatch.setenv("OPENBIRD_RETENTION_DAYS", "90")
    s = _settings_from_env()
    assert s.require_encryption is True
    assert s.retention_days == 90

    monkeypatch.setenv("OPENBIRD_REQUIRE_ENCRYPTION", "no")
    s2 = _settings_from_env()
    assert s2.require_encryption is False


def test_default_mode_still_falls_back_to_plaintext(monkeypatch, tmp_path):
    """Default (require_encryption=False) keeps the backward-compatible fallback."""
    db = str(tmp_path / "fallback.db")
    monkeypatch.setattr(crypto, "_get_or_create_key", lambda: None)
    settings = Settings(data_dir=tmp_path, db_path=db)
    handle = open_db_verified(settings=settings)
    try:
        assert handle.encrypted is False
        assert handle.backend == "sqlite3"
        assert settings.encryption_enabled is False
    finally:
        handle.conn.close()


# --------------------------------------------------------------------------- #
# delete atomicity / no orphans                                          #
# --------------------------------------------------------------------------- #


def _count_orphans(conn) -> int:
    """fts/vec rows whose chunk no longer has any blob_chunks mapping."""
    orphan_chunks = conn.execute(
        "SELECT rowid_int FROM chunks WHERE rowid_int IS NOT NULL "
        "AND chunk_hash NOT IN (SELECT chunk_hash FROM blob_chunks)"
    ).fetchall()
    # Also count vec/fts rows that point at no chunk at all.
    dangling_vec = conn.execute(
        "SELECT COUNT(*) c FROM vec_chunks WHERE chunk_rowid NOT IN "
        "(SELECT rowid_int FROM chunks WHERE rowid_int IS NOT NULL)"
    ).fetchone()["c"]
    dangling_fts = conn.execute(
        "SELECT COUNT(*) c FROM fts_chunks WHERE rowid NOT IN "
        "(SELECT rowid_int FROM chunks WHERE rowid_int IS NOT NULL)"
    ).fetchone()["c"]
    return len(orphan_chunks) + int(dangling_vec) + int(dangling_fts)


def test_delete_leaves_no_orphans_and_integrity_ok(tmp_path):
    db = str(tmp_path / "del.db")
    settings = Settings(data_dir=tmp_path, embed_dim=64)
    s = MemoryStore(db_path=db, settings=settings, provider=FakeProvider(embed_dim=64))
    try:
        for i in range(20):
            s.add_observation(f"observation number {i} with distinct content", source="t", ts=float(i))
        s.delete(since_ts=10.0)
        assert _count_orphans(s.conn) == 0
        ic = s.conn.execute("PRAGMA integrity_check").fetchone()
        assert next(iter(ic.values())) == "ok"
    finally:
        s.close()


def test_full_purge_deletes_reasoning_send_ledger(tmp_path):
    db = str(tmp_path / "ledger-purge.db")
    settings = Settings(data_dir=tmp_path, embed_dim=64)
    s = MemoryStore(db_path=db, settings=settings, provider=FakeProvider(embed_dim=64))
    try:
        s.add_observation("packet source", source="capture", ts=1.0)
        s.record_reasoning_send(
            feature="deep_brain.ask",
            packet_route="deep_brain.preview",
            reasoning_route="cloud_reasoning_active",
            egress="active_model_route",
            route_class="third-party-cloud",
            provider_family="openai",
            model="gpt-5.5",
            packet_hash="abc123",
            packet_bytes=123,
            selected_source_count=2,
            citation_count=1,
            excluded_observations=3,
            excluded_by={"source": 3, "capture": 99, "observation_id": -2},
            outcome="success",
        )
        rows = s.list_reasoning_send_ledger()
        assert len(rows) == 1
        assert rows[0]["excluded_by"] == {"source": 3}

        s.delete(all=True)

        assert s.list_reasoning_send_ledger() == []
    finally:
        s.close()


def test_selective_delete_keeps_redacted_reasoning_send_ledger(tmp_path):
    db = str(tmp_path / "ledger-selective-delete.db")
    settings = Settings(data_dir=tmp_path, embed_dim=64)
    s = MemoryStore(db_path=db, settings=settings, provider=FakeProvider(embed_dim=64))
    try:
        s.add_observation("old packet source", source="capture", ts=1.0)
        s.add_observation("new packet source", source="capture", ts=10.0)
        s.record_reasoning_send(
            feature="deep_brain.ask",
            packet_route="deep_brain.preview",
            reasoning_route="cloud_reasoning_active",
            egress="active_model_route",
            route_class="third-party-cloud",
            provider_family="openai",
            model="gpt-5.5",
            packet_hash="abc123",
            packet_bytes=123,
            selected_source_count=2,
            citation_count=1,
            excluded_observations=0,
            excluded_by={},
            outcome="success",
        )

        s.delete(before_ts=5.0)

        rows = s.list_reasoning_send_ledger()
        assert len(rows) == 1
        assert rows[0]["feature"] == "deep_brain.ask"
        assert rows[0]["packet_hash"] == "abc123"
    finally:
        s.close()


def test_reasoning_send_ledger_sanitizes_error_kind(tmp_path):
    db = str(tmp_path / "ledger-error-kind.db")
    settings = Settings(data_dir=tmp_path, embed_dim=64)
    s = MemoryStore(db_path=db, settings=settings, provider=FakeProvider(embed_dim=64))
    try:
        safe = s.record_reasoning_send(
            feature="deep_brain.ask",
            packet_route="deep_brain.preview",
            reasoning_route="cloud_reasoning_active",
            egress="active_model_route",
            route_class="third-party-cloud",
            provider_family="openai",
            model="gpt-5.5",
            packet_hash="abc123",
            packet_bytes=123,
            selected_source_count=2,
            citation_count=1,
            excluded_observations=0,
            excluded_by={},
            outcome="error",
            error_kind="RuntimeError",
        )
        redacted = s.record_reasoning_send(
            feature="deep_brain.ask",
            packet_route="deep_brain.preview",
            reasoning_route="cloud_reasoning_active",
            egress="active_model_route",
            route_class="third-party-cloud",
            provider_family="openai",
            model="gpt-5.5",
            packet_hash="def456",
            packet_bytes=456,
            selected_source_count=2,
            citation_count=1,
            excluded_observations=0,
            excluded_by={},
            outcome="error",
            error_kind="RuntimeError: secret provider text",
        )

        assert safe["error_kind"] == "RuntimeError"
        assert redacted["error_kind"] == "error"
        assert "secret provider text" not in json.dumps(
            s.list_reasoning_send_ledger(), sort_keys=True
        )
    finally:
        s.close()


def test_delete_rolls_back_on_error(tmp_path):
    """If a DELETE fails mid-way, the whole delete rolls back (no partial state).

    We install a BEFORE-DELETE trigger that RAISEs once observations have already
    been deleted within the transaction — proving the explicit BEGIN/ROLLBACK
    guard restores ALL the deleted rows (previously the multi-DELETE+commit path
    had no rollback, stranding orphaned chunks/fts/vec).
    """
    db = str(tmp_path / "delroll.db")
    settings = Settings(data_dir=tmp_path, embed_dim=64)
    s = MemoryStore(db_path=db, settings=settings, provider=FakeProvider(embed_dim=64))
    try:
        for i in range(5):
            s.add_observation(f"row {i} unique", source="t", ts=float(i))
        before = s.stats()
        assert before["observations"] == 5

        # Abort partway: fail when content_blobs deletion is attempted, AFTER the
        # observations DELETE has already run inside the same transaction.
        s.conn.execute(
            "CREATE TRIGGER blow_up BEFORE DELETE ON content_blobs "
            "BEGIN SELECT RAISE(ABORT, 'simulated crash mid-delete'); END"
        )
        with pytest.raises(Exception, match="simulated crash mid-delete"):
            s.delete(all=True)
        s.conn.execute("DROP TRIGGER blow_up")

        after = s.stats()
        # Everything rolled back: all observations restored, no orphans.
        assert after["observations"] == before["observations"]
        assert after["chunks"] == before["chunks"]
        assert _count_orphans(s.conn) == 0
    finally:
        s.close()


# --------------------------------------------------------------------------- #
# H10 — retention prune + vacuum reclaim                                       #
# --------------------------------------------------------------------------- #


def test_prune_older_than_days(tmp_path):
    db = str(tmp_path / "prune.db")
    settings = Settings(data_dir=tmp_path, embed_dim=64)
    s = MemoryStore(db_path=db, settings=settings, provider=FakeProvider(embed_dim=64))
    try:
        now = time.time()
        s.add_observation("old one", source="t", ts=now - 40 * 86400)
        s.add_observation("recent one", source="t", ts=now - 1 * 86400)
        deleted = s.prune(older_than_days=30)
        assert deleted == 1
        remaining = s.time_range(0.0, now + 1)
        assert len(remaining) == 1
        assert _count_orphans(s.conn) == 0
    finally:
        s.close()


def test_vacuum_reclaims_space(tmp_path):
    db = str(tmp_path / "vac.db")
    settings = Settings(data_dir=tmp_path, embed_dim=64)
    s = MemoryStore(db_path=db, settings=settings, provider=FakeProvider(embed_dim=64))
    try:
        # Insert a lot of distinct content to grow the file.
        for i in range(200):
            s.add_observation(
                f"distinct content block number {i} " * 20, source="t", ts=float(i)
            )
        s.delete(all=True)
        result = s.vacuum()
        # After deleting everything + VACUUM, the file is smaller than before.
        assert result["bytes_after"] <= result["bytes_before"]
        assert result["freelist_after"] == 0
        assert result["bytes_reclaimed"] >= 0
    finally:
        s.close()


def test_prune_requires_a_cutoff(tmp_path):
    db = str(tmp_path / "prune2.db")
    settings = Settings(data_dir=tmp_path, embed_dim=64)
    s = MemoryStore(db_path=db, settings=settings, provider=FakeProvider(embed_dim=64))
    try:
        with pytest.raises(ValueError, match="retention_days"):
            s.prune()
    finally:
        s.close()


# ---------------------------------------------------------------------------
# v4: activity spans + typed day-memory source refs (Phase B)
# ---------------------------------------------------------------------------


def test_empty_db_ladder_reaches_v4_cleanly(tmp_path):
    """Fresh DB: schema.sql -> v1 stamp -> v2 -> v3 -> v4, objects exactly once."""
    db = str(tmp_path / "fresh-v4.db")
    s = MemoryStore(db_path=db, settings=Settings(data_dir=tmp_path, embed_dim=64),
                    provider=FakeProvider(embed_dim=64))
    try:
        assert s.conn.execute("PRAGMA user_version").fetchone()["user_version"] == SCHEMA_VERSION
        # Every v4 object present exactly once (idempotency between schema.sql
        # and the migration would otherwise duplicate or fail).
        for kind, name in [
            ("table", "activity_spans"),
            ("table", "day_memory_source_refs"),
            ("trigger", "trg_day_memory_source_observation_delete"),
            ("trigger", "trg_day_memory_source_span_delete"),
            ("trigger", "trg_day_memory_source_refs_obs_exists"),
            ("trigger", "trg_day_memory_source_refs_span_exists"),
            ("index", "idx_observations_span"),
            ("index", "idx_activity_spans_start"),
        ]:
            rows = s.conn.execute(
                "SELECT COUNT(*) c FROM sqlite_master WHERE type=? AND name=?",
                (kind, name),
            ).fetchone()
            assert rows["c"] == 1, f"{kind} {name} count={rows['c']}"
        cols = {
            r["name"] for r in s.conn.execute("PRAGMA table_info(observations)").fetchall()
        }
        assert "span_id" in cols
    finally:
        s.close()


def test_v3_db_with_existing_day_memory_sources_backfills_refs(tmp_path):
    """Upgrade path: existing junction rows migrate into typed refs, table drops."""
    conn = _make_v1_shaped_db(tmp_path / "v3-backfill.db")
    try:
        # Model a real v3 DB: run the ladder up to v3 only by stamping v1 and
        # temporarily truncating the ladder... simpler: stamp v1, run full
        # ladder is v4 — so instead build the v3 state by hand: day-memory
        # tables per the v2 migration + a seeded row, stamped 3.
        conn.execute("DROP TABLE IF EXISTS day_memory_source_refs")
        conn.execute("DROP TABLE IF EXISTS activity_spans")
        conn.execute("DROP TRIGGER IF EXISTS trg_day_memory_source_observation_delete")
        conn.execute("DROP TRIGGER IF EXISTS trg_day_memory_source_span_delete")
        conn.execute("DROP TRIGGER IF EXISTS trg_day_memory_source_refs_obs_exists")
        conn.execute("DROP TRIGGER IF EXISTS trg_day_memory_source_refs_span_exists")
        # Recreate the LEGACY observations shape (no span_id column) so the v4
        # guarded ALTER path is exercised exactly as on a real v3 DB.
        conn.execute("DROP INDEX IF EXISTS idx_observations_span")
        conn.execute("DROP TABLE observations")
        conn.execute(
            "CREATE TABLE observations ("
            "id TEXT PRIMARY KEY,"
            "content_hash TEXT NOT NULL REFERENCES content_blobs(content_hash) "
            "ON DELETE CASCADE,"
            "ts REAL NOT NULL, app TEXT, window TEXT, url TEXT, "
            "session_id TEXT, source TEXT NOT NULL)"
        )
        # Recreate the LEGACY junction + trigger exactly as the old v2 shipped.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS day_memory_sources ("
            "day_memory_id TEXT NOT NULL REFERENCES day_memories(id) ON DELETE CASCADE,"
            "observation_id TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,"
            "PRIMARY KEY(day_memory_id, observation_id))"
        )
        conn.execute(
            "INSERT INTO content_blobs(content_hash, text) VALUES ('h1', 'txt')"
        )
        conn.execute(
            "INSERT INTO observations(id, content_hash, ts, source) "
            "VALUES ('obs1', 'h1', 1.0, 'capture')"
        )
        conn.execute(
            "INSERT INTO day_memories(id, local_date, source_scope, "
            "extractor_version, generated_at, payload_json, source_count) "
            "VALUES ('dm1', '2026-01-01', 'capture', 'v1', 1.0, '{}', 1)"
        )
        conn.execute(
            "INSERT INTO day_memory_sources(day_memory_id, observation_id) "
            "VALUES ('dm1', 'obs1')"
        )
        conn.execute("PRAGMA user_version = 3")
        conn.commit()

        assert ensure_schema_version(conn) == SCHEMA_VERSION
        refs = conn.execute(
            "SELECT source_kind, source_id FROM day_memory_source_refs "
            "WHERE day_memory_id = 'dm1'"
        ).fetchall()
        assert [tuple(r) for r in refs] == [("observation", "obs1")]
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='day_memory_sources'"
        ).fetchone() is None
        # The preserved day memory is still invalidated by observation delete
        # (through the NEW trigger).
        conn.execute("DELETE FROM observations WHERE id = 'obs1'")
        assert conn.execute("SELECT COUNT(*) FROM day_memories").fetchone()[0] == 0
    finally:
        conn.close()


def test_day_memory_source_refs_integrity_triggers(tmp_path):
    db = str(tmp_path / "refs-integrity.db")
    s = MemoryStore(db_path=db, settings=Settings(data_dir=tmp_path, embed_dim=64),
                    provider=FakeProvider(embed_dim=64))
    try:
        s.conn.execute(
            "INSERT INTO day_memories(id, local_date, source_scope, "
            "extractor_version, generated_at, payload_json, source_count) "
            "VALUES ('dm1', '2026-01-01', 'capture', 'v1', 1.0, '{}', 0)"
        )
        with pytest.raises(Exception, match="unknown observation ref"):
            s.conn.execute(
                "INSERT INTO day_memory_source_refs VALUES ('dm1', 'observation', 'nope')"
            )
        with pytest.raises(Exception, match="unknown span ref"):
            s.conn.execute(
                "INSERT INTO day_memory_source_refs VALUES ('dm1', 'span', 'nope')"
            )
    finally:
        s.close()


def test_activity_spans_check_constraints(tmp_path):
    db = str(tmp_path / "span-checks.db")
    s = MemoryStore(db_path=db, settings=Settings(data_dir=tmp_path, embed_dim=64),
                    provider=FakeProvider(embed_dim=64))
    try:
        # Tier 0 with a window title violates the tier CHECK.
        with pytest.raises(Exception, match="CHECK constraint failed"):
            s.conn.execute(
                "INSERT INTO activity_spans(span_id, epoch_id, start_ts, end_ts, "
                "bundle_id, detail_tier, window, reason) "
                "VALUES ('s1', 'e', 1.0, 2.0, 'b', 0, 'TITLE', 'blocklisted')"
            )
        # Tier 1 with a reason violates it too.
        with pytest.raises(Exception, match="CHECK constraint failed"):
            s.conn.execute(
                "INSERT INTO activity_spans(span_id, epoch_id, start_ts, end_ts, "
                "bundle_id, detail_tier, reason) "
                "VALUES ('s2', 'e', 1.0, 2.0, 'b', 1, 'blocklisted')"
            )
        # Reason outside the closed enum rejected.
        with pytest.raises(Exception, match="CHECK constraint failed"):
            s.conn.execute(
                "INSERT INTO activity_spans(span_id, epoch_id, start_ts, end_ts, "
                "bundle_id, detail_tier, reason) "
                "VALUES ('s3', 'e', 1.0, 2.0, 'b', 0, 'made_up')"
            )
    finally:
        s.close()


# ---------------------------------------------------------------------------
# v5: block summaries + taxonomy cache + summary source kind (Phase D)
# ---------------------------------------------------------------------------


# The six day-memory-source-refs triggers the v5 rebuild owns, name -> the
# load-bearing fragments their sqlite_master SQL must carry.
_V5_TRIGGER_EXPECTATIONS: dict[str, tuple[str, ...]] = {
    "trg_day_memory_source_refs_obs_exists": (
        "BEFORE INSERT ON day_memory_source_refs",
        "NEW.source_kind = 'observation'",
        "unknown observation ref",
    ),
    "trg_day_memory_source_refs_span_exists": (
        "BEFORE INSERT ON day_memory_source_refs",
        "NEW.source_kind = 'span'",
        "unknown span ref",
    ),
    "trg_day_memory_source_refs_summary_exists": (
        "BEFORE INSERT ON day_memory_source_refs",
        "NEW.source_kind = 'summary'",
        "SELECT 1 FROM block_summaries WHERE id = NEW.source_id",
        "unknown summary ref",
    ),
    "trg_day_memory_source_observation_delete": (
        "BEFORE DELETE ON observations",
        "DELETE FROM day_memories",
        "source_kind = 'observation' AND source_id = OLD.id",
    ),
    "trg_day_memory_source_span_delete": (
        "BEFORE DELETE ON activity_spans",
        "DELETE FROM day_memories",
        "source_kind = 'span' AND source_id = OLD.span_id",
    ),
    "trg_day_memory_source_summary_delete": (
        "BEFORE DELETE ON block_summaries",
        "DELETE FROM day_memories",
        "source_kind = 'summary' AND source_id = OLD.id",
    ),
}


def _normalized_trigger_sql(conn, name: str) -> str:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?", (name,)
    ).fetchone()
    assert row is not None, f"trigger {name} is missing"
    sql = row["sql"] if isinstance(row, dict) else row[0]
    return " ".join(str(sql).split())


def _assert_v5_trigger_sql(conn) -> None:
    """Assert every rebuilt trigger's sqlite_master SQL matches the expected text."""
    for name, fragments in _V5_TRIGGER_EXPECTATIONS.items():
        sql = _normalized_trigger_sql(conn, name)
        for fragment in fragments:
            assert fragment in sql, f"{name}: expected {fragment!r} in {sql!r}"


def _make_v4_shaped_db(path) -> sqlite3.Connection:
    """Build a realistic v4-stamped DB (pre-Phase-D shape) from current schema.sql.

    Strips every v5-only object and rebuilds day_memory_source_refs with the OLD
    two-kind CHECK, exactly as a released v4 build shipped it, then stamps 4.
    """
    conn = _make_v1_shaped_db(path)
    for name in _V5_TRIGGER_EXPECTATIONS:
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    conn.execute("DROP TRIGGER IF EXISTS trg_block_summary_source_refs_obs_exists")
    conn.execute("DROP TRIGGER IF EXISTS trg_block_summary_source_refs_span_exists")
    conn.execute("DROP TRIGGER IF EXISTS trg_block_summary_source_observation_delete")
    conn.execute("DROP TRIGGER IF EXISTS trg_block_summary_source_span_delete")
    conn.execute("DROP TABLE IF EXISTS block_summary_source_refs")
    conn.execute("DROP TABLE IF EXISTS block_summaries")
    conn.execute("DROP TABLE IF EXISTS category_assignments")
    conn.execute("DROP TABLE IF EXISTS day_memory_source_refs")
    conn.execute(
        "CREATE TABLE day_memory_source_refs ("
        "day_memory_id TEXT NOT NULL REFERENCES day_memories(id) ON DELETE CASCADE,"
        "source_kind   TEXT NOT NULL CHECK (source_kind IN ('observation','span')),"
        "source_id     TEXT NOT NULL,"
        "PRIMARY KEY (day_memory_id, source_kind, source_id))"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_day_memory_source_refs_source "
        "ON day_memory_source_refs(source_kind, source_id)"
    )
    # Recreate the four v4 triggers as the released v4 build shipped them.
    conn.executescript(
        """
        CREATE TRIGGER trg_day_memory_source_refs_obs_exists
        BEFORE INSERT ON day_memory_source_refs
        WHEN NEW.source_kind = 'observation'
            AND NOT EXISTS (SELECT 1 FROM observations WHERE id = NEW.source_id)
        BEGIN
            SELECT RAISE(ABORT, 'day_memory_source_refs: unknown observation ref');
        END;
        CREATE TRIGGER trg_day_memory_source_refs_span_exists
        BEFORE INSERT ON day_memory_source_refs
        WHEN NEW.source_kind = 'span'
            AND NOT EXISTS (SELECT 1 FROM activity_spans WHERE span_id = NEW.source_id)
        BEGIN
            SELECT RAISE(ABORT, 'day_memory_source_refs: unknown span ref');
        END;
        CREATE TRIGGER trg_day_memory_source_observation_delete
        BEFORE DELETE ON observations
        BEGIN
            DELETE FROM day_memories
            WHERE id IN (
                SELECT day_memory_id
                FROM day_memory_source_refs
                WHERE source_kind = 'observation' AND source_id = OLD.id
            );
        END;
        CREATE TRIGGER trg_day_memory_source_span_delete
        BEFORE DELETE ON activity_spans
        BEGIN
            DELETE FROM day_memories
            WHERE id IN (
                SELECT day_memory_id
                FROM day_memory_source_refs
                WHERE source_kind = 'span' AND source_id = OLD.span_id
            );
        END;
        """
    )
    conn.execute("PRAGMA user_version = 4")
    conn.commit()
    return conn


def test_empty_db_ladder_reaches_v5_cleanly(tmp_path):
    """Fresh DB: every v5 object exists exactly once, with the final trigger SQL."""
    db = str(tmp_path / "fresh-v5.db")
    s = MemoryStore(db_path=db, settings=Settings(data_dir=tmp_path, embed_dim=64),
                    provider=FakeProvider(embed_dim=64))
    try:
        assert s.conn.execute("PRAGMA user_version").fetchone()["user_version"] == SCHEMA_VERSION
        for kind, name in [
            ("table", "block_summaries"),
            ("table", "block_summary_source_refs"),
            ("table", "category_assignments"),
            ("trigger", "trg_block_summary_source_refs_obs_exists"),
            ("trigger", "trg_block_summary_source_refs_span_exists"),
            ("trigger", "trg_block_summary_source_observation_delete"),
            ("trigger", "trg_block_summary_source_span_delete"),
            ("trigger", "trg_day_memory_source_refs_summary_exists"),
            ("trigger", "trg_day_memory_source_summary_delete"),
            ("index", "idx_block_summaries_date"),
            ("index", "idx_block_summaries_start"),
            ("index", "idx_block_summary_source_refs_source"),
        ]:
            rows = s.conn.execute(
                "SELECT COUNT(*) c FROM sqlite_master WHERE type=? AND name=?",
                (kind, name),
            ).fetchone()
            assert rows["c"] == 1, f"{kind} {name} count={rows['c']}"
        # The fresh table carries the final three-kind CHECK.
        table_sql = s.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='day_memory_source_refs'"
        ).fetchone()["sql"]
        assert "'summary'" in table_sql
        _assert_v5_trigger_sql(s.conn)
    finally:
        s.close()


def test_v4_db_rebuild_preserves_rows_and_replaces_all_six_triggers(tmp_path):
    """v4 -> v5: the refs rebuild keeps rows, extends the CHECK, and the migration
    explicitly replaces all six day-memory triggers (asserted via sqlite_master)."""
    conn = _make_v4_shaped_db(tmp_path / "v4-to-v5.db")
    try:
        conn.execute("INSERT INTO content_blobs(content_hash, text) VALUES ('h1', 'txt')")
        conn.execute(
            "INSERT INTO observations(id, content_hash, ts, source) "
            "VALUES ('obs1', 'h1', 1.0, 'capture')"
        )
        conn.execute(
            "INSERT INTO activity_spans(span_id, epoch_id, start_ts, end_ts, "
            "bundle_id, detail_tier) VALUES ('sp1', 'e', 1.0, 2.0, 'b', 1)"
        )
        conn.execute(
            "INSERT INTO day_memories(id, local_date, source_scope, "
            "extractor_version, generated_at, payload_json, source_count) "
            "VALUES ('dm1', '2026-01-01', 'capture', 'v7', 1.0, '{}', 2)"
        )
        conn.execute(
            "INSERT INTO day_memory_source_refs VALUES ('dm1', 'observation', 'obs1')"
        )
        conn.execute("INSERT INTO day_memory_source_refs VALUES ('dm1', 'span', 'sp1')")
        conn.commit()
        # The pre-v5 CHECK rejects the new kind — proving the fixture is real.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO day_memory_source_refs VALUES ('dm1', 'summary', 'x')"
            )

        # Apply the current baseline first, exactly like MemoryStore._apply_schema
        # (schema.sql runs BEFORE the ladder on every open).
        conn.executescript(_SCHEMA_SQL)
        assert ensure_schema_version(conn) == SCHEMA_VERSION

        refs = conn.execute(
            "SELECT source_kind, source_id FROM day_memory_source_refs "
            "WHERE day_memory_id='dm1' ORDER BY source_kind"
        ).fetchall()
        assert [tuple(r) for r in refs] == [("observation", "obs1"), ("span", "sp1")]
        table_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='day_memory_source_refs'"
        ).fetchone()[0]
        assert "'summary'" in table_sql
        # Exactly one of each trigger, with the expected post-rebuild SQL.
        for name in _V5_TRIGGER_EXPECTATIONS:
            count = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name=?",
                (name,),
            ).fetchone()[0]
            assert count == 1, f"{name} count={count}"
        _assert_v5_trigger_sql(conn)
        # The rebuilt index came back.
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' "
            "AND name='idx_day_memory_source_refs_source'"
        ).fetchone()
        # Old triggers still function after the rebuild.
        conn.execute("DELETE FROM observations WHERE id='obs1'")
        assert conn.execute("SELECT COUNT(*) FROM day_memories").fetchone()[0] == 0
    finally:
        conn.close()


def test_v5_migration_is_idempotent_on_rerun(tmp_path):
    """Re-running the v5 step on an already-migrated DB leaves one of everything."""
    from openbird.memory.migrations import _apply_v5_block_summaries

    conn = _make_v4_shaped_db(tmp_path / "v5-rerun.db")
    try:
        conn.executescript(_SCHEMA_SQL)
        assert ensure_schema_version(conn) == SCHEMA_VERSION
        # Second application must not duplicate or fail.
        _apply_v5_block_summaries(conn)
        conn.commit()
        for name in _V5_TRIGGER_EXPECTATIONS:
            count = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name=?",
                (name,),
            ).fetchone()[0]
            assert count == 1
        _assert_v5_trigger_sql(conn)
    finally:
        conn.close()


def test_summary_ref_insert_rejected_for_unknown_id(tmp_path):
    db = str(tmp_path / "summary-ref.db")
    s = MemoryStore(db_path=db, settings=Settings(data_dir=tmp_path, embed_dim=64),
                    provider=FakeProvider(embed_dim=64))
    try:
        s.conn.execute(
            "INSERT INTO day_memories(id, local_date, source_scope, "
            "extractor_version, generated_at, payload_json, source_count) "
            "VALUES ('dm1', '2026-01-01', 'capture', 'v8', 1.0, '{}', 0)"
        )
        with pytest.raises(Exception, match="unknown summary ref"):
            s.conn.execute(
                "INSERT INTO day_memory_source_refs VALUES ('dm1', 'summary', 'nope')"
            )
    finally:
        s.close()


def _seed_summary_chain(conn) -> None:
    """Insert span -> block summary (citing the span) -> day memory (citing the summary)."""
    conn.execute(
        "INSERT INTO activity_spans(span_id, epoch_id, start_ts, end_ts, "
        "bundle_id, detail_tier) VALUES ('sp1', 'e', 100.0, 200.0, 'b', 1)"
    )
    conn.execute(
        "INSERT INTO block_summaries(id, local_date, block_key, block_fingerprint, "
        "start_ts, end_ts, dominant_bundle, level, summary_text, model, "
        "extractor_version, generated_at, source_count) "
        "VALUES ('bs1', '2026-01-01', 'k1', 'f1', 100.0, 200.0, 'b', NULL, "
        "'summary body', 'm', 'block-summary-v1', 1.0, 1)"
    )
    conn.execute(
        "INSERT INTO block_summary_source_refs VALUES ('bs1', 'span', 'sp1')"
    )
    conn.execute(
        "INSERT INTO day_memories(id, local_date, source_scope, extractor_version, "
        "generated_at, payload_json, source_count) "
        "VALUES ('dm1', '2026-01-01', 'capture', 'v8', 1.0, '{}', 1)"
    )
    conn.execute(
        "INSERT INTO day_memory_source_refs VALUES ('dm1', 'summary', 'bs1')"
    )


def test_recursive_trigger_chain_span_to_summary_to_day_memory(tmp_path):
    """span delete -> its block summary deleted -> the day memory citing it deleted.

    The middle hop is a trigger firing another trigger, so this is the direct
    proof that PRAGMA recursive_triggers is on and load-bearing.
    """
    db = str(tmp_path / "chain.db")
    s = MemoryStore(db_path=db, settings=Settings(data_dir=tmp_path, embed_dim=64),
                    provider=FakeProvider(embed_dim=64))
    try:
        _seed_summary_chain(s.conn)
        s.conn.execute("DELETE FROM activity_spans WHERE span_id='sp1'")
        assert s.conn.execute("SELECT COUNT(*) c FROM block_summaries").fetchone()["c"] == 0
        assert s.conn.execute("SELECT COUNT(*) c FROM day_memories").fetchone()["c"] == 0
        assert s.conn.execute(
            "SELECT COUNT(*) c FROM day_memory_source_refs"
        ).fetchone()["c"] == 0
    finally:
        s.close()


def test_recursive_trigger_chain_on_sqlcipher_backend(tmp_path):
    """The same chain holds on the SQLCipher backend when the extra is importable."""
    dbapi = pytest.importorskip("sqlcipher3").dbapi2
    conn = dbapi.connect(str(tmp_path / "chain-enc.db"))
    try:
        conn.execute("PRAGMA key = 'test-chain-key'")
        conn.execute("PRAGMA recursive_triggers = ON")
        assert int(conn.execute("PRAGMA recursive_triggers").fetchone()[0]) == 1
        conn.executescript(_SCHEMA_SQL)
        _seed_summary_chain(conn)
        conn.execute("DELETE FROM activity_spans WHERE span_id='sp1'")
        assert conn.execute("SELECT COUNT(*) FROM block_summaries").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM day_memories").fetchone()[0] == 0
    finally:
        conn.close()


def test_recursive_triggers_pragma_verified_and_raises_on_silent_noop(tmp_path):
    """The store sets + reads back recursive_triggers; a no-op backend is refused."""
    from openbird.memory.store import _enable_recursive_triggers

    db = str(tmp_path / "pragma.db")
    s = MemoryStore(db_path=db, settings=Settings(data_dir=tmp_path, embed_dim=64),
                    provider=FakeProvider(embed_dim=64))
    try:
        row = s.conn.execute("PRAGMA recursive_triggers").fetchone()
        assert int(next(iter(row.values()))) == 1
    finally:
        s.close()

    class _NoOpPragmaConn:
        """Backend that silently ignores the pragma (reads back nothing)."""

        def execute(self, sql, *args):
            class _Cur:
                @staticmethod
                def fetchone():
                    return None

            return _Cur()

    with pytest.raises(RuntimeError, match="recursive_triggers"):
        _enable_recursive_triggers(_NoOpPragmaConn())
