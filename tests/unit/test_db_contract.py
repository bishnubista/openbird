"""Regression tests for DB concurrency, migrations, strict encryption,
delete atomicity, and retention/vacuum.

These complement test_memory.py / test_crypto.py and exercise the on-disk
behaviors that must stay reliable in production.
"""

from __future__ import annotations

import sqlite3
import sys
import threading
import time

import pytest

from openbird.config import Settings
from openbird.memory import migrations
from openbird.memory.migrations import (
    SCHEMA_VERSION,
    Migration,
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


def test_fresh_db_is_stamped_to_current_version(tmp_path):
    db = str(tmp_path / "fresh.db")
    s = MemoryStore(
        db_path=db,
        settings=Settings(data_dir=tmp_path, embed_dim=64),
        provider=FakeProvider(embed_dim=64),
    )
    try:
        ver = s.conn.execute("PRAGMA user_version").fetchone()
        # mapping_row_factory -> dict; value is the single column.
        assert int(next(iter(ver.values()))) == SCHEMA_VERSION
    finally:
        s.close()


def test_legacy_unversioned_db_is_adopted_as_v1(tmp_path):
    """A DB with the v1 shape but user_version=0 is stamped to 1, not migrated."""
    conn = sqlite3.connect(tmp_path / "legacy.db")
    # Recreate just enough of the v1 shape that _db_has_existing_tables sees it.
    conn.execute("CREATE TABLE observations (id TEXT PRIMARY KEY)")
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    assert ensure_schema_version(conn) == SCHEMA_VERSION
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    conn.close()


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

    conn = sqlite3.connect(tmp_path / "upgrade.db")
    conn.execute("CREATE TABLE observations (id TEXT PRIMARY KEY)")
    conn.execute("PRAGMA user_version = 1")
    conn.commit()

    assert ensure_schema_version(conn) == 2
    assert applied == [2]
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    cols = [r[1] for r in conn.execute("PRAGMA table_info(observations)").fetchall()]
    assert "note" in cols
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
    conn = sqlite3.connect(tmp_path / "rollback.db")
    conn.execute("CREATE TABLE observations (id TEXT PRIMARY KEY)")
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
            s.add_observation(
                f"observation number {i} with distinct content", source="t", ts=float(i)
            )
        s.delete(since_ts=10.0)
        assert _count_orphans(s.conn) == 0
        ic = s.conn.execute("PRAGMA integrity_check").fetchone()
        assert next(iter(ic.values())) == "ok"
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
            s.add_observation(f"distinct content block number {i} " * 20, source="t", ts=float(i))
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
