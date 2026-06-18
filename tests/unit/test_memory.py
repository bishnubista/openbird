"""Unit tests for MemoryStore: chunk-level dedup, cascade delete, time_range.

Embeddings are supplied by a deterministic fake provider (see conftest) so these
tests need no Ollama/network.
"""

from __future__ import annotations

import pytest

from openbird.config import Settings
from openbird.memory.store import MemoryStore


@pytest.fixture
def store(mem_settings, fake_provider) -> MemoryStore:
    s = MemoryStore(db_path=":memory:", settings=mem_settings, provider=fake_provider)
    yield s
    s.close()


def test_reserved_mlx_backend_fails_before_db_open(tmp_path):
    settings = Settings(data_dir=tmp_path, llm_backend="mlx")
    db_path = tmp_path / "reserved-backend.db"
    with pytest.raises(NotImplementedError, match="experiments/mlx-runtime"):
        MemoryStore(db_path=str(db_path), settings=settings)
    assert not db_path.exists()


def test_dedup_keeps_one_blob_but_n_observations(store):
    text = "The status report is ready for review."
    obs1 = store.add_observation(text, source="capture", app="Mail", ts=100.0)
    obs2 = store.add_observation(text, source="capture", app="Mail", ts=200.0)
    obs3 = store.add_observation(text, source="capture", app="Mail", ts=300.0)

    assert obs1.id != obs2.id != obs3.id
    assert obs1.content_hash == obs2.content_hash == obs3.content_hash

    stats = store.stats()
    assert stats["observations"] == 3
    assert stats["blobs"] == 1  # deduped
    # The single short text is one chunk, embedded/indexed once.
    assert stats["chunks"] == 1
    assert stats["vectors"] == 1


def test_distinct_text_creates_distinct_blobs(store):
    store.add_observation("apple pie recipe", source="capture", ts=1.0)
    store.add_observation("orange marmalade recipe", source="capture", ts=2.0)
    stats = store.stats()
    assert stats["observations"] == 2
    assert stats["blobs"] == 2
    assert stats["vectors"] == 2


def test_chunk_shared_across_windows_stored_once(store, mem_settings, fake_provider):
    """Two long captures sharing an identical leading chunk must store/embed it once.

    Regression for the full-window-hash bug: dedup must be CHUNK-level, so an
    identical chunk recurring in two otherwise-different windows is stored,
    embedded, and indexed exactly once (mapped to both blobs via blob_chunks).
    """
    # A shared block well over CHUNK_SIZE so it forms its own leading chunk(s),
    # then a tail that differs between the two documents.
    shared = ("Shared sentence number one. " * 60).strip()  # ~1.6k chars
    doc_a = shared + "\n\n" + ("Alpha tail content for document A. " * 6)
    doc_b = shared + "\n\n" + ("Bravo tail content for document B. " * 6)

    store.add_observation(doc_a, source="capture", app="A", ts=1.0)
    store.add_observation(doc_b, source="capture", app="B", ts=2.0)

    # At least one chunk is shared (mapped to both blobs).
    shared_rows = store.conn.execute(
        "SELECT chunk_hash, COUNT(*) c FROM blob_chunks GROUP BY chunk_hash HAVING c >= 2"
    ).fetchall()
    assert shared_rows, "expected at least one chunk shared across both blobs"

    # Every stored chunk is embedded exactly once.
    s = store.stats()
    assert s["chunks"] == s["vectors"]
    assert s["blobs"] == 2

    # Sharing must REDUCE total chunks vs. storing each doc independently.
    def standalone_chunks(text: str) -> int:
        solo = MemoryStore(db_path=":memory:", settings=mem_settings, provider=fake_provider)
        solo.add_observation(text, source="t", ts=1.0)
        n = solo.stats()["chunks"]
        solo.close()
        return n

    independent = standalone_chunks(doc_a) + standalone_chunks(doc_b)
    assert s["chunks"] < independent, (
        f"expected sharing to dedup chunks: combined={s['chunks']} independent={independent}"
    )


def test_shared_chunk_resolves_to_most_recent_occurrence(store):
    """A deduped chunk shared by multiple observations cites the MOST RECENT one.

    This is an explicit, documented policy (not an accidental heuristic): the same
    content captured in two apps/times shares one chunk, and a retrieval hit
    resolves to the latest occurrence (see PLAN accepted residual risks). Locking
    it here makes the behavior intentional and regression-guarded.
    """
    store.add_observation("identical shared note text", source="capture", app="AppA", ts=100.0)
    store.add_observation("identical shared note text", source="capture", app="AppB", ts=200.0)

    stats = store.stats()
    assert stats["observations"] == 2 and stats["chunks"] == 1  # shared chunk

    hits = store.search("identical shared note text", k=5)
    assert hits and hits[0].observation is not None
    assert hits[0].observation.app == "AppB"   # most-recent occurrence
    assert hits[0].observation.ts == 200.0


def test_cascade_delete_all(store):
    store.add_observation("one", source="capture", ts=1.0)
    store.add_observation("two", source="capture", ts=2.0)
    deleted = store.delete(all=True)
    assert deleted == 2
    stats = store.stats()
    assert stats == {
        **stats,
        "observations": 0,
        "blobs": 0,
        "chunks": 0,
        "vectors": 0,
    }


def test_cascade_delete_since_orphans_only(store):
    # Same text seen at two times -> 1 blob, 2 observations.
    store.add_observation("shared content here", source="capture", ts=100.0)
    store.add_observation("shared content here", source="capture", ts=300.0)
    # A different blob only in the deleted window.
    store.add_observation("doomed unique content", source="capture", ts=400.0)

    deleted = store.delete(since_ts=250.0)
    assert deleted == 2  # the ts=300 shared obs + the ts=400 unique obs

    stats = store.stats()
    # Shared blob survives (its ts=100 observation remains).
    # Doomed blob is fully orphaned and cascade-deleted.
    assert stats["observations"] == 1
    assert stats["blobs"] == 1
    assert stats["chunks"] == 1
    assert stats["vectors"] == 1

    # The surviving observation is the ts=100 one.
    remaining = store.time_range(0.0, 1000.0)
    assert len(remaining) == 1
    assert remaining[0].ts == 100.0


def test_time_range_correctness(store):
    store.add_observation("a", source="capture", ts=10.0)
    store.add_observation("b", source="capture", ts=20.0)
    store.add_observation("c", source="capture", ts=30.0)
    store.add_observation("d", source="capture", ts=40.0)

    mid = store.time_range(15.0, 35.0)
    assert [o.ts for o in mid] == [20.0, 30.0]

    inclusive = store.time_range(10.0, 40.0)
    assert [o.ts for o in inclusive] == [10.0, 20.0, 30.0, 40.0]

    empty = store.time_range(100.0, 200.0)
    assert empty == []


def test_search_finds_relevant_chunk(store):
    store.add_observation(
        "The quarterly budget meeting is scheduled for Friday afternoon.",
        source="capture",
        app="Calendar",
        ts=500.0,
    )
    store.add_observation(
        "Remember to water the office plants on Monday.",
        source="capture",
        ts=600.0,
    )
    hits = store.search("budget meeting", k=5)
    assert hits, "expected at least one hit"
    top = hits[0]
    assert "budget" in top.text.lower()
    # Hit resolves back to a concrete observation (occurrence-aware citation).
    assert top.observation is not None
    assert top.observation.app == "Calendar"


def test_search_empty_query(store):
    store.add_observation("anything", source="capture", ts=1.0)
    assert store.search("   ") == []


def test_stats_reports_cohort_and_dim(store):
    s = store.stats()
    assert s["embed_dim"] == 768
    assert s["cohort_key"] is not None


def test_purge_all_allows_reopen_with_different_cohort(mem_settings, fake_provider, tmp_path):
    """delete(all=True) clears the cohort marker so a new embedding model is accepted.

    Regression: a stale cohort_key must not reject a fresh provider after every
    vector has been purged.
    """
    db = str(tmp_path / "cohort.db")
    s1 = MemoryStore(db_path=db, settings=mem_settings, provider=fake_provider)
    s1.add_observation("hello world", source="capture", ts=1.0)
    s1.delete(all=True)
    s1.close()

    # Reopen with a DIFFERENT embedding cohort — must not raise (store is empty).
    class OtherCohortProvider(fake_provider.__class__):
        def cohort_key(self) -> str:
            return "fake:other-embed:768:beadfeed"

    other = OtherCohortProvider(embed_dim=fake_provider.embed_dim)
    s2 = MemoryStore(db_path=db, settings=mem_settings, provider=other)
    assert s2.stats()["observations"] == 0
    assert s2.stats()["cohort_key"] == "fake:other-embed:768:beadfeed"
    s2.close()


def test_purge_all_allows_reopen_with_different_dimension(mem_settings, fake_provider, tmp_path):
    """After a full purge, reopening with a DIFFERENT embedding dimension works.

    The vector table must be rebuilt at the new dim; otherwise inserts against the
    stale FLOAT[old] table would fail.
    """
    from openbird.config import Settings

    db = str(tmp_path / "dim.db")
    s1 = MemoryStore(db_path=db, settings=mem_settings, provider=fake_provider)
    s1.add_observation("hello world", source="capture", ts=1.0)
    s1.delete(all=True)
    s1.close()

    # Reopen with a 512-dim provider + matching settings, then ingest (insert path).
    small = fake_provider.__class__(embed_dim=512)
    s2 = MemoryStore(db_path=db, settings=Settings(data_dir=tmp_path, embed_dim=512), provider=small)
    obs = s2.add_observation("a brand new note", source="capture", ts=2.0)
    assert obs.id
    assert s2.stats()["embed_dim"] == 512
    assert s2.stats()["vectors"] == 1  # inserted into the rebuilt FLOAT[512] table
    s2.close()
