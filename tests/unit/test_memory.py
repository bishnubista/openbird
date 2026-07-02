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


def test_integrity_check_ok_on_healthy_db(store):
    store.add_observation("integrity check payload", source="capture", app="Mail", ts=100.0)
    full = store.integrity_check()
    assert full == {"ok": True, "problems": []}
    quick = store.integrity_check(quick=True)
    assert quick["ok"] is True and quick["problems"] == []


def test_check_database_integrity_healthy(tmp_path):
    import sqlite3

    from openbird.memory.store import check_database_integrity

    db = tmp_path / "healthy.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t(x)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    conn.close()
    res = check_database_integrity(str(db), opener=lambda: sqlite3.connect(db))
    assert res == {"ok": True, "problems": []}


def test_check_database_integrity_corrupt_reports_not_raises(tmp_path):
    import sqlite3

    from openbird.memory.store import check_database_integrity

    db = tmp_path / "corrupt.db"
    db.write_bytes(b"definitely not a sqlite file " * 64)
    # Must REPORT a problem, never raise (this is the whole point of the command).
    res = check_database_integrity(str(db), opener=lambda: sqlite3.connect(db))
    assert res["ok"] is False
    assert res["problems"]


def test_check_database_integrity_open_failure_reported():
    from openbird.memory.store import check_database_integrity

    def boom():
        raise RuntimeError("locked keychain / unreadable")

    res = check_database_integrity("/nonexistent.db", opener=boom)
    assert res["ok"] is False
    assert res["problems"][0].startswith("cannot-open:")
    assert "locked keychain" not in str(res)  # only the exception type, no message


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
    resolves to the latest occurrence. Locking
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


def _save_test_day_memory(store, source_ids):
    return store.save_day_memory(
        local_date="2026-06-12",
        source_scope="capture",
        extractor_version="test",
        payload={
            "schema": 1,
            "narrative_status": "not_persisted",
            "coverage": {"source_ids": source_ids},
        },
        source_ids=source_ids,
        generated_at=123.0,
    )


def test_day_memory_deleted_when_source_observation_deleted_since(store):
    keep = store.add_observation("keep", source="capture", ts=100.0)
    doomed = store.add_observation("doomed", source="capture", ts=300.0)
    saved = _save_test_day_memory(store, [keep.id, doomed.id])
    assert saved["source_count"] == 2

    assert store.delete(since_ts=250.0) == 1

    assert store.get_day_memory(local_date="2026-06-12", source_scope="capture") is None
    assert store.conn.execute("SELECT COUNT(*) c FROM day_memory_source_refs").fetchone()["c"] == 0


def test_day_memory_deleted_when_source_observation_deleted_before(store):
    old = store.add_observation("old", source="capture", ts=100.0)
    new = store.add_observation("new", source="capture", ts=300.0)
    _save_test_day_memory(store, [old.id, new.id])

    assert store.delete(before_ts=200.0) == 1

    assert store.get_day_memory(local_date="2026-06-12", source_scope="capture") is None


def test_day_memory_deleted_by_prune(tmp_path, fake_provider):
    db = str(tmp_path / "daymem-prune.db")
    settings = Settings(data_dir=tmp_path, embed_dim=fake_provider.embed_dim)
    store = MemoryStore(db_path=db, settings=settings, provider=fake_provider)
    try:
        old = store.add_observation("old", source="capture", ts=100.0)
        store.add_observation("new", source="capture", ts=300.0)
        _save_test_day_memory(store, [old.id])

        assert store.prune(older_than_ts=200.0) == 1

        assert store.get_day_memory(local_date="2026-06-12", source_scope="capture") is None
    finally:
        store.close()


def test_day_memory_deleted_by_full_wipe(store):
    obs = store.add_observation("one", source="capture", ts=1.0)
    _save_test_day_memory(store, [obs.id])

    assert store.delete(all=True) == 1

    assert store.get_day_memory(local_date="2026-06-12", source_scope="capture") is None
    assert store.conn.execute("SELECT COUNT(*) c FROM day_memories").fetchone()["c"] == 0
    assert store.conn.execute("SELECT COUNT(*) c FROM day_memory_source_refs").fetchone()["c"] == 0


def test_ensure_day_memory_rebuilds_after_closed_day_backfill(store):
    start, end = 100.0, 199.0
    first = store.add_observation("first work", source="capture", ts=150.0)
    saved1 = store.ensure_day_memory(
        local_date="1970-01-01",
        start_ts=start,
        end_ts=end,
        day_offset=0,
    )
    assert saved1["source_ids"] == [first.id]

    second = store.add_observation("backfilled issue follow up", source="capture", ts=125.0)
    saved2 = store.ensure_day_memory(
        local_date="1970-01-01",
        start_ts=start,
        end_ts=end,
        day_offset=0,
    )

    assert saved2["id"] != saved1["id"]
    assert saved2["payload"]["source_fingerprint"] != saved1["payload"]["source_fingerprint"]
    assert sorted(saved2["source_ids"]) == sorted([first.id, second.id])


def test_ensure_day_memory_reuses_unchanged_timestamp_ties(store):
    start, end = 100.0, 199.0
    store.add_observation("alpha", source="capture", ts=150.0)
    store.add_observation("beta", source="capture", ts=150.0)
    saved1 = store.ensure_day_memory(
        local_date="1970-01-01", start_ts=start, end_ts=end, day_offset=0
    )
    saved2 = store.ensure_day_memory(
        local_date="1970-01-01", start_ts=start, end_ts=end, day_offset=0
    )

    assert saved2["id"] == saved1["id"]
    assert saved2["payload"]["source_fingerprint"] == saved1["payload"]["source_fingerprint"]


def test_ensure_day_memory_rebuilds_old_extractor_payload(store):
    start, end = 100.0, 199.0
    obs = store.add_observation("old extractor work", source="capture", ts=150.0)
    rows = store.time_range_text(start, end, source="capture")
    old = store.save_day_memory(
        local_date="1970-01-01",
        source_scope="capture",
        extractor_version="day-memory-v0",
        payload={
            "schema": 1,
            "extractor_version": "day-memory-v0",
            "narrative_status": "not_persisted",
            "source_fingerprint": store.day_memory_source_fingerprint_from_rows(rows),
            "coverage": {"source_ids": [obs.id]},
        },
        source_ids=[obs.id],
    )

    saved = store.ensure_day_memory(
        local_date="1970-01-01", start_ts=start, end_ts=end, day_offset=0
    )

    assert saved["id"] != old["id"]
    assert saved["extractor_version"] != "day-memory-v0"
    assert "workstreams" in saved["payload"]
    assert saved["payload"]["extractor_version"] == saved["extractor_version"]
    assert saved["payload"]["sessions"][0]["cues"]
    assert "session_id" in saved["payload"]["sessions"][0]


def test_concurrent_ensure_day_memory_converges_to_one_row(tmp_path, fake_provider):
    from concurrent.futures import ThreadPoolExecutor

    db = str(tmp_path / "daymem-concurrent.db")
    settings = Settings(data_dir=tmp_path, embed_dim=fake_provider.embed_dim)
    seed = MemoryStore(db_path=db, settings=settings, provider=fake_provider)
    try:
        seed.add_observation("concurrent day memory", source="capture", ts=150.0)
    finally:
        seed.close()

    def run_once():
        s = MemoryStore(db_path=db, settings=settings, provider=fake_provider)
        try:
            return s.ensure_day_memory(
                local_date="1970-01-01",
                start_ts=100.0,
                end_ts=199.0,
                day_offset=0,
            )["id"]
        finally:
            s.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(lambda _i: run_once(), range(2)))

    check = MemoryStore(db_path=db, settings=settings, provider=fake_provider)
    try:
        count = check.conn.execute("SELECT COUNT(*) c FROM day_memories").fetchone()["c"]
    finally:
        check.close()

    assert count == 1
    assert len(set(ids)) == 1


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


def test_search_falls_back_to_bm25_when_embedding_fails(store):
    """A failing embedding provider must NOT break search.

    Regression: hybrid search discarded already-successful BM25 hits when the
    vector stage raised (Ollama down / timeout / transport). It must degrade to
    BM25-only ranking, mirroring the reranker's RRF fallback.
    """
    store.add_observation(
        "The quarterly budget meeting is scheduled for Friday afternoon.",
        source="capture",
        app="Calendar",
        ts=500.0,
    )

    # Ingestion is done; now make the SEARCH-time embedding call fail.
    def _boom(texts):
        raise RuntimeError("ollama unreachable")

    store.provider.embed = _boom  # type: ignore[method-assign]

    # semantic=True previously raised; it must now return the BM25 hit instead.
    hits = store.search("budget meeting", k=5, semantic=True)
    assert hits, "search must still return BM25 results when embedding fails"
    assert "budget" in hits[0].text.lower()


def test_search_embedding_failure_log_is_content_free(store, caplog):
    """The vector fallback log carries only an exception-type reason code.

    Privacy guard: never log the query text or the provider's (possibly
    content-echoing) exception message.
    """
    store.add_observation("alpha bravo charlie secret payload", source="capture", ts=1.0)

    def _boom(texts):
        raise RuntimeError("ECHOED SENSITIVE TEXT from provider body")

    store.provider.embed = _boom  # type: ignore[method-assign]

    with caplog.at_level("INFO", logger="openbird.memory"):
        store.search("SENSITIVE QUERY alpha", k=5, semantic=True)
    blob = " ".join(r.getMessage() for r in caplog.records)
    assert "vector_skipped" in blob
    assert "RuntimeError" in blob  # exception type is the reason code
    assert "SENSITIVE QUERY" not in blob
    assert "ECHOED SENSITIVE TEXT" not in blob


def test_search_semantic_false_unaffected_by_broken_embedding(store):
    """semantic=False must never touch the embedding provider."""
    store.add_observation("budget meeting notes", source="capture", ts=1.0)

    def _boom(texts):  # would raise if (incorrectly) called
        raise AssertionError("embed() must not be called when semantic=False")

    store.provider.embed = _boom  # type: ignore[method-assign]
    hits = store.search("budget meeting", k=5, semantic=False)
    assert hits, "BM25-only search must work"
    assert "budget" in hits[0].text.lower()


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


# ---------------------------------------------------------------------------
# activity spans (Phase B): store APIs + lifecycle cascades
# ---------------------------------------------------------------------------


def _open_full_span(store, *, start=100.0, end=110.0, bundle="com.apple.mail", **kw):
    return store.open_span(
        epoch_id="epoch1", start_ts=start, end_ts=end, bundle_id=bundle,
        detail_tier=1, window=kw.pop("window", "Inbox"), **kw,
    )


def test_open_span_validates_tier_contract(store):
    # Coarse span with a window must fail in PYTHON (before SQL).
    with pytest.raises(ValueError, match="coarse span must not carry"):
        store.open_span(
            epoch_id="e", start_ts=1.0, end_ts=2.0, bundle_id="b",
            detail_tier=0, window="TITLE", reason="blocklisted",
        )
    with pytest.raises(ValueError, match="requires a reason"):
        store.open_span(
            epoch_id="e", start_ts=1.0, end_ts=2.0, bundle_id="b", detail_tier=0
        )
    with pytest.raises(ValueError, match="must not carry a reason"):
        store.open_span(
            epoch_id="e", start_ts=1.0, end_ts=2.0, bundle_id="b",
            detail_tier=1, reason="blocklisted",
        )
    with pytest.raises(ValueError, match="end_ts"):
        store.open_span(
            epoch_id="e", start_ts=2.0, end_ts=1.0, bundle_id="b", detail_tier=1
        )


def test_open_span_persists_meeting_and_defaults_off(store):
    # Phase C1: the meeting bit persists on both tiers and defaults to 0.
    flagged = _open_full_span(store, start=100.0, end=110.0, meeting=True)
    plain = store.open_span(
        epoch_id="e", start_ts=200.0, end_ts=210.0, bundle_id="us.zoom.xos",
        detail_tier=0, reason="not_allowlisted", meeting=True,
    )
    default = store.open_span(
        epoch_id="e", start_ts=300.0, end_ts=310.0, bundle_id="b", detail_tier=1
    )
    rows = {r["span_id"]: r for r in store.spans_in_range(0.0, 1_000.0)}
    assert rows[flagged]["meeting"] == 1
    assert rows[plain]["meeting"] == 1  # legal on the coarse tier too
    assert rows[default]["meeting"] == 0  # default off


def test_extend_span_is_monotone(store):
    sid = _open_full_span(store, start=100.0, end=110.0)
    store.extend_span(sid, 120.0)
    store.extend_span(sid, 105.0)  # stale update must not regress
    row = store.spans_in_range(0.0, 1_000.0)[0]
    assert row["end_ts"] == 120.0
    # close_span is a final extend (no status column by design).
    store.close_span(sid, 125.0)
    assert store.spans_in_range(0.0, 1_000.0)[0]["end_ts"] == 125.0


def test_spans_in_range_overlap_semantics(store):
    a = _open_full_span(store, start=100.0, end=200.0)
    _open_full_span(store, start=300.0, end=400.0)
    got = store.spans_in_range(150.0, 250.0)  # overlaps only span A
    assert [r["span_id"] for r in got] == [a]
    assert len(store.spans_in_range(0.0, 500.0)) == 2
    assert store.spans_in_range(201.0, 299.0) == []


def test_observation_links_to_span_and_set_null_on_span_delete(store):
    sid = _open_full_span(store)
    obs = store.add_observation("linked text", source="capture", ts=105.0, span_id=sid)
    assert obs.span_id == sid
    row = store.conn.execute(
        "SELECT span_id FROM observations WHERE id = ?", (obs.id,)
    ).fetchone()
    assert row["span_id"] == sid
    # Deleting the span nulls the link (FK SET NULL), never the observation.
    store.conn.execute("DELETE FROM activity_spans WHERE span_id = ?", (sid,))
    row = store.conn.execute(
        "SELECT span_id FROM observations WHERE id = ?", (obs.id,)
    ).fetchone()
    assert row["span_id"] is None


def test_prune_deletes_crossing_span_entirely_and_invalidates_day_memory(store):
    # Span crossing the cutoff: began before, ended after.
    crossing = _open_full_span(store, start=100.0, end=300.0)
    late = _open_full_span(store, start=400.0, end=500.0)
    obs = store.add_observation("txt", source="capture", ts=450.0)
    store.save_day_memory(
        local_date="2026-01-01", source_scope="capture", extractor_version="v1",
        payload={}, source_ids=[obs.id], span_ids=[crossing],
    )
    deleted = store.delete(before_ts=200.0)
    assert deleted == 0  # no observations before the cutoff
    ids = {r["span_id"] for r in store.spans_in_range(0.0, 1_000.0)}
    # The crossing span is deleted ENTIRELY (start < cutoff); the late one stays.
    assert ids == {late}
    # The day memory citing the deleted span was invalidated by the trigger.
    assert store.get_day_memory(local_date="2026-01-01", source_scope="capture") is None


def test_purge_since_deletes_touching_spans(store):
    early = _open_full_span(store, start=100.0, end=150.0)
    touching = _open_full_span(store, start=180.0, end=250.0)
    store.delete(since_ts=200.0)
    ids = {r["span_id"] for r in store.spans_in_range(0.0, 1_000.0)}
    assert early in ids and touching not in ids


def test_full_wipe_clears_spans_and_refs(store):
    sid = _open_full_span(store)
    obs = store.add_observation("txt", source="capture", ts=105.0, span_id=sid)
    store.save_day_memory(
        local_date="2026-01-01", source_scope="capture", extractor_version="v1",
        payload={}, source_ids=[obs.id], span_ids=[sid],
    )
    store.delete(all=True)
    assert store.spans_in_range(0.0, 1e12) == []
    assert store.conn.execute(
        "SELECT COUNT(*) c FROM day_memory_source_refs"
    ).fetchone()["c"] == 0
    assert store.stats()["activity_spans"] == 0


def test_day_memory_span_refs_round_trip(store):
    sid = _open_full_span(store)
    obs = store.add_observation("txt", source="capture", ts=105.0)
    saved = store.save_day_memory(
        local_date="2026-01-02", source_scope="capture", extractor_version="v1",
        payload={}, source_ids=[obs.id], span_ids=[sid],
    )
    assert saved["source_ids"] == [obs.id]
    assert saved["span_ids"] == [sid]
    assert saved["source_count"] == 2
    # Deleting the SPAN invalidates the day memory (typed trigger).
    store.conn.execute("DELETE FROM activity_spans WHERE span_id = ?", (sid,))
    assert store.get_day_memory(local_date="2026-01-02", source_scope="capture") is None


def test_span_id_survives_readback(store):
    sid = _open_full_span(store)
    store.add_observation("linked body text", source="capture", ts=105.0, span_id=sid)
    rows = store.time_range_text(0.0, 1_000.0)
    assert rows, "expected the observation back"
    assert rows[0][0].span_id == sid


# -- block summaries + taxonomy cache (Phase D / v5) ---------------------------


def _save_test_block_summary(store, *, span_ids, observation_ids=(), key="k1",
                             fingerprint="f1", text="worked on the memory schema"):
    return store.save_block_summary(
        local_date="2026-01-01",
        block_key=key,
        block_fingerprint=fingerprint,
        start_ts=100.0,
        end_ts=200.0,
        dominant_bundle="com.apple.mail",
        level=None,
        summary_text=text,
        model="ollama/qwen3:8b",
        extractor_version="block-summary-v1",
        observation_ids=list(observation_ids),
        span_ids=list(span_ids),
    )


def test_save_block_summary_round_trip_with_typed_refs(store):
    sid = _open_full_span(store)
    obs = store.add_observation("body", source="capture", ts=105.0, span_id=sid)
    saved = _save_test_block_summary(store, span_ids=[sid], observation_ids=[obs.id])
    assert saved["block_key"] == "k1"
    assert saved["summary_text"] == "worked on the memory schema"
    assert saved["source_count"] == 2
    assert {(r["source_kind"], r["source_id"]) for r in saved["source_refs"]} == {
        ("observation", obs.id),
        ("span", sid),
    }
    assert store.block_summary_keys() == {"k1": "f1"}
    by_date = store.block_summaries_for_date("2026-01-01")
    assert len(by_date) == 1 and by_date[0]["id"] == saved["id"]
    by_range = store.block_summaries_for_range(150.0, 300.0)
    assert len(by_range) == 1 and by_range[0]["id"] == saved["id"]
    assert store.block_summaries_for_range(300.0, 400.0) == []


def test_save_block_summary_regenerates_same_block_key(store):
    sid = _open_full_span(store)
    _save_test_block_summary(store, span_ids=[sid], fingerprint="f1", text="one")
    saved = _save_test_block_summary(store, span_ids=[sid], fingerprint="f2", text="two")
    rows = store.block_summaries_for_date("2026-01-01")
    assert len(rows) == 1
    assert rows[0]["id"] == saved["id"]
    assert rows[0]["block_fingerprint"] == "f2"
    assert rows[0]["summary_text"] == "two"


def test_block_summary_ref_integrity_rejects_unknown_sources(store):
    import pytest as _pytest

    with _pytest.raises(Exception, match="unknown span ref"):
        _save_test_block_summary(store, span_ids=["missing-span"])
    with _pytest.raises(Exception, match="unknown observation ref"):
        _save_test_block_summary(store, span_ids=[], observation_ids=["missing-obs"])
    # The failed transaction rolled back — no parent row survived.
    assert store.block_summary_keys() == {}


def test_block_summary_deleted_when_cited_span_deleted(store):
    sid = _open_full_span(store)
    _save_test_block_summary(store, span_ids=[sid])
    store.conn.execute("DELETE FROM activity_spans WHERE span_id = ?", (sid,))
    assert store.block_summaries_for_date("2026-01-01") == []
    assert store.conn.execute(
        "SELECT COUNT(*) c FROM block_summary_source_refs"
    ).fetchone()["c"] == 0


def test_block_summary_deleted_when_cited_observation_deleted(store):
    obs = store.add_observation("cited body", source="capture", ts=105.0)
    _save_test_block_summary(store, span_ids=[], observation_ids=[obs.id])
    store.delete(since_ts=100.0)
    assert store.block_summaries_for_date("2026-01-01") == []


def test_full_wipe_clears_block_summaries_and_taxonomy_cache(store):
    sid = _open_full_span(store)
    _save_test_block_summary(store, span_ids=[sid])
    store.save_category_assignment("bundle:com.apple.mail", "other_work", "m")
    assert store.stats()["block_summaries"] == 1
    assert store.stats()["category_assignments"] == 1
    store.delete(all=True)
    assert store.stats()["block_summaries"] == 0
    assert store.stats()["category_assignments"] == 0
    assert store.block_summary_keys() == {}
    assert store.get_category_assignments() == {}


def test_category_assignment_round_trip_and_level_validation(store):
    import pytest as _pytest

    store.save_category_assignment("host:github.com", "focus_work", "m")
    store.save_category_assignment("host:github.com", "other_work", "m2")  # upsert
    assert store.get_category_assignments() == {"host:github.com": "other_work"}
    with _pytest.raises(ValueError, match="unknown taxonomy level"):
        store.save_category_assignment("host:x.com", "productive", "m")
    with _pytest.raises(ValueError, match="unknown taxonomy level"):
        store.save_block_summary(
            local_date="2026-01-01", block_key="bad", block_fingerprint="f",
            start_ts=0.0, end_ts=1.0, dominant_bundle=None, level="productive",
            summary_text="t", model="m", extractor_version="v",
            observation_ids=[], span_ids=[],
        )


def test_save_block_summary_rejects_zero_refs(store):
    with pytest.raises(ValueError, match="at least one source ref"):
        store.save_block_summary(
            local_date="2026-01-01", block_key="k", block_fingerprint="f",
            start_ts=1.0, end_ts=2.0, dominant_bundle=None, level=None,
            summary_text="orphan prose", model="m", extractor_version="v",
            observation_ids=[], span_ids=[],
        )


# --------------------------------------------------------------------------- #
# Summary index (Phase E1)                                                    #
# --------------------------------------------------------------------------- #


def _saved_block(store, *, key="k1", fingerprint="f1", start=1000.0,
                 text="Reviewed the openbird retrieval design and citations."):
    span_id = store.open_span(
        epoch_id="e", start_ts=start, end_ts=start + 900.0, bundle_id="b",
        detail_tier=1,
    )
    return store.save_block_summary(
        local_date="2026-06-29", block_key=key, block_fingerprint=fingerprint,
        start_ts=start, end_ts=start + 900.0, dominant_bundle="b", level=None,
        summary_text=text, model="m", extractor_version="block-summary-v1",
        observation_ids=[], span_ids=[span_id],
    )


def test_stats_week_scope_split_and_summary_index_count(store):
    import json as _json

    store.conn.execute(
        "INSERT INTO day_memories(id, local_date, source_scope, extractor_version, "
        "generated_at, payload_json, source_count) "
        "VALUES ('d1', '2026-06-29', 'capture', 'v9', 1.0, '{}', 0)"
    )
    saved = _saved_block(store)
    store.conn.execute(
        "INSERT INTO day_memories(id, local_date, source_scope, extractor_version, "
        "generated_at, payload_json, source_count) VALUES ('w1', '2026-06-29', "
        "'week', 'week-memory-v1', 1.0, ?, 1)",
        (_json.dumps({"digest_text": "d", "member_fingerprint": "wf"}),),
    )
    store.conn.execute(
        "INSERT INTO day_memory_source_refs VALUES ('w1', 'summary', ?)",
        (saved["id"],),
    )
    store.index_summary(
        summary_kind="block", summary_id=saved["id"], fingerprint="f1",
        text=saved["summary_text"],
    )
    stats = store.stats()
    # Week rows are NOT counted as day memories (verification scripts parse
    # day_memories as the DAY count); they get their own key.
    assert stats["day_memories"] == 1
    assert stats["week_memories"] == 1
    assert stats["summary_index_entries"] == 1


def test_index_summary_replaces_stale_entries_and_chunks_long_text(store):
    saved = _saved_block(store)
    n = store.index_summary(
        summary_kind="block", summary_id=saved["id"], fingerprint="f1",
        text=saved["summary_text"],
    )
    assert n == 1
    assert store.summary_index_state() == {("block", saved["id"]): "f1"}

    # Re-index under a drifted fingerprint: old entries replaced, not appended.
    # The TOCTOU guard requires a LIVE week row whose member_fingerprint
    # matches, so create/refresh a real one per pass (fabricated ids refuse).
    long_text = " ".join(f"sentence number {i} about deep work." for i in range(80))

    def _week(fp):
        return store.save_week_memory(
            week_start_date="2026-06-22",
            payload={"member_fingerprint": fp, "digest": long_text, "model": "m"},
            extractor_version="week-memory-v1",
            summary_ids=[saved["id"]],
        )

    wk = _week("wf1")
    n2 = store.index_summary(
        summary_kind="week", summary_id=wk["id"], fingerprint="wf1", text=long_text
    )
    assert n2 >= 2  # ingest.chunk split the long digest into seq pieces
    wk = _week("wf2")  # regeneration: same Monday, new fingerprint/new id
    n3 = store.index_summary(
        summary_kind="week", summary_id=wk["id"], fingerprint="wf2", text=long_text
    )
    assert n3 == n2
    entries = store.conn.execute(
        "SELECT COUNT(*) c FROM summary_index_entries WHERE summary_kind='week'"
    ).fetchone()["c"]
    assert entries == n2  # replaced in place, zero stale rows
    assert store.summary_index_orphan_counts()["fts_orphans"] == 0
    assert store.summary_index_orphan_counts()["vec_orphans"] == 0


def test_summary_index_pending_reports_missing_and_stale(store):
    import json as _json

    saved = _saved_block(store)
    pending = store.summary_index_pending(limit=32)
    assert [(p["summary_kind"], p["summary_id"]) for p in pending] == [
        ("block", saved["id"])
    ]
    store.index_summary(
        summary_kind="block", summary_id=saved["id"], fingerprint="f1",
        text=saved["summary_text"],
    )
    assert store.summary_index_pending(limit=32) == []

    # Drift the block (regeneration re-keys the id), then add a never-indexed
    # week row citing the NEW summary: both become pending, weeks first.
    regen = _saved_block(store, key="k1", fingerprint="f2")
    store.conn.execute(
        "INSERT INTO day_memories(id, local_date, source_scope, extractor_version, "
        "generated_at, payload_json, source_count) VALUES ('w1', '2026-06-29', "
        "'week', 'week-memory-v1', 1.0, ?, 1)",
        (_json.dumps({"digest_text": "Week digest.", "member_fingerprint": "wf1"}),),
    )
    store.conn.execute(
        "INSERT INTO day_memory_source_refs VALUES ('w1', 'summary', ?)",
        (regen["id"],),
    )
    pending = store.summary_index_pending(limit=32)
    kinds = [(p["summary_kind"], p["summary_id"], p["fingerprint"]) for p in pending]
    assert ("week", "w1", "wf1") in kinds
    assert ("block", regen["id"], "f2") in kinds
    assert kinds[0][0] == "week"


def test_search_summaries_hybrid_finds_block_and_week(store):
    import json as _json

    saved = _saved_block(
        store, text="Debugged the sqlite vector index and citation validation."
    )
    store.index_summary(
        summary_kind="block", summary_id=saved["id"], fingerprint="f1",
        text=saved["summary_text"],
    )
    store.conn.execute(
        "INSERT INTO day_memories(id, local_date, source_scope, extractor_version, "
        "generated_at, payload_json, source_count) VALUES ('w1', '2026-06-29', "
        "'week', 'week-memory-v1', 1.0, ?, 1)",
        (
            _json.dumps(
                {
                    "digest_text": "A week focused on homebrew packaging release.",
                    "member_fingerprint": "wf1",
                    "window": {"start": 100.0, "end": 700.0},
                }
            ),
        ),
    )
    store.conn.execute(
        "INSERT INTO day_memory_source_refs VALUES ('w1', 'summary', ?)",
        (saved["id"],),
    )
    store.index_summary(
        summary_kind="week", summary_id="w1", fingerprint="wf1",
        text="A week focused on homebrew packaging release.",
    )

    hits = store.search_summaries("sqlite vector citation", k=5)
    assert hits and hits[0]["summary_kind"] == "block"
    assert hits[0]["summary_id"] == saved["id"]
    assert hits[0]["source_refs"]
    assert hits[0]["start_ts"] == saved["start_ts"]

    week_hits = store.search_summaries("homebrew packaging", k=5)
    assert week_hits and week_hits[0]["summary_kind"] == "week"
    assert week_hits[0]["summary_id"] == "w1"
    assert week_hits[0]["start_ts"] == 100.0
    assert week_hits[0]["source_refs"] == [
        {"source_kind": "summary", "source_id": saved["id"]}
    ]

    assert store.search_summaries("", k=5) == []


def test_search_summaries_drops_dead_entries(store):
    saved = _saved_block(store, text="Prototype week rollup digest pipeline.")
    store.index_summary(
        summary_kind="block", summary_id=saved["id"], fingerprint="f1",
        text=saved["summary_text"],
    )
    # Forbidden raw delete strands fts/vec rows; search must not resurrect them.
    store.conn.execute("DELETE FROM block_summaries WHERE id = ?", (saved["id"],))
    assert store.search_summaries("week rollup digest", k=5) == []


# --------------------------------------------------------------------------- #
# Week memories (Phase E1): day_memories rows with source_scope='week'        #
# --------------------------------------------------------------------------- #


def _week_payload(monday="2026-06-22", **extra):
    payload = {
        "week_start_date": monday,
        "local_date": monday,
        "digest_text": "A week spent shipping the summary index.",
        "member_fingerprint": "mf1",
        "as_of": 1000.0,
        "model": "m",
        "window": {"start": 100.0, "end": 700.0},
    }
    payload.update(extra)
    return payload


def test_save_week_memory_roundtrip_with_typed_provenance(store):
    saved_block = _saved_block(store)
    week = store.save_week_memory(
        week_start_date="2026-06-22",
        extractor_version="week-memory-v1",
        payload=_week_payload(),
        summary_ids=[saved_block["id"]],
    )
    assert week["local_date"] == "2026-06-22"
    assert week["source_scope"] == "week"
    assert week["payload"]["digest_text"].startswith("A week")
    # Provenance shape fix: summary refs are returned (typed + summary_ids),
    # while the legacy observation/span keys stay present and untouched.
    assert week["summary_ids"] == [saved_block["id"]]
    assert week["source_refs"] == [
        {"source_kind": "summary", "source_id": saved_block["id"]}
    ]
    assert week["source_ids"] == []
    assert week["span_ids"] == []
    assert store.get_week_memory("2026-06-22") == week
    # A chat-path day read for the same date never returns the week row.
    assert store.get_day_memory(local_date="2026-06-22") is None


def test_save_week_memory_refuses_zero_refs_and_non_monday(store):
    with pytest.raises(ValueError, match="at least one summary ref"):
        store.save_week_memory(
            week_start_date="2026-06-22",
            extractor_version="week-memory-v1",
            payload=_week_payload(),
            summary_ids=[],
        )
    saved_block = _saved_block(store)
    with pytest.raises(ValueError, match="Monday"):
        store.save_week_memory(
            week_start_date="2026-06-23",  # a Tuesday
            extractor_version="week-memory-v1",
            payload=_week_payload(),
            summary_ids=[saved_block["id"]],
        )
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        store.save_week_memory(
            week_start_date="junk",
            extractor_version="week-memory-v1",
            payload=_week_payload(),
            summary_ids=[saved_block["id"]],
        )


def test_save_week_memory_unknown_summary_ref_aborts_atomically(store):
    import sqlite3 as _sqlite3

    with pytest.raises(_sqlite3.IntegrityError, match="unknown summary ref"):
        store.save_week_memory(
            week_start_date="2026-06-22",
            extractor_version="week-memory-v1",
            payload=_week_payload(),
            summary_ids=["not-a-real-summary"],
        )
    # The parent insert rolled back with the failed ref insert.
    assert store.get_week_memory("2026-06-22") is None


def test_save_week_memory_regeneration_replaces_row_and_sweeps_index(store):
    saved_block = _saved_block(store)
    first = store.save_week_memory(
        week_start_date="2026-06-22",
        extractor_version="week-memory-v1",
        payload=_week_payload(),
        summary_ids=[saved_block["id"]],
    )
    store.index_summary(
        summary_kind="week", summary_id=first["id"], fingerprint="mf1",
        text=first["payload"]["digest_text"],
    )
    second = store.save_week_memory(
        week_start_date="2026-06-22",
        extractor_version="week-memory-v1",
        payload=_week_payload(member_fingerprint="mf2"),
        summary_ids=[saved_block["id"]],
    )
    assert second["id"] != first["id"]
    # UNIQUE(local_date, 'week') semantics: exactly one row per week.
    count = store.conn.execute(
        "SELECT COUNT(*) c FROM day_memories WHERE source_scope='week'"
    ).fetchone()["c"]
    assert count == 1
    # The OLD week row's index rows were swept in the same transaction.
    assert store.conn.execute(
        "SELECT COUNT(*) c FROM summary_index_entries WHERE summary_kind='week'"
    ).fetchone()["c"] == 0
    assert store.summary_index_orphan_counts()["ok"] is True


def test_week_memory_dies_with_cited_block_summary_recursively(store):
    """delete span -> block summary (trigger) -> week row (recursive trigger)."""
    saved_block = _saved_block(store, start=1000.0)
    week = store.save_week_memory(
        week_start_date="2026-06-22",
        extractor_version="week-memory-v1",
        payload=_week_payload(),
        summary_ids=[saved_block["id"]],
    )
    store.index_summary(
        summary_kind="week", summary_id=week["id"], fingerprint="mf1",
        text=week["payload"]["digest_text"],
    )
    deleted = store.delete(since_ts=0.0)  # production path sweeps the index too
    assert deleted == 0  # span-backed only; no observations existed
    assert store.get_week_memory("2026-06-22") is None
    assert store.conn.execute(
        "SELECT COUNT(*) c FROM block_summaries"
    ).fetchone()["c"] == 0
    assert store.summary_index_orphan_counts()["ok"] is True
    assert store.conn.execute(
        "SELECT COUNT(*) c FROM summary_index_entries"
    ).fetchone()["c"] == 0


def test_full_purge_wipes_week_memories(store):
    saved_block = _saved_block(store)
    store.save_week_memory(
        week_start_date="2026-06-22",
        extractor_version="week-memory-v1",
        payload=_week_payload(),
        summary_ids=[saved_block["id"]],
    )
    store.delete(all=True)
    assert store.get_week_memory("2026-06-22") is None
    assert store.stats()["week_memories"] == 0


def test_week_memories_overlapping_windows(store):
    import datetime as _dt

    saved_block = _saved_block(store)
    mondays = ["2026-06-15", "2026-06-22", "2026-06-29"]
    for monday in mondays:
        store.save_week_memory(
            week_start_date=monday,
            extractor_version="week-memory-v1",
            payload=_week_payload(monday),
            summary_ids=[saved_block["id"]],
        )

    def ts(day: str, hour: int = 12) -> float:
        return _dt.datetime.strptime(day, "%Y-%m-%d").replace(hour=hour).timestamp()

    # A mid-week single-day window hits exactly its own week.
    hits = store.week_memories_overlapping(ts("2026-06-24"), ts("2026-06-25"))
    assert [w["local_date"] for w in hits] == ["2026-06-22"]
    # A window crossing a Monday boundary hits both adjacent weeks.
    hits = store.week_memories_overlapping(ts("2026-06-21"), ts("2026-06-23"))
    assert [w["local_date"] for w in hits] == ["2026-06-15", "2026-06-22"]
    # A window before all stored weeks hits nothing.
    assert store.week_memories_overlapping(ts("2026-05-01"), ts("2026-05-02")) == []


# --------------------------------------------------------------------------- #
# entity ledger store APIs (Phase E2)                                         #
# --------------------------------------------------------------------------- #


def test_upsert_entity_deterministic_id_and_min_max_ts(store):
    from openbird.memory.store import entity_id_for

    a = store.upsert_entity("repo", "bbista/openbird", seen_ts=200.0)
    assert a["id"] == entity_id_for("repo", "bbista/openbird")
    assert a["first_ts"] == 200.0 and a["last_ts"] == 200.0
    assert a["aliases"] == []

    # Casefolded identity: a differently-cased sighting upserts the SAME row.
    b = store.upsert_entity("repo", "BBista/OpenBird", seen_ts=100.0)
    assert b["id"] == a["id"]
    assert b["name"] == "bbista/openbird"  # original stored name kept
    assert b["first_ts"] == 100.0 and b["last_ts"] == 200.0

    c = store.upsert_entity("repo", "bbista/openbird", seen_ts=300.0)
    assert c["first_ts"] == 100.0 and c["last_ts"] == 300.0
    assert len(store.list_entities(kind="repo")) == 1


def test_upsert_entity_last_seen_ref_updates_only_when_newer(store):
    obs_old = store.add_observation("old", source="capture", ts=100.0)
    obs_new = store.add_observation("new", source="capture", ts=200.0)
    store.upsert_entity(
        "repo", "bbista/openbird", seen_ts=200.0,
        source_kind="observation", source_id=obs_new.id,
    )
    stale = store.upsert_entity(
        "repo", "bbista/openbird", seen_ts=100.0,
        source_kind="observation", source_id=obs_old.id,
    )
    # The older sighting neither regresses last_ts nor replaces the ref.
    assert stale["last_ts"] == 200.0
    assert stale["last_seen_source_id"] == obs_new.id


def test_entity_status_transitions_and_user_marked_done_immunity(store):
    e = store.upsert_entity("repo", "bbista/openbird", seen_ts=100.0)
    assert e["status"] == "active"
    assert store.mark_dormant_entities(cutoff_ts=200.0) == 1
    assert store.get_entity(e["id"])["status"] == "dormant"

    # NEW activity (strictly newer ts) flips dormant back to active…
    again = store.upsert_entity("repo", "bbista/openbird", seen_ts=300.0)
    assert again["status"] == "active"

    # …but user_marked_done is immune to BOTH directions.
    store.set_entity_status(e["id"], "user_marked_done")
    assert store.upsert_entity(
        "repo", "bbista/openbird", seen_ts=400.0
    )["status"] == "user_marked_done"
    assert store.mark_dormant_entities(cutoff_ts=999.0) == 0
    assert store.get_entity(e["id"])["status"] == "user_marked_done"


def test_entities_matching_casefolded_both_directions(store):
    e = store.upsert_entity("repo", "bbista/openbird", seen_ts=100.0)
    store.set_entity_aliases(e["id"], ["openbird"])
    store.upsert_entity("domain", "github.com", seen_ts=100.0)

    # Query contains the alias.
    assert [x["name"] for x in store.entities_matching("the openbird repo")] == [
        "bbista/openbird"
    ]
    # Query IS a substring of the name.
    assert [x["name"] for x in store.entities_matching("OPENBIRD")] == [
        "bbista/openbird"
    ]
    # Exact full name.
    assert [x["name"] for x in store.entities_matching("bbista/openbird")] == [
        "bbista/openbird"
    ]
    assert store.entities_matching("unrelated thing") == []
    assert store.entities_matching("") == []


def test_entity_evidence_for_newest_first_and_dedup(store):
    obs = store.add_observation("merged text", source="capture", ts=100.0)
    obs2 = store.add_observation("closed text", source="capture", ts=200.0)
    e = store.upsert_entity("repo", "bbista/openbird", seen_ts=100.0)
    assert store.add_entity_evidence(
        e["id"], ts=100.0, kind="pr_merged", source_kind="observation",
        source_id=obs.id, detail="github:bbista/openbird#1",
    )
    # Identical row: UNIQUE dedup, returns False.
    assert not store.add_entity_evidence(
        e["id"], ts=100.0, kind="pr_merged", source_kind="observation",
        source_id=obs.id, detail="github:bbista/openbird#1",
    )
    assert store.add_entity_evidence(
        e["id"], ts=200.0, kind="ticket_closed", source_kind="observation",
        source_id=obs2.id, detail="github:bbista/openbird#2",
    )
    rows = store.entity_evidence_for(e["id"])
    assert [r["kind"] for r in rows] == ["ticket_closed", "pr_merged"]

    with pytest.raises(ValueError, match="unknown evidence kind"):
        store.add_entity_evidence(
            e["id"], ts=1.0, kind="vibes", source_kind="observation",
            source_id=obs.id,
        )
    with pytest.raises(ValueError, match="unknown entity kind"):
        store.upsert_entity("project", "x", seen_ts=1.0)
    with pytest.raises(ValueError, match="together"):
        store.upsert_entity("repo", "x", seen_ts=1.0, source_kind="observation")


def test_stats_has_entity_counts(store):
    stats = store.stats()
    assert stats["entities"] == 0 and stats["entity_evidence"] == 0
    obs = store.add_observation("t", source="capture", ts=1.0)
    e = store.upsert_entity("repo", "a/b", seen_ts=1.0)
    store.add_entity_evidence(
        e["id"], ts=1.0, kind="open_loop", source_kind="observation",
        source_id=obs.id, detail="github:a/b#1",
    )
    stats = store.stats()
    assert stats["entities"] == 1 and stats["entity_evidence"] == 1


def test_entity_evidence_orphan_counts_clean(store):
    obs = store.add_observation("t", source="capture", ts=1.0)
    e = store.upsert_entity("repo", "a/b", seen_ts=1.0)
    store.add_entity_evidence(
        e["id"], ts=1.0, kind="open_loop", source_kind="observation",
        source_id=obs.id,
    )
    counts = store.entity_evidence_orphan_counts()
    assert counts["ok"] is True
    store.delete(all=True)
    assert store.entity_evidence_orphan_counts()["ok"] is True


def test_observations_text_page_row_cap_and_composite_cursor(store):
    ids = []
    for i in range(5):
        obs = store.add_observation(f"row {i} text", source="capture", ts=100.0 + i)
        ids.append(obs.id)
    page1 = store.observations_text_page(0.0, "", limit=2)
    assert [o.id for o, _ in page1] == ids[:2]
    last = page1[-1][0]
    page2 = store.observations_text_page(last.ts, last.id, limit=10)
    assert [o.id for o, _ in page2] == ids[2:]
    # max_chars caps text per row.
    capped = store.observations_text_page(0.0, "", limit=1, max_chars=4)
    assert capped[0][1] == "row "


def test_observations_text_page_ties_on_same_ts(store):
    a = store.add_observation("tie a", source="capture", ts=100.0)
    b = store.add_observation("tie b", source="capture", ts=100.0)
    first, second = sorted([a.id, b.id])
    page = store.observations_text_page(100.0, first, limit=10)
    assert [o.id for o, _ in page] == [second]


def test_kv_roundtrip(store):
    assert store.get_kv("entity_aggregation.obs_ts") is None
    store.set_kv("entity_aggregation.obs_ts", "123.5")
    assert store.get_kv("entity_aggregation.obs_ts") == "123.5"
    store.set_kv("entity_aggregation.obs_ts", "456.0")
    assert store.get_kv("entity_aggregation.obs_ts") == "456.0"
