"""The local memory store: observations, deduped blobs, chunk-level retrieval.

Data model:
  * ``content_blobs`` — deduped canonical text, embedded once.
  * ``observations`` — one row per occurrence, never deduped.
  * ``chunks`` + ``fts_chunks`` (FTS5) + ``vec_chunks`` (sqlite-vec, per chunk).

``add_observation`` ALWAYS inserts a new observation row, dedups content at the
*chunk* level, embeds each unique chunk once, and indexes it in both FTS and the
vector table. ``search`` fuses vector + BM25 with RRF, then dedups with MMR, and
resolves every hit back to a concrete observation for occurrence-aware citations.
"""

from __future__ import annotations

import struct
import time
import uuid
from pathlib import Path

from openbird.config import Settings, get_settings
from openbird.llm.base import LLMProviderProtocol
from openbird.llm.provider import create_llm_provider
from openbird.memory import ingest
from openbird.memory.search import mmr, rrf
from openbird.storage.crypto import mapping_row_factory, open_encrypted_db
from openbird.types import Observation, SearchHit

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _serialize_f32(vector: list[float]) -> bytes:
    """Pack a float vector into sqlite-vec's little-endian float32 blob format."""
    return struct.pack(f"<{len(vector)}f", *vector)


class MemoryStore:
    """SQLite-backed hybrid (FTS5 + sqlite-vec) personal memory."""

    def __init__(
        self,
        db_path: str | None = None,
        *,
        settings: Settings | None = None,
        provider: LLMProviderProtocol | None = None,
    ) -> None:
        """Open the store, load sqlite-vec, and apply the schema.

        Args:
            db_path: Override DB path (``":memory:"`` is supported for tests).
            settings: Settings; defaults to :func:`get_settings`.
            provider: LLM provider for embeddings; defaults to the configured
                provider implementation. Injectable so tests can mock embeddings.
        """
        self.settings = settings or get_settings()
        self.provider = provider or create_llm_provider(self.settings)
        self.embed_dim = self.settings.embed_dim

        resolved = db_path if db_path is not None else self.settings.db_path
        if resolved == ":memory:":
            import sqlite_vec

            self.conn = __import__("sqlite3").connect(":memory:")
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
        else:
            self.conn = open_encrypted_db(resolved, settings=self.settings)

        self.conn.row_factory = mapping_row_factory
        self.conn.execute("PRAGMA foreign_keys = ON")
        try:
            self._apply_schema()
            self._record_cohort()
        except Exception:
            # Don't leak the open connection if construction fails (e.g. an
            # embedding-cohort mismatch on an existing DB).
            self.conn.close()
            raise

    # -- setup ----------------------------------------------------------------

    def _apply_schema(self) -> None:
        """Apply schema.sql and create the dimension-specific vec table."""
        sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        self.conn.executescript(sql)
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
            f"chunk_rowid INTEGER PRIMARY KEY, embedding FLOAT[{self.embed_dim}])"
        )
        self.conn.commit()

    def _record_cohort(self) -> None:
        """Persist the embedding cohort key; refuse mixing incompatible cohorts."""
        cohort = self.provider.cohort_key()
        row = self.conn.execute(
            "SELECT value FROM embedding_meta WHERE key = 'cohort_key'"
        ).fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO embedding_meta(key, value) VALUES ('cohort_key', ?)",
                (cohort,),
            )
            self.conn.commit()
        elif row["value"] != cohort:
            # Tolerate a cohort change ONLY when the store holds no vectors (e.g.
            # after a full purge): there is nothing to mix, so adopt the new
            # provider's cohort. Otherwise refuse to mix incompatible embeddings.
            vec_count = self.conn.execute(
                "SELECT COUNT(*) AS c FROM vec_chunks"
            ).fetchone()["c"]
            if vec_count == 0:
                # Adopt the new cohort AND rebuild the vector table at the new
                # provider's dimension — the old vec_chunks was created FLOAT[old]
                # and CREATE ... IF NOT EXISTS would otherwise keep the stale dim,
                # breaking inserts when the dimension changed.
                self.conn.execute("DROP TABLE IF EXISTS vec_chunks")
                self.conn.execute(
                    f"CREATE VIRTUAL TABLE vec_chunks USING vec0("
                    f"chunk_rowid INTEGER PRIMARY KEY, embedding FLOAT[{self.embed_dim}])"
                )
                self.conn.execute(
                    "UPDATE embedding_meta SET value = ? WHERE key = 'cohort_key'",
                    (cohort,),
                )
                self.conn.commit()
            else:
                raise ValueError(
                    f"Embedding cohort mismatch: store was built with {row['value']!r} "
                    f"but provider reports {cohort!r}. Reindex before reuse."
                )

    # -- ingest ---------------------------------------------------------------

    def add_observation(
        self,
        text: str,
        *,
        app: str | None = None,
        window: str | None = None,
        url: str | None = None,
        session_id: str | None = None,
        source: str,
        ts: float | None = None,
    ) -> Observation:
        """Record one occurrence of captured content.

        Normalizes and chunks ``text``; performs CHUNK-LEVEL content-hash dedup
        (a unique chunk's text/embedding/index entries are created once); ALWAYS
        inserts a new observation row (occurrences are never deduped). Returns
        the created :class:`Observation`.
        """
        ts = time.time() if ts is None else ts
        blob_hash = ingest.content_hash(text)
        norm = ingest.normalize(text)
        chunks = ingest.chunk(text)

        cur = self.conn
        try:
            return self._add_observation_txn(
                blob_hash, norm, chunks, ts,
                app=app, window=window, url=url, session_id=session_id, source=source,
            )
        except Exception:
            # Atomicity: embeddings are generated mid-transaction; if provider.embed
            # (or any insert) raises, roll back so we never leave blobs/observations/
            # chunks/FTS rows without their vectors (or a dangling open transaction).
            cur.rollback()
            raise

    def _add_observation_txn(
        self,
        blob_hash: str,
        norm: str,
        chunks: list,
        ts: float,
        *,
        app: str | None,
        window: str | None,
        url: str | None,
        session_id: str | None,
        source: str,
    ) -> Observation:
        """Body of :meth:`add_observation`, run inside a rollback-guarded txn."""
        cur = self.conn

        # 1) Blob (deduped).
        cur.execute(
            "INSERT OR IGNORE INTO content_blobs(content_hash, text) VALUES (?, ?)",
            (blob_hash, norm),
        )

        # 2) Observation (NEVER deduped).
        obs_id = uuid.uuid4().hex
        cur.execute(
            "INSERT INTO observations(id, content_hash, ts, app, window, url, session_id, source)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (obs_id, blob_hash, ts, app, window, url, session_id, source),
        )

        # 3) Chunks: deduped GLOBALLY by chunk_hash (normalized chunk text), so an
        #    identical chunk recurring across different windows is stored, embedded,
        #    and indexed exactly once. blob_chunks records the occurrence + span.
        new_chunk_texts: list[str] = []
        new_chunk_rowids: list[int] = []
        for (start, end), ctext in chunks:
            chunk_hash = ingest.content_hash(ctext)
            # Ensure the (deduped) chunk row exists BEFORE the mapping that FKs it.
            existing = cur.execute(
                "SELECT rowid_int FROM chunks WHERE chunk_hash = ?", (chunk_hash,)
            ).fetchone()
            if existing is None:
                rowid_int = cur.execute(
                    "INSERT INTO chunks(chunk_hash, text) VALUES (?, ?)", (chunk_hash, ctext)
                ).lastrowid
                cur.execute(
                    "UPDATE chunks SET rowid_int = ? WHERE chunk_hash = ?", (rowid_int, chunk_hash)
                )
                cur.execute(
                    "INSERT INTO fts_chunks(rowid, text) VALUES (?, ?)", (rowid_int, ctext)
                )
                new_chunk_texts.append(ctext)
                new_chunk_rowids.append(int(rowid_int))
            cur.execute(
                "INSERT OR IGNORE INTO blob_chunks(content_hash, chunk_hash, span_start, span_end)"
                " VALUES (?, ?, ?, ?)",
                (blob_hash, chunk_hash, start, end),
            )

        if new_chunk_texts:
            vectors = self.provider.embed(new_chunk_texts)
            for rowid_int, vector in zip(new_chunk_rowids, vectors):
                cur.execute(
                    "INSERT INTO vec_chunks(chunk_rowid, embedding) VALUES (?, ?)",
                    (int(rowid_int), _serialize_f32(vector)),
                )

        cur.commit()

        return Observation(
            id=obs_id,
            content_hash=blob_hash,
            ts=ts,
            app=app,
            window=window,
            url=url,
            session_id=session_id,
            source=source,
        )

    # -- search ---------------------------------------------------------------

    def search(self, query: str, k: int = 10, *, semantic: bool = True) -> list[SearchHit]:
        """Hybrid search: vector + BM25 -> RRF -> MMR dedup.

        Each surviving hit is resolved back to its most recent observation
        (app/window/ts) so citations are occurrence-aware. ``semantic=False``
        runs BM25 only (no embedding call).
        """
        if not query.strip():
            return []

        pool = max(k * 5, 20)
        rankings: list[list[str]] = []

        bm25_ids = self._bm25(query, pool)
        if bm25_ids:
            rankings.append(bm25_ids)

        if semantic:
            vec_ids = self._vector(query, pool)
            if vec_ids:
                rankings.append(vec_ids)

        if not rankings:
            return []

        fused = rrf(rankings)
        fused = fused[: pool]

        hits = [self._build_hit(rowid_int, score) for rowid_int, score in fused]
        hits = [h for h in hits if h is not None]
        deduped = mmr(hits, k=k)  # type: ignore[arg-type]
        return deduped

    def _bm25(self, query: str, limit: int) -> list[str]:
        """Return chunk rowids ranked by BM25 (best first)."""
        match = self._fts_query(query)
        if not match:
            return []
        rows = self.conn.execute(
            "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH ? ORDER BY bm25(fts_chunks) LIMIT ?",
            (match, limit),
        ).fetchall()
        return [str(r["rowid"]) for r in rows]

    @staticmethod
    def _fts_query(query: str) -> str:
        """Build a safe FTS5 MATCH expression (OR of quoted tokens)."""
        tokens = [t for t in "".join(c if c.isalnum() else " " for c in query).split() if t]
        if not tokens:
            return ""
        return " OR ".join(f'"{t}"' for t in tokens)

    def _vector(self, query: str, limit: int) -> list[str]:
        """Return chunk rowids ranked by vector similarity (best first)."""
        vector = self.provider.embed([query])[0]
        rows = self.conn.execute(
            "SELECT chunk_rowid FROM vec_chunks WHERE embedding MATCH ? "
            "ORDER BY distance LIMIT ?",
            (_serialize_f32(vector), limit),
        ).fetchall()
        return [str(r["chunk_rowid"]) for r in rows]

    def _build_hit(self, rowid_int: str, score: float) -> SearchHit | None:
        """Construct a :class:`SearchHit` and resolve its latest observation.

        A chunk may occur in several blobs; we join through ``blob_chunks`` to the
        most recent observation that contains it, so the citation names a concrete
        occurrence (app/window/ts).
        """
        chunk = self.conn.execute(
            "SELECT chunk_hash, text FROM chunks WHERE rowid_int = ?",
            (int(rowid_int),),
        ).fetchone()
        if chunk is None:
            return None
        obs_row = self.conn.execute(
            "SELECT o.* FROM observations o "
            "JOIN blob_chunks bc ON bc.content_hash = o.content_hash "
            "WHERE bc.chunk_hash = ? ORDER BY o.ts DESC LIMIT 1",
            (chunk["chunk_hash"],),
        ).fetchone()
        observation = self._row_to_observation(obs_row) if obs_row else None
        return SearchHit(
            chunk_id=chunk["chunk_hash"],
            content_hash=obs_row["content_hash"] if obs_row else chunk["chunk_hash"],
            text=chunk["text"],
            score=score,
            observation=observation,
        )

    # -- time-range -----------------------------------------------------------

    def time_range(self, start_ts: float, end_ts: float) -> list[Observation]:
        """Return observations with ``start_ts <= ts <= end_ts`` (range scan).

        This is the non-semantic activity-timeline path ("what did I do
        yesterday"), ordered chronologically.
        """
        rows = self.conn.execute(
            "SELECT * FROM observations WHERE ts >= ? AND ts <= ? ORDER BY ts ASC",
            (start_ts, end_ts),
        ).fetchall()
        return [self._row_to_observation(r) for r in rows]

    def time_range_text(
        self, start_ts: float, end_ts: float, *, max_chars: int = 2000
    ) -> list[tuple[Observation, str]]:
        """Like :meth:`time_range`, but also returns each observation's blob text.

        Joins observations to their deduped ``content_blobs`` body so routines and
        activity summaries can ground in actual captured text (not just app/window
        titles). Each body is truncated to ``max_chars``. The returned text is
        **untrusted captured content** and must be fenced as data by callers.
        """
        rows = self.conn.execute(
            "SELECT o.*, b.text AS blob_text FROM observations o "
            "JOIN content_blobs b ON b.content_hash = o.content_hash "
            "WHERE o.ts >= ? AND o.ts <= ? ORDER BY o.ts ASC",
            (start_ts, end_ts),
        ).fetchall()
        out: list[tuple[Observation, str]] = []
        for r in rows:
            text = r["blob_text"] or ""
            if max_chars and len(text) > max_chars:
                text = text[:max_chars]
            out.append((self._row_to_observation(r), text))
        return out

    @staticmethod
    def _row_to_observation(row) -> Observation:
        """Map a sqlite Row to an :class:`Observation`."""
        return Observation(
            id=row["id"],
            content_hash=row["content_hash"],
            ts=row["ts"],
            app=row["app"],
            window=row["window"],
            url=row["url"],
            session_id=row["session_id"],
            source=row["source"],
        )

    # -- delete ---------------------------------------------------------------

    def delete(self, *, since_ts: float | None = None, all: bool = False) -> int:
        """Delete observations and cascade-clean orphaned content.

        With ``all=True`` removes everything. With ``since_ts`` removes
        observations at/after that timestamp. After removing observations, any
        blob with no remaining observations is deleted along with its chunks,
        FTS entries, and vectors. Returns the number of observations deleted.

        Exactly one of ``all`` / ``since_ts`` should be given.
        """
        cur = self.conn
        if all:
            count = cur.execute("SELECT COUNT(*) AS c FROM observations").fetchone()["c"]
            cur.execute("DELETE FROM observations")
            cur.execute("DELETE FROM blob_chunks")
            cur.execute("DELETE FROM chunks")
            cur.execute("DELETE FROM content_blobs")
            cur.execute("DELETE FROM fts_chunks")
            cur.execute("DELETE FROM vec_chunks")
            # NOTE: we deliberately KEEP embedding_meta. With vec_chunks now empty,
            # a reopen under a different provider is detected as a cohort mismatch
            # on an empty store, which _record_cohort tolerates by rebuilding the
            # vector table at the new dimension and adopting the new cohort. Clearing
            # it here would instead look like a fresh store and skip that rebuild.
            cur.commit()
            return int(count)

        if since_ts is None:
            raise ValueError("delete() requires either all=True or since_ts")

        victims = cur.execute(
            "SELECT id, content_hash FROM observations WHERE ts >= ?", (since_ts,)
        ).fetchall()
        count = len(victims)
        affected_hashes = {r["content_hash"] for r in victims}
        cur.execute("DELETE FROM observations WHERE ts >= ?", (since_ts,))

        # Drop blobs now orphaned (no remaining observations). FK ON DELETE CASCADE
        # removes their blob_chunks mappings.
        for h in affected_hashes:
            remaining = cur.execute(
                "SELECT 1 FROM observations WHERE content_hash = ? LIMIT 1", (h,)
            ).fetchone()
            if remaining is not None:
                continue
            cur.execute("DELETE FROM content_blobs WHERE content_hash = ?", (h,))

        # Reclaim chunks no blob references any more, plus their fts/vec entries.
        orphans = cur.execute(
            "SELECT rowid_int FROM chunks WHERE rowid_int IS NOT NULL "
            "AND chunk_hash NOT IN (SELECT chunk_hash FROM blob_chunks)"
        ).fetchall()
        for r in orphans:
            rid = r["rowid_int"]
            cur.execute("DELETE FROM fts_chunks WHERE rowid = ?", (rid,))
            cur.execute("DELETE FROM vec_chunks WHERE chunk_rowid = ?", (rid,))
        cur.execute(
            "DELETE FROM chunks WHERE chunk_hash NOT IN (SELECT chunk_hash FROM blob_chunks)"
        )

        cur.commit()
        return count

    # -- stats ----------------------------------------------------------------

    def stats(self) -> dict:
        """Return row counts and the recorded embedding cohort key."""
        def count(table: str) -> int:
            return int(self.conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"])

        cohort_row = self.conn.execute(
            "SELECT value FROM embedding_meta WHERE key = 'cohort_key'"
        ).fetchone()
        return {
            "observations": count("observations"),
            "blobs": count("content_blobs"),
            "chunks": count("chunks"),
            "vectors": count("vec_chunks"),
            "embed_dim": self.embed_dim,
            "cohort_key": cohort_row["value"] if cohort_row else None,
            "encryption_enabled": self.settings.encryption_enabled,
        }

    def close(self) -> None:
        """Close the underlying connection."""
        self.conn.close()


__all__ = ["MemoryStore"]
