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

import json
import logging
import math
import re
import sqlite3
import struct
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from openbird.config import Settings, get_settings
from openbird.llm.base import LLMProviderProtocol
from openbird.llm.provider import create_llm_provider
from openbird.memory import ingest
from openbird.memory.migrations import (
    ensure_schema_version,
    preflight_legacy_shape_guard,
)
from openbird.memory.search import mmr, rrf
from openbird.storage.crypto import mapping_row_factory, open_encrypted_db
from openbird.types import Observation, SearchHit

_log = logging.getLogger("openbird.memory")


def _log_rerank_skip(reason: str) -> None:
    """Log a reranker fallback with a STRUCTURED, content-free reason code.

    The rerank server may echo documents in its error body, so we never log the
    exception text, response body, query, or chunk text — only the short reason
    code (``timeout|transport|http_<status>|bad_response|non_finite|error``).
    Search continues on the RRF order.
    """
    _log.info("rerank_skipped reason=%s (fell back to RRF order)", reason)


def _log_vector_skip(reason: str) -> None:
    """Log a vector/embedding fallback with a STRUCTURED, content-free reason code.

    The embedding provider's error may echo the query or contents in its message,
    so we never log the exception text, response body, or query — only the short
    exception-type ``reason`` code. Search continues on BM25-only ranking, exactly
    as :func:`_log_rerank_skip` keeps search alive when a reranker is down.
    """
    _log.info("vector_skipped reason=%s (fell back to BM25-only ranking)", reason)


class EmbeddingCohortMismatch(ValueError):
    """Opening a populated store whose embedding cohort differs from the provider's.

    Raised by :meth:`MemoryStore._record_cohort` when a store already holds vectors
    embedded under one cohort and the configured provider reports a different one
    (e.g. the user switched ``OPENBIRD_EMBED_MODEL`` or its dimension). The remedy is
    ``openbird reindex``, which re-embeds every chunk under the new cohort. Subclasses
    :class:`ValueError` so existing ``except ValueError`` handlers still catch it,
    while carrying the ``stored`` and ``current`` cohort keys so callers can render a
    precise, content-free recovery hint.
    """

    def __init__(self, stored: str, current: str) -> None:
        self.stored = stored
        self.current = current
        super().__init__(
            f"Embedding cohort mismatch: store was built with {stored!r} "
            f"but provider reports {current!r}. Reindex before reuse."
        )


@dataclass(frozen=True)
class SessionSummary:
    """One capture session within a day's timeline (for the Today/day view).

    ``session_id`` is the real (nullable) episodic id — legacy rows captured
    before episodic sessions have ``None`` and are NOT collapsed together (each
    falls in its own bucket; see :meth:`MemoryStore.day_sessions`).
    """

    session_id: str | None
    app: str | None
    start_ts: float
    end_ts: float
    count: int
    # The session's representative window title (the most-frequent non-empty
    # ``window`` among its observations; ties broken by the latest timestamp).
    # Powers the Today timeline card title (e.g. "rag.py — openbird"). ``None``
    # when no observation in the session carried a window title.
    window: str | None = None

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")
# Wait up to this long on a busy lock for the :memory: branch too; on-disk
# connections get this from storage.crypto.
_BUSY_TIMEOUT_MS = 5000


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
        reranker: object | None = None,
    ) -> None:
        """Open the store, load sqlite-vec, and apply the schema.

        Args:
            db_path: Override DB path (``":memory:"`` is supported for tests).
            settings: Settings; defaults to :func:`get_settings`.
            provider: LLM provider for embeddings; defaults to the configured
                provider implementation. Injectable so tests can mock embeddings.
            reranker: Optional cross-encoder reranker (``rerank(query, docs) ->
                scores``). ``None`` builds from settings (disabled unless
                ``rerank_model`` is set); injectable so tests can supply a fake.
        """
        self.settings = settings or get_settings()
        self.provider = provider or create_llm_provider(self.settings)
        self.embed_dim = self.settings.embed_dim
        if reranker is None:
            from openbird.llm.rerank import build_reranker, rerank_is_remote

            # Fail closed: a remote (non-loopback) rerank host sends query+chunk
            # text off-device, so auto-building it without cloud opt-in must refuse
            # — even on the store-direct path where a caller injected `provider`
            # and skipped the CLI/provider cloud gate. Explicitly injected
            # rerankers are a deliberate caller choice and bypass this.
            if rerank_is_remote(self.settings) and not self.settings.allow_cloud:
                from openbird.llm.provider import CloudOptInRequired

                raise CloudOptInRequired({"rerank": self.settings.rerank_model})
            reranker = build_reranker(self.settings)
        self.reranker = reranker

        resolved = db_path if db_path is not None else self.settings.db_path
        if resolved == ":memory:":
            import sqlite_vec

            self.conn = __import__("sqlite3").connect(":memory:")
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            # Queue on a busy lock instead of erroring immediately. Harmless
            # for a single-connection in-memory DB; keeps behavior uniform.
            self.conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        else:
            self.conn = open_encrypted_db(resolved, settings=self.settings)

        self.conn.row_factory = mapping_row_factory
        # Take explicit control of transaction boundaries: we use
        # BEGIN IMMEDIATE / COMMIT / ROLLBACK ourselves so the embedding network
        # call happens OUTSIDE the write lock and multi-statement deletes are
        # atomic. autocommit (isolation_level=None) disables sqlite3's implicit
        # BEGIN-before-DML so a stray write can't silently open a long txn.
        try:
            self.conn.isolation_level = None
        except AttributeError:
            # sqlcipher3's dbapi2 connection also exposes isolation_level; if a
            # backend doesn't, fall back to manual COMMIT (still correct).
            pass
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
        """Apply schema.sql, create the vec table, and reconcile the version.

        ``schema.sql`` is idempotent (CREATE ... IF NOT EXISTS), so applying it on
        every open is safe for both fresh and existing DBs. After the baseline
        shape exists we run :func:`ensure_schema_version`, which stamps
        ``PRAGMA user_version`` and runs any pending forward migrations — and
        refuses to open a DB written by a newer build.

        BEFORE applying schema.sql we run :func:`preflight_legacy_shape_guard`: a
        legacy unstamped DB whose tables don't match the v1 shape would otherwise
        make schema.sql's ``CREATE INDEX ... ON observations(ts)`` raise a cryptic
        ``no such column: ts`` (because ``CREATE TABLE IF NOT EXISTS`` skips the
        wrong-shaped table). The guard turns that into a clear migration error.
        """
        # Reject a partial / pre-v1 / foreign legacy DB up front, before any of
        # schema.sql's column-dependent DDL (e.g. idx_observations_ts) can fail
        # with a raw OperationalError.
        preflight_legacy_shape_guard(self.conn)
        sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        # executescript runs in autocommit (it issues its own COMMIT); fine since
        # we manage explicit transactions elsewhere.
        self.conn.executescript(sql)
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
            f"chunk_rowid INTEGER PRIMARY KEY, embedding FLOAT[{self.embed_dim}])"
        )
        ensure_schema_version(self.conn)

    def _begin(self) -> None:
        """Open a write transaction, grabbing the writer lock up front.

        ``BEGIN IMMEDIATE`` acquires the RESERVED lock immediately rather than
        lazily on first write, so the writer never has to upgrade a read lock
        mid-transaction (which is the classic SQLite deadlock/`database is
        locked` source). Combined with ``isolation_level = None`` this gives us
        precise, short write windows.
        """
        self.conn.execute("BEGIN IMMEDIATE")

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
            else:
                raise EmbeddingCohortMismatch(row["value"], cohort)

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
        span_id: str | None = None,
    ) -> Observation:
        """Record one occurrence of captured content.

        Normalizes and chunks ``text``; performs CHUNK-LEVEL content-hash dedup
        (a unique chunk's text/embedding/index entries are created once); ALWAYS
        inserts a new observation row (occurrences are never deduped). Returns
        the created :class:`Observation`.

        ``span_id`` is EVENT-SCOPED: the caller (capture daemon) resolves the
        activity span at trigger-handling time and passes the immutable id in —
        this method never queries a "current open span" (the span row was
        committed by ``open_span`` before this call, so the FK holds).
        """
        ts = time.time() if ts is None else ts
        blob_hash = ingest.content_hash(text)
        norm = ingest.normalize(text)
        chunks = ingest.chunk(text)

        # -- Phase 1: embed OUTSIDE any write transaction -------------------
        # Previously provider.embed() ran INSIDE the write txn, so a slow/wedged
        # Ollama held the single WAL writer slot across a network round-trip and a
        # concurrent reader/writer hit `database is locked`. We now (a) read which
        # chunk_hashes are already present (no lock held), then (b) embed only the
        # genuinely-new chunk texts with NO write lock held, and finally (c) open a
        # short INSERT-only transaction.
        chunk_hashes = [ingest.content_hash(ctext) for _span, ctext in chunks]
        # Map each unique new chunk_hash -> its text (dedup within THIS document).
        unique_new: dict[str, str] = {}
        for (_span, ctext), chash in zip(chunks, chunk_hashes):
            if chash in unique_new:
                continue
            existing = self.conn.execute(
                "SELECT 1 FROM chunks WHERE chunk_hash = ? LIMIT 1", (chash,)
            ).fetchone()
            if existing is None:
                unique_new[chash] = ctext

        embeddings: dict[str, bytes] = {}
        if unique_new:
            new_hashes = list(unique_new)
            vectors = self.provider.embed([unique_new[h] for h in new_hashes])
            embeddings = {h: _serialize_f32(v) for h, v in zip(new_hashes, vectors)}

        # -- Phase 2: short INSERT-only write transaction -----------------------
        try:
            self._begin()
            obs = self._write_observation(
                blob_hash, norm, chunks, chunk_hashes, embeddings, ts,
                app=app, window=window, url=url, session_id=session_id, source=source,
                span_id=span_id,
            )
            self.conn.commit()
            return obs
        except Exception:
            # Roll back so we never leave blobs/observations/chunks/FTS rows
            # without their vectors (or a dangling open transaction).
            self.conn.rollback()
            raise

    def _write_observation(
        self,
        blob_hash: str,
        norm: str,
        chunks: list,
        chunk_hashes: list[str],
        embeddings: dict[str, bytes],
        ts: float,
        *,
        app: str | None,
        window: str | None,
        url: str | None,
        session_id: str | None,
        source: str,
        span_id: str | None = None,
    ) -> Observation:
        """Insert blob/observation/chunks/index rows. Runs inside an open txn.

        ``embeddings`` maps chunk_hash -> packed vector for chunks that did NOT
        exist at probe time. A concurrent writer may have created a chunk between
        the probe and this transaction (TOCTOU); ``INSERT OR IGNORE`` + a rowcount
        check makes us insert the FTS/vec rows ONLY for chunks we actually create,
        so a chunk is never double-embedded or double-indexed.
        """
        cur = self.conn

        # 1) Blob (deduped).
        cur.execute(
            "INSERT OR IGNORE INTO content_blobs(content_hash, text) VALUES (?, ?)",
            (blob_hash, norm),
        )

        # 2) Observation (NEVER deduped).
        obs_id = uuid.uuid4().hex
        cur.execute(
            "INSERT INTO observations("
            "id, content_hash, ts, app, window, url, session_id, source, span_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (obs_id, blob_hash, ts, app, window, url, session_id, source, span_id),
        )

        # 3) Chunks: deduped GLOBALLY by chunk_hash (normalized chunk text), so an
        #    identical chunk recurring across different windows is stored, embedded,
        #    and indexed exactly once. blob_chunks records the occurrence + span.
        for ((start, end), ctext), chunk_hash in zip(chunks, chunk_hashes):
            # Atomically create-or-skip the chunk row. rowcount tells us whether
            # WE inserted it (1) or it already existed / a concurrent writer won (0).
            ins = cur.execute(
                "INSERT OR IGNORE INTO chunks(chunk_hash, text) VALUES (?, ?)",
                (chunk_hash, ctext),
            )
            if ins.rowcount == 1:
                # We created it: assign the stable integer key and index it.
                rowid_int = cur.execute(
                    "SELECT rowid FROM chunks WHERE chunk_hash = ?", (chunk_hash,)
                ).fetchone()["rowid"]
                cur.execute(
                    "UPDATE chunks SET rowid_int = ? WHERE chunk_hash = ?",
                    (rowid_int, chunk_hash),
                )
                cur.execute(
                    "INSERT INTO fts_chunks(rowid, text) VALUES (?, ?)", (rowid_int, ctext)
                )
                vector = embeddings.get(chunk_hash)
                if vector is None:
                    # Defensive: a chunk we created but didn't pre-embed (e.g. it
                    # appeared to exist at probe time then vanished). Embed it now;
                    # this is the rare slow path, not the common one.
                    vector = _serialize_f32(self.provider.embed([ctext])[0])
                cur.execute(
                    "INSERT INTO vec_chunks(chunk_rowid, embedding) VALUES (?, ?)",
                    (int(rowid_int), vector),
                )
            cur.execute(
                "INSERT OR IGNORE INTO blob_chunks(content_hash, chunk_hash, span_start, span_end)"
                " VALUES (?, ?, ?, ?)",
                (blob_hash, chunk_hash, start, end),
            )

        return Observation(
            id=obs_id,
            content_hash=blob_hash,
            ts=ts,
            app=app,
            window=window,
            url=url,
            session_id=session_id,
            source=source,
            span_id=span_id,
        )

    # -- search ---------------------------------------------------------------

    def search(self, query: str, k: int = 10, *, semantic: bool = True) -> list[SearchHit]:
        """Hybrid search: vector + BM25 -> RRF -> (optional rerank) -> MMR dedup.

        Each surviving hit is resolved back to its most recent observation
        (app/window/ts) so citations are occurrence-aware. ``semantic=False``
        runs BM25 only (no embedding call). When ``semantic=True`` but the
        embedding provider fails (Ollama down, timeout, transport, ...), the
        vector stage is skipped and search degrades to BM25-only ranking rather
        than crashing. When a cross-encoder reranker is configured it reorders the
        fused candidates by query-relevance before the MMR diversity pass; any
        reranker failure falls back to the RRF order so search never breaks.
        """
        if not query.strip():
            return []

        pool = max(k * 5, 20)
        rankings: list[list[str]] = []

        bm25_ids = self._bm25(query, pool)
        if bm25_ids:
            rankings.append(bm25_ids)

        if semantic:
            try:
                vec_ids = self._vector(query, pool)
            except Exception as exc:  # noqa: BLE001 - embedding failure must never break search
                # An embedding/provider failure (Ollama down, timeout, transport,
                # dim mismatch, ...) must degrade to BM25-only ranking, not discard
                # the already-successful BM25 hits. Mirrors _rerank's RRF fallback.
                _log_vector_skip(type(exc).__name__)
            else:
                if vec_ids:
                    rankings.append(vec_ids)

        if not rankings:
            return []

        fused = rrf(rankings)
        fused = fused[: pool]

        hits = [self._build_hit(rowid_int, score) for rowid_int, score in fused]
        hits = [h for h in hits if h is not None]
        hits = self._rerank(query, hits)  # type: ignore[arg-type]
        deduped = mmr(hits, k=k)  # type: ignore[arg-type]
        return deduped

    def _rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        """Reorder fused hits by a cross-encoder reranker; fall back to RRF order.

        No-op when no reranker is configured. Cross-encoder scores are uncalibrated
        (logits, negatives, narrow probabilities), so they are NOT assigned to
        ``SearchHit.score`` raw — MMR uses ``score`` directly. We min-max normalize
        them to ``[0, 1]`` over THIS candidate set (all-equal keeps the RRF order),
        reject non-finite, and tie-break by the original RRF position. ANY reranker
        failure logs a structured, content-free reason and returns the RRF order —
        search must never break because a reranker is down.
        """
        if self.reranker is None or len(hits) < 2:
            return hits
        from openbird.llm.rerank import RerankError

        try:
            scores = self.reranker.rerank(query, [h.text for h in hits])
        except RerankError as exc:
            _log_rerank_skip(exc.reason)
            return hits
        except Exception:  # noqa: BLE001 - a reranker must never break search
            _log_rerank_skip("error")
            return hits
        if not isinstance(scores, list) or len(scores) != len(hits):
            _log_rerank_skip("bad_response")
            return hits
        if any(not isinstance(s, (int, float)) or not math.isfinite(s) for s in scores):
            _log_rerank_skip("non_finite")
            return hits
        lo, hi = min(scores), max(scores)
        if hi <= lo:
            return hits  # all-equal -> reranker added no signal; keep RRF order
        span = hi - lo
        # Pair each hit with (normalized score, original index) and sort by score
        # desc, tie-breaking by the original RRF position for determinism.
        order = sorted(
            range(len(hits)),
            key=lambda i: (-(scores[i] - lo) / span, i),
        )
        return [
            hits[i].model_copy(update={"score": (scores[i] - lo) / span})
            for i in order
        ]

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

    def day_sessions(
        self, start_ts: float, end_ts: float, *, source: str = "capture"
    ) -> list[SessionSummary]:
        """Per-session activity summary for the ``[start_ts, end_ts]`` window.

        Powers the Today/day-view timeline: one row per capture session with its
        app, span, and observation count. Restricted to ``source`` (default
        ``"capture"``) so ingested files / MCP reads don't masquerade as capture
        sessions. Grouped by a coalesced key so legacy rows with a ``NULL``
        session_id each bucket on their own ``id`` (SQLite groups NULLs together,
        which would otherwise merge a whole day into one false session). Pure
        indexed read — no embedding.

        Each summary also carries a representative ``window`` title: the most
        frequent non-empty window in the session (ties → latest timestamp). The
        window pick reuses the EXACT same bucket key as the session grouping (a
        shared ``tagged`` CTE), so a legacy NULL-session row never borrows another
        row's window. Computed in one pass — no per-session subquery rescan.
        """
        rows = self.conn.execute(
            "WITH tagged AS ("
            "  SELECT (CASE WHEN session_id IS NULL THEN id ELSE session_id END) AS bucket, "
            "         session_id, app, ts, window "
            "  FROM observations WHERE ts >= ? AND ts <= ? AND source = ?"
            "), "
            "agg AS ("
            "  SELECT bucket, session_id, app, MIN(ts) AS start_ts, MAX(ts) AS end_ts, "
            "         COUNT(*) AS cnt FROM tagged GROUP BY bucket, app"
            "), "
            "win AS ("
            "  SELECT bucket, app, window, COUNT(*) AS wcnt, MAX(ts) AS wlast "
            "  FROM tagged WHERE window IS NOT NULL AND window != '' "
            "  GROUP BY bucket, app, window"
            "), "
            "win_ranked AS ("
            "  SELECT bucket, app, window, "
            "         ROW_NUMBER() OVER ("
            "           PARTITION BY bucket, app ORDER BY wcnt DESC, wlast DESC"
            "         ) AS rn FROM win"
            ") "
            "SELECT a.session_id, a.app, a.start_ts, a.end_ts, a.cnt, wr.window AS window "
            "FROM agg a "
            "LEFT JOIN win_ranked wr "
            "  ON wr.bucket IS a.bucket AND wr.app IS a.app AND wr.rn = 1 "
            "ORDER BY a.start_ts ASC",
            (start_ts, end_ts, source),
        ).fetchall()
        return [
            SessionSummary(
                session_id=r["session_id"],
                app=r["app"],
                start_ts=r["start_ts"],
                end_ts=r["end_ts"],
                count=r["cnt"],
                window=r["window"],
            )
            for r in rows
        ]

    def active_seconds(
        self, start_ts: float, end_ts: float, gap_seconds: float, *, source: str = "capture"
    ) -> float:
        """Gap-capped active time across the window: the sum of deltas between
        consecutive observations, each clipped to ``gap_seconds`` so idle gaps
        don't inflate it. A lone observation contributes 0 (no engaged span). This
        is a better "active" stat than ``sum(session end - start)``, which
        undercounts singletons and can overlap. Restricted to ``source`` (default
        ``"capture"``) so it measures capture activity, matching the timeline.
        """
        row = self.conn.execute(
            "WITH ordered AS ("
            "  SELECT ts, LAG(ts) OVER (ORDER BY ts) AS prev "
            "  FROM observations WHERE ts >= ? AND ts <= ? AND source = ?"
            ") SELECT COALESCE(SUM(MIN(ts - prev, ?)), 0) AS active "
            "FROM ordered WHERE prev IS NOT NULL",
            (start_ts, end_ts, source, gap_seconds),
        ).fetchone()
        return float(row["active"]) if row and row["active"] is not None else 0.0

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
        self,
        start_ts: float,
        end_ts: float,
        *,
        max_chars: int = 2000,
        source: str | None = None,
    ) -> list[tuple[Observation, str]]:
        """Like :meth:`time_range`, but also returns each observation's blob text.

        Joins observations to their deduped ``content_blobs`` body so routines and
        activity summaries can ground in actual captured text (not just app/window
        titles). Each body is truncated to ``max_chars``. When ``source`` is given,
        restricts to that source (the Today briefing passes ``"capture"`` to match
        its timeline; scheduled routines pass ``None`` for all sources). The
        returned text is **untrusted captured content** and must be fenced as data
        by callers.
        """
        sql = (
            "SELECT o.*, b.text AS blob_text FROM observations o "
            "JOIN content_blobs b ON b.content_hash = o.content_hash "
            "WHERE o.ts >= ? AND o.ts <= ?"
        )
        params: list[object] = [start_ts, end_ts]
        if source is not None:
            sql += " AND o.source = ?"
            params.append(source)
        sql += " ORDER BY o.ts ASC"
        rows = self.conn.execute(sql, params).fetchall()
        out: list[tuple[Observation, str]] = []
        for r in rows:
            text = r["blob_text"] or ""
            if max_chars and len(text) > max_chars:
                text = text[:max_chars]
            out.append((self._row_to_observation(r), text))
        return out

    def export_observations(
        self,
        *,
        since_ts: float | None = None,
        until_ts: float | None = None,
        source: str | None = None,
    ) -> Iterator[dict]:
        """Return decrypted observations plus blob text for explicit user export.

        Export is a local maintenance read: it never embeds and never contacts a
        provider. Callers are responsible for warning that the destination path
        may leave OpenBird's retention boundary (for example via iCloud/Dropbox).
        """
        sql = (
            "SELECT o.*, b.text AS text FROM observations o "
            "JOIN content_blobs b ON b.content_hash = o.content_hash WHERE 1=1"
        )
        params: list[object] = []
        if since_ts is not None:
            sql += " AND o.ts >= ?"
            params.append(since_ts)
        if until_ts is not None:
            sql += " AND o.ts <= ?"
            params.append(until_ts)
        if source is not None:
            sql += " AND o.source = ?"
            params.append(source)
        sql += " ORDER BY o.ts ASC"
        for row in self.conn.execute(sql, params):
            yield {
                "id": row["id"],
                "content_hash": row["content_hash"],
                "ts": row["ts"],
                "app": row["app"],
                "window": row["window"],
                "url": row["url"],
                "session_id": row["session_id"],
                "source": row["source"],
                "text": row["text"],
            }

    @staticmethod
    def _row_to_observation(row) -> Observation:
        """Map a sqlite Row to an :class:`Observation`.

        ``span_id`` is guarded: a SELECT that predates the v4 column (or a
        projection that omits it) must not break readback — it degrades to
        ``None`` rather than raising.
        """
        try:
            span_id = row["span_id"]
        except (KeyError, IndexError):
            span_id = None
        return Observation(
            id=row["id"],
            content_hash=row["content_hash"],
            ts=row["ts"],
            app=row["app"],
            window=row["window"],
            url=row["url"],
            session_id=row["session_id"],
            source=row["source"],
            span_id=span_id,
        )

    # -- activity spans (Phase B) ----------------------------------------------

    def open_span(
        self,
        *,
        epoch_id: str,
        start_ts: float,
        end_ts: float,
        bundle_id: str | None,
        detail_tier: int,
        window: str | None = None,
        url_host: str | None = None,
        identity_key: str | None = None,
        afk: bool = False,
        reason: str | None = None,
        span_id: str | None = None,
    ) -> str:
        """Insert one activity span and return its id (own short transaction).

        Enforces the tier contract in Python BEFORE SQL (fail-closed, testable
        error) in addition to the schema CHECKs: tier 0 must carry a reason from
        the closed enum and NO window/url_host/identity_key; tier 1 carries no
        reason. Metadata only — nothing here is captured content (window titles
        arrive already scrubbed by the caller).
        """
        from openbird.capture.redact import SPAN_REASONS, SPAN_TIER_COARSE, SPAN_TIER_FULL

        if detail_tier == SPAN_TIER_COARSE:
            if reason not in SPAN_REASONS:
                raise ValueError("coarse span requires a reason from SPAN_REASONS")
            if window is not None or url_host is not None or identity_key is not None:
                raise ValueError("coarse span must not carry window/url_host/identity_key")
        elif detail_tier == SPAN_TIER_FULL:
            if reason is not None:
                raise ValueError("full span must not carry a reason")
        else:
            raise ValueError("detail_tier must be 0 or 1")
        if not (end_ts >= start_ts):
            raise ValueError("span end_ts must be >= start_ts")

        new_id = span_id or uuid.uuid4().hex
        try:
            self._begin()
            self.conn.execute(
                "INSERT INTO activity_spans("
                "span_id, epoch_id, start_ts, end_ts, bundle_id, app, detail_tier, "
                "window, url_host, identity_key, afk, meeting, reason"
                ") VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, 0, ?)",
                (
                    new_id,
                    epoch_id,
                    float(start_ts),
                    float(end_ts),
                    bundle_id,
                    int(detail_tier),
                    window,
                    url_host,
                    identity_key,
                    1 if afk else 0,
                    reason,
                ),
            )
            self.conn.commit()
            return new_id
        except Exception:
            self.conn.rollback()
            raise

    def extend_span(self, span_id: str, end_ts: float) -> None:
        """Advance a span's end (monotone: never regresses on stale updates)."""
        try:
            self._begin()
            self.conn.execute(
                "UPDATE activity_spans SET end_ts = MAX(end_ts, ?) WHERE span_id = ?",
                (float(end_ts), span_id),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def close_span(self, span_id: str, end_ts: float) -> None:
        """Set a span's FINAL end (set-exact, floored at start_ts).

        Unlike :meth:`extend_span` (monotone MAX for stale mid-span updates),
        closing may TRUNCATE: an AFK boundary is backdated to when input
        actually stopped, which can precede a flushed end_ts (idle-tick frames
        captured after the user left). There is deliberately NO open/closed
        status column: the tracker is the single writer, and a crash leaves
        end_ts at the last flush — which is the correct closed value (never
        "now").
        """
        try:
            self._begin()
            self.conn.execute(
                "UPDATE activity_spans SET end_ts = MAX(start_ts, ?) "
                "WHERE span_id = ?",
                (float(end_ts), span_id),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def spans_in_range(self, start_ts: float, end_ts: float) -> list[dict]:
        """Return spans OVERLAPPING [start_ts, end_ts], ordered by start."""
        rows = self.conn.execute(
            "SELECT * FROM activity_spans WHERE start_ts <= ? AND end_ts >= ? "
            "ORDER BY start_ts",
            (float(end_ts), float(start_ts)),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- day memories ---------------------------------------------------------

    def save_day_memory(
        self,
        *,
        local_date: str,
        source_scope: str,
        extractor_version: str,
        payload: dict,
        source_ids: list[str],
        span_ids: list[str] | None = None,
        generated_at: float | None = None,
    ) -> dict:
        """Persist one deterministic daily memory and its source dependencies.

        Rebuild semantics are current-row only: any existing row for
        ``(local_date, source_scope)`` is deleted first, which cascades its old
        dependency rows. The new parent and dependencies are inserted in the same
        transaction. Payloads are derived sensitive data, so callers must not log
        them; this method logs nothing.
        """
        day_memory_id = uuid.uuid4().hex
        generated = time.time() if generated_at is None else generated_at
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        unique_source_ids = sorted(set(source_ids))
        unique_span_ids = sorted(set(span_ids or []))
        try:
            self._begin()
            self.conn.execute(
                "DELETE FROM day_memories WHERE local_date = ? AND source_scope = ?",
                (local_date, source_scope),
            )
            self.conn.execute(
                "INSERT INTO day_memories("
                "id, local_date, source_scope, extractor_version, generated_at, "
                "payload_json, source_count"
                ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    day_memory_id,
                    local_date,
                    source_scope,
                    extractor_version,
                    generated,
                    payload_json,
                    len(unique_source_ids) + len(unique_span_ids),
                ),
            )
            self.conn.executemany(
                "INSERT INTO day_memory_source_refs("
                "day_memory_id, source_kind, source_id) VALUES (?, ?, ?)",
                [(day_memory_id, "observation", i) for i in unique_source_ids]
                + [(day_memory_id, "span", i) for i in unique_span_ids],
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return self.get_day_memory(local_date=local_date, source_scope=source_scope) or {}

    def ensure_day_memory(
        self,
        *,
        local_date: str,
        start_ts: float,
        end_ts: float,
        day_offset: int,
        source_scope: str = "capture",
        force: bool = False,
    ) -> dict:
        """Return a fresh persisted day memory for ``local_date``.

        This is the safe reader entrypoint. It compares the stored extractor
        version and source fingerprint against the current source rows under a
        ``BEGIN IMMEDIATE`` write lock, so concurrent readers rebuilding today's
        open day converge on one row rather than racing a stale cache.
        """
        from openbird.day_memory import (
            EXTRACTOR_VERSION,
            build_day_memory,
            span_fingerprint_for_spans,
        )

        attempts = 3
        for attempt in range(attempts):
            try:
                self._begin()
                rows = self.time_range_text(start_ts, end_ts, source=source_scope)
                fingerprint = self.day_memory_source_fingerprint_from_rows(rows)
                spans = self.spans_in_range(start_ts, end_ts)
                span_fp = span_fingerprint_for_spans(spans)
                existing = self._get_day_memory_unchecked(
                    local_date=local_date, source_scope=source_scope
                )
                if (
                    not force
                    and existing is not None
                    and existing["extractor_version"] == EXTRACTOR_VERSION
                    and existing["payload"].get("source_fingerprint") == fingerprint
                    # Span freshness: an EXTENDED span fires no delete trigger,
                    # so the fingerprint (which includes end_ts) catches it.
                    and existing["payload"].get("span_fingerprint") == span_fp
                ):
                    self.conn.commit()
                    return existing

                built = build_day_memory(
                    rows,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    day_offset=day_offset,
                    source_scope=source_scope,
                    gap_seconds=self.settings.session_gap_seconds,
                    source_fingerprint=fingerprint,
                    as_of=min(end_ts, time.time()),
                    spans=spans,
                )
                day_memory_id = uuid.uuid4().hex
                generated = time.time()
                payload_json = json.dumps(
                    built.payload, ensure_ascii=False, sort_keys=True
                )
                unique_source_ids = sorted(set(built.source_ids))
                unique_span_ids = sorted(set(built.span_ids))
                self.conn.execute(
                    "DELETE FROM day_memories WHERE local_date = ? AND source_scope = ?",
                    (local_date, source_scope),
                )
                self.conn.execute(
                    "INSERT INTO day_memories("
                    "id, local_date, source_scope, extractor_version, generated_at, "
                    "payload_json, source_count"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        day_memory_id,
                        local_date,
                        source_scope,
                        EXTRACTOR_VERSION,
                        generated,
                        payload_json,
                        len(unique_source_ids) + len(unique_span_ids),
                    ),
                )
                self.conn.executemany(
                    "INSERT INTO day_memory_source_refs("
                    "day_memory_id, source_kind, source_id) VALUES (?, ?, ?)",
                    [
                        (day_memory_id, "observation", obs_id)
                        for obs_id in unique_source_ids
                    ]
                    + [(day_memory_id, "span", sid) for sid in unique_span_ids],
                )
                self.conn.commit()
                return self._get_day_memory_unchecked(
                    local_date=local_date, source_scope=source_scope
                ) or {}
            except self._retryable_db_errors():
                self.conn.rollback()
                if attempt + 1 >= attempts:
                    raise
                time.sleep(0.05 * (attempt + 1))
            except Exception:
                self.conn.rollback()
                raise
        raise RuntimeError("failed to ensure day memory")

    def _retryable_db_errors(self) -> tuple[type[BaseException], ...]:
        """Return lock/unique exception classes for the active DB-API backend."""
        errors: set[type[BaseException]] = {
            sqlite3.OperationalError,
            sqlite3.IntegrityError,
        }
        try:
            module = __import__(
                type(self.conn).__module__,
                fromlist=["OperationalError", "IntegrityError"],
            )
        except ImportError:
            return tuple(errors)
        for name in ("OperationalError", "IntegrityError"):
            exc = getattr(module, name, None)
            if isinstance(exc, type) and issubclass(exc, BaseException):
                errors.add(exc)
        return tuple(errors)

    @staticmethod
    def day_memory_source_fingerprint_from_rows(
        rows: list[tuple[Observation, str]]
    ) -> dict:
        from openbird.day_memory import source_fingerprint_for_rows

        return source_fingerprint_for_rows(rows)

    def get_day_memory(self, *, local_date: str, source_scope: str = "capture") -> dict | None:
        """Return a stored day memory, or ``None`` when it has not been built."""
        return self._get_day_memory_unchecked(
            local_date=local_date, source_scope=source_scope
        )

    def _get_day_memory_unchecked(
        self, *, local_date: str, source_scope: str = "capture"
    ) -> dict | None:
        """Return a stored day memory without checking source freshness."""
        row = self.conn.execute(
            "SELECT * FROM day_memories WHERE local_date = ? AND source_scope = ?",
            (local_date, source_scope),
        ).fetchone()
        if row is None:
            return None
        ref_rows = self.conn.execute(
            "SELECT source_kind, source_id FROM day_memory_source_refs "
            "WHERE day_memory_id = ? ORDER BY source_kind, source_id",
            (row["id"],),
        ).fetchall()
        return {
            "id": row["id"],
            "local_date": row["local_date"],
            "source_scope": row["source_scope"],
            "extractor_version": row["extractor_version"],
            "generated_at": row["generated_at"],
            "source_count": row["source_count"],
            # Back-compat key: observation refs only, as before.
            "source_ids": [
                r["source_id"] for r in ref_rows if r["source_kind"] == "observation"
            ],
            "span_ids": [
                r["source_id"] for r in ref_rows if r["source_kind"] == "span"
            ],
            "payload": json.loads(row["payload_json"]),
        }

    # -- reasoning send ledger ------------------------------------------------

    def record_reasoning_send(
        self,
        *,
        feature: str,
        packet_route: str | None,
        reasoning_route: str | None,
        egress: str,
        route_class: str,
        provider_family: str,
        model: str | None,
        packet_hash: str | None,
        packet_bytes: int | None,
        selected_source_count: int,
        citation_count: int,
        excluded_observations: int,
        excluded_by: dict[str, int],
        outcome: str,
        error_kind: str | None = None,
        deletion_caveat: str = (
            "This redacted local audit row may outlive selective source deletion; "
            "full purge removes it."
        ),
    ) -> dict:
        """Persist redacted metadata for a remote reasoning packet send attempt.

        The ledger intentionally stores only counts, route metadata, and a
        packet-content hash. It must never receive raw packet JSON, question text,
        answer text, snippets, source IDs, citation IDs, titles, URLs, or
        configured exclusion names.
        """
        row_id = uuid.uuid4().hex
        created_at = time.time()
        safe_excluded_by: dict[str, int] = {}
        for key, value in sorted((excluded_by or {}).items()):
            reason = str(key)
            if reason not in {"app", "source", "observation_id"}:
                continue
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            if count < 0:
                continue
            safe_excluded_by[reason] = count
        safe_excluded_observations = max(0, int(excluded_observations))
        safe_error_kind: str | None = None
        if error_kind is not None:
            candidate = str(error_kind).strip()
            safe_error_kind = (
                candidate
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}", candidate)
                else "error"
            )
        try:
            self._begin()
            self.conn.execute(
                """
                INSERT INTO reasoning_send_ledger(
                    id, created_at, feature, packet_route, reasoning_route, egress,
                    route_class, provider_family, model, packet_hash, packet_bytes,
                    selected_source_count, citation_count, excluded_observations,
                    excluded_by_json, outcome, error_kind, deletion_caveat
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    created_at,
                    feature,
                    packet_route,
                    reasoning_route,
                    egress,
                    route_class,
                    provider_family,
                    model,
                    packet_hash,
                    packet_bytes,
                    int(selected_source_count),
                    int(citation_count),
                    safe_excluded_observations,
                    json.dumps(safe_excluded_by, sort_keys=True, separators=(",", ":")),
                    outcome,
                    safe_error_kind,
                    deletion_caveat,
                ),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return self._reasoning_send_ledger_row(row_id)

    def list_reasoning_send_ledger(self, *, limit: int = 50) -> list[dict]:
        """Return newest redacted reasoning-send ledger rows."""
        rows = self.conn.execute(
            """
            SELECT * FROM reasoning_send_ledger
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (max(0, int(limit)),),
        ).fetchall()
        out: list[dict] = []
        for row in rows:
            item = dict(row)
            try:
                item["excluded_by"] = json.loads(item.pop("excluded_by_json") or "{}")
            except json.JSONDecodeError:
                item["excluded_by"] = {}
            out.append(item)
        return out

    def _reasoning_send_ledger_row(self, row_id: str) -> dict:
        row = self.conn.execute(
            "SELECT * FROM reasoning_send_ledger WHERE id = ?", (row_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError("reasoning send ledger row was not inserted")
        item = dict(row)
        try:
            item["excluded_by"] = json.loads(item.pop("excluded_by_json") or "{}")
        except json.JSONDecodeError:
            item["excluded_by"] = {}
        return item

    # -- delete ---------------------------------------------------------------

    def delete(
        self,
        *,
        since_ts: float | None = None,
        before_ts: float | None = None,
        all: bool = False,
    ) -> int:
        """Delete observations and cascade-clean orphaned content.

        Selectors (give exactly one):
          * ``all=True``        — remove everything.
          * ``since_ts``        — remove observations at/after that timestamp.
          * ``before_ts``       — remove observations strictly before that
            timestamp (retention pruning; see :meth:`prune`).

        After removing observations, any blob with no remaining observations is
        deleted along with its chunks, FTS entries, and vectors. The whole
        operation runs inside a single ``BEGIN IMMEDIATE`` transaction with
        rollback-on-error, so a crash or mid-delete failure cannot leave orphaned
        chunks/fts_chunks/vec_chunks (previously it issued many DELETEs + one
        commit with no rollback guard). Returns the number of observations
        deleted.
        """
        selectors = [all, since_ts is not None, before_ts is not None]
        if sum(1 for s in selectors if s) != 1:
            raise ValueError(
                "delete() requires exactly one of all=True, since_ts, or before_ts"
            )

        cur = self.conn
        try:
            self._begin()
            if all:
                count = cur.execute(
                    "SELECT COUNT(*) AS c FROM observations"
                ).fetchone()["c"]
                # Derived day memories are sensitive distilled data. Remove them
                # explicitly before the raw observation wipe so no synthesized
                # daily artifact survives a full purge.
                cur.execute("DELETE FROM day_memories")
                cur.execute("DELETE FROM day_memory_source_refs")
                cur.execute("DELETE FROM reasoning_send_ledger")
                cur.execute("DELETE FROM observations")
                cur.execute("DELETE FROM activity_spans")
                cur.execute("DELETE FROM blob_chunks")
                cur.execute("DELETE FROM chunks")
                cur.execute("DELETE FROM content_blobs")
                cur.execute("DELETE FROM fts_chunks")
                cur.execute("DELETE FROM vec_chunks")
                # NOTE: we deliberately KEEP embedding_meta. With vec_chunks now
                # empty, a reopen under a different provider is detected as a
                # cohort mismatch on an empty store, which _record_cohort tolerates
                # by rebuilding the vector table at the new dimension and adopting
                # the new cohort. Clearing it here would look like a fresh store
                # and skip that rebuild.
                cur.commit()
                return int(count)

            if since_ts is not None:
                where, param = "ts >= ?", since_ts
                # Purge-of-recent: any span TOUCHING the purged window goes —
                # a purge must not leave partial evidence of the erased time.
                span_where = "end_ts >= ?"
            else:
                where, param = "ts < ?", before_ts
                # Retention: a span that BEGAN before the cutoff is deleted
                # ENTIRELY (never truncated/retained) — no span data from the
                # pruned window may survive, and truncation would bypass the
                # span-delete invalidation trigger.
                span_where = "start_ts < ?"

            victims = cur.execute(
                f"SELECT id, content_hash FROM observations WHERE {where}", (param,)
            ).fetchall()
            count = len(victims)
            affected_hashes = {r["content_hash"] for r in victims}
            cur.execute(f"DELETE FROM observations WHERE {where}", (param,))
            # Span deletion fires trg_day_memory_source_span_delete, which
            # invalidates any day memory citing a deleted span; observations
            # referencing a deleted span get span_id NULLed by the FK.
            cur.execute(f"DELETE FROM activity_spans WHERE {span_where}", (param,))

            # Drop blobs now orphaned (no remaining observations). FK ON DELETE
            # CASCADE removes their blob_chunks mappings.
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
                "DELETE FROM chunks WHERE chunk_hash NOT IN "
                "(SELECT chunk_hash FROM blob_chunks)"
            )

            cur.commit()
            return count
        except Exception:
            cur.rollback()
            raise

    # -- retention / maintenance (H10) ----------------------------------------

    def prune(
        self, *, older_than_ts: float | None = None, older_than_days: float | None = None
    ) -> int:
        """Delete observations older than a cutoff (retention) and cascade-clean.

        Provide either an absolute ``older_than_ts`` (delete observations with
        ``ts < older_than_ts``) or ``older_than_days`` (cutoff = now - N days).
        Falls back to ``settings.retention_days`` when neither is given and that
        setting is > 0. Returns the number of observations deleted. Storage space
        is only reclaimed to the OS after :meth:`vacuum`.
        """
        if older_than_ts is None:
            days = older_than_days
            if days is None:
                days = self.settings.retention_days
            if not days or days <= 0:
                raise ValueError(
                    "prune() needs older_than_ts, older_than_days, or a positive "
                    "settings.retention_days"
                )
            older_than_ts = time.time() - float(days) * 86400.0
        return self.delete(before_ts=older_than_ts)

    def vacuum(self) -> dict:
        """Reclaim space: checkpoint the WAL and run VACUUM. Returns reclaim stats.

        Deletes only mark pages free in the SQLite file; the file does not shrink
        until ``VACUUM`` rewrites it (this DB uses ``auto_vacuum=NONE``). We first
        ``wal_checkpoint(TRUNCATE)`` to fold the WAL back and truncate it, then
        ``VACUUM`` to compact the main file. Returns page/freelist counts before
        and after so callers can show the reclaim. No-op-safe on ``:memory:``.

        VACUUM cannot run inside a transaction; this method must not be called
        with an open txn (it manages its own autocommit statements).
        """
        def _pragma_int(name: str) -> int:
            row = self.conn.execute(f"PRAGMA {name}").fetchone()
            if row is None:
                return 0
            # row_factory yields a dict; the pragma's column name varies across
            # SQLite builds, so read the single value regardless of its key.
            value = next(iter(row.values())) if isinstance(row, dict) else row[0]
            return int(value) if value is not None else 0

        page_size = _pragma_int("page_size")

        def _gauge() -> tuple[int, int]:
            return _pragma_int("page_count"), _pragma_int("freelist_count")

        before_pages, before_free = _gauge()
        # Best-effort WAL checkpoint+truncate (no-op on non-WAL / :memory:).
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        # VACUUM requires autocommit (no open transaction). isolation_level is
        # None for this store, so a bare execute runs outside any txn.
        self.conn.execute("VACUUM")
        # In WAL mode VACUUM writes the compacted image into the WAL, so the main
        # DB file is not physically truncated until the next checkpoint. Force one
        # so `openbird data vacuum` reclaims space on disk, not just logically.
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        after_pages, after_free = _gauge()
        return {
            "page_size": page_size,
            "pages_before": before_pages,
            "pages_after": after_pages,
            "freelist_before": before_free,
            "freelist_after": after_free,
            "bytes_before": before_pages * page_size,
            "bytes_after": after_pages * page_size,
            "bytes_reclaimed": (before_pages - after_pages) * page_size,
        }

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
            "day_memories": count("day_memories"),
            "activity_spans": count("activity_spans"),
            "embed_dim": self.embed_dim,
            "cohort_key": cohort_row["value"] if cohort_row else None,
            "encryption_enabled": self.settings.encryption_enabled,
        }

    def capture_app_activity(
        self, *, recent_since_ts: float | None = None, source: str = "capture"
    ) -> dict[str, dict[str, float | int | None]]:
        """Return per-app capture activity using metadata only.

        This intentionally reads only ``observations.app`` and timestamp/count
        metadata. It never joins ``content_blobs`` and never reads captured text,
        window titles, or URLs, so Settings can display capture health without
        widening the privacy surface.
        """
        recent_cutoff = float("-inf") if recent_since_ts is None else float(recent_since_ts)
        rows = self.conn.execute(
            """
            SELECT
                app,
                COUNT(*) AS total_observations,
                SUM(CASE WHEN ts >= ? THEN 1 ELSE 0 END) AS recent_observations,
                MAX(ts) AS last_captured_ts
            FROM observations
            WHERE source = ? AND app IS NOT NULL AND app != ''
            GROUP BY app
            """,
            (recent_cutoff, source),
        ).fetchall()
        return {
            row["app"]: {
                "total_observations": int(row["total_observations"] or 0),
                "recent_observations": int(row["recent_observations"] or 0),
                "last_captured_ts": row["last_captured_ts"],
            }
            for row in rows
        }

    def integrity_check(self, *, quick: bool = False) -> dict:
        """Verify the database is not corrupt via SQLite's integrity check.

        ``PRAGMA integrity_check`` returns a single ``"ok"`` row when the file is
        sound, or one row per problem otherwise. ``quick=True`` uses the faster
        ``quick_check`` (skips some cross-page/index checks). Read-only; safe to
        run anytime.
        """
        pragma = "quick_check" if quick else "integrity_check"
        return _integrity_result(self.conn.execute(f"PRAGMA {pragma}").fetchall())

    def close(self) -> None:
        """Close the underlying connection."""
        self.conn.close()


def _integrity_result(rows) -> dict:
    """Parse ``PRAGMA integrity_check``/``quick_check`` rows into a result dict.

    SQLite returns a single ``"ok"`` row when sound, else one row per problem.
    Handles both the dict row_factory and the default tuple shape.
    """
    problems: list[str] = []
    for row in rows:
        value = next(iter(row.values())) if isinstance(row, dict) else row[0]
        if value is not None:
            problems.append(str(value))
    ok = problems == ["ok"]
    return {"ok": ok, "problems": [] if ok else problems}


def check_database_integrity(
    db_path: str,
    *,
    settings=None,
    quick: bool = False,
    opener=None,
) -> dict:
    """Open the DB **raw** (no schema/migrations) and run the integrity PRAGMA.

    Used by ``openbird data integrity``. Opening through :class:`MemoryStore`
    would run schema/migration statements that themselves raise on a corrupt
    file — so a corruption diagnostic must NOT use it. This opens a bare
    connection and treats both open failures and PRAGMA errors as *findings*
    (``ok=False`` with a problem code) rather than raising. Never raises.
    """
    import sqlite3

    def _default_opener():
        from openbird.storage.crypto import open_encrypted_db

        return open_encrypted_db(db_path, settings=settings)

    try:
        conn = (opener or _default_opener)()
    except Exception as exc:  # noqa: BLE001 - a diagnostic must report, not crash
        return {"ok": False, "problems": [f"cannot-open: {type(exc).__name__}"]}
    try:
        pragma = "quick_check" if quick else "integrity_check"
        return _integrity_result(conn.execute(f"PRAGMA {pragma}").fetchall())
    except sqlite3.DatabaseError as exc:
        return {"ok": False, "problems": [f"check-failed: {type(exc).__name__}"]}
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


__all__ = ["EmbeddingCohortMismatch", "MemoryStore", "check_database_integrity"]
