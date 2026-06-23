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
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from openbird.config import Settings, get_settings
from openbird.llm.base import LLMProviderProtocol
from openbird.llm.provider import create_llm_provider
from openbird.memory import ingest
from openbird.memory.migrations import ensure_schema_version
from openbird.memory.search import mmr, rrf
from openbird.storage.crypto import mapping_row_factory, open_encrypted_db
from openbird.types import Observation, SearchHit


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
        """
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
            "INSERT INTO observations(id, content_hash, ts, app, window, url, session_id, source)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (obs_id, blob_hash, ts, app, window, url, session_id, source),
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
                cur.execute("DELETE FROM observations")
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
            else:
                where, param = "ts < ?", before_ts

            victims = cur.execute(
                f"SELECT id, content_hash FROM observations WHERE {where}", (param,)
            ).fetchall()
            count = len(victims)
            affected_hashes = {r["content_hash"] for r in victims}
            cur.execute(f"DELETE FROM observations WHERE {where}", (param,))

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
            "embed_dim": self.embed_dim,
            "cohort_key": cohort_row["value"] if cohort_row else None,
            "encryption_enabled": self.settings.encryption_enabled,
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


__all__ = ["MemoryStore", "check_database_integrity"]
