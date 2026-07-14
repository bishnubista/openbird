"""The local memory store: observations, deduped blobs, chunk-level retrieval.

Data model:
  * ``content_blobs`` — deduped canonical text, embedded once.
  * ``observations`` — one row per occurrence, never deduped.
  * ``chunks`` + ``fts_chunks`` (FTS5) + ``vec_chunks`` (sqlite-vec, per chunk).

``add_observation`` ALWAYS inserts a new observation row, dedups content at the
*chunk* level, embeds each unique chunk once, and indexes it in both FTS and the
vector table. ``search`` fuses vector + BM25 with RRF, then dedups with MMR, and
resolves every hit back to a concrete observation for occurrence-aware citations.

SUMMARY-INDEX DELETION CONTRACT (v6, Phase E1): ``summary_index_entries`` +
``fts_summaries`` + ``vec_summaries`` mirror derived block-summary / week-digest
prose for retrieval. SQLite triggers clean only the plain entries table (virtual
tables cannot be written from triggers), so every PRODUCTION deletion path MUST
go through the MemoryStore APIs — ``delete()``/``prune()``,
``save_block_summary()`` regeneration, and ``save_week_memory()`` regeneration —
which call :meth:`MemoryStore._sweep_summary_index_for_blocks` (or the pair-level
sweep) in the SAME transaction, BEFORE the source delete fires the entry-cleanup
triggers. Raw SQL deletes against ``block_summaries`` or week-scope
``day_memories`` rows outside these APIs are FORBIDDEN: they strand fts/vec
orphans. ``openbird data integrity`` reports orphan counts as the safety net.
"""

from __future__ import annotations

import hashlib
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

_CAPTURE_ATTEMPT_REASON_CODES = frozenset(
    {
        "paused", "self_capture", "not_allowlisted", "dangerous_app",
        "private_window", "no_frontmost_app", "no_window", "ax_timeout",
        "budget_exhausted", "empty_text", "unchanged", "normalized_empty",
        "policy_rejected", "ingest_failed",
    }
)
_CAPTURE_BUNDLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,254}$")


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

# The five-level activity taxonomy (v5). Mirrors the CHECK constraints on
# block_summaries.level / category_assignments.level and taxonomy.LEVELS —
# validated here in Python too so a bad level fails loudly before SQL.
_TAXONOMY_LEVELS = frozenset(
    {"focus_work", "other_work", "neutral", "personal", "distracting"}
)

# Entity ledger enums (v7, Phase E2). Mirror the schema CHECK constraints so a
# bad value fails loudly in Python before SQL.
_ENTITY_KINDS = frozenset({"repo", "domain", "document", "topic"})
_ENTITY_SOURCE_KINDS = frozenset({"observation", "span", "summary"})
_ENTITY_EVIDENCE_KINDS = frozenset(
    {"pr_merged", "ticket_closed", "shipped_language", "open_loop",
     "open_loop_resolved"}
)
# Prefix for the entity-aggregation watermark keys in the embedding_meta KV
# table. A FULL purge clears every key under this prefix (the watermarks are
# derived from captured activity positions); the cohort_key stays.
ENTITY_AGGREGATION_KV_PREFIX = "entity_aggregation."


def entity_id_for(kind: str, name: str) -> str:
    """Deterministic entity id: sha256 over ``kind:casefold(name)``.

    Deterministic ids make aggregation upserts idempotent across runs; entity
    identity is EXACT casefolded name within a kind — no fuzzy merging.
    """
    payload = f"{kind}:{str(name).casefold()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _enable_recursive_triggers(conn) -> None:
    """Set ``PRAGMA recursive_triggers = ON`` and VERIFY it took effect.

    The v5 deletion chain (span delete -> trigger deletes its block summary ->
    trigger invalidates day memories citing that summary) only runs when a
    trigger body can fire further triggers, which SQLite gates behind this
    per-connection pragma (default OFF). Some backends silently ignore unknown
    pragmas, and a silent no-op here would silently break the deletion-cascade
    privacy contract — so the value is READ BACK and a mismatch RAISES at
    startup instead of degrading.
    """
    conn.execute("PRAGMA recursive_triggers = ON")
    row = conn.execute("PRAGMA recursive_triggers").fetchone()
    value = None
    if row is not None:
        value = next(iter(row.values())) if isinstance(row, dict) else row[0]
    try:
        ok = int(value) == 1  # type: ignore[arg-type]
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise RuntimeError(
            "PRAGMA recursive_triggers did not read back as 1; this backend "
            "cannot enforce the derived-artifact deletion cascade. Refusing to "
            "open the store."
        )


def _serialize_f32(vector: list[float]) -> bytes:
    """Pack a float vector into sqlite-vec's little-endian float32 blob format."""
    return struct.pack(f"<{len(vector)}f", *vector)


def _dt_date_from_iso(value: str):
    """Parse a strict ``YYYY-MM-DD`` local date (raises ValueError otherwise)."""
    import datetime as _dt

    return _dt.datetime.strptime(value, "%Y-%m-%d").date()


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
        # Load-bearing pragma (v5): the deletion chain span -> block summary ->
        # citing day memory is trigger-fired-by-a-trigger, which SQLite only runs
        # with recursive triggers on. Set + VERIFY before any schema/migration
        # DDL so a backend that silently ignores the pragma cannot silently break
        # the deletion-cascade privacy contract.
        _enable_recursive_triggers(self.conn)
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
        # Summary-index vectors (v6): created here, not in schema.sql or the
        # migration ladder, because the dimension comes from Settings.embed_dim —
        # exactly like vec_chunks.
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_summaries USING vec0("
            f"entry_rowid INTEGER PRIMARY KEY, embedding FLOAT[{self.embed_dim}])"
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
            # Tolerate a cohort change ONLY when the store holds no vectors AT
            # ALL — chunk vectors AND summary-index vectors (e.g. after a full
            # purge): there is nothing to mix, so adopt the new provider's
            # cohort. Otherwise refuse to mix incompatible embeddings.
            vec_count = self.conn.execute(
                "SELECT (SELECT COUNT(*) FROM vec_chunks) "
                "+ (SELECT COUNT(*) FROM vec_summaries) AS c"
            ).fetchone()["c"]
            if vec_count == 0:
                # Adopt the new cohort AND rebuild BOTH vector tables at the new
                # provider's dimension — the old tables were created FLOAT[old]
                # and CREATE ... IF NOT EXISTS would otherwise keep the stale dim,
                # breaking inserts when the dimension changed.
                self.conn.execute("DROP TABLE IF EXISTS vec_chunks")
                self.conn.execute(
                    f"CREATE VIRTUAL TABLE vec_chunks USING vec0("
                    f"chunk_rowid INTEGER PRIMARY KEY, embedding FLOAT[{self.embed_dim}])"
                )
                self.conn.execute("DROP TABLE IF EXISTS vec_summaries")
                self.conn.execute(
                    f"CREATE VIRTUAL TABLE vec_summaries USING vec0("
                    f"entry_rowid INTEGER PRIMARY KEY, embedding FLOAT[{self.embed_dim}])"
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

    def record_capture_attempt(self, **attempt: object) -> dict:
        """Upsert one strictly metadata-only capture-attempt event.

        The daemon owns validation against closed vocabularies. This store API
        owns atomic started->finished updates and the observation/successor FKs.
        Captured text, titles, URLs, and hashes are intentionally absent from
        both the signature and table schema.
        """
        reason_codes = attempt.get("reason_codes")
        if (
            not isinstance(reason_codes, (list, tuple))
            or len(reason_codes) > 16
            or any(
                not isinstance(reason, str)
                or reason not in _CAPTURE_ATTEMPT_REASON_CODES
                for reason in reason_codes
            )
        ):
            raise ValueError("invalid capture-attempt reason codes")
        bundle_id = attempt.get("bundle_id")
        if bundle_id is not None and (
            not isinstance(bundle_id, str)
            or _CAPTURE_BUNDLE_ID_RE.fullmatch(bundle_id) is None
        ):
            raise ValueError("invalid capture-attempt bundle id")
        for key in (
            "trigger_ts", "started_ts", "finished_ts", "earliest_coalesced_ts"
        ):
            value = attempt.get(key)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError("invalid capture-attempt timestamp")
        for key in (
            "trigger_seq", "nodes_visited", "bytes_emitted", "elapsed_ms",
            "coalesced_trigger_count",
        ):
            value = attempt.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("invalid capture-attempt counter")
        for key in ("attempt_id", "helper_epoch", "successor_attempt_id"):
            value = attempt.get(key)
            if value is None and key == "successor_attempt_id":
                continue
            if not isinstance(value, str):
                raise ValueError("invalid capture-attempt id")
            try:
                uuid.UUID(value)
            except ValueError as exc:
                raise ValueError("invalid capture-attempt id") from exc
        reason_json = json.dumps(list(reason_codes), separators=(",", ":"))
        values = (
            attempt["attempt_id"],
            attempt["helper_epoch"],
            attempt["trigger_seq"],
            attempt["trigger_ts"],
            attempt.get("started_ts"),
            attempt.get("finished_ts"),
            attempt["status"],
            attempt.get("bundle_id"),
            attempt["trigger"],
            attempt.get("adapter_id"),
            attempt.get("extractor_version"),
            attempt.get("policy_tier"),
            attempt.get("outcome"),
            attempt.get("nodes_visited", 0),
            attempt.get("bytes_emitted", 0),
            attempt.get("elapsed_ms", 0),
            attempt.get("completeness"),
            reason_json,
            attempt.get("coalesced_trigger_count", 0),
            attempt.get("earliest_coalesced_ts"),
            attempt.get("successor_attempt_id"),
            attempt.get("observation_id"),
        )
        try:
            self._begin()
            self.conn.execute(
                """
                INSERT INTO capture_attempts(
                    attempt_id, helper_epoch, trigger_seq, trigger_ts,
                    started_ts, finished_ts, status, bundle_id, trigger,
                    adapter_id, extractor_version, policy_tier, outcome,
                    nodes_visited, bytes_emitted, elapsed_ms, completeness,
                    reason_codes_json, coalesced_trigger_count,
                    earliest_coalesced_ts, successor_attempt_id, observation_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(attempt_id) DO UPDATE SET
                    helper_epoch = excluded.helper_epoch,
                    trigger_seq = excluded.trigger_seq,
                    trigger_ts = excluded.trigger_ts,
                    started_ts = COALESCE(excluded.started_ts, capture_attempts.started_ts),
                    finished_ts = COALESCE(excluded.finished_ts, capture_attempts.finished_ts),
                    status = CASE
                        WHEN capture_attempts.status = 'finished'
                         AND excluded.status = 'started' THEN capture_attempts.status
                        ELSE excluded.status
                    END,
                    bundle_id = COALESCE(excluded.bundle_id, capture_attempts.bundle_id),
                    trigger = excluded.trigger,
                    adapter_id = COALESCE(excluded.adapter_id, capture_attempts.adapter_id),
                    extractor_version = COALESCE(
                        excluded.extractor_version, capture_attempts.extractor_version),
                    policy_tier = COALESCE(excluded.policy_tier, capture_attempts.policy_tier),
                    outcome = COALESCE(excluded.outcome, capture_attempts.outcome),
                    nodes_visited = CASE WHEN excluded.status = 'finished'
                        THEN excluded.nodes_visited ELSE capture_attempts.nodes_visited END,
                    bytes_emitted = CASE WHEN excluded.status = 'finished'
                        THEN excluded.bytes_emitted ELSE capture_attempts.bytes_emitted END,
                    elapsed_ms = CASE WHEN excluded.status = 'finished'
                        THEN excluded.elapsed_ms ELSE capture_attempts.elapsed_ms END,
                    completeness = COALESCE(
                        excluded.completeness, capture_attempts.completeness),
                    reason_codes_json = CASE WHEN excluded.status = 'finished'
                        THEN excluded.reason_codes_json
                        ELSE capture_attempts.reason_codes_json END,
                    coalesced_trigger_count = CASE WHEN excluded.status = 'finished'
                        THEN excluded.coalesced_trigger_count
                        ELSE capture_attempts.coalesced_trigger_count END,
                    earliest_coalesced_ts = COALESCE(
                        excluded.earliest_coalesced_ts,
                        capture_attempts.earliest_coalesced_ts),
                    successor_attempt_id = COALESCE(
                        excluded.successor_attempt_id,
                        capture_attempts.successor_attempt_id),
                    observation_id = COALESCE(
                        excluded.observation_id, capture_attempts.observation_id)
                """,
                values,
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        row = self.conn.execute(
            "SELECT * FROM capture_attempts WHERE attempt_id = ?",
            (attempt["attempt_id"],),
        ).fetchone()
        if row is None:
            raise RuntimeError("capture attempt row was not inserted")
        item = dict(row)
        try:
            item["reason_codes"] = json.loads(item.pop("reason_codes_json"))
        except json.JSONDecodeError:
            item["reason_codes"] = []
        return item

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

    def recent_capture_text(
        self,
        start_ts: float,
        end_ts: float,
        *,
        limit: int,
        max_chars: int = 2000,
        before: tuple[float, str] | None = None,
    ) -> list[tuple[Observation, str]]:
        """Return a newest-first, row-bounded page of capture text.

        This is an assistant-safe local read: it performs no embedding, rerank,
        or completion call. App/source/id egress exclusions remain the caller's
        responsibility because they are policy rather than storage semantics.

        ``before`` is an exclusive keyset boundary ``(ts, id)``: only rows
        strictly below it in ``(ts, id)`` descending order are returned, so a
        caller can resume a page walk without offsets (stable under concurrent
        newer inserts — later pages are strictly older). Served by
        ``idx_observations_source_ts_id`` (schema v8).
        """
        sql = (
            "SELECT o.*, b.text AS blob_text FROM observations o "
            "JOIN content_blobs b ON b.content_hash = o.content_hash "
            "WHERE o.ts >= ? AND o.ts <= ? AND o.source = 'capture' "
        )
        params: list[object] = [float(start_ts), float(end_ts)]
        if before is not None:
            before_ts, before_id = before
            sql += "AND (o.ts < ? OR (o.ts = ? AND o.id < ?)) "
            params.extend([float(before_ts), float(before_ts), str(before_id)])
        sql += "ORDER BY o.ts DESC, o.id DESC LIMIT ?"
        params.append(max(0, int(limit)))
        rows = self.conn.execute(sql, params).fetchall()
        return [
            (self._row_to_observation(row), str(row["blob_text"] or "")[:max_chars])
            for row in rows
        ]

    def capture_spans_overlapping(
        self, start_ts: float, end_ts: float, *, limit: int
    ) -> list[dict]:
        """Return a bounded, chronological page of spans overlapping the window.

        Unlike :meth:`spans_in_range` this read is hard-bounded (``limit``) and
        projected to the columns the assistant summary aggregates — it never
        returns ``window``, ``url_host``, or ``identity_key``, so tier-1 titles
        cannot reach an assistant even by a serialization mistake. Callers pass
        ``limit = cap + 1`` and treat a full result as overflow (fail closed).
        """
        rows = self.conn.execute(
            "SELECT span_id, start_ts, end_ts, bundle_id, detail_tier, afk, meeting "
            "FROM activity_spans WHERE start_ts <= ? AND end_ts >= ? "
            "ORDER BY start_ts, span_id LIMIT ?",
            (float(end_ts), float(start_ts), max(0, int(limit))),
        ).fetchall()
        return [dict(r) for r in rows]

    def lexical_capture_text(
        self,
        query: str,
        *,
        limit: int,
        max_chars: int = 2000,
    ) -> list[tuple[Observation, str]]:
        """BM25-only capture search with no model, vector, or reranker path."""
        match = self._fts_query(query)
        if not match or limit <= 0:
            return []
        rows = self.conn.execute(
            "WITH ranked_chunks AS ("
            "  SELECT c.chunk_hash, c.text AS chunk_text, "
            "         bm25(fts_chunks) AS rank "
            "  FROM fts_chunks "
            "  JOIN chunks c ON c.rowid_int = fts_chunks.rowid "
            "  WHERE fts_chunks MATCH ? AND EXISTS ("
            "    SELECT 1 FROM blob_chunks capture_bc "
            "    JOIN observations capture_o "
            "      ON capture_o.content_hash = capture_bc.content_hash "
            "    WHERE capture_bc.chunk_hash = c.chunk_hash "
            "      AND capture_o.source = 'capture'"
            "  ) "
            "  ORDER BY rank ASC LIMIT ?"
            "), ranked_occurrences AS ("
            "  SELECT o.*, rc.chunk_hash AS matched_chunk, rc.chunk_text, rc.rank, "
            "         ROW_NUMBER() OVER ("
            "           PARTITION BY rc.chunk_hash ORDER BY o.ts DESC, o.id DESC"
            "         ) AS occurrence_rank "
            "  FROM ranked_chunks rc "
            "  JOIN blob_chunks bc ON bc.chunk_hash = rc.chunk_hash "
            "  JOIN observations o ON o.content_hash = bc.content_hash "
            "  WHERE o.source = 'capture'"
            ") "
            "SELECT * FROM ranked_occurrences WHERE occurrence_rank = 1 "
            "ORDER BY rank ASC, ts DESC, id DESC LIMIT ?",
            (match, int(limit), int(limit)),
        ).fetchall()
        return [
            (
                self._row_to_observation(row),
                str(row["chunk_text"] or "")[:max_chars],
            )
            for row in rows
        ]

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
        meeting: bool = False,
        reason: str | None = None,
        span_id: str | None = None,
    ) -> str:
        """Insert one activity span and return its id (own short transaction).

        Enforces the tier contract in Python BEFORE SQL (fail-closed, testable
        error) in addition to the schema CHECKs: tier 0 must carry a reason from
        the closed enum and NO window/url_host/identity_key; tier 1 carries no
        reason. ``meeting`` (Phase C1) is legal on BOTH tiers — a coarse
        non-allowlisted Zoom span may still be a meeting. Metadata only —
        nothing here is captured content (window titles arrive already
        scrubbed by the caller).
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
                ") VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?)",
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
                    1 if meeting else 0,
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
        from openbird.taxonomy import (
            levels_for_spans,
            load_overrides,
            taxonomy_fingerprint,
        )

        attempts = 3
        for attempt in range(attempts):
            try:
                self._begin()
                rows = self.time_range_text(start_ts, end_ts, source=source_scope)
                fingerprint = self.day_memory_source_fingerprint_from_rows(rows)
                # Spans are CAPTURE ground truth: only capture-scope memories
                # cite them (a meetings-scope memory must neither include
                # span metrics nor be invalidated by unrelated capture spans).
                if source_scope == "capture":
                    spans = self.spans_in_range(start_ts, end_ts)
                    span_fp = span_fingerprint_for_spans(spans)
                    # Pre-resolved identity -> level mapping (overrides + rules
                    # + LLM cache) for the measured span_time_by_level block;
                    # its fingerprint joins the freshness check so an edited
                    # taxonomy.json or a new cached level rebuilds the row.
                    overrides = load_overrides(self.settings)
                    taxonomy_map = levels_for_spans(
                        spans,
                        overrides=overrides,
                        cache=self.get_category_assignments(),
                    )
                    tax_fp = taxonomy_fingerprint(taxonomy_map, overrides)
                else:
                    spans = None
                    span_fp = None
                    taxonomy_map = None
                    tax_fp = None
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
                    and existing["payload"].get("taxonomy_fingerprint") == tax_fp
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
                    spans=spans,  # None outside capture scope (no span block)
                    taxonomy=taxonomy_map,
                    taxonomy_fingerprint=tax_fp,
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
        """Return a stored day memory without checking source freshness.

        Provenance shape (Phase E1): this shared reader used to DROP
        ``source_kind='summary'`` refs (it mapped only observation/span kinds),
        so a week row would come back without the provenance its answer path
        needs. It now also returns ``summary_ids`` and a full typed
        ``source_refs`` list (all kinds) while keeping the legacy
        ``source_ids``/``span_ids`` keys untouched.
        """
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
            "summary_ids": [
                r["source_id"] for r in ref_rows if r["source_kind"] == "summary"
            ],
            "source_refs": [
                {"source_kind": r["source_kind"], "source_id": r["source_id"]}
                for r in ref_rows
            ],
            "payload": json.loads(row["payload_json"]),
        }

    # -- week memories (Phase E1) -----------------------------------------------

    def save_week_memory(
        self,
        *,
        week_start_date: str,
        extractor_version: str,
        payload: dict,
        summary_ids: list[str],
        generated_at: float | None = None,
    ) -> dict:
        """Persist one week digest as a ``day_memories`` row with scope ``week``.

        Week rows reuse the day-memory storage: ``local_date`` is the ISO week's
        MONDAY (``YYYY-MM-DD``) and ``source_scope='week'`` — the existing
        ``UNIQUE(local_date, source_scope)`` gives per-week uniqueness for free.
        Refs are SUMMARY-KIND ONLY (the member block-summary ids the digest
        actually cited); an empty ref list is REFUSED in Python (mirroring
        ``save_block_summary``) because derived-sensitive prose with no typed
        refs would have no invalidation path.

        Regeneration runs in a single transaction: the OLD week row's
        summary-index rows are swept FIRST (deletion contract — the row delete
        below fires the entry-cleanup trigger, which cannot clean fts/vec),
        then the old row is deleted and the new parent + refs inserted.
        ``payload`` (digest text, member_fingerprint, window, ...) is derived
        sensitive; this method logs nothing.
        """
        try:
            parsed = _dt_date_from_iso(week_start_date)
        except ValueError as exc:
            raise ValueError(
                f"week_start_date must be YYYY-MM-DD, got {week_start_date!r}"
            ) from exc
        if parsed.weekday() != 0:
            raise ValueError(
                f"week_start_date must be a Monday, got {week_start_date!r}"
            )
        unique_summary_ids = sorted({str(i) for i in summary_ids if i})
        if not unique_summary_ids:
            raise ValueError("week memory requires at least one summary ref")
        week_id = uuid.uuid4().hex
        generated = time.time() if generated_at is None else generated_at
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        try:
            self._begin()
            old_rows = self.conn.execute(
                "SELECT id FROM day_memories "
                "WHERE local_date = ? AND source_scope = 'week'",
                (week_start_date,),
            ).fetchall()
            self._sweep_summary_index_pairs(
                [("week", r["id"]) for r in old_rows]
            )
            self.conn.execute(
                "DELETE FROM day_memories "
                "WHERE local_date = ? AND source_scope = 'week'",
                (week_start_date,),
            )
            self.conn.execute(
                "INSERT INTO day_memories("
                "id, local_date, source_scope, extractor_version, generated_at, "
                "payload_json, source_count"
                ") VALUES (?, ?, 'week', ?, ?, ?, ?)",
                (
                    week_id,
                    week_start_date,
                    extractor_version,
                    generated,
                    payload_json,
                    len(unique_summary_ids),
                ),
            )
            self.conn.executemany(
                "INSERT INTO day_memory_source_refs("
                "day_memory_id, source_kind, source_id) VALUES (?, 'summary', ?)",
                [(week_id, i) for i in unique_summary_ids],
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        saved = self.get_week_memory(week_start_date)
        if saved is None:  # pragma: no cover - insert above would have raised
            raise RuntimeError("week memory row was not inserted")
        return saved

    def get_week_memory(self, week_start_date: str) -> dict | None:
        """Return the stored week digest for a Monday date, or ``None``."""
        return self._get_day_memory_unchecked(
            local_date=week_start_date, source_scope="week"
        )

    def week_memories_overlapping(self, start_ts: float, end_ts: float) -> list[dict]:
        """Return week rows whose local Mon..Sun window overlaps [start_ts, end_ts].

        Week rows are few (one per ISO week), so this reads them all and
        filters by the local-time window derived from each row's Monday
        ``local_date`` (inclusive end at the following Monday minus a tick,
        mirroring the day-window convention).
        """
        import datetime as _dt

        rows = self.conn.execute(
            "SELECT local_date FROM day_memories WHERE source_scope = 'week' "
            "ORDER BY local_date",
        ).fetchall()
        out: list[dict] = []
        for row in rows:
            try:
                monday = _dt.datetime.strptime(row["local_date"], "%Y-%m-%d")
            except ValueError:
                continue
            week_start = monday.timestamp()
            week_end = (monday + _dt.timedelta(days=7)).timestamp() - 1e-6
            if week_start <= float(end_ts) and week_end >= float(start_ts):
                item = self.get_week_memory(row["local_date"])
                if item is not None:
                    out.append(item)
        return out

    # -- block summaries + taxonomy cache (Phase D) -----------------------------

    def save_block_summary(
        self,
        *,
        local_date: str,
        block_key: str,
        block_fingerprint: str,
        start_ts: float,
        end_ts: float,
        dominant_bundle: str | None,
        level: str | None,
        summary_text: str,
        model: str,
        extractor_version: str,
        observation_ids: list[str],
        span_ids: list[str],
        generated_at: float | None = None,
    ) -> dict:
        """Persist one block summary and its typed source refs (single txn).

        Regenerate semantics: any existing row for the same ``block_key`` is
        deleted first (cascading its old refs), then the new parent + refs are
        inserted in the same transaction. ``summary_text`` is DERIVED SENSITIVE
        content — this method logs nothing, and callers must not log it either.
        """
        if level is not None and level not in _TAXONOMY_LEVELS:
            raise ValueError(f"unknown taxonomy level: {level!r}")
        if not observation_ids and not span_ids:
            # Derived-sensitive prose with no typed refs would have NO
            # invalidation path on selective source deletion — refuse it.
            raise ValueError("block summary requires at least one source ref")
        summary_id = uuid.uuid4().hex
        generated = time.time() if generated_at is None else generated_at
        unique_obs = sorted(set(observation_ids))
        unique_spans = sorted(set(span_ids))
        try:
            self._begin()
            # Deletion contract: sweep the OLD row's summary-index rows (and any
            # dependent week row's — the delete below trigger-deletes weeks
            # citing it) BEFORE the source delete fires the entry triggers.
            old_rows = self.conn.execute(
                "SELECT id FROM block_summaries WHERE block_key = ?", (block_key,)
            ).fetchall()
            self._sweep_summary_index_for_blocks([r["id"] for r in old_rows])
            self.conn.execute(
                "DELETE FROM block_summaries WHERE block_key = ?", (block_key,)
            )
            self.conn.execute(
                "INSERT INTO block_summaries("
                "id, local_date, block_key, block_fingerprint, start_ts, end_ts, "
                "dominant_bundle, level, summary_text, model, extractor_version, "
                "generated_at, source_count"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    summary_id,
                    local_date,
                    block_key,
                    block_fingerprint,
                    float(start_ts),
                    float(end_ts),
                    dominant_bundle,
                    level,
                    summary_text,
                    model,
                    extractor_version,
                    generated,
                    len(unique_obs) + len(unique_spans),
                ),
            )
            self.conn.executemany(
                "INSERT INTO block_summary_source_refs("
                "summary_id, source_kind, source_id) VALUES (?, ?, ?)",
                [(summary_id, "observation", i) for i in unique_obs]
                + [(summary_id, "span", i) for i in unique_spans],
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        saved = self._block_summary_by_id(summary_id)
        if saved is None:  # pragma: no cover - insert above would have raised
            raise RuntimeError("block summary row was not inserted")
        return saved

    def block_summaries_for_range(self, start_ts: float, end_ts: float) -> list[dict]:
        """Return block summaries OVERLAPPING [start_ts, end_ts] plus their refs."""
        rows = self.conn.execute(
            "SELECT * FROM block_summaries WHERE start_ts <= ? AND end_ts >= ? "
            "ORDER BY start_ts",
            (float(end_ts), float(start_ts)),
        ).fetchall()
        return [self._block_summary_with_refs(dict(r)) for r in rows]

    def block_summaries_for_date(self, local_date: str) -> list[dict]:
        """Return block summaries for one local date plus their refs."""
        rows = self.conn.execute(
            "SELECT * FROM block_summaries WHERE local_date = ? ORDER BY start_ts",
            (local_date,),
        ).fetchall()
        return [self._block_summary_with_refs(dict(r)) for r in rows]

    def block_summary_keys(self) -> dict[str, str]:
        """Return ``block_key -> block_fingerprint`` (cheap pending-work probe)."""
        rows = self.conn.execute(
            "SELECT block_key, block_fingerprint FROM block_summaries"
        ).fetchall()
        return {r["block_key"]: r["block_fingerprint"] for r in rows}

    def _block_summary_by_id(self, summary_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM block_summaries WHERE id = ?", (summary_id,)
        ).fetchone()
        if row is None:
            return None
        return self._block_summary_with_refs(dict(row))

    def _block_summary_with_refs(self, item: dict) -> dict:
        refs = self.conn.execute(
            "SELECT source_kind, source_id FROM block_summary_source_refs "
            "WHERE summary_id = ? ORDER BY source_kind, source_id",
            (item["id"],),
        ).fetchall()
        item["source_refs"] = [
            {"source_kind": r["source_kind"], "source_id": r["source_id"]}
            for r in refs
        ]
        return item

    def get_category_assignments(self) -> dict[str, str]:
        """Return the LLM-fallback taxonomy cache as ``identity_key -> level``."""
        rows = self.conn.execute(
            "SELECT identity_key, level FROM category_assignments"
        ).fetchall()
        return {r["identity_key"]: r["level"] for r in rows}

    def save_category_assignment(self, identity_key: str, level: str, model: str) -> None:
        """Cache one LLM-derived taxonomy level (identity key + level only)."""
        if level not in _TAXONOMY_LEVELS:
            raise ValueError(f"unknown taxonomy level: {level!r}")
        try:
            self._begin()
            self.conn.execute(
                "INSERT INTO category_assignments(identity_key, level, model, generated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(identity_key) DO UPDATE SET "
                "level = excluded.level, model = excluded.model, "
                "generated_at = excluded.generated_at",
                (identity_key, level, model, time.time()),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # -- summary index (Phase E1) -----------------------------------------------

    def _sweep_summary_index_pairs(self, pairs: list[tuple[str, str]]) -> None:
        """Delete index rows (entries + fts + vec) for ``(kind, id)`` pairs.

        MUST run inside an already-open write transaction, BEFORE the source
        delete fires the plain-table cleanup triggers — this is the API half of
        the summary-index deletion contract (triggers cannot write the fts/vec
        virtual tables). Deleting the entries here as well is idempotent: the
        trigger's later DELETE simply finds nothing.
        """
        for kind, summary_id in pairs:
            rows = self.conn.execute(
                "SELECT entry_rowid FROM summary_index_entries "
                "WHERE summary_kind = ? AND summary_id = ?",
                (kind, summary_id),
            ).fetchall()
            for r in rows:
                rid = int(r["entry_rowid"])
                self.conn.execute("DELETE FROM fts_summaries WHERE rowid = ?", (rid,))
                self.conn.execute(
                    "DELETE FROM vec_summaries WHERE entry_rowid = ?", (rid,)
                )
            self.conn.execute(
                "DELETE FROM summary_index_entries "
                "WHERE summary_kind = ? AND summary_id = ?",
                (kind, summary_id),
            )

    def _sweep_summary_index_for_blocks(self, doomed_block_ids: list[str]) -> None:
        """Sweep index rows for doomed block summaries AND their dependent weeks.

        Covers the CASCADE: deleting a block summary trigger-deletes any week
        row citing it (``trg_day_memory_source_summary_delete``), whose OWN
        index rows would otherwise orphan — so this sweeps (i) ``('block', id)``
        for every doomed block id and (ii) ``('week', week_row_id)`` for every
        week-scope ``day_memories`` row holding a summary ref to any doomed
        block. MUST run inside the caller's open transaction, before the
        source deletes. Shared by ``delete()``/``prune()``,
        ``save_block_summary()`` regeneration, and (transitively) week-row
        replacement.
        """
        doomed = sorted({str(i) for i in doomed_block_ids if i})
        if not doomed:
            return
        placeholders = ",".join("?" for _ in doomed)
        week_rows = self.conn.execute(
            "SELECT DISTINCT d.id AS id FROM day_memories d "
            "JOIN day_memory_source_refs r ON r.day_memory_id = d.id "
            "WHERE d.source_scope = 'week' AND r.source_kind = 'summary' "
            f"AND r.source_id IN ({placeholders})",
            doomed,
        ).fetchall()
        pairs = [("block", block_id) for block_id in doomed]
        pairs += [("week", str(r["id"])) for r in week_rows]
        self._sweep_summary_index_pairs(pairs)

    def summary_index_state(self) -> dict[tuple[str, str], str]:
        """Return ``(summary_kind, summary_id) -> fingerprint`` for indexed rows.

        One fingerprint per summary (all seq pieces of one summary share it);
        cheap pending-work probe for the indexing runner.
        """
        rows = self.conn.execute(
            "SELECT DISTINCT summary_kind, summary_id, fingerprint "
            "FROM summary_index_entries"
        ).fetchall()
        return {
            (r["summary_kind"], r["summary_id"]): r["fingerprint"] for r in rows
        }

    def summary_index_pending(self, *, limit: int = 32) -> list[dict]:
        """Return stored summaries whose index rows are missing or stale.

        Weeks first (few, high leverage), then block summaries newest-first;
        bounded by ``limit``. Each item carries ``summary_kind``, ``summary_id``,
        ``fingerprint`` (the CURRENT source fingerprint), and ``text`` — the
        caller re-indexes via :meth:`index_summary`. Never logs the text.
        """
        state = self.summary_index_state()
        pending: list[dict] = []

        week_rows = self.conn.execute(
            "SELECT id, payload_json FROM day_memories "
            "WHERE source_scope = 'week' ORDER BY local_date DESC"
        ).fetchall()
        for row in week_rows:
            try:
                payload = json.loads(row["payload_json"])
            except (TypeError, ValueError):
                continue
            text = str(payload.get("digest_text") or "").strip()
            fingerprint = str(payload.get("member_fingerprint") or "")
            if not text or not fingerprint:
                continue
            if state.get(("week", row["id"])) == fingerprint:
                continue
            pending.append(
                {
                    "summary_kind": "week",
                    "summary_id": row["id"],
                    "fingerprint": fingerprint,
                    "text": text,
                }
            )

        block_rows = self.conn.execute(
            "SELECT id, block_fingerprint, summary_text FROM block_summaries "
            "ORDER BY generated_at DESC, id"
        ).fetchall()
        for row in block_rows:
            text = str(row["summary_text"] or "").strip()
            if not text:
                continue
            if state.get(("block", row["id"])) == row["block_fingerprint"]:
                continue
            pending.append(
                {
                    "summary_kind": "block",
                    "summary_id": row["id"],
                    "fingerprint": row["block_fingerprint"],
                    "text": text,
                }
            )

        return pending[: max(0, int(limit))]

    def _summary_source_fingerprint(
        self, summary_kind: str, summary_id: str
    ) -> str | None:
        """Return the LIVE source fingerprint for an index target, or None.

        Used by :meth:`index_summary` to revalidate inside the write
        transaction that the summary still exists with the same fingerprint
        after the (lock-free) embedding phase.
        """
        if summary_kind == "block":
            row = self.conn.execute(
                "SELECT block_fingerprint AS fp FROM block_summaries WHERE id = ?",
                (summary_id,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT payload_json FROM day_memories "
                "WHERE id = ? AND source_scope = 'week'",
                (summary_id,),
            ).fetchone()
            if row is None:
                return None
            try:
                payload = json.loads(row["payload_json"])
            except (TypeError, ValueError):
                return None
            fp = payload.get("member_fingerprint")
            return str(fp) if fp is not None else None
        return row["fp"] if row is not None else None

    def index_summary(
        self, *, summary_kind: str, summary_id: str, fingerprint: str, text: str
    ) -> int:
        """(Re)index one summary's text: entries + FTS + vectors. Returns pieces.

        Two-phase like :meth:`add_observation`: the embedding provider call runs
        OUTSIDE the write lock, then a short transaction deletes any stale index
        rows for this summary (all three tables, honoring the deletion contract)
        and inserts the fresh ones. Long week digests are split with
        ``ingest.chunk``; ``seq`` preserves piece order. Embedding is
        egress-bearing under a remote embed route — callers must sit behind the
        provider's cloud gate (they do: the routines pass / summaries build).
        """
        if summary_kind not in ("block", "week"):
            raise ValueError(f"unknown summary_kind: {summary_kind!r}")
        pieces = [ctext for _span, ctext in ingest.chunk(text)]
        pieces = [p for p in pieces if p.strip()]
        if not pieces:
            return 0
        vectors = self.provider.embed(pieces)
        packed = [_serialize_f32(v) for v in vectors]
        try:
            self._begin()
            # TOCTOU guard: the embed ran outside the lock, so the source may
            # have been deleted or regenerated meanwhile. Re-validate inside
            # the transaction; on miss/drift, sweep any stale rows and no-op —
            # indexing a dead/stale summary would violate the zero-orphan
            # contract the deletion APIs enforce.
            live_fp = self._summary_source_fingerprint(summary_kind, summary_id)
            if live_fp is None or live_fp != fingerprint:
                self._sweep_summary_index_pairs([(summary_kind, summary_id)])
                self.conn.commit()
                return 0
            self._sweep_summary_index_pairs([(summary_kind, summary_id)])
            for seq, (piece, vector) in enumerate(zip(pieces, packed)):
                cur = self.conn.execute(
                    "INSERT INTO summary_index_entries("
                    "summary_kind, summary_id, seq, text, fingerprint"
                    ") VALUES (?, ?, ?, ?, ?)",
                    (summary_kind, summary_id, seq, piece, fingerprint),
                )
                rid = int(cur.lastrowid)
                self.conn.execute(
                    "INSERT INTO fts_summaries(rowid, text) VALUES (?, ?)",
                    (rid, piece),
                )
                self.conn.execute(
                    "INSERT INTO vec_summaries(entry_rowid, embedding) VALUES (?, ?)",
                    (rid, vector),
                )
            self.conn.commit()
            return len(pieces)
        except Exception:
            self.conn.rollback()
            raise

    def search_summaries(self, query: str, k: int = 6, *, semantic: bool = True) -> list[dict]:
        """Hybrid search over the summary index: BM25 + vector, RRF-fused.

        Returns at most ``k`` dicts ``{summary_kind, summary_id, text, score,
        start_ts, end_ts, local_date, source_refs}`` resolved back to the live
        ``block_summaries`` / week ``day_memories`` rows (dead entries resolve
        to nothing and drop). ``search()`` itself is untouched — the occurrence
        retrieval model stays pure; merging happens in RAG. Embedding failures
        degrade to BM25-only ranking, mirroring :meth:`search`.
        """
        if not query.strip():
            return []
        pool = max(k * 5, 20)
        rankings: list[list[str]] = []

        match = self._fts_query(query)
        if match:
            rows = self.conn.execute(
                "SELECT rowid FROM fts_summaries WHERE fts_summaries MATCH ? "
                "ORDER BY bm25(fts_summaries) LIMIT ?",
                (match, pool),
            ).fetchall()
            if rows:
                rankings.append([str(r["rowid"]) for r in rows])

        if semantic:
            try:
                vector = self.provider.embed([query])[0]
                rows = self.conn.execute(
                    "SELECT entry_rowid FROM vec_summaries WHERE embedding MATCH ? "
                    "ORDER BY distance LIMIT ?",
                    (_serialize_f32(vector), pool),
                ).fetchall()
            except Exception as exc:  # noqa: BLE001 - embedding failure must not break search
                _log_vector_skip(type(exc).__name__)
            else:
                if rows:
                    rankings.append([str(r["entry_rowid"]) for r in rows])

        if not rankings:
            return []

        fused = rrf(rankings)
        results: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for rowid, score in fused:
            entry = self.conn.execute(
                "SELECT summary_kind, summary_id FROM summary_index_entries "
                "WHERE entry_rowid = ?",
                (int(rowid),),
            ).fetchone()
            if entry is None:
                continue
            key = (entry["summary_kind"], entry["summary_id"])
            if key in seen:
                continue  # best-ranked piece of a multi-piece summary wins
            seen.add(key)
            resolved = self._resolve_summary_result(key[0], key[1], score)
            if resolved is not None:
                results.append(resolved)
            if len(results) >= k:
                break
        return results

    def _resolve_summary_result(
        self, summary_kind: str, summary_id: str, score: float
    ) -> dict | None:
        """Resolve one indexed summary back to its live source row."""
        if summary_kind == "block":
            item = self._block_summary_by_id(summary_id)
            if item is None:
                return None
            return {
                "summary_kind": "block",
                "summary_id": summary_id,
                "text": item["summary_text"],
                "score": float(score),
                "start_ts": item["start_ts"],
                "end_ts": item["end_ts"],
                "local_date": item["local_date"],
                "source_refs": item["source_refs"],
            }
        row = self.conn.execute(
            "SELECT * FROM day_memories WHERE id = ? AND source_scope = 'week'",
            (summary_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, ValueError):
            return None
        window = payload.get("window") or {}
        refs = self.conn.execute(
            "SELECT source_kind, source_id FROM day_memory_source_refs "
            "WHERE day_memory_id = ? ORDER BY source_kind, source_id",
            (summary_id,),
        ).fetchall()
        return {
            "summary_kind": "week",
            "summary_id": summary_id,
            "text": str(payload.get("digest_text") or ""),
            "score": float(score),
            "start_ts": window.get("start"),
            "end_ts": window.get("end"),
            "local_date": row["local_date"],
            "source_refs": [
                {"source_kind": r["source_kind"], "source_id": r["source_id"]}
                for r in refs
            ],
        }

    def summary_index_orphan_counts(self) -> dict:
        """Count summary-index rows violating the deletion contract (probe).

        Returns counts of (a) fts/vec rows without a ``summary_index_entries``
        row and (b) entries without a live source summary. Non-zero counts mean
        a production path bypassed the sweep APIs — `openbird data integrity`
        surfaces them so contract violations are detected instead of silently
        degrading search. Counts only; never summary text.
        """
        return _summary_index_orphan_counts(self.conn)

    # -- entity ledger (Phase E2) -------------------------------------------------

    def upsert_entity(
        self,
        kind: str,
        name: str,
        *,
        seen_ts: float,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> dict:
        """Create-or-refresh one ledger entity; returns the stored row.

        The id is deterministic — ``sha256("kind:casefold(name)")`` — so
        repeated aggregation runs upsert idempotently. On conflict:
        ``first_ts`` keeps the minimum, ``last_ts`` the maximum, the TYPED
        ``last_seen_source_kind``/``last_seen_source_id`` pair is replaced only
        when this sighting is at least as new as the stored ``last_ts``, and a
        ``dormant`` entity flips back to ``active`` on strictly newer activity.
        ``user_marked_done`` is NEVER touched here (user intent outranks
        activity). ``name`` is DERIVED SENSITIVE — this method logs nothing.
        """
        if kind not in _ENTITY_KINDS:
            raise ValueError(f"unknown entity kind: {kind!r}")
        if (source_kind is None) != (source_id is None):
            raise ValueError("source_kind and source_id must be given together")
        if source_kind is not None and source_kind not in _ENTITY_SOURCE_KINDS:
            raise ValueError(f"unknown source kind: {source_kind!r}")
        entity_id = entity_id_for(kind, name)
        ts = float(seen_ts)
        try:
            self._begin()
            self.conn.execute(
                """
                INSERT INTO entities(
                    id, kind, name, aliases, first_ts, last_ts, status,
                    last_seen_source_kind, last_seen_source_id
                ) VALUES (?, ?, ?, '[]', ?, ?, 'active', ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    first_ts = MIN(COALESCE(first_ts, excluded.first_ts),
                                   excluded.first_ts),
                    last_seen_source_kind = CASE
                        WHEN excluded.last_seen_source_kind IS NOT NULL
                             AND excluded.last_ts >= COALESCE(last_ts, excluded.last_ts)
                        THEN excluded.last_seen_source_kind
                        ELSE last_seen_source_kind END,
                    last_seen_source_id = CASE
                        WHEN excluded.last_seen_source_kind IS NOT NULL
                             AND excluded.last_ts >= COALESCE(last_ts, excluded.last_ts)
                        THEN excluded.last_seen_source_id
                        ELSE last_seen_source_id END,
                    status = CASE
                        WHEN status = 'dormant'
                             AND (last_ts IS NULL OR excluded.last_ts > last_ts)
                        THEN 'active' ELSE status END,
                    last_ts = MAX(COALESCE(last_ts, excluded.last_ts),
                                  excluded.last_ts)
                """,
                (entity_id, kind, name, ts, ts, source_kind, source_id),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        saved = self.get_entity(entity_id)
        if saved is None:  # pragma: no cover - the insert above would have raised
            raise RuntimeError("entity row was not inserted")
        return saved

    def get_entity(self, entity_id: str) -> dict | None:
        """Return one entity row (aliases decoded), or ``None``."""
        row = self.conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        return self._entity_row_to_dict(row) if row is not None else None

    @staticmethod
    def _entity_row_to_dict(row) -> dict:
        item = dict(row)
        try:
            aliases = json.loads(item.get("aliases") or "[]")
        except (TypeError, ValueError):
            aliases = []
        item["aliases"] = [str(a) for a in aliases] if isinstance(aliases, list) else []
        return item

    def set_entity_aliases(self, entity_id: str, aliases: list[str]) -> None:
        """Replace one entity's alias list (aggregation-only writer)."""
        payload = json.dumps(sorted({str(a) for a in aliases if a}), ensure_ascii=False)
        try:
            self._begin()
            self.conn.execute(
                "UPDATE entities SET aliases = ? WHERE id = ?", (payload, entity_id)
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def set_entity_status(self, entity_id: str, status: str) -> None:
        """Set one entity's status (schema-ready for the mark-done affordance)."""
        if status not in ("active", "dormant", "user_marked_done"):
            raise ValueError(f"unknown entity status: {status!r}")
        try:
            self._begin()
            self.conn.execute(
                "UPDATE entities SET status = ? WHERE id = ?", (status, entity_id)
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def add_entity_evidence(
        self,
        entity_id: str,
        *,
        ts: float,
        kind: str,
        source_kind: str,
        source_id: str,
        detail: str = "",
    ) -> bool:
        """Insert one evidence row; returns whether a row was actually inserted.

        ``INSERT OR IGNORE`` + the UNIQUE constraint make re-runs idempotent
        (the aggregation overlap re-scan relies on this); the per-kind BEFORE
        INSERT triggers still fail loudly on an unknown source ref (RAISE ABORT
        overrides OR IGNORE). ``detail`` is DERIVED SENSITIVE — never logged.
        """
        if kind not in _ENTITY_EVIDENCE_KINDS:
            raise ValueError(f"unknown evidence kind: {kind!r}")
        if source_kind not in _ENTITY_SOURCE_KINDS:
            raise ValueError(f"unknown source kind: {source_kind!r}")
        try:
            self._begin()
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO entity_evidence("
                "id, entity_id, ts, kind, source_kind, source_id, detail"
                ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                (uuid.uuid4().hex, entity_id, float(ts), kind, source_kind,
                 source_id, detail or ""),
            )
            inserted = cur.rowcount == 1
            self.conn.commit()
            return inserted
        except Exception:
            self.conn.rollback()
            raise

    def list_entities(
        self, *, kind: str | None = None, status: str | None = None
    ) -> list[dict]:
        """Return entity rows (aliases decoded), newest activity first."""
        sql = "SELECT * FROM entities WHERE 1=1"
        params: list[object] = []
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY last_ts DESC, name ASC"
        rows = self.conn.execute(sql, params).fetchall()
        return [self._entity_row_to_dict(r) for r in rows]

    def entities_matching(self, text: str) -> list[dict]:
        """Entities whose name/alias matches ``text`` (casefolded, both directions).

        A candidate matches when the casefolded query equals a name/alias,
        contains one, or is contained by one — the RAG completion path then
        prefers exact matches and refuses to guess between several candidates.
        The entity table is small (exact-identity rows), so this is a plain
        scan; no fts/vec indexing by design.
        """
        needle = " ".join(str(text or "").split()).casefold()
        if not needle:
            return []
        out: list[dict] = []
        for item in self.list_entities():
            keys = [str(item.get("name") or "").casefold()]
            keys += [a.casefold() for a in item.get("aliases") or []]
            keys = [k for k in keys if k]
            if any(k == needle or k in needle or needle in k for k in keys):
                out.append(item)
        return out

    def entity_evidence_for(
        self,
        entity_id: str,
        limit: int = 50,
        *,
        start_ts: float | None = None,
        end_ts: float | None = None,
    ) -> list[dict]:
        """Return one entity's evidence rows, newest first (bounded).

        The optional ``[start_ts, end_ts]`` window is applied IN SQL, BEFORE
        the newest-first LIMIT — a windowed completion answer must surface an
        older period's evidence even when more than ``limit`` newer rows
        exist (filtering a newest-N fetch afterwards would falsely report
        "no activity in that period": a grounding-rule violation).
        """
        sql = "SELECT * FROM entity_evidence WHERE entity_id = ?"
        params: list[object] = [entity_id]
        if start_ts is not None:
            sql += " AND ts >= ?"
            params.append(float(start_ts))
        if end_ts is not None:
            sql += " AND ts <= ?"
            params.append(float(end_ts))
        sql += " ORDER BY ts DESC, id DESC LIMIT ?"
        params.append(max(0, int(limit)))
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def mark_dormant_entities(self, cutoff_ts: float) -> int:
        """Flip ACTIVE entities with no activity since ``cutoff_ts`` to dormant.

        ``user_marked_done`` rows are immune (the WHERE targets 'active' only).
        Returns the number of rows flipped. Own short transaction — the
        aggregation-time dormancy path; the delete()/prune() paths run their
        own synchronous in-transaction pass instead.
        """
        try:
            self._begin()
            cur = self.conn.execute(
                "UPDATE entities SET status = 'dormant' "
                "WHERE status = 'active' AND (last_ts IS NULL OR last_ts < ?)",
                (float(cutoff_ts),),
            )
            count = int(cur.rowcount or 0)
            self.conn.commit()
            return count
        except Exception:
            self.conn.rollback()
            raise

    def _mark_evidence_less_entities_dormant(self) -> int:
        """Flip ACTIVE entities whose evidence set is now empty to dormant.

        MUST run inside an already-open write transaction — this is the
        synchronous delete()/prune() pass (never inside a trigger), so a
        selective purge cannot leave evidence-less entities looking active
        until the next routine run. ``user_marked_done`` rows are immune.
        """
        cur = self.conn.execute(
            "UPDATE entities SET status = 'dormant' WHERE status = 'active' "
            "AND id NOT IN (SELECT DISTINCT entity_id FROM entity_evidence)"
        )
        return int(cur.rowcount or 0)

    def observations_text_page(
        self,
        after_ts: float,
        after_id: str,
        *,
        limit: int,
        max_chars: int = 4000,
    ) -> list[tuple[Observation, str]]:
        """One ROW-CAPPED page of observations+text after a composite cursor.

        ``ORDER BY ts ASC, id ASC LIMIT ?`` with a strict ``(ts, id) >
        (after_ts, after_id)`` predicate — the entity aggregation pass's real
        batching bound (``time_range_text``'s ``max_chars`` caps text per row,
        NOT row count, so it cannot bound a first 14-day scan). ``max_chars``
        truncates each blob body; returned text is untrusted captured content.
        """
        rows = self.conn.execute(
            "SELECT o.*, b.text AS blob_text FROM observations o "
            "JOIN content_blobs b ON b.content_hash = o.content_hash "
            "WHERE (o.ts > ? OR (o.ts = ? AND o.id > ?)) "
            "ORDER BY o.ts ASC, o.id ASC LIMIT ?",
            (float(after_ts), float(after_ts), str(after_id), max(0, int(limit))),
        ).fetchall()
        out: list[tuple[Observation, str]] = []
        for r in rows:
            text = r["blob_text"] or ""
            if max_chars and len(text) > max_chars:
                text = text[:max_chars]
            out.append((self._row_to_observation(r), text))
        return out

    def observations_text_for_ids(
        self, ids: list[str], *, max_chars: int = 4000
    ) -> list[tuple[Observation, str]]:
        """Observations+text for explicit ids, ordered by (ts, id).

        Used by the open-loop promotion REHYDRATION: day-memory payloads carry
        sorted/capped source_ids, so the aggregation pass re-reads the actual
        rows to recover the item identity and the true earliest occurrence.
        Missing ids are silently absent (their sources were deleted).
        """
        unique = sorted({str(i) for i in ids if i})
        if not unique:
            return []
        placeholders = ",".join("?" for _ in unique)
        rows = self.conn.execute(
            "SELECT o.*, b.text AS blob_text FROM observations o "
            "JOIN content_blobs b ON b.content_hash = o.content_hash "
            f"WHERE o.id IN ({placeholders}) ORDER BY o.ts ASC, o.id ASC",
            unique,
        ).fetchall()
        out: list[tuple[Observation, str]] = []
        for r in rows:
            text = r["blob_text"] or ""
            if max_chars and len(text) > max_chars:
                text = text[:max_chars]
            out.append((self._row_to_observation(r), text))
        return out

    def spans_page(
        self, after_start_ts: float, after_id: str, *, limit: int
    ) -> list[dict]:
        """One ROW-CAPPED page of activity spans after a composite cursor.

        ``ORDER BY start_ts ASC, span_id ASC LIMIT ?`` with a strict
        ``(start_ts, span_id) > (after_start_ts, after_id)`` predicate — the
        entity aggregation's INDEPENDENT span cursor (mirrors
        :meth:`observations_text_page`): a dense span history must page
        forward run over run instead of repeatedly re-slicing the earliest
        overlapping spans and never reaching later span-derived domains.
        """
        rows = self.conn.execute(
            "SELECT * FROM activity_spans "
            "WHERE (start_ts > ? OR (start_ts = ? AND span_id > ?)) "
            "ORDER BY start_ts ASC, span_id ASC LIMIT ?",
            (float(after_start_ts), float(after_start_ts), str(after_id),
             max(0, int(limit))),
        ).fetchall()
        return [dict(r) for r in rows]

    def block_summaries_generated_since(
        self, after_generated_at: float, after_id: str, *, limit: int
    ) -> list[dict]:
        """One page of block summaries after a (generated_at, id) cursor.

        The entity aggregation's SUMMARY cursor is generation-time, not
        activity-time: ``save_block_summary`` regeneration REPLACES a
        historical row in place with a fresh ``generated_at``, so this cursor
        always re-mines a regenerated old block (an activity-time watermark
        would strand it forever).
        """
        rows = self.conn.execute(
            "SELECT * FROM block_summaries "
            "WHERE (generated_at > ? OR (generated_at = ? AND id > ?)) "
            "ORDER BY generated_at ASC, id ASC LIMIT ?",
            (float(after_generated_at), float(after_generated_at), str(after_id),
             max(0, int(limit))),
        ).fetchall()
        return [self._block_summary_with_refs(dict(r)) for r in rows]

    def entity_open_loop_candidates(self) -> list[dict]:
        """Unresolved open loops paired with their exact-detail resolving row.

        The precise resolution rule (Phase E2): an ``open_loop`` row is
        resolved iff a ``pr_merged``/``ticket_closed`` row exists on the SAME
        entity with the IDENTICAL ``detail`` key and a LATER ``ts`` — never
        loop-text similarity. A loop counts as already resolved ONLY when an
        ``open_loop_resolved`` row for that detail is LATER than THAT loop
        row's ts — a REOPENED loop (newer than the last resolution) becomes a
        candidate again and resolves on the next later completion.
        """
        rows = self.conn.execute(
            """
            SELECT l.entity_id AS entity_id, l.detail AS detail, l.ts AS loop_ts,
                   c.ts AS resolved_ts, c.source_kind AS source_kind,
                   c.source_id AS source_id
            FROM entity_evidence l
            JOIN entity_evidence c
              ON c.entity_id = l.entity_id AND c.detail = l.detail
            WHERE l.kind = 'open_loop'
              AND c.kind IN ('pr_merged', 'ticket_closed')
              AND c.ts > l.ts
              AND NOT EXISTS (
                  SELECT 1 FROM entity_evidence r
                  WHERE r.entity_id = l.entity_id AND r.detail = l.detail
                    AND r.kind = 'open_loop_resolved'
                    AND r.ts > l.ts
              )
            ORDER BY l.entity_id, l.detail, c.ts ASC
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def entity_evidence_orphan_counts(self) -> dict:
        """Count evidence rows whose source row is gone (belt-and-braces probe).

        The BEFORE DELETE triggers should keep these at zero always; non-zero
        counts mean something deleted sources outside SQLite (or a trigger was
        dropped) — `openbird data integrity` surfaces them. Counts only; never
        entity names or details.
        """
        return _entity_evidence_orphan_counts(self.conn)

    # -- aggregation watermarks (embedding_meta KV) ------------------------------

    def get_kv(self, key: str) -> str | None:
        """Read one small metadata value from the ``embedding_meta`` KV table.

        ``embedding_meta`` is the store's only key/value table; the entity
        aggregation watermarks (``entity_aggregation.*``) live here as plain
        positions (timestamps/row ids — metadata, never captured content).
        """
        row = self.conn.execute(
            "SELECT value FROM embedding_meta WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else row["value"]

    def set_kv(self, key: str, value: str) -> None:
        """Write one small metadata value into the ``embedding_meta`` KV table."""
        try:
            self._begin()
            self.conn.execute(
                "INSERT INTO embedding_meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

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
                # daily artifact survives a full purge. Block summaries and the
                # taxonomy cache are likewise LLM-derived from captured content
                # — a full purge wipes them too.
                cur.execute("DELETE FROM day_memories")
                cur.execute("DELETE FROM day_memory_source_refs")
                cur.execute("DELETE FROM block_summaries")
                cur.execute("DELETE FROM block_summary_source_refs")
                cur.execute("DELETE FROM category_assignments")
                # The summary index mirrors derived-sensitive prose — wipe all
                # three tables explicitly (triggers cannot clean the virtual
                # fts/vec tables).
                cur.execute("DELETE FROM summary_index_entries")
                cur.execute("DELETE FROM fts_summaries")
                cur.execute("DELETE FROM vec_summaries")
                # Entity ledger (v7): names/aliases/details are derived
                # sensitive — a full purge wipes both tables AND the
                # aggregation watermarks (positions derived from captured
                # activity). The cohort_key row stays (see the NOTE below).
                cur.execute("DELETE FROM entity_evidence")
                cur.execute("DELETE FROM entities")
                cur.execute(
                    "DELETE FROM embedding_meta WHERE key LIKE ?",
                    (ENTITY_AGGREGATION_KV_PREFIX + "%",),
                )
                cur.execute("DELETE FROM reasoning_send_ledger")
                # Capture attempts contain no content, but bundle ids and
                # timestamps are still private activity metadata. A full purge
                # removes them; selective/retention purges keep accountability
                # and let observation_id clear through ON DELETE SET NULL.
                cur.execute("DELETE FROM capture_attempts")
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
            # Deletion contract (v6): the observation/span deletes below cascade
            # through the recursive trigger chain into block summaries and week
            # rows, whose fts/vec index rows the triggers CANNOT clean. Pre-
            # select the doomed block-summary ids (sources intersecting the
            # victim set) and sweep their index rows — including dependent week
            # rows' — in this same transaction, BEFORE the source deletes.
            doomed_summaries = cur.execute(
                "SELECT DISTINCT summary_id FROM block_summary_source_refs "
                "WHERE (source_kind = 'observation' AND source_id IN "
                f"  (SELECT id FROM observations WHERE {where})) "
                "OR (source_kind = 'span' AND source_id IN "
                f"  (SELECT span_id FROM activity_spans WHERE {span_where}))",
                (param, param),
            ).fetchall()
            self._sweep_summary_index_for_blocks(
                [r["summary_id"] for r in doomed_summaries]
            )
            cur.execute(f"DELETE FROM observations WHERE {where}", (param,))
            # Span deletion fires trg_day_memory_source_span_delete, which
            # invalidates any day memory citing a deleted span; observations
            # referencing a deleted span get span_id NULLed by the FK.
            # No new SQL is needed for block summaries here: the observation/
            # span deletes above fire trg_block_summary_source_*_delete, and
            # (recursive_triggers ON) each deleted summary in turn fires
            # trg_day_memory_source_summary_delete for day memories citing it.
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

            # Entity dormancy (v7): the source deletes above trigger-deleted
            # matching entity_evidence rows; flip any ACTIVE entity whose
            # evidence set emptied to dormant SYNCHRONOUSLY, in this same
            # transaction (user_marked_done is immune) — a selective purge
            # must not leave evidence-less entities looking active until the
            # next aggregation run.
            self._mark_evidence_less_entities_dormant()

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
        # ``day_memories`` deliberately EXCLUDES week-scope rows: verification
        # scripts (script/verify_ask_app.sh, script/beta_rehearsal.py) parse it
        # as the DAY count, so existing consumers stay truthful; week rows are
        # reported separately under ``week_memories``.
        day_count = int(
            self.conn.execute(
                "SELECT COUNT(*) AS c FROM day_memories WHERE source_scope != 'week'"
            ).fetchone()["c"]
        )
        week_count = int(
            self.conn.execute(
                "SELECT COUNT(*) AS c FROM day_memories WHERE source_scope = 'week'"
            ).fetchone()["c"]
        )
        return {
            "observations": count("observations"),
            "blobs": count("content_blobs"),
            "chunks": count("chunks"),
            "vectors": count("vec_chunks"),
            "day_memories": day_count,
            "week_memories": week_count,
            "summary_index_entries": count("summary_index_entries"),
            "activity_spans": count("activity_spans"),
            "block_summaries": count("block_summaries"),
            "category_assignments": count("category_assignments"),
            "entities": count("entities"),
            "entity_evidence": count("entity_evidence"),
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

    def capture_content_quality(
        self, *, recent_since_ts: float, source: str = "capture"
    ) -> dict[str, dict[str, float | int]]:
        """Return privacy-safe per-app aggregates of captured context richness.

        Text is read only inside SQLite to calculate lengths and line counts.
        The query emits one aggregate row per app and never returns captured
        text, titles, URLs, or individual content hashes to Python.
        """
        rows = self.conn.execute(
            """
            WITH base AS (
                SELECT
                    o.app AS app,
                    o.content_hash AS content_hash,
                    LENGTH(COALESCE(b.text, '')) AS chars,
                    CASE
                        WHEN COALESCE(b.text, '') = '' THEN 0
                        ELSE 1 + LENGTH(b.text) - LENGTH(REPLACE(b.text, CHAR(10), ''))
                    END AS lines
                FROM observations o
                JOIN content_blobs b ON b.content_hash = o.content_hash
                WHERE o.source = ?
                  AND o.ts >= ?
                  AND o.app IS NOT NULL
                  AND o.app != ''
            ),
            grouped AS (
                SELECT
                    app,
                    COUNT(*) AS sample_count,
                    COUNT(DISTINCT content_hash) AS distinct_contexts,
                    SUM(CASE WHEN chars >= 120 THEN 1 ELSE 0 END) AS substantive_count,
                    SUM(CASE WHEN chars >= 400 OR lines >= 8 THEN 1 ELSE 0 END) AS rich_count
                FROM base
                GROUP BY app
            ),
            ranked AS (
                SELECT
                    app,
                    chars,
                    lines,
                    ROW_NUMBER() OVER (PARTITION BY app ORDER BY chars) AS char_rank,
                    ROW_NUMBER() OVER (PARTITION BY app ORDER BY lines) AS line_rank,
                    COUNT(*) OVER (PARTITION BY app) AS sample_count
                FROM base
            ),
            percentiles AS (
                SELECT
                    app,
                    MIN(CASE WHEN char_rank >= (sample_count + 1) / 2 THEN chars END)
                        AS chars_p50,
                    MIN(CASE WHEN char_rank >= (9 * sample_count + 9) / 10 THEN chars END)
                        AS chars_p90,
                    MIN(CASE WHEN line_rank >= (sample_count + 1) / 2 THEN lines END)
                        AS lines_p50,
                    MIN(CASE WHEN line_rank >= (9 * sample_count + 9) / 10 THEN lines END)
                        AS lines_p90
                FROM ranked
                GROUP BY app
            )
            SELECT
                g.app,
                g.sample_count,
                g.distinct_contexts,
                p.chars_p50,
                p.chars_p90,
                p.lines_p50,
                p.lines_p90,
                CAST(g.substantive_count AS REAL) / g.sample_count AS substantive_ratio,
                CAST(g.rich_count AS REAL) / g.sample_count AS rich_ratio
            FROM grouped g
            JOIN percentiles p ON p.app = g.app
            ORDER BY g.app
            """,
            (source, float(recent_since_ts)),
        ).fetchall()
        return {
            row["app"]: {
                "sample_count": int(row["sample_count"]),
                "distinct_contexts": int(row["distinct_contexts"]),
                "chars_p50": int(row["chars_p50"]),
                "chars_p90": int(row["chars_p90"]),
                "lines_p50": int(row["lines_p50"]),
                "lines_p90": int(row["lines_p90"]),
                "substantive_ratio": float(row["substantive_ratio"]),
                "rich_ratio": float(row["rich_ratio"]),
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


def _scalar_count(row) -> int:
    """Read one COUNT(*) value regardless of row_factory (dict or tuple)."""
    if row is None:
        return 0
    value = next(iter(row.values())) if isinstance(row, dict) else row[0]
    return int(value or 0)


def _summary_index_orphan_counts(conn) -> dict:
    """Compute the summary-index orphan counts over an open connection.

    Works with either row_factory (MemoryStore's dict rows or a raw tuple
    connection). Counts only; never summary text.
    """
    fts_orphans = _scalar_count(
        conn.execute(
            "SELECT COUNT(*) FROM fts_summaries "
            "WHERE rowid NOT IN (SELECT entry_rowid FROM summary_index_entries)"
        ).fetchone()
    )
    vec_orphans = _scalar_count(
        conn.execute(
            "SELECT COUNT(*) FROM vec_summaries "
            "WHERE entry_rowid NOT IN (SELECT entry_rowid FROM summary_index_entries)"
        ).fetchone()
    )
    entry_orphans = _scalar_count(
        conn.execute(
            "SELECT COUNT(*) FROM summary_index_entries e "
            "WHERE (e.summary_kind = 'block' AND NOT EXISTS "
            "  (SELECT 1 FROM block_summaries b WHERE b.id = e.summary_id)) "
            "OR (e.summary_kind = 'week' AND NOT EXISTS "
            "  (SELECT 1 FROM day_memories d WHERE d.id = e.summary_id "
            "   AND d.source_scope = 'week'))"
        ).fetchone()
    )
    return {
        "fts_orphans": fts_orphans,
        "vec_orphans": vec_orphans,
        "entry_orphans": entry_orphans,
        "ok": fts_orphans == 0 and vec_orphans == 0 and entry_orphans == 0,
    }


def check_summary_index_orphans(
    db_path: str,
    *,
    settings=None,
    opener=None,
) -> dict:
    """Raw-open probe for summary-index orphans (``openbird data integrity``).

    Mirrors :func:`check_database_integrity`'s never-raise contract: open
    failures and query errors become findings, and a pre-v6 DB (tables absent)
    is a clean skip (``{"ok": True, "counts": None}``). Reports counts only.
    """
    import sqlite3

    def _default_opener():
        from openbird.storage.crypto import open_encrypted_db

        return open_encrypted_db(db_path, settings=settings)

    try:
        conn = (opener or _default_opener)()
    except Exception as exc:  # noqa: BLE001 - a diagnostic must report, not crash
        return {
            "ok": False,
            "counts": None,
            "problems": [f"cannot-open: {type(exc).__name__}"],
        }
    try:
        tables = {
            str(_scalar_name(row))
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN "
                "('summary_index_entries', 'fts_summaries', 'vec_summaries')"
            ).fetchall()
        }
        if not {"summary_index_entries", "fts_summaries", "vec_summaries"} <= tables:
            # Pre-v6 DB, or an upgraded DB not yet opened by MemoryStore (which
            # creates vec_summaries): nothing to probe.
            return {"ok": True, "counts": None, "problems": []}
        counts = _summary_index_orphan_counts(conn)
        problems: list[str] = []
        if not counts["ok"]:
            problems.append(
                "summary-index-orphans: "
                f"fts={counts['fts_orphans']} vec={counts['vec_orphans']} "
                f"entries={counts['entry_orphans']}"
            )
        return {"ok": counts["ok"], "counts": counts, "problems": problems}
    except sqlite3.DatabaseError as exc:
        return {
            "ok": False,
            "counts": None,
            "problems": [f"check-failed: {type(exc).__name__}"],
        }
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def _scalar_name(row) -> object:
    """Read one single-column value regardless of row_factory."""
    if isinstance(row, dict):
        return next(iter(row.values()))
    return row[0]


def _entity_evidence_orphan_counts(conn) -> dict:
    """Compute entity-evidence orphan counts over an open connection.

    Evidence whose typed source row is gone should ALWAYS be zero (the
    BEFORE DELETE triggers cover every path); non-zero means a trigger was
    bypassed or dropped. Counts only; never entity names or details.
    """
    obs_orphans = _scalar_count(
        conn.execute(
            "SELECT COUNT(*) FROM entity_evidence e "
            "WHERE e.source_kind = 'observation' AND NOT EXISTS "
            "  (SELECT 1 FROM observations o WHERE o.id = e.source_id)"
        ).fetchone()
    )
    span_orphans = _scalar_count(
        conn.execute(
            "SELECT COUNT(*) FROM entity_evidence e "
            "WHERE e.source_kind = 'span' AND NOT EXISTS "
            "  (SELECT 1 FROM activity_spans s WHERE s.span_id = e.source_id)"
        ).fetchone()
    )
    summary_orphans = _scalar_count(
        conn.execute(
            "SELECT COUNT(*) FROM entity_evidence e "
            "WHERE e.source_kind = 'summary' AND NOT EXISTS "
            "  (SELECT 1 FROM block_summaries b WHERE b.id = e.source_id)"
        ).fetchone()
    )
    return {
        "observation_orphans": obs_orphans,
        "span_orphans": span_orphans,
        "summary_orphans": summary_orphans,
        "ok": obs_orphans == 0 and span_orphans == 0 and summary_orphans == 0,
    }


def check_entity_evidence_orphans(
    db_path: str,
    *,
    settings=None,
    opener=None,
) -> dict:
    """Raw-open probe for entity-evidence orphans (``openbird data integrity``).

    Mirrors :func:`check_summary_index_orphans`'s never-raise contract: open
    failures and query errors become findings, and a pre-v7 DB (tables absent)
    is a clean skip (``{"ok": True, "counts": None}``). Reports counts only.
    """
    import sqlite3

    def _default_opener():
        from openbird.storage.crypto import open_encrypted_db

        return open_encrypted_db(db_path, settings=settings)

    try:
        conn = (opener or _default_opener)()
    except Exception as exc:  # noqa: BLE001 - a diagnostic must report, not crash
        return {
            "ok": False,
            "counts": None,
            "problems": [f"cannot-open: {type(exc).__name__}"],
        }
    try:
        tables = {
            str(_scalar_name(row))
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN "
                "('entities', 'entity_evidence')"
            ).fetchall()
        }
        if not {"entities", "entity_evidence"} <= tables:
            # Pre-v7 DB: nothing to probe.
            return {"ok": True, "counts": None, "problems": []}
        counts = _entity_evidence_orphan_counts(conn)
        problems: list[str] = []
        if not counts["ok"]:
            problems.append(
                "entity-evidence-orphans: "
                f"observations={counts['observation_orphans']} "
                f"spans={counts['span_orphans']} "
                f"summaries={counts['summary_orphans']}"
            )
        return {"ok": counts["ok"], "counts": counts, "problems": problems}
    except sqlite3.DatabaseError as exc:
        return {
            "ok": False,
            "counts": None,
            "problems": [f"check-failed: {type(exc).__name__}"],
        }
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


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


__all__ = [
    "ENTITY_AGGREGATION_KV_PREFIX",
    "EmbeddingCohortMismatch",
    "MemoryStore",
    "check_database_integrity",
    "check_entity_evidence_orphans",
    "check_summary_index_orphans",
    "entity_id_for",
]
