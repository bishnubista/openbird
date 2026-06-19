"""Durable persistence for scheduled-routine executions.

A routine "run" is a single firing of a routine at a particular scheduled time.
Runs are persisted in a dedicated ``routine_runs`` SQLite table keyed by an
**idempotency key** (``<routine>@<scheduled_ts>`` by default) so that:

  * the same scheduled occurrence is never executed twice (idempotency), and
  * occurrences that were *missed* while the daemon was down can be detected and
    re-run on the next startup (missed-job catchup).

Privacy boundary
------------------------
Routine bookkeeping is operational metadata, but routine *output* (the generated
summary) is derived from captured on-screen content and therefore must not sit
in plaintext outside the encrypted memory boundary. This store opens its DB
through :func:`openbird.storage.crypto.open_encrypted_db`, which prefers
SQLCipher (whole-DB encryption) and otherwise degrades **honestly** to a
``0600`` plaintext file (``settings.encryption_enabled = False``).

  * When encryption is active, the full summary text may be persisted (it is
    encrypted at rest like the rest of memory).
  * When encryption is NOT active, only **non-content metadata** is persisted
    (status, an output length, a content hash, an error class/code). The
    plaintext column never receives captured-content-derived summary bodies.

Thread safety
-------------
APScheduler fires jobs on a background executor thread, so ``claim()`` /
``finish()`` are called from a different thread than the one that constructed
the store. SQLite connections are not safe to share across threads, so this
store keeps **one connection per thread** (thread-local) and guards all writes
with a process-wide :class:`threading.Lock`. Each thread-local connection is
opened through the same encrypted-DB helper.

Lease model
-----------
``claim()`` inserts a row in ``running`` state with a fresh ``lease_ts``. A
crash between claim and finish would otherwise strand the occurrence forever
(its idempotency key exists, so it is never retried). :meth:`reclaim_stale`
atomically marks ``running`` rows whose lease has expired as ``error`` (a
content-safe ``LEASE_EXPIRED`` code), freeing the occurrence to be retried while
guaranteeing only one active execution at a time.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from datetime import date, datetime, time as dt_time, timedelta, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from collections.abc import Callable

from openbird.config import Settings, get_settings
from openbird.storage.crypto import mapping_row_factory, open_encrypted_db
from openbird.types import RoutineRun

# A clock callable returning the current unix time; injectable for tests so that
# lease timestamps and reclamation use the same (possibly fake) time base.
Clock = Callable[[], float]

logger = logging.getLogger("openbird.routines")

# Run lifecycle statuses persisted in the ``status`` column.
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_MISSED = "missed"

# Terminal statuses: a run in one of these states will never be (re-)executed.
_TERMINAL = frozenset({STATUS_DONE, STATUS_ERROR})

# Default lease timeout (seconds): a ``running`` row older than this is assumed
# to belong to a crashed worker and is reclaimable.
DEFAULT_LEASE_TIMEOUT = 900.0

# Content-safe error code recorded when a stale lease is reclaimed.
ERROR_CODE_LEASE_EXPIRED = "LEASE_EXPIRED"

# Content-safe error code for a delivery-sink failure that occurred AFTER the
# runner produced output. The run is recorded terminally, but (like a reclaimed
# crash) the occurrence is freed for a later retry — see :meth:`RoutineStore.fail_delivery`.
ERROR_CODE_DELIVERY = "DELIVERY_EXCEPTION"

# Error codes whose rows represent an occurrence that was NOT durably completed:
# the grid attempt was freed (re-keyed) and the occurrence is eligible for retry.
# These rows must never anchor the catch-up grid or they would swallow the very
# occurrence they freed.
_FREED_FOR_RETRY = (ERROR_CODE_LEASE_EXPIRED, ERROR_CODE_DELIVERY)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS routine_runs (
    id              TEXT PRIMARY KEY,
    routine         TEXT NOT NULL,
    scheduled_ts    REAL NOT NULL,
    started_ts      REAL,
    finished_ts     REAL,
    lease_ts        REAL,
    status          TEXT NOT NULL,
    attempt         INTEGER NOT NULL DEFAULT 1,
    output          TEXT,
    output_len      INTEGER,
    output_hash     TEXT,
    error_code      TEXT,
    error_class     TEXT,
    idempotency_key TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_routine_runs_routine ON routine_runs(routine);
CREATE INDEX IF NOT EXISTS idx_routine_runs_status  ON routine_runs(status);
CREATE INDEX IF NOT EXISTS idx_routine_runs_sched   ON routine_runs(scheduled_ts);
CREATE INDEX IF NOT EXISTS idx_routine_runs_lease   ON routine_runs(lease_ts);
"""


def default_idempotency_key(routine: str, scheduled_ts: float) -> str:
    """Build the default idempotency key for a (routine, scheduled time) pair.

    The key collapses sub-second jitter to whole seconds so that two firings
    nominally targeting the same occurrence map to the same key.
    """
    return f"{routine}@{int(scheduled_ts)}"


def next_scheduled_occurrence(
    interval: float,
    *,
    after: float,
    timezone: tzinfo | str | None = None,
) -> float:
    """Return the next scheduled timestamp strictly after ``after``.

    Sub-day intervals stay on a stable unix-second grid. Whole-day intervals are
    calendar intervals in local wall-clock time, so daily/weekly routines keep
    the same local clock slot across DST transitions.
    """
    if interval <= 0:
        raise ValueError("interval must be positive")

    days = _calendar_interval_days(interval)
    if days is None:
        return (math.floor(after / interval) + 1) * interval

    tz = _coerce_timezone(timezone)
    floor = _calendar_grid_floor(after, days=days, timezone=tz)
    if floor > after:
        return floor
    return _add_calendar_days(floor, days=days, timezone=tz)


def previous_scheduled_occurrence(
    interval: float,
    *,
    at: float,
    timezone: tzinfo | str | None = None,
) -> float:
    """Return the most recent scheduled timestamp at or before ``at``."""
    if interval <= 0:
        raise ValueError("interval must be positive")

    days = _calendar_interval_days(interval)
    if days is None:
        return math.floor(at / interval) * interval

    return _calendar_grid_floor(at, days=days, timezone=_coerce_timezone(timezone))


def _calendar_interval_days(interval: float) -> int | None:
    """Return whole calendar days for day-based intervals, else ``None``."""
    day = 86400.0
    days = interval / day
    if days >= 1 and math.isclose(days, round(days)):
        return int(round(days))
    return None


def _coerce_timezone(value: tzinfo | str | None) -> tzinfo:
    """Resolve a timezone object, IANA timezone name, or local default."""
    if isinstance(value, str):
        return ZoneInfo(value)
    if value is not None:
        return value
    return _local_timezone()


def _local_timezone() -> tzinfo:
    """Best-effort local timezone lookup with UTC-safe fallback."""
    tz_env = os.environ.get("TZ")
    if tz_env:
        try:
            return ZoneInfo(tz_env)
        except ZoneInfoNotFoundError:
            pass

    try:
        target = Path("/etc/localtime").resolve()
        parts = target.parts
        if "zoneinfo" in parts:
            idx = parts.index("zoneinfo")
            name = "/".join(parts[idx + 1 :])
            if name:
                return ZoneInfo(name)
    except Exception as exc:  # noqa: BLE001 - fallback keeps scheduling alive
        logger.debug(
            "local timezone resolution failed: error_class=%s",
            type(exc).__name__,
        )

    return datetime.now().astimezone().tzinfo or ZoneInfo("UTC")


def _calendar_grid_floor(ts: float, *, days: int, timezone: tzinfo) -> float:
    """Return the calendar-grid timestamp at or before ``ts``."""
    local = datetime.fromtimestamp(ts, timezone)
    local_day = local.date()
    epoch_day = date(1970, 1, 1)
    offset = (local_day - epoch_day).days % days
    scheduled_day = local_day - timedelta(days=offset)
    scheduled = datetime.combine(scheduled_day, dt_time.min, tzinfo=timezone)
    if scheduled.timestamp() > ts:
        scheduled_day -= timedelta(days=days)
        scheduled = datetime.combine(scheduled_day, dt_time.min, tzinfo=timezone)
    return scheduled.timestamp()


def _add_calendar_days(ts: float, *, days: int, timezone: tzinfo) -> float:
    """Add calendar days in local time, preserving wall-clock fields."""
    local = datetime.fromtimestamp(ts, timezone)
    return (local + timedelta(days=days)).timestamp()


class RoutineStore:
    """SQLite-backed durable log of routine runs with idempotency guarantees.

    The store is safe to call from multiple threads: it keeps one connection per
    thread and serializes all access with an internal lock.
    """

    def __init__(
        self,
        db_path: str | None = None,
        *,
        settings: Settings | None = None,
        lease_timeout: float = DEFAULT_LEASE_TIMEOUT,
        clock: Clock | None = None,
    ) -> None:
        """Open (or create) the routine-runs database.

        Args:
            db_path: Override DB path; ``":memory:"`` is supported for tests.
                Defaults to ``<data_dir>/routines.db`` so routine bookkeeping is
                isolated from the captured-content memory DB.
            settings: Settings; defaults to :func:`get_settings`.
            lease_timeout: Seconds after which a ``running`` row is considered
                stale (its worker presumed crashed) and reclaimable.
            clock: Returns current unix time, used for lease timestamps and the
                default reclamation ``now``. Defaults to :func:`time.time`.
                Injecting a fake clock keeps lease/reclaim comparisons coherent.
        """
        self.settings = settings or get_settings()
        self.lease_timeout = lease_timeout
        self._clock: Clock = clock or time.time
        if db_path is not None:
            self._resolved = db_path
        else:
            self._resolved = str(Path(self.settings.data_dir) / "routines.db")

        # A single in-memory DB cannot be reopened per-thread (each connection
        # would get its own empty database), so for ":memory:" we share one
        # connection across threads (guarded by the lock) using
        # check_same_thread=False. File-backed and SQLCipher DBs get a real
        # per-thread connection.
        self._is_memory = self._resolved == ":memory:"
        self._lock = threading.Lock()
        self._local = threading.local()
        self._shared_conn: sqlite3.Connection | None = None
        # Every per-thread connection ever opened (e.g. by APScheduler worker
        # threads), so close() can reclaim them all — not just the caller's.
        self._all_conns: list[sqlite3.Connection] = []

        if self._is_memory:
            self._shared_conn = self._make_memory_conn()
        else:
            # Open eagerly once to initialize the schema and surface errors now.
            self._conn()

    # -- connection management -----------------------------------------------

    def _make_memory_conn(self) -> sqlite3.Connection:
        """Create the shared in-memory connection (test-only path)."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = mapping_row_factory
        conn.executescript(_SCHEMA)
        conn.commit()
        return conn

    def _conn(self) -> sqlite3.Connection:
        """Return this thread's connection, opening + initializing it lazily.

        For ``":memory:"`` the single shared connection is returned (its access
        is serialized by :attr:`_lock`). For file/SQLCipher DBs a fresh
        connection is opened per thread so SQLite is never used cross-thread.
        """
        if self._is_memory:
            assert self._shared_conn is not None
            return self._shared_conn

        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = open_encrypted_db(self._resolved, settings=self.settings)
            conn.row_factory = mapping_row_factory
            conn.executescript(_SCHEMA)
            conn.commit()
            self._local.conn = conn
            # NOTE: _conn() is always called while the caller already holds
            # self._lock (claim/finish/etc.), so this append is serialized — do
            # NOT re-acquire self._lock here (threading.Lock is not reentrant; that
            # would deadlock). list.append is atomic under the GIL regardless.
            self._all_conns.append(conn)
        return conn

    @property
    def encryption_enabled(self) -> bool:
        """Whether the underlying DB is encrypted at rest (SQLCipher active)."""
        return bool(self.settings.encryption_enabled)

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Close ALL opened connections (every worker thread's), and the shared one."""
        if self._is_memory:
            if self._shared_conn is not None:
                self._shared_conn.close()
                self._shared_conn = None
            return
        with self._lock:
            conns = list(self._all_conns)
            self._all_conns.clear()
        for conn in conns:
            try:
                conn.close()
            except Exception:
                pass
        self._local.conn = None

    def __enter__(self) -> RoutineStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- claiming / idempotency ----------------------------------------------

    def claim(
        self,
        routine: str,
        scheduled_ts: float,
        *,
        idempotency_key: str | None = None,
    ) -> RoutineRun | None:
        """Atomically claim a run slot for ``(routine, scheduled_ts)``.

        Returns a freshly-inserted :class:`RoutineRun` in ``running`` state if
        this occurrence has not been claimed before, or ``None`` if a run with
        the same idempotency key already exists (i.e. it was already executed or
        is in progress). This is the core idempotency primitive: the ``UNIQUE``
        constraint on ``idempotency_key`` makes the claim race-free.

        The new row carries a ``lease_ts`` so that a worker which crashes before
        :meth:`finish` can be detected and reclaimed by :meth:`reclaim_stale`.
        """
        key = idempotency_key or default_idempotency_key(routine, scheduled_ts)
        now = self._clock()
        run_id = uuid.uuid4().hex
        with self._lock:
            conn = self._conn()
            try:
                with conn:
                    conn.execute(
                        "INSERT INTO routine_runs"
                        "(id, routine, scheduled_ts, started_ts, finished_ts,"
                        " lease_ts, status, attempt, output, output_len,"
                        " output_hash, error_code, error_class, idempotency_key)"
                        " VALUES (?, ?, ?, ?, NULL, ?, ?, 1, NULL, NULL, NULL,"
                        " NULL, NULL, ?)",
                        (run_id, routine, scheduled_ts, now, now,
                         STATUS_RUNNING, key),
                    )
            except sqlite3.IntegrityError:
                # Idempotency key already present -> someone else owns this run.
                return None
        return RoutineRun(
            id=run_id,
            routine=routine,
            scheduled_ts=scheduled_ts,
            started_ts=now,
            finished_ts=None,
            status=STATUS_RUNNING,
            output=None,
            idempotency_key=key,
        )

    def has_run(self, idempotency_key: str) -> bool:
        """Return whether a run with ``idempotency_key`` already exists."""
        with self._lock:
            conn = self._conn()
            row = conn.execute(
                "SELECT 1 FROM routine_runs WHERE idempotency_key = ? LIMIT 1",
                (idempotency_key,),
            ).fetchone()
        return row is not None

    def reclaim_stale(self, *, now: float | None = None) -> list[str]:
        """Reclaim crashed (stale-lease) ``running`` rows, atomically.

        A row is stale if it is still ``running`` and its ``lease_ts`` is older
        than ``now - lease_timeout`` — i.e. the worker that claimed it is
        presumed to have crashed.

        Each stale row is, in a single transaction:

          * transitioned to ``error`` with a content-safe ``LEASE_EXPIRED`` code
            (preserving the audit trail of the failed attempt), and
          * **re-keyed** so its original grid idempotency key is freed
            (``<key>#crashed-<id>``). Freeing the key lets a later
            :meth:`claim` create a *new attempt* for the same scheduled
            occurrence, so :meth:`missed_occurrences` will detect and retry it.

        This models leases/attempts separately from occurrences: only one row is
        ever ``running`` for a given key at a time (single active execution),
        but a crashed attempt no longer blocks a retry of the occurrence.

        Returns:
            The ids of the rows that were reclaimed.
        """
        cutoff = (self._clock() if now is None else now) - self.lease_timeout
        end = self._clock() if now is None else now
        with self._lock:
            conn = self._conn()
            with conn:
                rows = conn.execute(
                    "SELECT id, idempotency_key FROM routine_runs"
                    " WHERE status = ? AND lease_ts IS NOT NULL AND lease_ts < ?",
                    (STATUS_RUNNING, cutoff),
                ).fetchall()
                for r in rows:
                    archived_key = f"{r['idempotency_key']}#crashed-{r['id']}"
                    conn.execute(
                        "UPDATE routine_runs"
                        " SET status = ?, error_code = ?, finished_ts = ?,"
                        " lease_ts = NULL, idempotency_key = ?"
                        " WHERE id = ?",
                        (STATUS_ERROR, ERROR_CODE_LEASE_EXPIRED, end,
                         archived_key, r["id"]),
                    )
        return [r["id"] for r in rows]

    # -- completion -----------------------------------------------------------

    def finish(
        self,
        run_id: str,
        *,
        status: str = STATUS_DONE,
        output: str | None = None,
        error_code: str | None = None,
        error_class: str | None = None,
        finished_ts: float | None = None,
    ) -> RoutineRun:
        """Mark a claimed run as finished (``done`` or ``error``).

        Args:
            run_id: The ``id`` returned by :meth:`claim`.
            status: Terminal status, normally :data:`STATUS_DONE`.
            output: The delivered routine summary text. This is **content
                derived from captured data**: it is only persisted in the
                ``output`` column when the DB is encrypted at rest
                (:attr:`encryption_enabled`). When encryption is off we store
                only non-content metadata (length + hash), never the body.
            error_code: A stable, content-safe error code (e.g. ``"RUNNER"``).
            error_class: The exception class name (content-safe). Never store
                exception *messages*: they can embed captured content.
            finished_ts: Completion time; defaults to now.

        Returns:
            The updated :class:`RoutineRun`.

        Raises:
            KeyError: If ``run_id`` does not exist.
        """
        end = self._clock() if finished_ts is None else finished_ts
        stored_output, output_len, output_hash = self._content_safe_output(output)

        with self._lock:
            conn = self._conn()
            with conn:
                cur = conn.execute(
                    "UPDATE routine_runs"
                    " SET status = ?, output = ?, output_len = ?,"
                    " output_hash = ?, error_code = ?, error_class = ?,"
                    " finished_ts = ?, lease_ts = NULL"
                    " WHERE id = ?",
                    (status, stored_output, output_len, output_hash,
                     error_code, error_class, end, run_id),
                )
                if cur.rowcount == 0:
                    raise KeyError(f"unknown routine run id: {run_id!r}")
        return self.get(run_id)

    def _content_safe_output(
        self, output: str | None
    ) -> tuple[str | None, int | None, str | None]:
        """Derive ``(stored_output, output_len, output_hash)`` for ``output``.

        The body itself is only persisted inside the encrypted boundary; with
        encryption off we keep non-content metadata (length + sha256) only.
        Shared by :meth:`finish` and :meth:`fail_delivery` so the privacy rule
        is expressed in exactly one place.
        """
        if output is None:
            return None, None, None
        output_len = len(output)
        output_hash = hashlib.sha256(output.encode("utf-8")).hexdigest()
        stored_output = output if self.encryption_enabled else None
        return stored_output, output_len, output_hash

    def fail_delivery(
        self,
        run_id: str,
        *,
        output: str | None,
        error_class: str,
        finished_ts: float | None = None,
    ) -> RoutineRun:
        """Record a *retryable* delivery failure for a claimed run.

        The runner already produced ``output`` (generation succeeded); only the
        delivery sink raised. Unlike :meth:`finish`, this method:

          * **persists the generated body** (within the encrypted boundary,
            exactly like a successful run) so the work is not silently lost, and
          * **re-keys** the row's idempotency key (``<key>#delivery-failed-<id>``)
            so the occurrence's grid key is freed and :meth:`missed_occurrences`
            re-detects it — identical to crash reclamation (:meth:`reclaim_stale`).

        The exception *message* is never stored: only ``error_class`` and the
        content-safe :data:`ERROR_CODE_DELIVERY` are recorded.

        Raises:
            KeyError: If ``run_id`` does not exist.
        """
        end = self._clock() if finished_ts is None else finished_ts
        stored_output, output_len, output_hash = self._content_safe_output(output)

        with self._lock:
            conn = self._conn()
            with conn:
                row = conn.execute(
                    "SELECT idempotency_key FROM routine_runs WHERE id = ?",
                    (run_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"unknown routine run id: {run_id!r}")
                # Free the occurrence's grid key so a later claim can retry it,
                # while preserving the audit trail of the failed attempt.
                freed_key = f"{row['idempotency_key']}#delivery-failed-{run_id}"
                conn.execute(
                    "UPDATE routine_runs"
                    " SET status = ?, output = ?, output_len = ?, output_hash = ?,"
                    " error_code = ?, error_class = ?, finished_ts = ?,"
                    " lease_ts = NULL, idempotency_key = ?"
                    " WHERE id = ?",
                    (STATUS_ERROR, stored_output, output_len, output_hash,
                     ERROR_CODE_DELIVERY, error_class, end, freed_key, run_id),
                )
        return self.get(run_id)

    # -- reads ----------------------------------------------------------------

    def get(self, run_id: str) -> RoutineRun:
        """Fetch a single run by id, raising :class:`KeyError` if absent."""
        with self._lock:
            conn = self._conn()
            row = conn.execute(
                "SELECT * FROM routine_runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown routine run id: {run_id!r}")
        return self._row_to_run(row)

    def last_run(self, routine: str) -> RoutineRun | None:
        """Return the most recent run (by scheduled time) for ``routine``."""
        with self._lock:
            conn = self._conn()
            row = conn.execute(
                "SELECT * FROM routine_runs WHERE routine = ?"
                " ORDER BY scheduled_ts DESC LIMIT 1",
                (routine,),
            ).fetchone()
        return self._row_to_run(row) if row is not None else None

    def list_runs(self, routine: str | None = None) -> list[RoutineRun]:
        """List runs, optionally filtered by routine, newest scheduled first."""
        with self._lock:
            conn = self._conn()
            if routine is None:
                rows = conn.execute(
                    "SELECT * FROM routine_runs ORDER BY scheduled_ts DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM routine_runs WHERE routine = ?"
                    " ORDER BY scheduled_ts DESC",
                    (routine,),
                ).fetchall()
        return [self._row_to_run(r) for r in rows]

    # -- missed-job catchup ---------------------------------------------------

    def missed_occurrences(
        self,
        routine: str,
        interval: float,
        *,
        now: float,
        lookback: float | None = None,
        timezone: tzinfo | str | None = None,
    ) -> list[float]:
        """Compute scheduled occurrences that elapsed but were never run.

        Two sources are unioned:

          1. **Grid walk** — from the last *settled* run (or ``now - interval``
             if the routine has never run) forward to ``now``, every grid point
             whose idempotency key is not yet present in the store.
          2. **Freed-for-retry attempts** — occurrences whose in-flight attempt
             crashed and was freed by :meth:`reclaim_stale` (``LEASE_EXPIRED``),
             or whose delivery sink failed after generation
             (:meth:`fail_delivery`, ``DELIVERY_EXCEPTION``), provided the
             occurrence's grid key is now free (no later attempt has run). This
             guarantees neither a crash between ``claim`` and ``finish`` nor a
             transient delivery failure permanently loses the occurrence.

        Callers should invoke :meth:`reclaim_stale` before this so crashed
        in-flight runs become eligible for retry.

        Args:
            routine: Routine name.
            interval: Spacing between scheduled occurrences, in seconds.
            now: Current time (injectable for a fake clock in tests).
            lookback: Cap how far back to look for missed runs; occurrences
                older than ``now - lookback`` are ignored. ``None`` = no cap
                beyond the last recorded run.
            timezone: Timezone used for whole-day calendar cadences. ``None``
                uses the local system timezone.

        Returns:
            Ascending list of missed scheduled timestamps.
        """
        if interval <= 0:
            raise ValueError("interval must be positive")

        tz = _coerce_timezone(timezone)

        # Anchor on the newest occurrence that has *settled* (a real completion),
        # so neither a stale ``running`` row nor a reclaimed crash pushes the
        # anchor past occurrences that still need to fire.
        anchor_run = self._last_anchorable(routine)
        global_floor = None if lookback is None else now - lookback

        missed: set[float] = set()

        # (1) Grid walk forward from the anchor.
        if anchor_run is None:
            # Never settled: run the single most-recent due occurrence so a
            # brand-new routine fires promptly without replaying history.
            candidate = previous_scheduled_occurrence(
                interval, at=now, timezone=tz
            )
            floor = candidate if global_floor is None else max(candidate, global_floor)
        else:
            anchor = anchor_run.scheduled_ts
            floor = anchor if global_floor is None else max(anchor, global_floor)
            candidate = next_scheduled_occurrence(
                interval, after=anchor, timezone=tz
            )
        while candidate <= now:
            if candidate >= floor and not self.has_run(
                default_idempotency_key(routine, candidate)
            ):
                missed.add(candidate)
            candidate = next_scheduled_occurrence(
                interval, after=candidate, timezone=tz
            )

        # (2) Reclaimed crashes whose grid key is free again.
        for sched_ts in self._reclaimed_occurrences(routine):
            if sched_ts > now:
                continue
            if global_floor is not None and sched_ts < global_floor:
                continue
            if not self.has_run(default_idempotency_key(routine, sched_ts)):
                missed.add(sched_ts)

        return sorted(missed)

    def _reclaimed_occurrences(self, routine: str) -> list[float]:
        """Scheduled times of freed-for-retry attempts (crash or delivery)."""
        placeholders = ",".join("?" * len(_FREED_FOR_RETRY))
        with self._lock:
            conn = self._conn()
            rows = conn.execute(
                "SELECT DISTINCT scheduled_ts FROM routine_runs"
                f" WHERE routine = ? AND status = ? AND error_code IN ({placeholders})",
                (routine, STATUS_ERROR, *_FREED_FOR_RETRY),
            ).fetchall()
        return [r["scheduled_ts"] for r in rows]

    def _last_anchorable(self, routine: str) -> RoutineRun | None:
        """Return the newest run usable as a catch-up grid anchor.

        The anchor is the newest run that represents a *settled* occurrence:

          * terminal (``done``/``error``) — a run that actually completed, but
          * **not** a reclaimed crash (``error_code = LEASE_EXPIRED``): those
            occurrences still need retry, so they must remain *after* the anchor
            rather than becoming it (otherwise reclamation would silently swallow
            the very occurrence it just freed).

        This also ignores still-``running`` rows so a single stale lease left by
        a crash does not advance the anchor past later occurrences. Falls back to
        the most recent run of any status only if nothing settled exists.
        """
        placeholders = ",".join("?" * len(_FREED_FOR_RETRY))
        with self._lock:
            conn = self._conn()
            row = conn.execute(
                "SELECT * FROM routine_runs"
                " WHERE routine = ? AND status IN (?, ?)"
                f" AND (error_code IS NULL OR error_code NOT IN ({placeholders}))"
                " ORDER BY scheduled_ts DESC LIMIT 1",
                (routine, STATUS_DONE, STATUS_ERROR, *_FREED_FOR_RETRY),
            ).fetchone()
            if row is None:
                # Fall back to the newest run that is NOT freed-for-retry, so a
                # crash/delivery-only history does not anchor on (and thus
                # swallow) the very occurrence that still needs retry.
                row = conn.execute(
                    "SELECT * FROM routine_runs WHERE routine = ?"
                    f" AND (error_code IS NULL OR error_code NOT IN ({placeholders}))"
                    " ORDER BY scheduled_ts DESC LIMIT 1",
                    (routine, *_FREED_FOR_RETRY),
                ).fetchone()
        return self._row_to_run(row) if row is not None else None

    # -- mapping --------------------------------------------------------------

    @staticmethod
    def _row_to_run(row: dict[str, object]) -> RoutineRun:
        """Map a DB row to a :class:`RoutineRun`.

        ``output`` is whatever the (possibly encrypted) DB holds; when
        encryption is off it is ``None`` because only metadata was persisted.
        """
        keys = row.keys()
        return RoutineRun(
            id=row["id"],
            routine=row["routine"],
            scheduled_ts=row["scheduled_ts"],
            started_ts=row["started_ts"],
            finished_ts=row["finished_ts"],
            status=row["status"],
            output=row["output"] if "output" in keys else None,
            idempotency_key=row["idempotency_key"],
        )


__all__ = [
    "RoutineStore",
    "default_idempotency_key",
    "next_scheduled_occurrence",
    "previous_scheduled_occurrence",
    "STATUS_PENDING",
    "STATUS_RUNNING",
    "STATUS_DONE",
    "STATUS_ERROR",
    "STATUS_MISSED",
    "ERROR_CODE_LEASE_EXPIRED",
    "ERROR_CODE_DELIVERY",
    "DEFAULT_LEASE_TIMEOUT",
]
