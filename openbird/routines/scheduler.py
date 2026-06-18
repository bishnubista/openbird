"""APScheduler wrapper that runs durable, idempotent, catch-up routines.

The :class:`RoutineScheduler` ties three pieces together:

  * a :class:`~openbird.routines.store.RoutineStore` for durable, idempotent run
    bookkeeping;
  * :class:`~openbird.routines.templates.RoutineTemplate` callables that produce
    text from a time-range RAG query (read/summarize only);
  * an APScheduler ``BackgroundScheduler`` that fires each registered routine on
    its interval while the daemon is alive.

On :meth:`start` it first reclaims any crashed in-flight runs and runs any
**missed** occurrences (the daemon may have been down), then schedules future
firings. Every firing goes through :meth:`fire`, which **claims** the occurrence
(idempotency: a duplicate claim is a no-op), executes the template, delivers the
output through the configured deliverer, and records a terminal
:class:`~openbird.types.RoutineRun`. ``launchd`` is the recommended OS-level
supervisor for true always-on behavior; APScheduler is best-effort while the
process runs.

Privacy
---------------
Routine output is derived from captured on-screen content. The **daemon default
deliverer is metadata-only** (:func:`null_deliverer`): it never writes summary
bodies to stdout/stderr, which can land in launchd logs or terminal scrollback.
Printing the body to stdout (:func:`stdout_deliverer`) is reserved for explicit,
**interactive** use (a user running a routine in the foreground) and must be
opted into. Likewise, run *errors* are persisted as non-content metadata only
(exception class + a stable error code), never as exception message strings,
which commonly embed request payloads or captured text.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from apscheduler.schedulers.background import BackgroundScheduler

from openbird.routines import store as store_mod
from openbird.routines.store import RoutineStore, default_idempotency_key
from openbird.routines.templates import BUILTIN_TEMPLATES, RoutineTemplate
from openbird.types import RoutineRun

# Metadata-only logger. Per the privacy contract this logger MUST
# only ever emit non-content metadata — routine names, statuses, scheduled
# timestamps, and counts — never summary bodies or exception message strings
# (which can embed captured content). Routed to stderr by the CLI daemon.
logger = logging.getLogger("openbird.routines")

# Default cap on how far back startup catch-up reaches. Without a cap, a
# laptop asleep for weeks would, on the next `start()`, fire every missed
# occurrence back-to-back — a startup "LLM storm" that can appear to hang. Seven
# days is a sane default; pass ``lookback=None`` to ``start()`` to disable the
# cap explicitly, or a smaller value to tighten it.
DEFAULT_CATCHUP_LOOKBACK: float = 7 * 24 * 3600.0

# A clock callable returning the current unix time; injectable for tests.
Clock = Callable[[], float]

# A deliverer takes (routine_name, text) and is responsible for output side
# effects. The daemon default is the content-safe no-op (:func:`null_deliverer`).
Deliverer = Callable[[str, str], None]

# Stable, content-safe error code recorded when a routine runner raises.
ERROR_CODE_RUNNER = "RUNNER_EXCEPTION"


def null_deliverer(routine: str, text: str) -> None:
    """Content-safe default sink: deliver nothing (no stdout/stderr leak).

    Used as the unattended-daemon default so that summary bodies derived from
    captured content never reach stdout/stderr (and thus launchd logs or
    terminal scrollback). The run is still recorded durably in the (encrypted)
    routine store; an interactive caller can pass :func:`stdout_deliverer` or a
    notification sink to actually surface the text.
    """
    return None


def stdout_deliverer(routine: str, text: str) -> None:
    """Interactive-only delivery sink: print the routine output to stdout.

    This writes captured-content-derived summary text to stdout and is therefore
    **not** safe as an unattended-daemon default (stdout can be captured by
    launchd logs / terminal scrollback). Use it only for explicit, interactive
    foreground runs where the user is watching the terminal.
    """
    sys.stdout.write(f"=== {routine} ===\n{text}\n")
    sys.stdout.flush()


@dataclass
class Routine:
    """A registered routine: (name, prompt, interval) plus its template runner.

    ``runner(store, provider, now) -> str`` produces the text to deliver. For
    built-in templates this is :meth:`RoutineTemplate.run`; custom routines may
    supply any compatible callable.
    """

    name: str
    prompt: str
    interval: float
    runner: Callable[..., str]
    last_anchor: float | None = field(default=None)


class RoutineScheduler:
    """Schedules and executes durable, idempotent, catch-up routines."""

    def __init__(
        self,
        *,
        memory_store: object,
        provider: object,
        routine_store: RoutineStore | None = None,
        clock: Clock | None = None,
        deliverer: Deliverer | None = None,
        scheduler: BackgroundScheduler | None = None,
    ) -> None:
        """Wire the scheduler to its memory store, provider, and run store.

        Args:
            memory_store: Object exposing ``time_range(start, end)``.
            provider: Object exposing ``complete(messages)``.
            routine_store: Durable run store; defaults to the file-backed
                ``<data_dir>/routines.db`` if omitted (never ``:memory:``).
            clock: Returns current unix time; defaults to :func:`time.time`.
                Injecting a fake clock makes catchup/idempotency deterministic.
            deliverer: Output sink; defaults to :func:`stdout_deliverer`.
            scheduler: An APScheduler scheduler; a ``BackgroundScheduler`` is
                created lazily if omitted (only needed for live operation).
        """
        self.memory_store = memory_store
        self.provider = provider
        self.clock = clock or time.time
        # When we own the run store, default to the DURABLE file-backed routines DB
        # (<data_dir>/routines.db) — NOT :memory:, which would silently drop
        # routine_runs on every restart and defeat idempotency + missed-job
        # catchup. Share our clock so lease/reclaim time bases match (important
        # under a fake clock in tests). Tests that want ephemeral state inject an
        # explicit RoutineStore(db_path=":memory:").
        self._owns_store = routine_store is None
        self.run_store = routine_store or RoutineStore(clock=self.clock)
        # Default to the content-safe no-op so the unattended daemon never leaks
        # summary bodies to stdout/stderr. Interactive callers opt into
        # ``stdout_deliverer`` (or a notification sink) explicitly.
        self.deliverer = deliverer or null_deliverer
        self._scheduler = scheduler
        self._routines: dict[str, Routine] = {}

    # -- registration ---------------------------------------------------------

    def register(
        self,
        name: str,
        prompt: str,
        interval: float,
        *,
        runner: Callable[..., str] | None = None,
    ) -> Routine:
        """Register a routine = (name, prompt, interval).

        If ``runner`` is omitted and ``name`` matches a built-in template, that
        template's :meth:`run` is used. Otherwise ``runner`` is required.

        Raises:
            ValueError: If ``interval`` is non-positive, the name is already
                registered, or no runner can be resolved.
        """
        if interval <= 0:
            raise ValueError("interval must be positive")
        if name in self._routines:
            raise ValueError(f"routine already registered: {name!r}")

        resolved = runner
        if resolved is None:
            template: RoutineTemplate | None = BUILTIN_TEMPLATES.get(name)
            if template is None:
                raise ValueError(
                    f"no runner given and no built-in template named {name!r}"
                )
            resolved = template.run

        routine = Routine(name=name, prompt=prompt, interval=interval, runner=resolved)
        self._routines[name] = routine
        return routine

    def register_template(self, template: RoutineTemplate) -> Routine:
        """Register a :class:`RoutineTemplate` using its built-in cadence."""
        return self.register(
            template.name,
            template.prompt,
            template.interval,
            runner=template.run,
        )

    @property
    def routines(self) -> dict[str, Routine]:
        """Return the registered routines keyed by name."""
        return dict(self._routines)

    # -- execution ------------------------------------------------------------

    def fire(
        self,
        name: str,
        *,
        scheduled_ts: float | None = None,
    ) -> RoutineRun | None:
        """Execute one occurrence of ``name`` for ``scheduled_ts`` (idempotent).

        Claims the run via the durable store; if the occurrence was already
        executed (same idempotency key), returns ``None`` without re-running.
        On success runs the template, delivers the output, and records a
        terminal :class:`RoutineRun`. Template errors are captured and the run
        is recorded with ``error`` status (the scheduler must not crash on one
        bad routine).

        The returned :class:`RoutineRun` carries the live in-memory ``output``
        (the generated summary, or the ``error_class: error_code`` marker on
        failure) for the immediate caller's convenience. This is **not** the
        same as what is durably persisted: the store only writes the summary
        body when the DB is encrypted at rest, and never writes exception
        message strings (see the store's privacy notes).

        Returns:
            The completed :class:`RoutineRun`, or ``None`` if it was a duplicate.
        """
        routine = self._routines[name]
        sched = self.clock() if scheduled_ts is None else scheduled_ts

        run = self.run_store.claim(name, sched)
        if run is None:
            # Idempotency: this occurrence is already done or in progress.
            return None

        try:
            text = routine.runner(self.memory_store, self.provider, now=sched)
        except Exception as exc:  # noqa: BLE001 - one routine must not kill the loop
            # Persist NON-content metadata only: the exception class name and a
            # stable error code. The exception *message* is never stored — it
            # frequently embeds request payloads, prompts, or captured text, and
            # the routine DB may be plaintext when encryption is unavailable.
            error_class = type(exc).__name__
            persisted = self.run_store.finish(
                run.id,
                status=store_mod.STATUS_ERROR,
                error_class=error_class,
                error_code=ERROR_CODE_RUNNER,
            )
            # Metadata only: routine, scheduled occurrence, error class + code.
            # NEVER the exception message (it can embed captured content).
            logger.warning(
                "routine error: name=%s scheduled_ts=%.0f error_class=%s error_code=%s",
                name,
                sched,
                error_class,
                ERROR_CODE_RUNNER,
            )
            # Surface a content-safe marker to the immediate caller (not stored).
            return persisted.model_copy(
                update={"output": f"{error_class}: {ERROR_CODE_RUNNER}"}
            )

        self.deliverer(name, text)
        # Metadata only: output length, never the body.
        logger.info(
            "routine done: name=%s scheduled_ts=%.0f output_len=%d",
            name,
            sched,
            len(text),
        )
        # ``output`` is content derived from captured data; the store only
        # persists the body when the DB is encrypted at rest, otherwise it keeps
        # metadata (length + hash) only.
        persisted = self.run_store.finish(
            run.id, status=store_mod.STATUS_DONE, output=text
        )
        # Hand the live text back to the immediate caller without depending on
        # whether the durable store retained it (it does not when unencrypted).
        return persisted.model_copy(update={"output": text})

    # -- missed-job catchup ---------------------------------------------------

    def run_missed(self, *, lookback: float | None = None) -> list[RoutineRun]:
        """Run every missed occurrence across all registered routines.

        Intended to be called once on startup. For each routine it asks the
        durable store which scheduled occurrences elapsed without a run, then
        :meth:`fire`\\ s each in chronological order. Idempotency in
        :meth:`fire` guarantees no double execution even if catchup overlaps a
        live trigger.

        Args:
            lookback: Cap how far back to catch up (seconds). ``None`` = back to
                the routine's last recorded run only.

        Returns:
            The completed runs, in execution order.
        """
        now = self.clock()
        # Free occurrences whose worker crashed mid-run before computing what is
        # missed: a stale ``running`` row would otherwise strand its occurrence
        # forever (its idempotency key exists, so it is never retried).
        reclaimed = self.run_store.reclaim_stale(now=now)
        if reclaimed:
            logger.info("reclaimed stale running occurrences: count=%d", len(reclaimed))
        completed: list[RoutineRun] = []
        for name, routine in self._routines.items():
            missed = self.run_store.missed_occurrences(
                name, routine.interval, now=now, lookback=lookback
            )
            if missed:
                logger.info(
                    "catch-up: name=%s missed=%d lookback=%s",
                    name,
                    len(missed),
                    "none" if lookback is None else f"{lookback:.0f}s",
                )
            for scheduled_ts in missed:
                run = self.fire(name, scheduled_ts=scheduled_ts)
                if run is not None:
                    completed.append(run)
        return completed

    # -- live scheduling ------------------------------------------------------

    def _ensure_scheduler(self) -> BackgroundScheduler:
        if self._scheduler is None:
            self._scheduler = BackgroundScheduler()
        return self._scheduler

    def start(
        self,
        *,
        catch_up: bool = True,
        lookback: float | None = DEFAULT_CATCHUP_LOOKBACK,
    ) -> None:
        """Run missed jobs, then start firing each routine on its interval.

        Args:
            catch_up: Run missed occurrences before scheduling future ones.
            lookback: Cap how far back catch-up reaches (seconds). Defaults to
                :data:`DEFAULT_CATCHUP_LOOKBACK` to avoid a startup "LLM storm"
                after a long downtime. Pass ``None`` to disable the cap.
        """
        if catch_up:
            self.run_missed(lookback=lookback)

        scheduler = self._ensure_scheduler()
        for name, routine in self._routines.items():
            scheduler.add_job(
                self._scheduled_fire,
                trigger="interval",
                seconds=routine.interval,
                args=[name],
                id=name,
                replace_existing=True,
                coalesce=True,
                max_instances=1,
            )
        if not scheduler.running:
            scheduler.start()
        logger.info(
            "scheduler started: routines=%d catch_up=%s",
            len(self._routines),
            catch_up,
        )

    def _scheduled_fire(self, name: str) -> None:
        """APScheduler entrypoint: fire ``name`` at the current wall clock."""
        self.fire(name)

    def shutdown(self, *, wait: bool = True) -> None:
        """Stop the background scheduler and close the run store if we own it."""
        if self._scheduler is not None and self._scheduler.running:
            self._scheduler.shutdown(wait=wait)
            logger.info("scheduler stopped")
        # Close the run store's (per-thread) connections only if we created it; an
        # injected store is the caller's to close.
        if self._owns_store:
            self.run_store.close()

    # -- introspection --------------------------------------------------------

    def last_idempotency_key(self, name: str, scheduled_ts: float) -> str:
        """Return the idempotency key used for ``(name, scheduled_ts)``."""
        return default_idempotency_key(name, scheduled_ts)


__all__ = [
    "RoutineScheduler",
    "Routine",
    "stdout_deliverer",
    "null_deliverer",
    "ERROR_CODE_RUNNER",
    "DEFAULT_CATCHUP_LOOKBACK",
]
