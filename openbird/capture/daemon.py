"""Capture orchestration: run the helper, redact, normalize, then ingest.

The actual screen reading is done by a separate **Swift capture helper** (built
later) that emits one JSON object per capture event on stdout:

    {"app": "...", "window": "...", "url": "...", "text": "...", "ts": 1700000000.0,
     "incognito": false}

``app`` is the frontmost app's bundle id, ``window`` its title, ``url`` the
browser URL (if any), ``text`` the AX-extracted active-window text, ``ts`` an
epoch seconds timestamp, and an optional ``incognito`` flag. This daemon:

  1. spawns the helper (an injectable command, so tests use a fake emitter),
  2. parses each JSON line,
  3. runs the redaction policy (allowlist-first + secret scrubbing),
  4. applies per-app normalization,
  5. and ingests accepted events via :class:`MemoryStore.add_observation`.

Privacy by prevention: captured text is NEVER written to logs, exception
messages, argv, or env. On any parse/handle error we log a metadata-only
diagnostic (``reason`` codes, byte counts) and move on — the raw text never
leaves memory except into the (encrypted-at-rest) store.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from openbird.capture import adapters, redact
from openbird.config import Settings, get_settings
from openbird.types import Observation

logger = logging.getLogger("openbird.capture")

#: Environment variable pointing at the **packaged, signed** capture helper
#: binary inside the app/LaunchAgent bundle. TCC grants are per signed path, so
#: the dev build path (``capture-helper/.build/release/...``) must never be the
#: default — it has no stable Accessibility/Screen-Recording grant.
#: Operators/preflight set this to the bundled artifact.
HELPER_PATH_ENV = "OPENBIRD_CAPTURE_HELPER"

#: Conventional install location of the signed helper inside the packaged app
#: bundle. Used only when the env override is unset. There is intentionally NO
#: dev-build fallback: if neither resolves to a real binary, capture fails
#: closed rather than running an unsigned binary that will lose TCC grants.
DEFAULT_SIGNED_HELPER_PATH = (
    "/Applications/OpenBird.app/Contents/MacOS/capture-helper"
)


class HelperUnavailableError(RuntimeError):
    """Raised when no usable signed capture helper binary can be resolved.

    The daemon fails closed (raises this) rather than silently falling back to a
    dev/unsigned binary, because TCC Accessibility/Screen-Recording grants are
    bound to a specific signed path. The message contains only a path/metadata,
    never captured content.
    """


class HelperExitError(RuntimeError):
    """Raised when the helper exits non-zero *on its own* (not terminated by us).

    A non-zero exit signals a real helper-side failure — e.g. an Accessibility
    denial or a privacy-boundary refusal — so the supervisor's backoff/circuit
    breaker must apply rather than silently re-spawning forever. The message
    carries only the integer exit code, never captured content.
    """


class CaptureSupervisorError(RuntimeError):
    """Raised when the supervised loop aborts (circuit breaker tripped).

    Surfaces sustained helper failure so the CLI exits non-zero instead of
    reporting success. Message carries only the failure count, never content.
    """


def default_helper_cmd() -> tuple[str, ...]:
    """Resolve the default capture-helper command (signed bundle, fail-closed).

    Resolution order:

      1. ``$OPENBIRD_CAPTURE_HELPER`` if set.
      2. The conventional signed-bundle path :data:`DEFAULT_SIGNED_HELPER_PATH`.

    The returned tuple is a *candidate*; whether the binary actually exists and
    looks executable is enforced at spawn time by :func:`_resolve_helper`, which
    raises :class:`HelperUnavailableError` when it does not. This keeps the
    constructor cheap and tests able to inject a fake command.
    """
    configured = os.environ.get(HELPER_PATH_ENV)
    if configured:
        return (configured,)
    return (DEFAULT_SIGNED_HELPER_PATH,)


#: Back-compat module-level default (now points at the signed bundle path, never
#: the dev build). Prefer :func:`default_helper_cmd` so env overrides apply.
DEFAULT_HELPER_CMD: tuple[str, ...] = (DEFAULT_SIGNED_HELPER_PATH,)

# Cap on a single event's text to bound memory and ingest cost. Oversized
# payloads are truncated (a metadata diagnostic is logged, not the text).
_MAX_TEXT_BYTES = 1_000_000

# Hard cap on how many bytes of helper stderr we will drain/buffer. stderr must
# carry only non-content diagnostics; we drain it on a separate thread so a
# chatty helper can't deadlock capture by filling the pipe. We never log its
# contents — only a byte count.
_MAX_STDERR_BYTES = 64 * 1024

# Supervised-loop defaults. The Swift capture-helper captures the frontmost
# window and exits, so continuous capture means re-spawning it on an interval.
_DEFAULT_POLL_INTERVAL = 2.0  # seconds the supervisor idles between helper spawns
_DEFAULT_MAX_CONSECUTIVE_FAILURES = 5  # circuit breaker: stop after this many
_BACKOFF_BASE = 1.0  # first retry delay (seconds), doubled each consecutive fail
_BACKOFF_MAX = 60.0  # cap on the exponential backoff delay


class IngestSink(Protocol):
    """Minimal structural type the daemon needs from a memory store.

    Matches :class:`openbird.memory.store.MemoryStore.add_observation`, so the
    real store satisfies it and tests can pass a lightweight fake.
    """

    def add_observation(
        self,
        text: str,
        *,
        app: str | None = ...,
        window: str | None = ...,
        url: str | None = ...,
        session_id: str | None = ...,
        source: str,
        ts: float | None = ...,
    ) -> Observation: ...


@dataclass(frozen=True)
class CaptureStats:
    """Counters for a capture run (metadata only — never contains text)."""

    received: int = 0
    ingested: int = 0
    rejected: int = 0
    errors: int = 0

    def _with(self, **delta: int) -> "CaptureStats":
        return CaptureStats(
            received=self.received + delta.get("received", 0),
            ingested=self.ingested + delta.get("ingested", 0),
            rejected=self.rejected + delta.get("rejected", 0),
            errors=self.errors + delta.get("errors", 0),
        )

    def _add(self, other: "CaptureStats") -> "CaptureStats":
        """Sum two stats objects (used to aggregate across supervised cycles)."""
        return CaptureStats(
            received=self.received + other.received,
            ingested=self.ingested + other.ingested,
            rejected=self.rejected + other.rejected,
            errors=self.errors + other.errors,
        )


def _truncate(text: str) -> str:
    """Truncate text to the byte cap on a UTF-8 boundary (no logging of text)."""
    encoded = text.encode("utf-8")
    if len(encoded) <= _MAX_TEXT_BYTES:
        return text
    return encoded[:_MAX_TEXT_BYTES].decode("utf-8", errors="ignore")


def parse_event(line: str) -> dict | None:
    """Parse one helper JSON line into a normalized event dict.

    Returns ``None`` for blank lines or malformed JSON (a metadata-only warning
    is logged — never the offending text). On success returns a dict with keys
    ``app, window, url, text, ts, incognito`` (missing optionals defaulted).
    """
    line = line.strip()
    if not line:
        return None
    try:
        raw = json.loads(line)
    except json.JSONDecodeError:
        # Do NOT log the line contents (it may contain captured text).
        logger.warning("capture: dropping unparseable helper line (%d bytes)", len(line))
        return None
    if not isinstance(raw, dict):
        logger.warning("capture: helper line was not a JSON object; dropping")
        return None

    ts = raw.get("ts")
    try:
        ts_val = float(ts) if ts is not None else None
    except (TypeError, ValueError):
        ts_val = None

    def _str_or_none(value: object) -> str | None:
        """Accept only strings for free-text fields; coerce anything else to None."""
        return value if isinstance(value, str) else None

    # ``text`` drives redaction (string ops), so a non-string value would crash
    # downstream — normalize it to "" rather than trusting the helper's types.
    text = raw.get("text")
    text_val = text if isinstance(text, str) else ""

    return {
        "app": _str_or_none(raw.get("app")),
        "window": _str_or_none(raw.get("window")),
        "url": _str_or_none(raw.get("url")),
        "text": text_val,
        "ts": ts_val,
        "incognito": bool(raw.get("incognito", False)),
    }


class CaptureDaemon:
    """Orchestrates the capture helper -> redact -> normalize -> ingest flow."""

    def __init__(
        self,
        store: IngestSink,
        *,
        settings: Settings | None = None,
        helper_cmd: Iterable[str] | None = None,
        source: str = "capture",
        require_signed_helper: bool = True,
    ) -> None:
        """Construct the daemon.

        Args:
            store: A memory store exposing ``add_observation`` (the ingest sink).
            settings: Settings providing allow/blocklists; defaults to
                :func:`get_settings`.
            helper_cmd: Command to launch the capture helper. Defaults to the
                resolved **signed bundle** path (see :func:`default_helper_cmd`);
                tests inject a fake emitter command. The error counter and stats
                are unaffected by the choice.
            source: ``source`` tag stamped on every ingested observation.
            require_signed_helper: When True (default) the daemon fails closed
                with :class:`HelperUnavailableError` at spawn time if the helper
                binary is missing/non-executable, so an unsigned/dev binary can
                never be launched (TCC grants are per signed path). Tests that
                inject an explicit ``helper_cmd`` (e.g. a python emitter) pass
                ``False`` to opt out of the bundle-path requirement.
        """
        self.store = store
        self.settings = settings or get_settings()
        if helper_cmd is not None:
            self.helper_cmd = tuple(helper_cmd)
        else:
            self.helper_cmd = default_helper_cmd()
        self.source = source
        self.require_signed_helper = require_signed_helper
        self._stderr_thread: threading.Thread | None = None
        # Safe, content-free error counter for the whole-data-path privacy gate:
        # store/embed failures bump this instead of emitting tracebacks that
        # might embed captured text.
        self.error_count = 0

    def _pause_file(self) -> Path:
        """Path written by the macOS trust controller to pause ingestion."""
        return Path(self.settings.data_dir) / "capture.paused"

    def is_paused(self) -> bool:
        """Whether capture ingestion is currently paused by the trust surface."""
        return self._pause_file().exists()

    # -- per-event handling ---------------------------------------------------

    def handle_event(self, event: dict, stats: CaptureStats) -> CaptureStats:
        """Apply policy + normalization to one parsed event and maybe ingest.

        Returns the updated :class:`CaptureStats`. Never raises on policy
        rejection; rejections increment ``rejected`` and are logged with their
        metadata-only ``reason`` code.
        """
        stats = stats._with(received=1)
        if self.is_paused():
            logger.debug("capture: rejected event reason=paused")
            return stats._with(rejected=1)

        app = event.get("app")
        window = event.get("window")
        url = event.get("url")
        text = event.get("text")
        ts = event.get("ts")
        incognito = bool(event.get("incognito", False))

        decision, scrubbed = redact.apply(
            app=app,
            window=window,
            text=text,
            incognito=incognito,
            settings=self.settings,
        )
        if not decision.capture or scrubbed is None:
            logger.debug("capture: rejected event app=%s reason=%s", app, decision.reason)
            return stats._with(rejected=1)

        normalized = adapters.normalize_for_app(scrubbed, app)
        if not normalized.strip():
            # Everything was chrome/boilerplate -> nothing worth storing.
            logger.debug("capture: event reduced to empty after normalization app=%s", app)
            return stats._with(rejected=1)

        normalized = _truncate(normalized)

        # Scrub metadata too: URLs embed auth codes/tokens/emails/doc
        # ids in their query/fragment, and window titles can carry full message
        # content. Body text alone going through scrub() is insufficient.
        safe_window, safe_url, title_rules = redact.scrub_metadata(
            window=window, url=url
        )

        try:
            self.store.add_observation(
                normalized,
                app=app,
                window=safe_window,
                url=safe_url,
                session_id=None,
                source=self.source,
                ts=ts,
            )
        except Exception as exc:  # noqa: BLE001 - isolate one bad event from the loop
            # Some store/embed layers raise exceptions whose
            # message embeds the input text. NEVER log the exception message or
            # a traceback at default level (exc_info=False, and we log only the
            # exception *type* + safe app metadata). Full tracebacks are gated
            # behind an explicit, opt-in non-content debug path below.
            self.error_count += 1
            logger.error(
                "capture: add_observation failed app=%s error_type=%s",
                app,
                type(exc).__name__,
                exc_info=False,
            )
            # Opt-in deep debugging only. ``exc_info`` is still suppressed so the
            # exception's (potentially content-bearing) message/traceback is not
            # emitted; we surface only the type and a stable error counter.
            logger.debug(
                "capture: add_observation failure detail app=%s error_type=%s count=%d",
                app,
                type(exc).__name__,
                self.error_count,
                exc_info=False,
            )
            return stats._with(errors=1)

        matched = tuple(decision.matched_rules) + tuple(title_rules)
        if matched:
            logger.debug(
                "capture: scrubbed secrets app=%s rules=%s",
                app,
                ",".join(matched),
            )
        return stats._with(ingested=1)

    # -- stream drivers -------------------------------------------------------

    def run_lines(self, lines: Iterable[str]) -> CaptureStats:
        """Process an iterable of raw JSON lines (the testable core).

        Each line is parsed, policy-checked, normalized, and (if accepted)
        ingested. Unparseable lines are counted as errors. Returns aggregate
        :class:`CaptureStats`.
        """
        stats = CaptureStats()
        for line in lines:
            event = parse_event(line)
            if event is None:
                if line.strip():  # blank lines are not errors
                    stats = stats._with(errors=1)
                continue
            stats = self.handle_event(event, stats)
        return stats

    def _resolve_helper(self) -> list[str]:
        """Return the launch argv, failing closed if the signed helper is absent.

        Enforces the signed-bundle / TCC requirement: when
        ``require_signed_helper`` is set, the first argv element must resolve to
        an existing, executable file (either an absolute/relative path or a name
        on ``PATH``). Otherwise we raise :class:`HelperUnavailableError` rather
        than launch an unsigned/dev binary that would not carry stable TCC
        grants. The error carries only the path, never captured content.
        """
        argv = list(self.helper_cmd)
        if not argv:
            raise HelperUnavailableError("capture: empty helper command")
        argv = self._with_policy_args(argv)
        if not self.require_signed_helper:
            return argv

        binary = argv[0]
        resolved = binary if os.path.isabs(binary) else shutil.which(binary)
        if resolved is None:
            # Allow a relative/explicit path that exists on disk.
            candidate = Path(binary)
            resolved = str(candidate) if candidate.exists() else None

        if resolved is None or not Path(resolved).is_file():
            raise HelperUnavailableError(
                f"capture: signed helper not found at {binary!r}; refusing to "
                "launch an unsigned/dev binary (TCC grants are per signed path). "
                f"Set ${HELPER_PATH_ENV} to the packaged signed helper."
            )
        if not os.access(resolved, os.X_OK):
            raise HelperUnavailableError(
                f"capture: helper at {resolved!r} is not executable; refusing to launch."
            )
        argv[0] = resolved
        return argv

    def _with_policy_args(self, argv: list[str]) -> list[str]:
        """Append the allow/block policy so the helper gates content AT THE SOURCE.

        The capture helper enforces the allowlist-only policy before reading any AX
        text, so disallowed app content is never read or sent over IPC. The Python
        redaction pass (`redact.decide`) still runs as authoritative defense-in-depth.
        """
        allow = list(getattr(self.settings, "allowlist", None) or [])
        block = list(getattr(self.settings, "blocklist", None) or [])
        extra: list[str] = []
        if allow:
            extra += ["--allow", ",".join(allow)]
        if block:
            extra += ["--block", ",".join(block)]
        return argv + extra

    def _spawn(self) -> subprocess.Popen[str]:
        """Launch the helper as a text-mode subprocess yielding stdout lines.

        stdout carries the JSON event stream. stderr is **drained on a separate
        bounded thread** (:meth:`_drain_stderr`) so a helper that writes a lot to
        stderr cannot fill the pipe buffer and deadlock capture; its contents are
        never logged (only a capped byte count), honoring subprocess hygiene.
        """
        argv = self._resolve_helper()
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        self._start_stderr_drain(proc)
        return proc

    def _start_stderr_drain(self, proc: subprocess.Popen[str]) -> None:
        """Spawn a daemon thread that drains (and discards) helper stderr."""
        if proc.stderr is None:
            return
        thread = threading.Thread(
            target=self._drain_stderr,
            args=(proc.stderr,),
            name="capture-stderr-drain",
            daemon=True,
        )
        thread.start()
        self._stderr_thread = thread

    @staticmethod
    def _drain_stderr(stderr) -> None:
        """Continuously read helper stderr to prevent pipe-buffer deadlock.

        We never log stderr content (it could echo captured text); we only track
        a capped byte count so the buffer can always drain. Stops on EOF.
        """
        drained = 0
        try:
            for chunk in iter(lambda: stderr.read(4096), ""):
                if not chunk:
                    break
                if drained < _MAX_STDERR_BYTES:
                    drained += len(chunk)
                # Past the cap we keep reading to avoid deadlock but discard.
        except (OSError, ValueError):
            # Stream closed during shutdown — nothing to do.
            return

    def _iter_stdout(self, proc: subprocess.Popen[str]) -> Iterator[str]:
        """Yield decoded stdout lines from a running helper process."""
        assert proc.stdout is not None
        yield from proc.stdout

    def _terminate_on_stop(
        self, proc: subprocess.Popen[str], stop_event: "threading.Event"
    ) -> None:
        """Watcher: terminate the helper promptly when ``stop_event`` is set.

        Without this, a clean shutdown could block until a hung/long-lived helper
        closed its stdout. We poll so we also exit when the helper ends on its own.
        """
        while proc.poll() is None:
            if stop_event.wait(0.1):
                if proc.poll() is None:
                    proc.terminate()
                return

    def run(
        self,
        *,
        max_events: int | None = None,
        stop_event: "threading.Event | None" = None,
    ) -> CaptureStats:
        """Spawn the helper and process its event stream until it exits.

        Args:
            max_events: If set, stop after this many *received* events and
                terminate the helper (useful for bounded runs/tests against a
                long-lived emitter).
            stop_event: If set, a watcher terminates the helper when the event
                fires so a clean shutdown isn't blocked by a hung helper.

        Returns:
            Aggregate :class:`CaptureStats` for the run.

        Raises:
            HelperExitError: If the helper exits non-zero *on its own* (i.e. we
                did not terminate it via max_events/stop), so the supervisor can
                back off rather than hot-loop a failing helper.
        """
        proc = self._spawn()
        stats = CaptureStats()
        stopped_by_us = False
        watcher: threading.Thread | None = None
        if stop_event is not None:
            watcher = threading.Thread(
                target=self._terminate_on_stop,
                args=(proc, stop_event),
                name="capture-stop-watch",
                daemon=True,
            )
            watcher.start()
        try:
            for line in self._iter_stdout(proc):
                if stop_event is not None and stop_event.is_set():
                    stopped_by_us = True
                    break
                event = parse_event(line)
                if event is None:
                    if line.strip():
                        stats = stats._with(errors=1)
                    continue
                stats = self.handle_event(event, stats)
                if max_events is not None and stats.received >= max_events:
                    stopped_by_us = True
                    break
        finally:
            self._shutdown(proc)
            if watcher is not None:
                watcher.join(timeout=1.0)
        # A non-zero code only counts as a helper-side failure when the helper
        # exited on its OWN. We initiate termination two ways: the max_events/stop
        # break in the loop (``stopped_by_us``) AND the watcher thread, which
        # terminates the child when ``stop_event`` fires without the loop ever
        # observing it. A helper that *traps* SIGTERM can then exit with a
        # positive code; that is still our-initiated, so treat any set stop_event
        # as our-initiated to avoid a spurious HelperExitError on shutdown.
        we_stopped = stopped_by_us or (stop_event is not None and stop_event.is_set())
        if not we_stopped:
            rc = proc.returncode
            # Any non-zero code is a helper-side failure: positive = explicit
            # exit (Accessibility denied=2, privacy refusal=3), negative = death
            # by signal (e.g. SIGSEGV=-11). Both must reach the supervisor.
            if rc is not None and rc != 0:
                raise HelperExitError(f"capture helper exited with code {rc}")
        return stats

    def run_forever(
        self,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        stop_event: "threading.Event | None" = None,
        max_consecutive_failures: int = _DEFAULT_MAX_CONSECUTIVE_FAILURES,
        max_cycles: int | None = None,
    ) -> CaptureStats:
        """Supervise the one-shot helper, re-spawning it until stopped.

        The Swift capture-helper captures the current frontmost window and exits,
        so a single :meth:`run` yields one batch then EOF. Continuous capture
        therefore means re-spawning it on a cadence. This loop runs one
        :meth:`run` per cycle, then idles ``poll_interval`` seconds, until
        ``stop_event`` is set (the CLI wires it to SIGINT/SIGTERM).

        Resilience:
          * A cycle that *raises* (helper crash, broken pipe) is counted as a
            failure and retried with exponential backoff (capped). After
            ``max_consecutive_failures`` consecutive failures the loop trips a
            circuit breaker and returns — fail-closed, never a hot spin.
          * A successful cycle resets the failure counter.
          * :class:`HelperUnavailableError` is a *permanent* config error (no
            signed bundle) and is re-raised immediately, not retried.

        Args:
            poll_interval: Seconds to idle between successful cycles.
            stop_event: Set to request a clean stop; created if omitted.
            max_consecutive_failures: Circuit-breaker threshold.
            max_cycles: Stop after this many cycles (deterministic for tests).

        Returns:
            Aggregate :class:`CaptureStats` across all cycles.
        """
        stop = stop_event or threading.Event()
        total = CaptureStats()
        failures = 0
        cycles = 0
        logger.info("capture supervisor: starting (poll_interval=%.1fs)", poll_interval)
        while not stop.is_set():
            try:
                total = total._add(self.run(stop_event=stop))
            except HelperUnavailableError:
                # Missing signed bundle is permanent, not transient: don't retry.
                raise
            except Exception as exc:  # noqa: BLE001 - one bad cycle must not kill the loop
                failures += 1
                self.error_count += 1
                # Metadata only: class + consecutive count, never the message.
                logger.warning(
                    "capture cycle failed: error_class=%s consecutive=%d",
                    type(exc).__name__,
                    failures,
                )
                if failures >= max_consecutive_failures:
                    logger.error(
                        "capture supervisor: circuit breaker tripped after %d "
                        "consecutive failures; stopping",
                        failures,
                    )
                    # Surface sustained failure to the caller (CLI -> nonzero exit)
                    # rather than returning as if the session ended normally.
                    raise CaptureSupervisorError(
                        f"capture helper failed {failures} consecutive times"
                    )
                delay = min(_BACKOFF_MAX, _BACKOFF_BASE * (2 ** (failures - 1)))
                if stop.wait(delay):
                    break
                continue
            failures = 0
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            # Interruptible idle between cycles (returns True if stop was set).
            if stop.wait(poll_interval):
                break
        logger.info("capture supervisor: stopped (cycles=%d)", cycles)
        return total

    @staticmethod
    def _shutdown(proc: subprocess.Popen[str]) -> None:
        """Terminate the helper process, escalating to kill if needed."""
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass


__all__ = [
    "CaptureDaemon",
    "CaptureStats",
    "IngestSink",
    "HelperUnavailableError",
    "HelperExitError",
    "CaptureSupervisorError",
    "parse_event",
    "default_helper_cmd",
    "DEFAULT_HELPER_CMD",
    "HELPER_PATH_ENV",
    "DEFAULT_SIGNED_HELPER_PATH",
]
