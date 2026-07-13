"""Capture orchestration: run the helper, redact, normalize, then ingest.

The actual screen reading is done by a separate **Swift capture helper** (built
later) that emits one JSON object per capture event on stdout:

    {"app": "...", "window": "...", "url": "...", "text": "...", "ts": 1700000000.0,
     "incognito": false}

``app`` is the frontmost app's bundle id, ``window`` its title, ``url`` a
browser URL if a helper supplies one (the MVP Swift helper emits ``null``),
``text`` the AX-extracted active-window text, ``ts`` an epoch seconds timestamp,
and an optional ``incognito`` flag. This daemon:

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
import math
import os
import queue
import select
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol

from openbird.capture import adapters, redact, volatility
from openbird.capture.spans import NullSpanTracker, SpanTracker
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
_DEFAULT_DUPLICATE_WINDOW = 60.0  # keep one unchanged snapshot per minute
# Fallback session-gap (seconds) used when a settings object predates the
# ``session_gap_seconds`` field (e.g. an injected test ``Settings``). A new
# episodic session starts when the foreground app changes OR activity pauses
# longer than this. Kept in sync with ``config.Settings.session_gap_seconds``.
_DEFAULT_SESSION_GAP = 300.0
_DEFAULT_MAX_CONSECUTIVE_FAILURES = 5  # circuit breaker: stop after this many
_BACKOFF_BASE = 1.0  # first retry delay (seconds), doubled each consecutive fail
_BACKOFF_MAX = 60.0  # cap on the exponential backoff delay

# Persistent (stream) mode. The helper runs long-lived (`--stream`), pushing
# typed NDJSON events; the daemon reads them in short queue slices so a stop
# request is honored within ~1s regardless of the stall-detection timeout.
CAPTURE_MODE_ENV = "OPENBIRD_CAPTURE_MODE"  # "oneshot" | "persistent" (force)
_STREAM_READ_TIMEOUT_MIN = 30.0  # floor for the no-events stall timeout
_STREAM_STOP_SLICE = 1.0  # queue.get slice: stop-responsiveness bound
_STREAM_QUEUE_MAX = 1000  # bounded event queue (backpressure, not unbounded RAM)
_LIVENESS_WRITE_INTERVAL = 10.0  # max sidecar write frequency (seconds)
# A persistent cycle that survived at least this long resets the consecutive-
# failure counter BEFORE its failure is counted: a once-a-day crash must never
# accumulate into a breaker trip, while a restart storm still trips it fast.
_FAILURE_DECAY_SECONDS = 300.0
#: Liveness sidecar filename (inside data_dir). Metadata only — timestamps,
#: mode, AFK flag, heartbeat seq. Never content. capture-health reads it.
LIVENESS_FILENAME = "capture.liveness.json"


def _runtime_version() -> str:
    """Return the version of the package running this daemon."""
    try:
        return version("openbird")
    except PackageNotFoundError:
        return "unknown"

# App-supervised self-exit (orphan cleanup).
#
# An app-launched ``capture --loop`` daemon must self-exit when its supervising
# app dies non-gracefully (the child is then reparented to launchd/PID 1 and
# would otherwise run forever). We deliberately do NOT key this off ``getppid()
# == 1``: a user who backgrounds the daemon by hand (``capture --loop &``,
# ``nohup``) is ALSO reparented to PID 1 and must be left running untouched.
#
# Instead, the app stamps a per-launch random token into this env var AND writes
# that exact token through an inherited "death pipe" (the child's stdin). The
# daemon arms a stop-on-EOF watcher ONLY after it reads that exact token off
# stdin; when the app dies the pipe's write end closes and the daemon observes
# EOF and stops cleanly. A leaked env var alone cannot arm the watcher (the
# secret token must also arrive on stdin), so manually-started daemons — which
# never receive the token — keep today's behavior exactly. Every non-matching
# path (wrong/absent token, premature EOF, read error) fails OPEN: we prefer a
# missed cleanup over ever stopping the wrong daemon.
SUPERVISOR_TOKEN_ENV = "OPENBIRD_SUPERVISOR_TOKEN"
# Cap the handshake line read so a leaked env on an unrelated, chatty stdin
# cannot make us buffer unbounded data before deciding it is not our token.
_SUPERVISOR_TOKEN_MAX = 256
# How long the supervisor watcher blocks in select before re-checking the stop
# event. Short enough that a normal shutdown joins the watcher promptly; long
# enough that it is not a busy-spin while idling for the app to die.
_SUPERVISOR_SELECT_TIMEOUT = 0.2


# Typed stream-event vocabulary (Phase A, event-driven capture). A line without
# a ``type`` field is a capture frame — that is the back-compat contract with
# older one-shot helper binaries, which predate typing entirely. Unknown or
# non-string ``type`` values are treated as capture frames (fail-safe: the
# strictest path — they then pass through redaction and are rejected if empty).
_EVENT_TYPES = frozenset(
    {"capture", "afk_transition", "heartbeat", "system", "app_changed"}
)

# Closed vocabularies for helper-supplied enum-ish metadata. These strings can
# reach logs, so they are sanitized against a closed set — a helper bug can
# never route free text (which could embed content) into a log line.
_CAPTURE_TRIGGERS = frozenset(
    {
        "app_activated",
        "window_changed",
        "title_changed",
        "focus_changed",
        "typing_pause",
        "idle_tick",
        "force_ceiling",
        "return_from_afk",
        "startup",
    }
)
_SYSTEM_KINDS = frozenset(
    {
        "will_sleep",
        "did_wake",
        "screen_locked",
        "screen_unlocked",
        # Mic run-state edges (Phase C1). Raw hardware signal only — the
        # meeting judgment (conjunction with the frontmost surface) is applied
        # in SpanTracker. Old daemons coerce these unknown kinds to None and
        # on_system(None, ts) is a no-op — zero back-compat risk.
        "mic_started",
        "mic_stopped",
        # OCR fallback availability edges (Phase C2): the helper's Screen
        # Recording preflight reading at startup + observed flips. Metadata
        # only; feeds the liveness sidecar's ocr_state. Same back-compat
        # argument as mic_*: old daemons coerce unknown kinds to None.
        "ocr_available",
        "ocr_unavailable",
    }
)


def _finite_ts(value: object, fallback: float | None = None) -> float:
    """Return a FINITE timestamp from an untrusted value (shared guard).

    JSON permits ``NaN``/``Infinity`` literals and ``float('nan')`` does not
    raise, so a malformed helper ``ts`` could otherwise poison liveness math,
    session clocks, or time-range ordering. Non-finite or unparseable values
    fall back to ``fallback`` (wall-clock now when omitted). Used by the
    capture-ingest clock, typed-event dispatch, and liveness bookkeeping.
    """
    try:
        parsed = float(value) if value is not None else None  # type: ignore[arg-type]
    except (TypeError, ValueError):
        parsed = None
    if parsed is None or not math.isfinite(parsed):
        return time.time() if fallback is None else fallback
    return parsed


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
        span_id: str | None = ...,
    ) -> Observation: ...


@dataclass(frozen=True)
class CaptureStats:
    """Counters for a capture run (metadata only — never contains text)."""

    received: int = 0
    ingested: int = 0
    coalesced: int = 0
    rejected: int = 0
    errors: int = 0
    # Typed stream-event counters (Phase A). ``heartbeats`` counts liveness-only
    # events (heartbeat + system); ``afk_transitions`` counts AFK state flips;
    # ``span_markers`` counts app_changed boundary markers (Phase B).
    heartbeats: int = 0
    afk_transitions: int = 0
    span_markers: int = 0
    # Ingested capture frames whose text came from the OCR fallback (Phase C2,
    # ``ocr: true`` on the frame). A subset of ``ingested`` — metadata only.
    ocr_captures: int = 0

    def _with(self, **delta: int) -> "CaptureStats":
        return CaptureStats(
            received=self.received + delta.get("received", 0),
            ingested=self.ingested + delta.get("ingested", 0),
            coalesced=self.coalesced + delta.get("coalesced", 0),
            rejected=self.rejected + delta.get("rejected", 0),
            errors=self.errors + delta.get("errors", 0),
            heartbeats=self.heartbeats + delta.get("heartbeats", 0),
            afk_transitions=self.afk_transitions + delta.get("afk_transitions", 0),
            span_markers=self.span_markers + delta.get("span_markers", 0),
            ocr_captures=self.ocr_captures + delta.get("ocr_captures", 0),
        )

    def _add(self, other: "CaptureStats") -> "CaptureStats":
        """Sum two stats objects (used to aggregate across supervised cycles)."""
        return CaptureStats(
            received=self.received + other.received,
            ingested=self.ingested + other.ingested,
            coalesced=self.coalesced + other.coalesced,
            rejected=self.rejected + other.rejected,
            errors=self.errors + other.errors,
            heartbeats=self.heartbeats + other.heartbeats,
            afk_transitions=self.afk_transitions + other.afk_transitions,
            span_markers=self.span_markers + other.span_markers,
            ocr_captures=self.ocr_captures + other.ocr_captures,
        )


@dataclass(frozen=True)
class _CaptureSignature:
    """Content-free identity for deciding whether a capture is unchanged."""

    app: str | None
    window: str | None
    url: str | None
    text_hash: str


def _truncate(text: str) -> str:
    """Truncate text to the byte cap on a UTF-8 boundary (no logging of text)."""
    encoded = text.encode("utf-8")
    if len(encoded) <= _MAX_TEXT_BYTES:
        return text
    return encoded[:_MAX_TEXT_BYTES].decode("utf-8", errors="ignore")


def parse_event(line: str) -> dict | None:
    """Parse one helper JSON line into a normalized event dict.

    Returns ``None`` for blank lines or malformed JSON (a metadata-only warning
    is logged — never the offending text). Stream-mode helpers emit *typed*
    events (``type`` in :data:`_EVENT_TYPES`); a line without ``type`` — or with
    an unknown/non-string one — is a capture frame (back-compat with one-shot
    helpers, and fail-safe: unknown types go through the strictest path).

    Capture frames return the classic shape ``app, window, url, text, ts,
    incognito`` plus ``type="capture"`` and a sanitized optional ``trigger``.
    Non-capture events return metadata-only dicts (never window/URL/text).
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

    event_type = raw.get("type")
    if not isinstance(event_type, str) or event_type not in _EVENT_TYPES:
        event_type = "capture"

    ts = raw.get("ts")
    try:
        ts_val = float(ts) if ts is not None else None
    except (TypeError, ValueError):
        ts_val = None

    if event_type != "capture":
        # Metadata-only events: sanitize every enum-ish field against a closed
        # vocabulary (these values can reach logs; free text must not).
        kind = raw.get("kind")
        idle = raw.get("idle_seconds")
        try:
            idle_val = float(idle) if idle is not None else None
        except (TypeError, ValueError):
            idle_val = None
        seq = raw.get("seq")
        event = {
            "type": event_type,
            "ts": ts_val,
            "afk": bool(raw.get("afk", False)),
            "paused": bool(raw.get("paused", False)),
            "idle_seconds": idle_val,
            "seq": seq if isinstance(seq, int) else None,
            "kind": kind if isinstance(kind, str) and kind in _SYSTEM_KINDS else None,
        }
        if event_type == "app_changed":
            # Boundary markers carry the bundle id (identity metadata, not
            # content). The key exists ONLY on this type — other typed events
            # keep their strict no-identity key contract.
            app = raw.get("app")
            event["app"] = app if isinstance(app, str) else None
        return event

    trigger = raw.get("trigger")
    if not isinstance(trigger, str) or trigger not in _CAPTURE_TRIGGERS:
        trigger = None

    def _str_or_none(value: object) -> str | None:
        """Accept only strings for free-text fields; coerce anything else to None."""
        return value if isinstance(value, str) else None

    # ``text`` drives redaction (string ops), so a non-string value would crash
    # downstream — normalize it to "" rather than trusting the helper's types.
    text = raw.get("text")
    text_val = text if isinstance(text, str) else ""

    return {
        "type": "capture",
        "trigger": trigger,
        "app": _str_or_none(raw.get("app")),
        "window": _str_or_none(raw.get("window")),
        "url": _str_or_none(raw.get("url")),
        "text": text_val,
        "ts": ts_val,
        "incognito": bool(raw.get("incognito", False)),
        # Phase C2: the helper stamps ocr:true on frames whose text came from
        # the OCR fallback. Sanitized strictly (only JSON true counts; absent
        # or any non-bool -> False); drives only the ocr_captures counter —
        # never policy.
        "ocr": raw.get("ocr") is True,
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
        duplicate_window: float = _DEFAULT_DUPLICATE_WINDOW,
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
            duplicate_window: Seconds to suppress unchanged app/window/url/text
                repeats after an ingest. A fresh heartbeat is still stored once
                the window elapses, preserving duration without 2-second copies.
        """
        self.store = store
        self.settings = settings or get_settings()
        if helper_cmd is not None:
            self.helper_cmd = tuple(helper_cmd)
        else:
            self.helper_cmd = default_helper_cmd()
        self.source = source
        self.require_signed_helper = require_signed_helper
        self.duplicate_window = max(0.0, duplicate_window)
        self._last_ingested_signature: _CaptureSignature | None = None
        self._last_ingested_at: float | None = None
        # Generated once in memory for this daemon process. Post-release checks
        # use it to prove the newly installed app, rather than a stale process,
        # wrote the current liveness sidecar.
        self._instance_uuid = str(uuid.uuid4())
        self._runtime_version = _runtime_version()
        # Episodic-session segmentation (Layer 4). A session groups a contiguous
        # run of activity in one app; ``_assemble_context`` / ``_answer_temporal``
        # use it to keep "what did I do today" recall coherent. State advances on
        # every policy-accepted frame (see :meth:`_session_for`), so continuity is
        # independent of the coalesce window. Survives a rejected/paused frame.
        self._session_id: str | None = None
        self._session_app: str | None = None
        self._session_last_ts: float | None = None
        # AFK state mirrored from helper afk_transition events (stream mode).
        # Metadata only; used for diagnostics/liveness, never gates ingestion
        # (the helper already suppresses captures while AFK at the source).
        self._afk = False
        # OCR fallback availability mirrored from helper ocr_available /
        # ocr_unavailable system events (Phase C2). Metadata only; surfaces in
        # the liveness sidecar for capture-health. None = never reported
        # (OCR not opted in, or an old helper).
        self._ocr_state: str | None = None
        # Span tracker (Phase B): turns the typed event stream into
        # activity_spans rows. Only wired when the sink exposes the span API,
        # so lightweight test fakes keep working unchanged.
        if hasattr(store, "open_span"):
            self._span_tracker = SpanTracker(store, self.settings)
        else:
            self._span_tracker = NullSpanTracker()
        # Whether the helper binary supports --stream. Optimistic until proven
        # otherwise: a helper that EOFs rc=0 without EVER emitting a heartbeat
        # is an old one-shot binary that ignored the flag — we then downgrade
        # to poll mode for the rest of this daemon's life (no respawn storm).
        self._stream_supported = True
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

    def _reset_coalescing(self) -> None:
        """Forget the previous accepted foreground state after capture leaves it."""
        self._last_ingested_signature = None
        self._last_ingested_at = None

    @staticmethod
    def _event_clock(ts: object) -> float:
        """Return a FINITE comparison/storage timestamp from an untrusted ts.

        Delegates to the shared :func:`_finite_ts` guard (see its docstring for
        the NaN/Infinity rationale). Kept as a method for existing callers/tests.
        """
        return _finite_ts(ts)

    @staticmethod
    def _signature(
        *,
        app: str | None,
        window: str | None,
        url: str | None,
        text: str,
    ) -> _CaptureSignature:
        """Build a non-content signature for unchanged-capture suppression."""
        text_hash = sha256(text.encode("utf-8")).hexdigest()
        return _CaptureSignature(app=app, window=window, url=url, text_hash=text_hash)

    def _is_recent_duplicate(self, signature: _CaptureSignature, event_at: float) -> bool:
        if self.duplicate_window <= 0:
            return False
        if self._last_ingested_signature != signature or self._last_ingested_at is None:
            return False
        if event_at < self._last_ingested_at:
            return False
        return event_at - self._last_ingested_at < self.duplicate_window

    def _mark_ingested(self, signature: _CaptureSignature, event_at: float) -> None:
        self._last_ingested_signature = signature
        self._last_ingested_at = event_at

    def _session_for(self, app: str | None, event_at: float) -> str:
        """Return the current episodic-session id, starting a new one when needed.

        A NEW ``uuid4().hex`` session begins when the foreground app changes OR the
        gap since the last accepted frame exceeds ``session_gap_seconds``. Either
        way ``_session_last_ts`` is advanced to ``event_at``, so this MUST be called
        for every policy-accepted, non-empty frame — INCLUDING ones that go on to
        coalesce — to keep the session-activity clock current. That decouples
        session continuity from ``duplicate_window``: a long run of coalesced
        heartbeats keeps the session alive. Rejected/paused/empty frames never call
        this (a blocked app must not extend a session).

        ``getattr`` keeps an injected test ``Settings`` lacking the field working.
        """
        gap = getattr(self.settings, "session_gap_seconds", _DEFAULT_SESSION_GAP)
        prev = self._session_last_ts
        if (
            self._session_id is None
            or app != self._session_app
            or prev is None
            or event_at - prev > gap
        ):
            self._session_id = uuid.uuid4().hex
            self._session_app = app
            self._session_last_ts = event_at
        elif event_at > prev:
            # Same session: advance the activity clock, but NEVER regress it on a
            # backward / out-of-order timestamp (a coalesced stale frame could
            # otherwise rewind the clock and spuriously age the session, splitting
            # one continuous run into two). ``event_at > prev`` is also NaN-safe.
            self._session_last_ts = event_at
        return self._session_id

    # -- per-event handling ---------------------------------------------------

    def handle_event(self, event: dict, stats: CaptureStats) -> CaptureStats:
        """Apply policy + normalization to one parsed event and maybe ingest.

        Returns the updated :class:`CaptureStats`. Never raises on policy
        rejection; rejections increment ``rejected`` and are logged with their
        metadata-only ``reason`` code.
        """
        stats = stats._with(received=1)
        event_type = event.get("type", "capture")
        if event_type != "capture" and event_type in _EVENT_TYPES:
            return self._handle_typed_event(event_type, event, stats)

        app = event.get("app")
        window = event.get("window")
        url = event.get("url")
        text = event.get("text")
        ts = event.get("ts")
        incognito = bool(event.get("incognito", False))
        # OCR-fallback provenance flag (Phase C2). Counter-only: the ingest
        # path below is IDENTICAL for OCR and AX text (redact.apply ->
        # normalize -> volatility -> scrub -> truncate -> scrub_metadata) —
        # the no-bypass guarantee, asserted by test, not by new code.
        ocr = event.get("ocr") is True
        event_at = self._event_clock(ts)

        # Span resolution is EVENT-SCOPED and runs for EVERY frame — including
        # ones the content policy rejects below (a blocked terminal still gets
        # its coarse time span; that is the whole point of the spans layer).
        # The tracker applies classify_policy internally, so tier/reason are
        # decided structurally, never from text.
        paused = self.is_paused()
        span_id = self._span_tracker.on_frame(
            app=app,
            window=window,
            url=url,
            incognito=incognito,
            paused=paused,
            afk=self._afk,
            ts=event_at,
        )

        if paused:
            self._reset_coalescing()
            logger.debug("capture: rejected event reason=paused")
            return stats._with(rejected=1)

        decision, scrubbed = redact.apply(
            app=app,
            window=window,
            text=text,
            incognito=incognito,
            settings=self.settings,
        )
        if not decision.capture or scrubbed is None:
            self._reset_coalescing()
            logger.debug("capture: rejected event app=%s reason=%s", app, decision.reason)
            return stats._with(rejected=1)

        normalized = adapters.normalize_for_app(scrubbed, app)
        # Layer 1: strip high-churn UI animation (spinners, progress bars, ANSI) so
        # flicker frames hash identically and stop forking a fresh blob every poll.
        # Runs BEFORE the empty check so a pure-spinner frame de-flickers to "" and
        # is rejected through the existing branch below.
        normalized = volatility.normalize(normalized)
        # Re-scrub AFTER de-flickering: stripping ANSI/control sequences can rejoin a
        # secret that the first scrub missed because an escape split the token
        # (e.g. ``sk-<ESC>[31mABC…``). The stored body is therefore always the
        # output of scrub() applied to the final, fully-normalized text.
        normalized, body_rules = redact.scrub(normalized)
        if not normalized.strip():
            # Everything was chrome/boilerplate/animation -> nothing worth storing.
            self._reset_coalescing()
            logger.debug("capture: event reduced to empty after normalization app=%s", app)
            return stats._with(rejected=1)

        normalized = _truncate(normalized)

        # Scrub metadata too: URLs embed auth codes/tokens/emails/doc
        # ids in their query/fragment, and window titles can carry full message
        # content. Body text alone going through scrub() is insufficient.
        safe_window, safe_url, title_rules = redact.scrub_metadata(
            window=window, url=url
        )
        # Advance the episodic-session clock for EVERY accepted frame (before the
        # coalesce check), so continuity holds even across coalesced heartbeats.
        session_id = self._session_for(app, event_at)
        signature = self._signature(
            app=app,
            window=safe_window,
            url=safe_url,
            text=normalized,
        )
        if self._is_recent_duplicate(signature, event_at):
            logger.debug("capture: coalesced unchanged event app=%s", app)
            return stats._with(coalesced=1)

        try:
            # span_id is passed ONLY when a span was resolved: sinks without
            # the span API (NullSpanTracker path — e.g. minimal test fakes)
            # never see the kwarg, so their add_observation signature is
            # unchanged.
            span_kwargs = {"span_id": span_id} if span_id is not None else {}
            self.store.add_observation(
                normalized,
                app=app,
                window=safe_window,
                url=safe_url,
                session_id=session_id,
                source=self.source,
                ts=event_at,
                **span_kwargs,
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
            self._reset_coalescing()
            return stats._with(errors=1)

        self._mark_ingested(signature, event_at)
        matched = tuple(decision.matched_rules) + tuple(body_rules) + tuple(title_rules)
        if matched:
            logger.debug(
                "capture: scrubbed secrets app=%s rules=%s",
                app,
                ",".join(matched),
            )
        return stats._with(ingested=1, ocr_captures=1 if ocr else 0)

    def _handle_typed_event(
        self, event_type: str, event: dict, stats: CaptureStats
    ) -> CaptureStats:
        """Handle a metadata-only stream event (never ingests content).

        ``afk_transition`` flips the mirrored AFK state and resets coalescing —
        the first frame after a return from AFK must be stored even if its
        content is unchanged, so "resumed at HH:MM" is visible on the timeline.
        ``heartbeat`` and ``system`` are liveness-only. All logging here is
        reason-code metadata; the event dicts carry no window/URL/text by
        construction (see :func:`parse_event`).
        """
        ts = _finite_ts(event.get("ts"))
        if event_type == "afk_transition":
            afk = bool(event.get("afk", False))
            self._afk = afk
            self._reset_coalescing()
            self._span_tracker.on_afk_transition(afk=afk, ts=ts)
            logger.debug("capture: afk_transition afk=%s", afk)
            return stats._with(afk_transitions=1)
        if event_type == "system":
            # ``kind`` was sanitized against the closed vocabulary at parse time.
            kind = event.get("kind")
            if kind in ("mic_started", "mic_stopped"):
                # Mic edges route to the tracker's meeting machinery (Phase
                # C1); sleep/lock force-close stays in on_system.
                self._span_tracker.on_mic(kind == "mic_started", ts)
            elif kind in ("ocr_available", "ocr_unavailable"):
                # OCR availability edges (Phase C2) feed the liveness sidecar
                # only — spans are unaffected (metadata, not a boundary).
                self._ocr_state = (
                    "available" if kind == "ocr_available" else "unavailable"
                )
            else:
                self._span_tracker.on_system(kind, ts)
            logger.debug("capture: system event kind=%s", kind)
            return stats._with(heartbeats=1)
        if event_type == "app_changed":
            self._span_tracker.on_app_changed(event.get("app"), ts)
            return stats._with(span_markers=1)
        # heartbeat: pure liveness (and the paused/afk pulse for spans).
        self._span_tracker.on_heartbeat(
            afk=bool(event.get("afk", False)),
            paused=bool(event.get("paused", False)),
            ts=ts,
        )
        return stats._with(heartbeats=1)

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

    def _resolve_helper(self, *, stream: bool = False) -> list[str]:
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
        argv = self._with_policy_args(argv, stream=stream)
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

    def _with_policy_args(self, argv: list[str], *, stream: bool = False) -> list[str]:
        """Append the allow/block policy so the helper gates content AT THE SOURCE.

        The capture helper enforces the pause + allowlist-only policy before
        reading any AX text, so paused/disallowed app content is never read or
        sent over IPC. The Python redaction pass (`redact.decide`) still runs as
        authoritative defense-in-depth.

        With ``stream=True`` the persistent-mode flags are appended: ``--stream``
        plus the (config-clamped) timing knobs. Poll/one-shot spawns never pass
        them, so ``openbird doctor`` and ``--once`` behave exactly as before.
        """
        allow = list(getattr(self.settings, "allowlist", None) or [])
        block = list(getattr(self.settings, "blocklist", None) or [])
        extra: list[str] = ["--pause-file", str(self._pause_file())]
        if allow:
            extra += ["--allow", ",".join(allow)]
        if block:
            extra += ["--block", ",".join(block)]
        detailed_capture_apps = list(
            getattr(self.settings, "detailed_capture_apps", None) or []
        )
        if detailed_capture_apps:
            extra += ["--detailed-capture-apps", ",".join(detailed_capture_apps)]
        # Opt-in browser URL capture (Apple Events): only pass the flag when the
        # user enabled it, so the helper never scripts a browser — and never
        # triggers an Automation prompt — by default.
        if getattr(self.settings, "capture_urls", False):
            extra += ["--capture-urls"]
        if stream:
            s = self.settings
            extra += [
                "--stream",
                "--idle-tick",
                str(getattr(s, "capture_idle_tick_seconds", 5.0)),
                "--afk-threshold",
                str(getattr(s, "capture_afk_threshold_seconds", 150.0)),
                "--ceiling",
                str(getattr(s, "capture_force_ceiling_seconds", 60.0)),
                "--min-gap",
                str(getattr(s, "capture_min_gap_seconds", 1.0)),
            ]
            # Opt-in OCR fallback (Phase C2): STREAM-ONLY — the helper's
            # per-app throttle needs process-lifetime state a one-shot spawn
            # cannot hold. Passed only when apps are opted in, mirroring
            # --capture-urls, so an un-opted helper never touches SCK.
            ocr_apps = list(getattr(s, "capture_ocr_apps", None) or [])
            if ocr_apps:
                extra += [
                    "--ocr-apps",
                    ",".join(ocr_apps),
                    "--ocr-min-interval",
                    str(getattr(s, "capture_ocr_min_interval_seconds", 30.0)),
                ]
        return argv + extra

    def _spawn(self, *, stream: bool = False) -> subprocess.Popen[str]:
        """Launch the helper as a text-mode subprocess yielding stdout lines.

        stdout carries the JSON event stream. stderr is **drained on a separate
        bounded thread** (:meth:`_drain_stderr`) so a helper that writes a lot to
        stderr cannot fill the pipe buffer and deadlock capture; its contents are
        never logged (only a capped byte count), honoring subprocess hygiene.
        """
        argv = self._resolve_helper(stream=stream)
        proc = subprocess.Popen(
            argv,
            # Detach the helper from FD 0. When the app supervises us, FD 0 is the
            # death pipe carrying the supervisor token; an inherited stdin would let
            # the helper consume that token before _watch_supervisor() arms on it
            # (and the helper never reads stdin anyway).
            stdin=subprocess.DEVNULL,
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

    # -- persistent (stream) mode ---------------------------------------------

    def _stream_read_timeout(self) -> float:
        """No-events stall bound: 3 heartbeat intervals, floored at 30s."""
        tick = float(getattr(self.settings, "capture_idle_tick_seconds", 5.0))
        return max(_STREAM_READ_TIMEOUT_MIN, 3.0 * tick)

    def _liveness_path(self) -> Path:
        return Path(self.settings.data_dir) / LIVENESS_FILENAME

    def _write_liveness(
        self,
        *,
        mode: str,
        last_event_ts: float | None,
        last_capture_ts: float | None,
        heartbeat_seq: int | None,
    ) -> None:
        """Atomically write the metadata-only liveness sidecar (best-effort).

        capture-health reads this to distinguish "daemon alive and eventing"
        from "daemon gone / wedged" without ever reporting ok off a stale
        timestamp. Write-temp-then-rename inside data_dir keeps readers from
        ever observing a torn file. Failures are logged as a reason code only.
        """
        payload = {
            "instance_uuid": self._instance_uuid,
            "pid": os.getpid(),
            "runtime_version": self._runtime_version,
            "updated_at": time.time(),
            "last_event_ts": last_event_ts,
            "last_capture_ts": last_capture_ts,
            "mode": mode,
            "afk": self._afk,
            "heartbeat_seq": heartbeat_seq,
            # Meeting-live latch (Phase C1): a metadata BIT for the
            # background-LLM gate. getattr-guarded so bare NullSpanTracker
            # stand-ins (tests) stay valid.
            "meeting": bool(getattr(self._span_tracker, "meeting_live", False)),
            # OCR fallback availability (Phase C2): "available" /
            # "unavailable" from the helper's preflight edges, None when
            # never reported. Metadata only; capture-health renders it.
            "ocr_state": self._ocr_state,
        }
        path = self._liveness_path()
        tmp = path.with_name(path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, path)
        except OSError:
            logger.debug("capture: liveness_write_failed")

    def run_persistent(
        self, *, stop_event: "threading.Event | None" = None
    ) -> CaptureStats:
        """Run ONE persistent-helper cycle: spawn ``--stream``, consume events.

        A reader thread pushes decoded lines into a queue; this loop drains it
        in :data:`_STREAM_STOP_SLICE` slices so a stop request is honored
        within ~1s (never a full stall timeout). Outcomes:

          * ``stop_event`` set -> helper terminated, CLEAN return (stats).
          * helper stdout EOF, rc != 0 -> :class:`HelperExitError` (supervisor
            backoff applies; exit 2 = AX denial, 3 = pipe/EPIPE refusal).
          * EOF, rc == 0, NO heartbeat ever seen -> old binary that ignored
            ``--stream``: mark stream unsupported, CLEAN return (the caller
            falls back to poll mode — no error, no breaker hit).
          * EOF, rc == 0, heartbeats seen -> unexpected self-exit of a healthy
            stream helper: :class:`HelperExitError` (respawn with backoff).
          * no events for the stall timeout with the process alive -> reap it
            and raise :class:`HelperExitError` ("stalled").
        """
        stop = stop_event or threading.Event()
        # Restart epoch: a new helper process means a new scheduler/monotonic
        # context — spans never merge across helper lifetimes.
        self._span_tracker.begin_epoch()
        proc = self._spawn(stream=True)
        stats = CaptureStats()
        # Bounded: if ingest wedges (e.g. a stuck embed call), the reader
        # blocks once full, the pipe backs up, and the helper's writes stall —
        # bounded memory with natural backpressure instead of unbounded growth.
        lines: "queue.Queue[object]" = queue.Queue(maxsize=_STREAM_QUEUE_MAX)
        eof_sentinel = object()

        def _reader(stdout) -> None:
            try:
                for line in stdout:
                    lines.put(line)
            except (OSError, ValueError):
                pass  # stream closed during shutdown
            finally:
                lines.put(eof_sentinel)

        reader = threading.Thread(
            target=_reader, args=(proc.stdout,), name="capture-stream-reader",
            daemon=True,
        )
        reader.start()

        saw_heartbeat = False
        heartbeat_seq: int | None = None
        last_event_ts: float | None = None
        last_capture_ts: float | None = None
        read_timeout = self._stream_read_timeout()
        last_line_mono = time.monotonic()
        last_liveness_mono = 0.0
        stopped_by_us = False
        try:
            while True:
                if stop.is_set():
                    stopped_by_us = True
                    break
                try:
                    item = lines.get(timeout=_STREAM_STOP_SLICE)
                except queue.Empty:
                    if time.monotonic() - last_line_mono > read_timeout:
                        if proc.poll() is None:
                            # Alive but silent past 3 heartbeat intervals: a
                            # wedged helper. Reap and let the supervisor back
                            # off (metadata-only log).
                            logger.warning(
                                "capture stream: reason=helper_stalled "
                                "(silent for %.0fs)",
                                read_timeout,
                            )
                            raise HelperExitError(
                                "capture helper stalled"
                            ) from None
                        # Process died; the reader's EOF sentinel arrives next.
                    continue
                if item is eof_sentinel:
                    break
                last_line_mono = time.monotonic()
                event = parse_event(item)  # type: ignore[arg-type]
                if event is None:
                    if str(item).strip():
                        stats = stats._with(errors=1)
                    continue
                event_type = event.get("type", "capture")
                if event_type == "heartbeat":
                    saw_heartbeat = True
                    seq = event.get("seq")
                    heartbeat_seq = seq if isinstance(seq, int) else heartbeat_seq
                ts = _finite_ts(event.get("ts"))
                last_event_ts = ts
                if event_type == "capture":
                    last_capture_ts = ts
                stats = self.handle_event(event, stats)
                now_mono = time.monotonic()
                if now_mono - last_liveness_mono >= _LIVENESS_WRITE_INTERVAL:
                    self._write_liveness(
                        mode="stream",
                        last_event_ts=last_event_ts,
                        last_capture_ts=last_capture_ts,
                        heartbeat_seq=heartbeat_seq,
                    )
                    last_liveness_mono = now_mono
        finally:
            self._shutdown(proc)
            # Final flush so the sidecar reflects end-of-cycle state (the
            # periodic write is interval-bounded and may lag the last events).
            self._write_liveness(
                mode="stream",
                last_event_ts=last_event_ts,
                last_capture_ts=last_capture_ts,
                heartbeat_seq=heartbeat_seq,
            )

        if stopped_by_us or stop.is_set():
            return stats
        rc = proc.returncode
        if rc is not None and rc != 0:
            raise HelperExitError(f"capture helper exited with code {rc}")
        if not saw_heartbeat:
            # Old binary: ignored --stream, did its one-shot capture, exited 0.
            self._stream_supported = False
            logger.info("capture: stream_unsupported; falling back to one-shot polling")
            return stats
        # A healthy stream helper never exits 0 on its own (only on our signal).
        raise HelperExitError("stream helper exited unexpectedly (code 0)")

    def force_oneshot_mode(self) -> None:
        """Public switch to the legacy one-shot polling cadence (CLI --poll).

        Persistent/stream mode never activates for this daemon instance; the
        env force (:data:`CAPTURE_MODE_ENV`) still wins either way, matching
        the auto path's precedence.
        """
        self._stream_supported = False

    def _capture_mode(self) -> str:
        """Effective loop mode: env force wins, else downgrade-aware default."""
        forced = os.environ.get(CAPTURE_MODE_ENV, "").strip().lower()
        if forced in ("oneshot", "persistent"):
            return forced
        return "persistent" if self._stream_supported else "oneshot"

    @staticmethod
    def _watch_supervisor(token: str, fd: int, stop: "threading.Event") -> None:
        """Watch an app-owned "death pipe" and request a clean stop on EOF.

        Runs in a background daemon thread, blocking in the kernel on
        :func:`os.read` (no busy-spin). The contract with the supervising app
        (see :data:`SUPERVISOR_TOKEN_ENV`):

          1. The app writes ``token + "\\n"`` through the pipe right after launch.
             We arm ONLY after reading exactly that token off ``fd`` — a leaked
             env var on an unrelated stdin will not carry the secret token, so
             this fails open (returns without ever stopping the daemon).
          2. Once armed, we wait until EOF. The app never writes again; it holds
             the write end open for its lifetime, so EOF means the app process is
             gone (graceful exit, crash, or SIGKILL all close the fd). We then set
             ``stop`` for a clean shutdown.

        Every other path — wrong/short token, EOF before the token arrives, an
        over-long first line, or any OSError — returns WITHOUT setting ``stop``
        (fail-open). We would rather miss a cleanup than stop the wrong daemon.

        Stoppability: every read waits via :func:`select.select` with a short
        timeout, re-checking ``stop`` between waits, so a normal shutdown
        (``stop`` set by SIGINT / ``max_cycles`` / circuit breaker) makes this
        thread return promptly even though the pipe never EOFs — it does not have
        to block until the app dies. The thread is a daemon thread regardless.

        ``fd`` is the daemon's stdin (FD 0) in production; tests inject a pipe
        read end. The daemon never otherwise reads stdin, so FD 0 is reserved as
        this liveness channel when armed.
        """
        expected = token.encode("utf-8", "replace")

        def _read1() -> "bytes | None":
            # Return one byte, b"" on EOF, or None when ``stop`` was requested
            # before any byte arrived (lets the caller exit without blocking).
            while not stop.is_set():
                ready, _, _ = select.select([fd], [], [], _SUPERVISOR_SELECT_TIMEOUT)
                if ready:
                    return os.read(fd, 1)
            return None

        try:
            line = bytearray()
            while True:
                chunk = _read1()
                if chunk is None:
                    return  # stop requested -> exit promptly, no spurious stop
                if not chunk:
                    return  # EOF before a full token line -> fail open, no stop
                if chunk == b"\n":
                    break
                line += chunk
                if len(line) > _SUPERVISOR_TOKEN_MAX:
                    return  # over-long line: not our short token -> fail open
            if bytes(line) != expected:
                return  # not our token (e.g. leaked env on unrelated pipe)
            # Armed: the supervising app proved itself. Now wait for the pipe to
            # close (app gone) and request a clean stop.
            while True:
                chunk = _read1()
                if chunk is None:
                    return  # stop already requested elsewhere -> nothing to do
                if not chunk:
                    break  # EOF -> supervising app gone
        except OSError:
            return  # read error: fail open, never a spurious stop
        if not stop.is_set():
            # Privacy-safe: reason code only, never captured content/window/URL.
            logger.info("capture supervisor: supervising app gone, exiting")
            stop.set()

    def run_forever(
        self,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        stop_event: "threading.Event | None" = None,
        max_consecutive_failures: int = _DEFAULT_MAX_CONSECUTIVE_FAILURES,
        max_cycles: int | None = None,
        _supervisor_fd: int = 0,
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

        App-supervised self-exit:
          * When the app spawns this daemon it sets :data:`SUPERVISOR_TOKEN_ENV`
            and writes the matching token through the inherited death pipe. We
            then run :meth:`_watch_supervisor`, which sets ``stop`` when the app
            process disappears (the pipe EOFs). Manually-started daemons have no
            token and are unaffected. See :data:`SUPERVISOR_TOKEN_ENV`.

        Args:
            poll_interval: Seconds to idle between successful cycles.
            stop_event: Set to request a clean stop; created if omitted.
            max_consecutive_failures: Circuit-breaker threshold.
            max_cycles: Stop after this many cycles (deterministic for tests).
            _supervisor_fd: Death-pipe fd to watch for the supervisor handshake
                (defaults to stdin, FD 0). Injected by tests only.

        Returns:
            Aggregate :class:`CaptureStats` across all cycles.
        """
        stop = stop_event or threading.Event()
        total = CaptureStats()
        failures = 0
        cycles = 0
        # Arm the app-supervised self-exit watcher only when the app handed us a
        # token (env var). Manual daemons (no token) never start a watcher and so
        # keep today's behavior exactly. The watcher is a daemon thread blocked in
        # the kernel on os.read, so it never busy-spins and never blocks process
        # exit; we still best-effort join it after the loop.
        supervisor_thread: threading.Thread | None = None
        supervisor_token = os.environ.get(SUPERVISOR_TOKEN_ENV)
        if supervisor_token:
            supervisor_thread = threading.Thread(
                target=self._watch_supervisor,
                args=(supervisor_token, _supervisor_fd, stop),
                name="openbird-supervisor-watch",
                daemon=True,
            )
            supervisor_thread.start()
        logger.info("capture supervisor: starting (poll_interval=%.1fs)", poll_interval)
        try:
            while not stop.is_set():
                cycle_start = time.monotonic()
                try:
                    if self._capture_mode() == "persistent":
                        total = total._add(self.run_persistent(stop_event=stop))
                    else:
                        cycle_stats = self.run(stop_event=stop)
                        total = total._add(cycle_stats)
                        # Keep the liveness sidecar current in poll mode too:
                        # after an old-binary downgrade the last stream-mode
                        # flush would otherwise go stale within 30s and
                        # capture-health would report a healthy polling daemon
                        # as "stale". Poll cycles are near-instant, so cycle
                        # wall-clock stands in for the event time — but only
                        # when the cycle actually received events (honest
                        # nulls otherwise). Metadata only.
                        now = time.time()
                        got_events = cycle_stats.received > 0
                        self._write_liveness(
                            mode="oneshot",
                            last_event_ts=now if got_events else None,
                            last_capture_ts=now if got_events else None,
                            heartbeat_seq=None,
                        )
                except HelperUnavailableError:
                    # Missing signed bundle is permanent, not transient: don't retry.
                    raise
                except Exception as exc:  # noqa: BLE001 - one bad cycle must not kill the loop
                    # Duration-based decay: a persistent cycle that survived a
                    # long time before failing is NOT part of a failure streak.
                    # A once-a-day crash must never accumulate into a breaker
                    # trip; a restart storm (fast failures) still trips it.
                    if time.monotonic() - cycle_start >= _FAILURE_DECAY_SECONDS:
                        failures = 0
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
                        # Surface sustained failure to the caller (CLI -> nonzero
                        # exit) rather than returning as if the session ended OK.
                        # `from None`: surface only the circuit-breaker summary,
                        # never chain the helper exception's message/traceback into
                        # what reaches the CLI (privacy-safe; also silences B904).
                        raise CaptureSupervisorError(
                            f"capture helper failed {failures} consecutive times"
                        ) from None
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
        finally:
            # On EVERY exit path (clean stop, max_cycles, circuit-breaker raise,
            # HelperUnavailable raise): set stop so an armed watcher abandons its
            # select/read wait, then best-effort join it. The watcher is a daemon
            # thread, so this never blocks process exit regardless. The open
            # span closes at its last-event time — a daemon stop is a span
            # boundary (never leave a logically-open span behind).
            stop.set()
            self._span_tracker.close_open()
            self._join_supervisor(supervisor_thread)
        logger.info("capture supervisor: stopped (cycles=%d)", cycles)
        return total

    @staticmethod
    def _join_supervisor(thread: "threading.Thread | None") -> None:
        """Best-effort join of the supervisor-watch thread.

        The thread may still be blocked in :func:`os.read` on the death pipe (no
        EOF yet — e.g. a normal SIGINT/``max_cycles`` stop). It is a daemon
        thread, so we never hang process exit on it; we just join briefly so a
        long-lived embedding process does not accumulate live threads.
        """
        if thread is not None:
            thread.join(timeout=1.0)

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
    "SUPERVISOR_TOKEN_ENV",
    "CAPTURE_MODE_ENV",
    "LIVENESS_FILENAME",
]
