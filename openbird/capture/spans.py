"""Heartbeat-merged activity spans: the daemon-side span state machine.

``SpanTracker`` turns the typed capture-event stream (frames, heartbeats,
afk transitions, system events, app_changed boundary markers) into
``activity_spans`` rows — ground-truth "app X was frontmost from t1 to t2"
time, two-tier by the structural policy (:func:`redact.classify_policy`).

Design rules (all reached adversarial-review consensus; see
docs/design/capture-efficiency-redesign.md):

* Identity tuple = ``(bundle_id, detail_tier, reason, window, url_host,
  identity_key, afk)`` with NULL-equals-NULL semantics (plain ``None``
  equality). Tier-0 identities are built with ``window=None, url_host=None,
  identity_key=None`` unconditionally — the SECOND enforcement point after the
  helper (a malicious/old helper sending a title for a blocked app still
  yields a title-free span).
* Merge deadlines use the MONOTONIC clock only; wall time is stored, never
  compared. A wall-clock jump can neither extend nor split a span.
* Force-close cases: sleep, screen lock, AFK boundary, pause flip, epoch
  change (helper/daemon restart), pulsetime expiry, ceiling cap — each closes
  at the LAST EVENT's wall time, never at "now".
* Span writes are metadata-only and best-effort: a store failure logs a
  reason code and returns ``None`` (the observation still ingests with
  ``span_id=NULL`` — store-with-null, never skip).
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

from openbird.capture import redact
from openbird.config import Settings

logger = logging.getLogger("openbird.capture")

#: Max seconds between DB end_ts flushes for an open span. A crash loses at
#: most this much extension — and never GAINS time (end_ts only advances on
#: real events).
_SPAN_FLUSH_INTERVAL = 10.0

#: Floor/derivation constants for the merge pulsetime.
_PULSETIME_FLOOR = 15.0


class SpanSink(Protocol):
    """The store surface the tracker needs (MemoryStore satisfies it)."""

    def open_span(
        self,
        *,
        epoch_id: str,
        start_ts: float,
        end_ts: float,
        bundle_id: str | None,
        detail_tier: int,
        window: str | None = ...,
        url_host: str | None = ...,
        identity_key: str | None = ...,
        afk: bool = ...,
        reason: str | None = ...,
        span_id: str | None = ...,
    ) -> str: ...

    def extend_span(self, span_id: str, end_ts: float) -> None: ...

    def close_span(self, span_id: str, end_ts: float) -> None: ...


@dataclass(frozen=True)
class _SpanIdentity:
    """The merge-identity tuple (NULL-equals-NULL via plain None equality)."""

    bundle_id: str | None
    detail_tier: int
    reason: str | None
    window: str | None
    url_host: str | None
    identity_key: str | None
    afk: bool


@dataclass
class _OpenSpan:
    span_id: str
    identity: _SpanIdentity
    start_ts: float
    last_event_wall: float
    last_event_mono: float
    last_flush_mono: float


def _url_host(url: str | None) -> str | None:
    """Host component only (never path/query — those can carry content)."""
    if not url:
        return None
    try:
        return urlsplit(url).hostname
    except ValueError:
        return None


def policy_fingerprint(settings: Settings) -> str:
    """Stable hash over the capture policy (allow/block/url flag).

    Stored in tracker state (not the DB). Policy changes only take effect via
    daemon restart, and every restart begins a new epoch that force-closes the
    open span — so the design's force-close-on-policy-change rule holds by
    construction; the fingerprint makes that auditable in logs.
    """
    allow = ",".join(sorted(getattr(settings, "allowlist", None) or []))
    block = ",".join(sorted(getattr(settings, "blocklist", None) or []))
    urls = "1" if getattr(settings, "capture_urls", False) else "0"
    return hashlib.sha256(f"{allow}\n{block}\n{urls}".encode()).hexdigest()[:16]


class NullSpanTracker:
    """No-op tracker for sinks without the span API (lightweight test fakes)."""

    def begin_epoch(self) -> None: ...

    def on_frame(self, **_kw) -> None:
        return None

    def on_app_changed(self, *_a, **_kw) -> None: ...

    def on_heartbeat(self, *_a, **_kw) -> None: ...

    def on_afk_transition(self, *_a, **_kw) -> None: ...

    def on_system(self, *_a, **_kw) -> None: ...

    def close_open(self, *_a, **_kw) -> None: ...


class SpanTracker:
    """Pure span state machine; the daemon feeds it typed events in order.

    All inputs arrive on the daemon's single event-processing thread, so the
    tracker needs no locking. ``mono`` is injectable for deterministic tests.
    """

    def __init__(
        self,
        store: SpanSink,
        settings: Settings,
        *,
        mono=time.monotonic,
    ) -> None:
        self._store = store
        self._settings = settings
        self._mono = mono
        self._epoch_id: str | None = None
        self._fingerprint = policy_fingerprint(settings)
        self._open: _OpenSpan | None = None
        # Pending app-boundary marker: (bundle_id, wall_ts, mono_ts).
        self._pending: tuple[str | None, float, float] | None = None
        self.error_count = 0

        ceiling = float(getattr(settings, "capture_force_ceiling_seconds", 60.0))
        tick = float(getattr(settings, "capture_idle_tick_seconds", 5.0))
        configured = float(getattr(settings, "capture_span_pulsetime_seconds", 0.0))
        if configured > 0:
            pulse = configured
        else:
            pulse = max(2.0 * tick + 5.0, _PULSETIME_FLOOR)
        # Pulsetime may never exceed the ceiling (a silent gap longer than the
        # ceiling must split, not merge).
        self.pulsetime = min(pulse, ceiling)
        self.begin_epoch()

    # -- epoch / lifecycle -----------------------------------------------------

    def begin_epoch(self) -> None:
        """Start a new restart epoch: force-close any open span first.

        Called at construction (daemon start) and at the start of every
        persistent-helper cycle (helper restart). Merging never crosses an
        epoch: monotonic deadlines are only valid within one process lifetime
        of the clock feeding them.
        """
        import uuid

        self.close_open()
        self._pending = None
        self._epoch_id = uuid.uuid4().hex
        logger.debug(
            "capture: span_epoch begin policy_fp=%s", self._fingerprint
        )

    def close_open(self) -> None:
        """Force-close the open span at its LAST EVENT wall time (never now)."""
        if self._open is None:
            return
        span = self._open
        self._open = None
        self._safe_close(span.span_id, max(span.start_ts, span.last_event_wall))

    # -- store wrappers (best-effort, metadata-only logging) --------------------

    def _safe_close(self, span_id: str, end_ts: float) -> None:
        try:
            self._store.close_span(span_id, end_ts)
        except Exception as exc:  # noqa: BLE001 - span loss must not break capture
            self.error_count += 1
            logger.error(
                "capture: span_close failed error_type=%s", type(exc).__name__,
                exc_info=False,
            )

    def _safe_open(
        self, identity: _SpanIdentity, start_ts: float, end_ts: float
    ) -> str | None:
        try:
            return self._store.open_span(
                epoch_id=self._epoch_id or "unknown",
                start_ts=start_ts,
                end_ts=end_ts,
                bundle_id=identity.bundle_id,
                detail_tier=identity.detail_tier,
                window=identity.window,
                url_host=identity.url_host,
                identity_key=identity.identity_key,
                afk=identity.afk,
                reason=identity.reason,
            )
        except Exception as exc:  # noqa: BLE001
            self.error_count += 1
            logger.error(
                "capture: span_open failed error_type=%s app=%s",
                type(exc).__name__,
                identity.bundle_id,
                exc_info=False,
            )
            return None

    def _safe_extend(self, span_id: str, end_ts: float) -> None:
        try:
            self._store.extend_span(span_id, end_ts)
        except Exception as exc:  # noqa: BLE001
            self.error_count += 1
            logger.error(
                "capture: span_extend failed error_type=%s", type(exc).__name__,
                exc_info=False,
            )

    # -- identity ----------------------------------------------------------------

    def _identity_for_frame(
        self,
        *,
        app: str | None,
        window: str | None,
        url: str | None,
        incognito: bool,
        paused: bool,
        afk: bool,
    ) -> _SpanIdentity:
        cls = redact.classify_policy(
            app=app,
            window=window,
            incognito=incognito,
            paused=paused,
            settings=self._settings,
        )
        if cls.tier == redact.SPAN_TIER_COARSE:
            # Second enforcement point (after the helper): coarse spans carry
            # NO window/url/identity keys, unconditionally.
            return _SpanIdentity(
                bundle_id=app,
                detail_tier=0,
                reason=cls.reason,
                window=None,
                url_host=None,
                identity_key=None,
                afk=afk,
            )
        safe_window, _safe_url, _rules = redact.scrub_metadata(window=window, url=None)
        return _SpanIdentity(
            bundle_id=app,
            detail_tier=1,
            reason=None,
            window=safe_window,
            url_host=_url_host(url),
            identity_key=None,  # extraction deferred (storage shape decided)
            afk=afk,
        )

    # -- merge core ----------------------------------------------------------------

    def _can_extend(self, identity: _SpanIdentity, now_mono: float) -> bool:
        span = self._open
        if span is None:
            return False
        if span.identity != identity:
            return False
        return (now_mono - span.last_event_mono) <= self.pulsetime

    def _extend(self, ts: float, now_mono: float) -> str:
        span = self._open
        assert span is not None
        if ts > span.last_event_wall:
            span.last_event_wall = ts
        span.last_event_mono = now_mono
        if now_mono - span.last_flush_mono >= _SPAN_FLUSH_INTERVAL:
            self._safe_extend(span.span_id, span.last_event_wall)
            span.last_flush_mono = now_mono
        return span.span_id

    def _open_new(
        self, identity: _SpanIdentity, ts: float, now_mono: float
    ) -> str | None:
        self.close_open()
        start_ts = ts
        # A matching pending boundary marker backdates the start to the exact
        # (pre-debounce) switch instant.
        if self._pending is not None:
            p_bundle, p_wall, p_mono = self._pending
            if p_bundle == identity.bundle_id and (
                now_mono - p_mono
            ) <= self.pulsetime:
                start_ts = min(p_wall, ts)
            self._pending = None
        span_id = self._safe_open(identity, start_ts, max(start_ts, ts))
        if span_id is None:
            return None
        self._open = _OpenSpan(
            span_id=span_id,
            identity=identity,
            start_ts=start_ts,
            last_event_wall=max(start_ts, ts),
            last_event_mono=now_mono,
            last_flush_mono=now_mono,
        )
        return span_id

    # -- inputs ----------------------------------------------------------------

    def on_frame(
        self,
        *,
        app: str | None,
        window: str | None,
        url: str | None,
        incognito: bool,
        paused: bool,
        afk: bool,
        ts: float,
    ) -> str | None:
        """Merge-or-open for one capture frame; returns the frame's span id."""
        identity = self._identity_for_frame(
            app=app, window=window, url=url,
            incognito=incognito, paused=paused, afk=afk,
        )
        now_mono = self._mono()
        if self._can_extend(identity, now_mono):
            return self._extend(ts, now_mono)
        return self._open_new(identity, ts, now_mono)

    def on_app_changed(self, bundle_id: str | None, ts: float) -> None:
        """Pre-debounce boundary marker: exact span edges on app switches."""
        now_mono = self._mono()
        span = self._open
        if span is not None and span.identity.bundle_id == bundle_id:
            # Re-activation of the same app: not a boundary.
            return
        if self._pending is not None:
            # Fast A->B->A: the middle app never produced a frame. Materialize
            # its small span [pending.ts, ts] so the timeline stays truthful.
            p_bundle, p_wall, p_mono = self._pending
            if p_bundle != bundle_id and (now_mono - p_mono) <= self.pulsetime:
                identity = self._identity_for_frame(
                    app=p_bundle, window=None, url=None,
                    incognito=False, paused=False,
                    afk=span.identity.afk if span else False,
                )
                if ts > p_wall:
                    self._safe_open(identity, p_wall, ts)
        if span is not None:
            # Close the outgoing app's span at the exact switch instant.
            span.last_event_wall = max(span.last_event_wall, min(ts, time.time()))
            self.close_open()
        self._pending = (bundle_id, ts, now_mono)

    def on_heartbeat(self, *, afk: bool, paused: bool, ts: float) -> None:
        """Liveness pulse: extends a matching open span; manages paused spans."""
        now_mono = self._mono()
        span = self._open
        if paused:
            if span is not None and span.identity.reason != "paused":
                self.close_open()
                span = None
            if span is None:
                identity = _SpanIdentity(
                    bundle_id=None,  # the only NULL-bundle case: helper emits
                    detail_tier=0,   # no frames while paused
                    reason="paused",
                    window=None,
                    url_host=None,
                    identity_key=None,
                    afk=afk,
                )
                self._open_new(identity, ts, now_mono)
                return
            if self._can_extend(span.identity, now_mono):
                self._extend(ts, now_mono)
            else:
                self.close_open()
            return
        if span is not None and span.identity.reason == "paused":
            self.close_open()
            return
        if span is not None and span.identity.afk == afk:
            if self._can_extend(span.identity, now_mono):
                self._extend(ts, now_mono)
            else:
                self.close_open()

    def on_afk_transition(self, *, afk: bool, ts: float) -> None:
        """AFK boundary: truncate-close, then mirror the span with afk flipped."""
        now_mono = self._mono()
        span = self._open
        if afk:
            if span is None:
                return  # restart while away: honest untracked
            identity = span.identity
            span.last_event_wall = max(span.start_ts, min(span.last_event_wall, ts))
            self.close_open()
            afk_identity = _SpanIdentity(
                bundle_id=identity.bundle_id,
                detail_tier=identity.detail_tier,
                reason=identity.reason,
                window=identity.window,
                url_host=identity.url_host,
                identity_key=identity.identity_key,
                afk=True,
            )
            self._open_new(afk_identity, max(span.start_ts, ts), now_mono)
            return
        # Returning: close the AFK span at the return instant; the
        # return_from_afk frame opens the fresh active span.
        if span is not None and span.identity.afk:
            span.last_event_wall = max(span.last_event_wall, ts)
            self.close_open()

    def on_system(self, kind: str | None, ts: float) -> None:
        """Sleep/lock force-close; wake/unlock is a no-op (next frame opens)."""
        if kind in ("will_sleep", "screen_locked"):
            span = self._open
            if span is not None:
                span.last_event_wall = max(span.start_ts, min(ts, time.time()))
            self.close_open()
            self._pending = None


__all__ = ["NullSpanTracker", "SpanTracker", "policy_fingerprint"]
