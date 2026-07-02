"""Heartbeat-merged activity spans: the daemon-side span state machine.

``SpanTracker`` turns the typed capture-event stream (frames, heartbeats,
afk transitions, system events, app_changed boundary markers) into
``activity_spans`` rows — ground-truth "app X was frontmost from t1 to t2"
time, two-tier by the structural policy (:func:`redact.classify_policy`).

Design rules (all reached adversarial-review consensus; see
docs/design/capture-efficiency-redesign.md):

* Identity tuple = ``(bundle_id, detail_tier, reason, window, url_host,
  identity_key, afk, meeting)`` with NULL-equals-NULL semantics (plain
  ``None`` equality). Tier-0 identities are built with ``window=None, url_host=None,
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
from dataclasses import dataclass, replace
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

# Meeting-app sets (Phase C1). The meeting JUDGMENT is the conjunction
# ``mic_hot AND (bundle in MEETING_BUNDLES OR url_host in MEETING_HOSTS)`` and
# lives HERE, daemon-side: only Python has the tier-gated url_host (a Meet tab
# is a browser bundle; its host exists only on tier-1 spans), and mic-alone
# would flag dictation/Siri while app-alone would flag idling in Discord.
#
# Python-only by design — no Swift copy, no parity test: unlike the
# dangerous-app list (a privacy boundary enforced BEFORE any AX read, hence
# three lockstep copies), this set is an enrichment label. A stale entry
# mislabels time; it never leaks content. User override is deferred.
#
# Entries are stored CASEFOLDED and compared casefolded (bundle ids arrive raw
# from NSRunningApplication and are STORED raw — only the comparison folds).
MEETING_BUNDLES = frozenset(
    s.casefold()
    for s in (
        "us.zoom.xos",
        "com.microsoft.teams2",
        "com.microsoft.teams",
        "com.apple.FaceTime",
        "Cisco-Systems.Spark",
        "com.webex.meetingmanager",
        "com.hnc.Discord",
    )
)
MEETING_HOSTS = frozenset(
    s.casefold()
    for s in (
        "meet.google.com",
        "teams.microsoft.com",
        "teams.live.com",
        "app.zoom.us",
    )
)


def _meeting_app(bundle_id: str | None, url_host: str | None) -> bool:
    """Casefolded membership test against the meeting sets (mic not consulted)."""
    if bundle_id is not None and bundle_id.casefold() in MEETING_BUNDLES:
        return True
    return url_host is not None and url_host.casefold() in MEETING_HOSTS


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
        meeting: bool = ...,
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
    #: Meeting conjunction bit (Phase C1). Part of the identity so a mid-span
    #: flip SPLITS the span (extend-in-place would lie about time), exactly
    #: like ``afk``.
    meeting: bool = False


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

    #: Mirrors :attr:`SpanTracker.meeting_live` (liveness sidecar field).
    meeting_live: bool = False

    def begin_epoch(self) -> None: ...

    def on_frame(self, **_kw) -> None:
        return None

    def on_app_changed(self, *_a, **_kw) -> None: ...

    def on_heartbeat(self, *_a, **_kw) -> None: ...

    def on_afk_transition(self, *_a, **_kw) -> None: ...

    def on_system(self, *_a, **_kw) -> None: ...

    def on_mic(self, *_a, **_kw) -> None: ...

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
        # Pending app-boundary marker: (bundle_id, wall_ts, mono_ts, mic_hot).
        # The mic bit records the state AT MARKER TIME so backdating across the
        # marker can apply the Phase C1 exactness rule (see _open_new).
        self._pending: tuple[str | None, float, float, bool] | None = None
        self.error_count = 0
        # Mic run-state (Phase C1): the raw hardware bit from the helper's
        # mic_started/mic_stopped edges, plus the last edge's wall time.
        self._mic_hot = False
        self._mic_edge_wall: float | None = None
        # Deferral latch: a meeting span was seen since the mic went hot.
        # meeting_live stays ON while the user glances at Notes mid-call and
        # never fires for dictation (mic hot but no meeting surface).
        self._meeting_seen = False

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
        # Mic state re-learns from the new helper: a fresh helper re-emits
        # mic_started when the mic is ALREADY hot, so resetting here can never
        # lose a live call — but a stale hot bit from a dead helper (its
        # mic_stopped lost with it) must not survive into the new epoch.
        self._mic_hot = False
        self._mic_edge_wall = None
        self._meeting_seen = False
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
                meeting=identity.meeting,
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

    # -- meeting conjunction (Phase C1) -----------------------------------------

    @property
    def meeting_live(self) -> bool:
        """Deferral signal: mic currently hot AND a meeting span was seen.

        The latch keeps deferral ON while the user glances at a non-meeting
        app mid-call; it clears on mic-off and on epoch change. Written to the
        liveness sidecar (a metadata bit) for the background-LLM gate.
        """
        return self._mic_hot and self._meeting_seen

    def _is_meeting(self, bundle_id: str | None, url_host: str | None) -> bool:
        """The meeting judgment: mic hot AND a meeting bundle/host (casefolded)."""
        return self._mic_hot and _meeting_app(bundle_id, url_host)

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
            # NO window/url/identity keys, unconditionally. Meeting matches on
            # the bundle only (url_host is None here BY DESIGN, so a coarse
            # browser span never flags via a Meet tab — but a non-allowlisted
            # Zoom still gets meeting=1 on its coarse span).
            return _SpanIdentity(
                bundle_id=app,
                detail_tier=0,
                reason=cls.reason,
                window=None,
                url_host=None,
                identity_key=None,
                afk=afk,
                meeting=self._is_meeting(app, None),
            )
        safe_window, _safe_url, _rules = redact.scrub_metadata(window=window, url=None)
        host = _url_host(url)
        return _SpanIdentity(
            bundle_id=app,
            detail_tier=1,
            reason=None,
            window=safe_window,
            url_host=host,
            identity_key=None,  # extraction deferred (storage shape decided)
            afk=afk,
            meeting=self._is_meeting(app, host),
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
        if span.identity.meeting:
            self._meeting_seen = True
        if ts > span.last_event_wall:
            span.last_event_wall = ts
        span.last_event_mono = now_mono
        if now_mono - span.last_flush_mono >= _SPAN_FLUSH_INTERVAL:
            self._safe_extend(span.span_id, span.last_event_wall)
            span.last_flush_mono = now_mono
        return span.span_id

    def _open_new(
        self,
        identity: _SpanIdentity,
        ts: float,
        now_mono: float,
        *,
        consume_pending: bool = True,
    ) -> str | None:
        self.close_open()
        start_ts = ts
        # A matching pending boundary marker backdates the start to the exact
        # (pre-debounce) switch instant.
        if consume_pending and self._pending is not None:
            p_bundle, p_wall, p_mono, p_mic = self._pending
            if p_bundle == identity.bundle_id and (
                now_mono - p_mono
            ) <= self.pulsetime:
                # Exactness rule (Phase C1): backdating across the marker is
                # only truthful when the meeting bit did NOT flip between
                # marker time and frame time. On a flip (mic edge landed in
                # between), the pre-edge stretch is materialized with the
                # MARKER-time bit and the new span opens at the mic edge — an
                # app switch to Zoom at t=100 with the mic starting at t=101
                # yields non-meeting [100,101) + meeting [101,...), never a
                # meeting span backdated to 100 (and inversely after a stop).
                marker_meeting = p_mic and _meeting_app(
                    identity.bundle_id, identity.url_host
                )
                if marker_meeting == identity.meeting:
                    start_ts = min(p_wall, ts)
                else:
                    edge = (
                        self._mic_edge_wall
                        if self._mic_edge_wall is not None
                        else ts
                    )
                    start_ts = min(max(p_wall, edge), ts)
                    if start_ts > p_wall:
                        pre = replace(identity, meeting=marker_meeting)
                        self._safe_open(pre, p_wall, start_ts)
            self._pending = None
        span_id = self._safe_open(identity, start_ts, max(start_ts, ts))
        if span_id is None:
            return None
        if identity.meeting:
            self._meeting_seen = True
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
        if span is not None and span.identity.reason == "paused":
            # Capture is paused: app switches are not span boundaries (there
            # is nothing being captured to attribute). Record the marker so a
            # near-simultaneous unpause frame can still backdate its start,
            # but never fragment the paused span itself.
            self._pending = (bundle_id, ts, now_mono, self._mic_hot)
            return
        if span is not None and span.identity.bundle_id == bundle_id:
            # Re-activation of the same app: not a boundary.
            return
        if self._pending is not None:
            # Fast A->B->A: the middle app never produced a frame. Materialize
            # its small span [pending.ts, ts] so the timeline stays truthful.
            p_bundle, p_wall, p_mono, p_mic = self._pending
            if (
                p_bundle != bundle_id
                and (now_mono - p_mono) <= self.pulsetime
                and ts > p_wall
            ):
                base = self._identity_for_frame(
                    app=p_bundle, window=None, url=None,
                    incognito=False, paused=False,
                    afk=span.identity.afk if span else False,
                )
                # Meeting exactness for the materialized stretch: the bit must
                # reflect the MIC STATE DURING [p_wall, ts), not at judgment
                # time. If the mic flipped mid-stretch, split at the edge so
                # each side carries its true bit (same rule _open_new applies
                # to backdating).
                app_meet = _meeting_app(p_bundle, None)
                edge = self._mic_edge_wall
                if not app_meet or p_mic == self._mic_hot:
                    self._safe_open(base, p_wall, ts)
                elif edge is not None and p_wall < edge < ts:
                    self._safe_open(replace(base, meeting=p_mic), p_wall, edge)
                    self._safe_open(
                        replace(base, meeting=self._mic_hot), edge, ts
                    )
                elif edge is not None and edge <= p_wall:
                    # Flip happened before the stretch began: current bit holds
                    # for the whole range (base already carries it).
                    self._safe_open(base, p_wall, ts)
                else:
                    # Flip after the stretch ended (or edge unknown): the
                    # marker-time bit holds for the whole range.
                    self._safe_open(
                        replace(base, meeting=p_mic and app_meet), p_wall, ts
                    )
        if span is not None:
            # Close the outgoing app's span at the exact switch instant.
            span.last_event_wall = max(span.last_event_wall, min(ts, time.time()))
            self.close_open()
        self._pending = (bundle_id, ts, now_mono, self._mic_hot)

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
                    meeting=False,  # paused pseudo-spans never flag
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
                # Copied AS-IS (critical): on a call you don't type — meeting
                # spans routinely go AFK and must stay meeting time.
                meeting=identity.meeting,
            )
            self._open_new(afk_identity, max(span.start_ts, ts), now_mono)
            return
        # Returning: close the AFK span at the return instant; the
        # return_from_afk frame opens the fresh active span.
        if span is not None and span.identity.afk:
            span.last_event_wall = max(span.last_event_wall, ts)
            self.close_open()

    def on_mic(self, hot: bool, ts: float) -> None:
        """Mic run-state edge (mic_started/mic_stopped): recompute + split.

        The meeting bit is part of the merge identity, so when the edge flips
        the OPEN span's recomputed bit, the span truncate-closes at the exact
        edge instant and reopens with the bit flipped (backdated edge, exactly
        mirroring :meth:`on_afk_transition`). Duplicate edges are no-ops.
        """
        hot = bool(hot)
        if hot == self._mic_hot:
            return
        self._mic_hot = hot
        self._mic_edge_wall = ts
        if not hot:
            self._meeting_seen = False
        span = self._open
        if span is None:
            return
        identity = span.identity
        new_meeting = self._is_meeting(identity.bundle_id, identity.url_host)
        if new_meeting == identity.meeting:
            return  # paused pseudo-spans (NULL bundle) always land here
        now_mono = self._mono()
        # Close the outgoing span at the exact edge instant (clamped to now,
        # like the app-switch boundary), then reopen with the bit flipped.
        span.last_event_wall = max(span.last_event_wall, min(ts, time.time()))
        self.close_open()
        self._open_new(
            replace(identity, meeting=new_meeting),
            max(span.start_ts, ts),
            now_mono,
            consume_pending=False,
        )

    def on_system(self, kind: str | None, ts: float) -> None:
        """Sleep/lock force-close; wake/unlock is a no-op (next frame opens)."""
        if kind in ("will_sleep", "screen_locked"):
            span = self._open
            if span is not None:
                span.last_event_wall = max(span.start_ts, min(ts, time.time()))
            self.close_open()
            self._pending = None


__all__ = [
    "MEETING_BUNDLES",
    "MEETING_HOSTS",
    "NullSpanTracker",
    "SpanTracker",
    "policy_fingerprint",
]
