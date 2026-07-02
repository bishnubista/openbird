"""SpanTracker merge rules: identity, pulsetime, force-close matrix, epochs.

Pure state-machine tests with a fake store and injectable monotonic clock —
every rule from the reviewed Phase B plan has a case here.
"""

from __future__ import annotations

import tempfile

import pytest

from openbird.capture.spans import SpanTracker, policy_fingerprint
from openbird.config import Settings


class FakeSpanStore:
    """Records span calls; spans keyed by id with start/end/fields."""

    def __init__(self, *, fail_open: bool = False) -> None:
        self.spans: dict[str, dict] = {}
        self.fail_open = fail_open
        self._n = 0

    def open_span(self, **kw) -> str:
        if self.fail_open:
            raise RuntimeError("boom")
        self._n += 1
        span_id = kw.pop("span_id", None) or f"span{self._n}"
        self.spans[span_id] = dict(kw)
        return span_id

    def extend_span(self, span_id: str, end_ts: float) -> None:
        row = self.spans[span_id]
        row["end_ts"] = max(row["end_ts"], end_ts)

    def close_span(self, span_id: str, end_ts: float) -> None:
        # Set-exact (floored at start): closing may truncate (AFK backdating).
        row = self.spans[span_id]
        row["end_ts"] = max(row["start_ts"], end_ts)


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture()
def settings():
    return Settings(
        data_dir=tempfile.mkdtemp(prefix="openbird-spantracker-"),
        allowlist=["com.apple.mail", "com.apple.notes"],
        blocklist=["com.apple.Terminal"],
    )


def make_tracker(settings, *, fail_open: bool = False):
    store = FakeSpanStore(fail_open=fail_open)
    clock = Clock()
    tracker = SpanTracker(store, settings, mono=clock)
    return tracker, store, clock


def frame(tracker, *, app="com.apple.mail", window="Inbox", url=None,
          incognito=False, paused=False, afk=False, ts=100.0):
    return tracker.on_frame(
        app=app, window=window, url=url, incognito=incognito,
        paused=paused, afk=afk, ts=ts,
    )


# -- identity + merge ---------------------------------------------------------


def test_identical_frames_merge_into_one_span(settings):
    tracker, store, clock = make_tracker(settings)
    a = frame(tracker, ts=100.0)
    clock.t += 5.0
    b = frame(tracker, ts=105.0)
    assert a == b
    assert len(store.spans) == 1
    tracker.close_open()
    row = store.spans[a]
    assert (row["start_ts"], row["end_ts"]) == (100.0, 105.0)


def test_identity_change_splits(settings):
    tracker, store, clock = make_tracker(settings)
    a = frame(tracker, window="Inbox", ts=100.0)
    clock.t += 2.0
    b = frame(tracker, window="Drafts", ts=102.0)
    assert a != b
    assert len(store.spans) == 2
    # The first span closed at ITS last event, not at the split moment.
    assert store.spans[a]["end_ts"] == 100.0


def test_null_equals_null_merging(settings):
    # Tier-1 span with no window title: None == None must merge.
    tracker, store, clock = make_tracker(settings)
    a = frame(tracker, window=None, ts=100.0)
    clock.t += 3.0
    b = frame(tracker, window=None, ts=103.0)
    assert a == b


def test_coarse_tier_strips_window_and_url_python_side(settings):
    # Blocked app arriving WITH a title (malicious/old helper): span stores none.
    # Terminal is allowlisted here so the BLOCKLIST gate is the one that fires
    # (allowlist-first evaluation order).
    settings.allowlist = settings.allowlist + ["com.apple.Terminal"]
    tracker, store, clock = make_tracker(settings)
    sid = frame(tracker, app="com.apple.Terminal", window="secret title",
                url="https://x.test/p?q=1", ts=100.0)
    row = store.spans[sid]
    assert row["detail_tier"] == 0
    assert row["reason"] == "blocklisted"
    assert row["window"] is None
    assert row["url_host"] is None


def test_tier1_url_host_only(settings):
    tracker, store, clock = make_tracker(settings)
    sid = frame(tracker, url="https://github.com/a/b?token=x", ts=100.0)
    assert store.spans[sid]["url_host"] == "github.com"


def test_reason_is_part_of_identity(settings):
    # allow -> paused -> allow: three spans, never flattened.
    tracker, store, clock = make_tracker(settings)
    a = frame(tracker, ts=100.0)
    clock.t += 2.0
    b = frame(tracker, paused=True, ts=102.0)
    clock.t += 2.0
    c = frame(tracker, ts=104.0)
    assert len({a, b, c}) == 3
    assert store.spans[b]["reason"] == "paused"


# -- pulsetime / ceiling --------------------------------------------------------


def test_pulsetime_expiry_closes_at_last_event_not_now(settings):
    tracker, store, clock = make_tracker(settings)
    a = frame(tracker, ts=100.0)
    clock.t += tracker.pulsetime + 1.0  # silent gap past pulsetime
    b = frame(tracker, ts=100.0 + tracker.pulsetime + 1.0)
    assert a != b
    assert store.spans[a]["end_ts"] == 100.0  # never extended across the gap


def test_wall_clock_jump_with_steady_mono_still_merges(settings):
    tracker, store, clock = make_tracker(settings)
    a = frame(tracker, ts=100.0)
    clock.t += 2.0  # mono: 2s later
    b = frame(tracker, ts=100_000.0)  # wall clock jumped hours
    assert a == b  # monotonic governs merging; wall is stored only


def test_pulsetime_never_exceeds_ceiling(settings):
    import dataclasses

    s = dataclasses.replace(settings) if dataclasses.is_dataclass(settings) else settings
    s.capture_span_pulsetime_seconds = 500.0  # clamped by config to ceiling=60
    tracker, _, _ = make_tracker(s)
    assert tracker.pulsetime <= s.capture_force_ceiling_seconds


# -- force-close matrix -----------------------------------------------------------


def test_sleep_and_lock_force_close(settings):
    for kind in ("will_sleep", "screen_locked"):
        tracker, store, clock = make_tracker(settings)
        a = frame(tracker, ts=100.0)
        clock.t += 1.0
        tracker.on_system(kind, 101.0)
        assert tracker._open is None
        assert store.spans[a]["end_ts"] == 101.0
        # Wake is a no-op; the next frame opens a NEW span.
        tracker.on_system("did_wake", 102.0)
        b = frame(tracker, ts=103.0)
        assert b != a


def test_afk_transition_truncates_and_mirrors(settings):
    tracker, store, clock = make_tracker(settings)
    a = frame(tracker, ts=100.0)
    clock.t += 10.0
    frame(tracker, ts=110.0)
    clock.t += 5.0
    # User went AFK; helper backdates the boundary to when input stopped (105).
    tracker.on_afk_transition(afk=True, ts=105.0)
    assert store.spans[a]["end_ts"] == 105.0  # truncated, not extended
    # Mirrored AFK span opened with the same identity but afk=1.
    open_span = tracker._open
    assert open_span is not None and open_span.identity.afk is True
    afk_id = open_span.span_id
    # Heartbeats every 10s (inside pulsetime) keep the AFK span alive.
    hb_ts = 105.0
    for _ in range(6):
        clock.t += 10.0
        hb_ts += 10.0
        tracker.on_heartbeat(afk=True, paused=False, ts=hb_ts)
    # Return: AFK span closes at the return instant.
    clock.t += 5.0
    tracker.on_afk_transition(afk=False, ts=170.0)
    assert tracker._open is None
    assert store.spans[afk_id]["end_ts"] == 170.0


def test_afk_while_nothing_open_is_untracked(settings):
    tracker, store, clock = make_tracker(settings)
    tracker.on_afk_transition(afk=True, ts=100.0)
    assert tracker._open is None
    assert store.spans == {}


def test_paused_heartbeats_produce_null_bundle_paused_span(settings):
    tracker, store, clock = make_tracker(settings)
    a = frame(tracker, ts=100.0)
    clock.t += 1.0
    tracker.on_heartbeat(afk=False, paused=True, ts=101.0)
    paused_span = tracker._open
    assert paused_span is not None
    assert paused_span.identity.reason == "paused"
    assert paused_span.identity.bundle_id is None
    clock.t += 3.0
    tracker.on_heartbeat(afk=False, paused=True, ts=104.0)
    # Unpause closes the paused span; next frame opens the real one.
    clock.t += 1.0
    tracker.on_heartbeat(afk=False, paused=False, ts=105.0)
    assert tracker._open is None
    assert store.spans[paused_span.span_id]["end_ts"] == 104.0
    assert store.spans[a]["end_ts"] == 100.0


# -- restart epochs ----------------------------------------------------------------


def test_begin_epoch_closes_open_at_last_event(settings):
    tracker, store, clock = make_tracker(settings)
    a = frame(tracker, ts=100.0)
    clock.t += 2.0
    frame(tracker, ts=102.0)
    tracker.begin_epoch()
    assert tracker._open is None
    assert store.spans[a]["end_ts"] == 102.0
    # Post-epoch frames NEVER merge with pre-epoch spans, even if identical
    # and inside pulsetime.
    clock.t += 1.0
    b = frame(tracker, ts=103.0)
    assert b != a


def test_policy_fingerprint_stable_and_order_insensitive(settings):
    fp1 = policy_fingerprint(settings)
    settings.allowlist = list(reversed(settings.allowlist))
    assert policy_fingerprint(settings) == fp1
    settings.allowlist = settings.allowlist + ["com.new.app"]
    assert policy_fingerprint(settings) != fp1


# -- app_changed boundary markers -----------------------------------------------------


def test_app_changed_backdates_next_span(settings):
    tracker, store, clock = make_tracker(settings)
    a = frame(tracker, app="com.apple.mail", ts=100.0)
    clock.t += 1.0
    # Switch happens at 100.7 (pre-debounce marker); the frame lands at 101.0.
    tracker.on_app_changed("com.apple.notes", 100.7)
    assert store.spans[a]["end_ts"] <= 100.7  # outgoing span closed at the switch
    clock.t += 0.3
    b = frame(tracker, app="com.apple.notes", ts=101.0)
    assert store.spans[b]["start_ts"] == 100.7  # backdated to the marker


def test_fast_a_b_a_materializes_middle_span(settings):
    tracker, store, clock = make_tracker(settings)
    a = frame(tracker, app="com.apple.mail", ts=100.0)
    clock.t += 0.1
    tracker.on_app_changed("com.apple.notes", 100.1)  # A -> B
    clock.t += 0.15
    tracker.on_app_changed("com.apple.mail", 100.25)  # B -> A before B framed
    # The B span exists as a real (closed) row [100.1, 100.25].
    b_rows = [
        r for r in store.spans.values() if r["bundle_id"] == "com.apple.notes"
    ]
    assert len(b_rows) == 1
    assert (b_rows[0]["start_ts"], b_rows[0]["end_ts"]) == (100.1, 100.25)


def test_app_changed_same_app_is_not_a_boundary(settings):
    tracker, store, clock = make_tracker(settings)
    a = frame(tracker, ts=100.0)
    clock.t += 1.0
    tracker.on_app_changed("com.apple.mail", 101.0)  # re-activation, same app
    clock.t += 1.0
    b = frame(tracker, ts=102.0)
    assert a == b  # still one span


# -- store failure -----------------------------------------------------------------


def test_store_failure_returns_none_and_never_raises(settings):
    tracker, store, clock = make_tracker(settings, fail_open=True)
    sid = frame(tracker, ts=100.0)
    assert sid is None
    assert tracker.error_count >= 1


# -- meeting flag (Phase C1) --------------------------------------------------


def test_meeting_requires_conjunction_of_mic_and_meeting_app(settings):
    tracker, store, clock = make_tracker(settings)
    tracker.on_mic(True, 99.0)
    # Mic hot + meeting bundle -> meeting (coarse tier still matches bundles:
    # a non-allowlisted Zoom gets meeting=1 on its coarse span).
    a = frame(tracker, app="us.zoom.xos", window=None, ts=100.0)
    assert store.spans[a]["meeting"] is True
    assert store.spans[a]["detail_tier"] == 0
    # Mic hot + non-meeting app -> NOT a meeting (dictation/Siri guard).
    clock.t += 2.0
    b = frame(tracker, ts=102.0)
    assert store.spans[b]["meeting"] is False


def test_meeting_app_without_mic_never_flags(settings):
    # App-alone (idling in Discord, mic cold) -> not a meeting.
    tracker, store, clock = make_tracker(settings)
    a = frame(tracker, app="com.hnc.Discord", window=None, ts=100.0)
    assert store.spans[a]["meeting"] is False


def test_mid_span_mic_flip_splits_at_exact_edge_both_directions(settings):
    tracker, store, clock = make_tracker(settings)
    a = frame(tracker, app="us.zoom.xos", window=None, ts=100.0)
    clock.t += 3.0
    frame(tracker, app="us.zoom.xos", window=None, ts=103.0)
    clock.t += 2.0
    tracker.on_mic(True, 105.0)  # mic starts mid-span
    assert store.spans[a]["end_ts"] == 105.0  # closed at the exact edge
    meeting_span = tracker._open
    assert meeting_span is not None and meeting_span.identity.meeting is True
    assert store.spans[meeting_span.span_id]["start_ts"] == 105.0
    clock.t += 5.0
    tracker.on_mic(False, 110.0)  # mic stops mid-span: split back
    assert store.spans[meeting_span.span_id]["end_ts"] == 110.0
    after = tracker._open
    assert after is not None and after.identity.meeting is False
    assert store.spans[after.span_id]["start_ts"] == 110.0


def test_duplicate_mic_edges_are_noops(settings):
    tracker, store, clock = make_tracker(settings)
    tracker.on_mic(True, 99.0)
    frame(tracker, app="us.zoom.xos", window=None, ts=100.0)
    clock.t += 1.0
    tracker.on_mic(True, 101.0)  # duplicate edge: no split
    assert len(store.spans) == 1


def test_afk_mirror_preserves_meeting(settings):
    # On a call you don't type: meeting spans routinely go AFK and must stay
    # meeting time.
    tracker, store, clock = make_tracker(settings)
    tracker.on_mic(True, 99.0)
    frame(tracker, app="us.zoom.xos", window=None, ts=100.0)
    clock.t += 10.0
    tracker.on_afk_transition(afk=True, ts=105.0)
    mirror = tracker._open
    assert mirror is not None
    assert mirror.identity.afk is True and mirror.identity.meeting is True
    assert store.spans[mirror.span_id]["meeting"] is True


def test_tier1_meet_tab_flags_via_url_host(settings):
    settings.allowlist = settings.allowlist + ["com.google.Chrome"]
    tracker, store, clock = make_tracker(settings)
    tracker.on_mic(True, 99.0)
    sid = frame(tracker, app="com.google.Chrome", window="Meet",
                url="https://meet.google.com/abc-defg", ts=100.0)
    row = store.spans[sid]
    assert row["detail_tier"] == 1
    assert row["url_host"] == "meet.google.com"
    assert row["meeting"] is True


def test_tier0_browser_span_never_flags_via_url(settings):
    # Chrome NOT allowlisted -> coarse span strips the URL, so a Meet tab
    # cannot flag through a tier-0 browser span (documented, acceptable).
    tracker, store, clock = make_tracker(settings)
    tracker.on_mic(True, 99.0)
    sid = frame(tracker, app="com.google.Chrome",
                url="https://meet.google.com/abc", ts=100.0)
    row = store.spans[sid]
    assert row["detail_tier"] == 0
    assert row["url_host"] is None
    assert row["meeting"] is False


def test_meeting_matching_casefolds_but_stores_raw(settings):
    tracker, store, clock = make_tracker(settings)
    tracker.on_mic(True, 99.0)
    a = frame(tracker, app="COM.APPLE.facetime", window=None, ts=100.0)
    assert store.spans[a]["meeting"] is True
    assert store.spans[a]["bundle_id"] == "COM.APPLE.facetime"  # stored raw
    clock.t += 2.0
    b = frame(tracker, app="com.hnc.DISCORD", window=None, ts=102.0)
    assert store.spans[b]["meeting"] is True
    assert store.spans[b]["bundle_id"] == "com.hnc.DISCORD"


def test_marker_then_mic_start_never_backdates_the_meeting_span(settings):
    # App switch to Zoom at t=100, mic starts at t=101, frame at t=101.3:
    # must yield non-meeting [100,101) + meeting [101,...) — NEVER a meeting
    # span backdated to 100.
    tracker, store, clock = make_tracker(settings)
    frame(tracker, ts=99.0)
    clock.t += 1.0
    tracker.on_app_changed("us.zoom.xos", 100.0)  # marker records mic COLD
    clock.t += 1.0
    tracker.on_mic(True, 101.0)
    clock.t += 0.3
    sid = frame(tracker, app="us.zoom.xos", window=None, ts=101.3)
    assert store.spans[sid]["meeting"] is True
    assert store.spans[sid]["start_ts"] == 101.0  # opened at the mic edge
    pre = [r for r in store.spans.values()
           if r["bundle_id"] == "us.zoom.xos" and r["meeting"] is False]
    assert len(pre) == 1  # pre-edge stretch materialized with the marker bit
    assert (pre[0]["start_ts"], pre[0]["end_ts"]) == (100.0, 101.0)


def test_marker_then_mic_stop_never_backdates_the_nonmeeting_span(settings):
    # The inverse ordering: switch to Zoom mid-call at t=100, mic stops at
    # t=101 -> meeting [100,101) + non-meeting [101,...).
    tracker, store, clock = make_tracker(settings)
    tracker.on_mic(True, 98.0)
    frame(tracker, ts=99.0)  # mail: mic hot but not a meeting surface
    clock.t += 1.0
    tracker.on_app_changed("us.zoom.xos", 100.0)  # marker records mic HOT
    clock.t += 1.0
    tracker.on_mic(False, 101.0)
    clock.t += 0.3
    sid = frame(tracker, app="us.zoom.xos", window=None, ts=101.3)
    assert store.spans[sid]["meeting"] is False
    assert store.spans[sid]["start_ts"] == 101.0
    pre = [r for r in store.spans.values()
           if r["bundle_id"] == "us.zoom.xos" and r["meeting"] is True]
    assert len(pre) == 1
    assert (pre[0]["start_ts"], pre[0]["end_ts"]) == (100.0, 101.0)


def test_marker_with_unchanged_meeting_bit_backdates_normally(settings):
    tracker, store, clock = make_tracker(settings)
    tracker.on_mic(True, 98.0)
    frame(tracker, ts=99.0)
    clock.t += 1.0
    tracker.on_app_changed("us.zoom.xos", 100.0)
    clock.t += 0.3
    sid = frame(tracker, app="us.zoom.xos", window=None, ts=100.3)
    # No mic edge between marker and frame: the classic exact backdate holds.
    assert store.spans[sid]["start_ts"] == 100.0
    assert store.spans[sid]["meeting"] is True


def test_meeting_live_latch_survives_glance_and_clears_on_mic_off(settings):
    tracker, store, clock = make_tracker(settings)
    assert tracker.meeting_live is False
    tracker.on_mic(True, 99.0)
    assert tracker.meeting_live is False  # mic alone (dictation) never defers
    frame(tracker, app="us.zoom.xos", window=None, ts=100.0)
    assert tracker.meeting_live is True
    clock.t += 2.0
    frame(tracker, app="com.apple.notes", ts=102.0)  # glance mid-call
    assert tracker.meeting_live is True  # latch holds while the mic is hot
    tracker.on_mic(False, 104.0)
    assert tracker.meeting_live is False


def test_meeting_live_latch_clears_on_epoch(settings):
    tracker, store, clock = make_tracker(settings)
    tracker.on_mic(True, 99.0)
    frame(tracker, app="us.zoom.xos", window=None, ts=100.0)
    assert tracker.meeting_live is True
    tracker.begin_epoch()
    assert tracker.meeting_live is False


def test_paused_pseudo_span_never_flags_meeting(settings):
    tracker, store, clock = make_tracker(settings)
    tracker.on_mic(True, 99.0)
    tracker.on_heartbeat(afk=False, paused=True, ts=100.0)
    paused = tracker._open
    assert paused is not None and paused.identity.reason == "paused"
    assert paused.identity.meeting is False
    # A mic edge never splits the paused pseudo-span.
    clock.t += 1.0
    tracker.on_mic(False, 101.0)
    assert tracker._open is not None
    assert tracker._open.span_id == paused.span_id


def test_app_switch_never_fragments_paused_span(settings):
    # CodeRabbit regression: app switches while paused must not close the
    # paused span (capture is off; the switch attributes nothing).
    tracker, store, clock = make_tracker(settings)
    clock.t += 1.0
    tracker.on_heartbeat(afk=False, paused=True, ts=100.0)
    paused = tracker._open
    assert paused is not None and paused.identity.reason == "paused"
    clock.t += 3.0
    tracker.on_app_changed("com.apple.notes", 103.0)  # switch while paused
    clock.t += 3.0
    tracker.on_heartbeat(afk=False, paused=True, ts=106.0)
    # Still ONE paused span, alive across the switch.
    assert tracker._open is not None
    assert tracker._open.span_id == paused.span_id
    paused_rows = [r for r in store.spans.values() if r["reason"] == "paused"]
    assert len(paused_rows) == 1
