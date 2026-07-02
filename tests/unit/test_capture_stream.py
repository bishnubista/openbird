"""Persistent (stream) capture mode: supervisor, downgrade, liveness, health.

Uses the same injection seam as test_capture.py — fake helper emitters as
``python -c`` scripts with ``require_signed_helper=False`` — so no Swift binary
or Accessibility grant is needed. Every scenario from the reviewed Phase A
plan is covered: typed-event flow, hang detection (reap + HelperExitError),
death detection (exit-code propagation), old-binary downgrade (no heartbeat ->
poll fallback), stop responsiveness (~1s), sidecar atomicity, and the
health-block staleness rules.
"""

from __future__ import annotations

import json
import sys
import threading
import time

import pytest

from openbird.capture.daemon import (
    CAPTURE_MODE_ENV,
    LIVENESS_FILENAME,
    CaptureDaemon,
    CaptureStats,
    HelperExitError,
)
from openbird.capture.health import build_capture_health
from openbird.config import Settings
from openbird.types import Observation


class FakeStore:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def add_observation(self, text, *, app=None, window=None, url=None,
                        session_id=None, source, ts=None):
        self.calls.append({"text": text, "app": app, "window": window,
                           "url": url, "session_id": session_id,
                           "source": source, "ts": ts})
        return Observation(
            id="obs", content_hash="h", ts=ts or 0.0, app=app, window=window,
            url=url, session_id=session_id, source=source,
        )


@pytest.fixture()
def allow_settings(tmp_path):
    return Settings(
        data_dir=str(tmp_path),
        allowlist=["com.apple.mail"],
        blocklist=[],
    )


def _emitter(script: str) -> tuple[str, ...]:
    return (sys.executable, "-c", script)


def _daemon(settings, script: str) -> tuple[CaptureDaemon, FakeStore]:
    store = FakeStore()
    daemon = CaptureDaemon(
        store,
        settings=settings,
        helper_cmd=_emitter(script),
        require_signed_helper=False,
    )
    return daemon, store


# ---------------------------------------------------------------------------
# run_persistent — happy path
# ---------------------------------------------------------------------------


HAPPY_SCRIPT = r"""
import json, sys, time
def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n"); sys.stdout.flush()
emit({"type": "heartbeat", "ts": time.time(), "seq": 0, "afk": False, "paused": False})
emit({"type": "capture", "trigger": "startup", "app": "com.apple.mail",
      "window": "Inbox", "url": None, "text": "hello from stream",
      "ts": time.time(), "incognito": False})
emit({"type": "heartbeat", "ts": time.time(), "seq": 1, "afk": False, "paused": False})
time.sleep(30)  # stay alive until the daemon stops us
"""


def test_run_persistent_ingests_and_stops_cleanly(allow_settings):
    daemon, store = _daemon(allow_settings, HAPPY_SCRIPT)
    stop = threading.Event()
    result: dict = {}

    def _run():
        result["stats"] = daemon.run_persistent(stop_event=stop)

    t = threading.Thread(target=_run)
    t.start()
    deadline = time.monotonic() + 10.0
    while not store.calls and time.monotonic() < deadline:
        time.sleep(0.05)
    stop.set()
    t.join(timeout=10.0)
    assert not t.is_alive(), "run_persistent did not stop"

    stats: CaptureStats = result["stats"]
    assert stats.ingested == 1
    assert stats.heartbeats >= 2
    assert store.calls[0]["app"] == "com.apple.mail"
    assert "hello from stream" in store.calls[0]["text"]


def test_run_persistent_stop_is_fast(allow_settings):
    # Emitter that heartbeats once then sleeps forever: a set stop event must
    # return within ~1s (the queue slice), not the 30s+ stall timeout.
    script = HAPPY_SCRIPT.replace("time.sleep(30)", "time.sleep(600)")
    daemon, _ = _daemon(allow_settings, script)
    stop = threading.Event()
    result: dict = {}

    def _run():
        result["stats"] = daemon.run_persistent(stop_event=stop)

    t = threading.Thread(target=_run)
    t.start()
    time.sleep(1.0)  # let it spawn and read the heartbeat
    started = time.monotonic()
    stop.set()
    t.join(timeout=8.0)
    elapsed = time.monotonic() - started
    assert not t.is_alive(), "stop was not honored"
    # Slice (1s) + shutdown escalation headroom; far below the stall timeout.
    assert elapsed < 8.0
    assert "stats" in result  # clean return, no HelperExitError


# ---------------------------------------------------------------------------
# run_persistent — failure classification
# ---------------------------------------------------------------------------


def test_run_persistent_helper_death_raises(allow_settings):
    script = r"""
import json, sys, time
sys.stdout.write(json.dumps({"type": "heartbeat", "ts": time.time(), "seq": 0}) + "\n")
sys.stdout.flush()
sys.exit(2)
"""
    daemon, _ = _daemon(allow_settings, script)
    with pytest.raises(HelperExitError, match="code 2"):
        daemon.run_persistent()


def test_run_persistent_unexpected_clean_exit_raises(allow_settings):
    # A stream-capable helper (it heartbeated) exiting 0 on its own is a
    # failure: healthy stream helpers only exit on our signal.
    script = r"""
import json, sys, time
sys.stdout.write(json.dumps({"type": "heartbeat", "ts": time.time(), "seq": 0}) + "\n")
sys.stdout.flush()
"""
    daemon, _ = _daemon(allow_settings, script)
    with pytest.raises(HelperExitError, match="unexpectedly"):
        daemon.run_persistent()


def test_run_persistent_stall_reaps_and_raises(allow_settings, monkeypatch):
    # Alive-but-silent helper: after the stall timeout the daemon reaps it and
    # raises. Shrink the timeout via the idle-tick setting floor override.
    monkeypatch.setattr(
        "openbird.capture.daemon._STREAM_READ_TIMEOUT_MIN", 2.0
    )
    script = r"""
import json, sys, time
sys.stdout.write(json.dumps({"type": "heartbeat", "ts": time.time(), "seq": 0}) + "\n")
sys.stdout.flush()
time.sleep(600)
"""
    daemon, _ = _daemon(allow_settings, script)
    started = time.monotonic()
    with pytest.raises(HelperExitError, match="stalled"):
        daemon.run_persistent()
    # 2s timeout + slices + SIGTERM escalation, never the full 600s sleep.
    assert time.monotonic() - started < 30.0


# ---------------------------------------------------------------------------
# Old-binary downgrade
# ---------------------------------------------------------------------------

OLD_BINARY_SCRIPT = r"""
import json, sys, time
# An old one-shot helper: ignores --stream, emits ONE typeless capture, exits 0.
sys.stdout.write(json.dumps({"app": "com.apple.mail", "window": "Inbox",
    "url": None, "text": "one-shot frame", "ts": time.time(),
    "incognito": False}) + "\n")
sys.stdout.flush()
"""


def test_run_persistent_downgrades_for_old_binary(allow_settings):
    daemon, store = _daemon(allow_settings, OLD_BINARY_SCRIPT)
    stats = daemon.run_persistent()  # clean return, no raise
    assert daemon._stream_supported is False
    assert daemon._capture_mode() == "oneshot"
    assert stats.ingested == 1  # the one-shot frame still counted
    assert store.calls


def test_capture_mode_env_force(allow_settings, monkeypatch):
    daemon, _ = _daemon(allow_settings, OLD_BINARY_SCRIPT)
    monkeypatch.setenv(CAPTURE_MODE_ENV, "oneshot")
    assert daemon._capture_mode() == "oneshot"
    monkeypatch.setenv(CAPTURE_MODE_ENV, "persistent")
    assert daemon._capture_mode() == "persistent"
    monkeypatch.delenv(CAPTURE_MODE_ENV)
    assert daemon._capture_mode() == "persistent"  # optimistic default


def test_run_forever_downgrade_continues_polling(allow_settings):
    # One supervised session: cycle 1 (persistent) downgrades on the old
    # binary, later cycles run the one-shot path — no breaker, no error.
    daemon, store = _daemon(allow_settings, OLD_BINARY_SCRIPT)
    stats = daemon.run_forever(poll_interval=0.05, max_cycles=3)
    assert daemon._stream_supported is False
    assert stats.received == 3  # one frame per cycle, all three ingest paths ran
    assert len(store.calls) >= 1  # first ingests; later ones may coalesce


def test_downgraded_polling_daemon_keeps_liveness_fresh(allow_settings, tmp_path):
    # Regression (Codex diff review): after the old-binary downgrade, poll
    # cycles must keep writing the sidecar — otherwise the last stream-mode
    # flush goes stale within 30s and health calls a healthy daemon "stale".
    daemon, _ = _daemon(allow_settings, OLD_BINARY_SCRIPT)
    daemon.run_forever(poll_interval=0.05, max_cycles=3)
    assert daemon._stream_supported is False

    payload = json.loads((tmp_path / LIVENESS_FILENAME).read_text())
    assert payload["mode"] == "oneshot"
    # The sidecar was just written by the final poll cycle: health sees ok.
    health = build_capture_health(settings=allow_settings)
    assert health["daemon"]["state"] == "ok"
    assert health["daemon"]["mode"] == "oneshot"


def test_stream_args_only_in_stream_spawn(allow_settings):
    daemon, _ = _daemon(allow_settings, "pass")
    poll_argv = daemon._resolve_helper()
    stream_argv = daemon._resolve_helper(stream=True)
    assert "--stream" not in poll_argv
    assert "--stream" in stream_argv
    for flag in ("--idle-tick", "--afk-threshold", "--ceiling", "--min-gap"):
        assert flag in stream_argv
        assert flag not in poll_argv


# ---------------------------------------------------------------------------
# Liveness sidecar + health block
# ---------------------------------------------------------------------------


def test_liveness_sidecar_written_and_metadata_only(allow_settings, tmp_path):
    daemon, store = _daemon(allow_settings, HAPPY_SCRIPT)
    stop = threading.Event()
    result: dict = {}

    def _run():
        result["stats"] = daemon.run_persistent(stop_event=stop)

    t = threading.Thread(target=_run)
    t.start()
    path = tmp_path / LIVENESS_FILENAME
    deadline = time.monotonic() + 10.0
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    stop.set()
    t.join(timeout=10.0)

    assert path.exists(), "liveness sidecar never written"
    payload = json.loads(path.read_text())
    assert payload["mode"] == "stream"
    assert isinstance(payload["updated_at"], float)
    assert payload["last_capture_ts"] is not None
    # Metadata only: no content-bearing keys, and no captured text anywhere.
    assert set(payload) == {
        "updated_at", "last_event_ts", "last_capture_ts", "mode", "afk",
        "heartbeat_seq", "meeting", "ocr_state",
    }
    assert payload["meeting"] is False  # a metadata BIT, off by default
    assert payload["ocr_state"] is None  # never reported in this run
    assert "hello from stream" not in path.read_text()
    # No stray temp file left behind (atomic rename).
    assert not (tmp_path / (LIVENESS_FILENAME + ".tmp")).exists()


def test_health_daemon_block_states(allow_settings, tmp_path):
    # Absent sidecar -> unknown (never ok).
    health = build_capture_health(settings=allow_settings, generated_at=1000.0)
    assert health["daemon"] == {"state": "unknown"}

    # Fresh sidecar -> ok.
    path = tmp_path / LIVENESS_FILENAME
    path.write_text(json.dumps({
        "updated_at": 995.0, "last_event_ts": 994.0, "last_capture_ts": 990.0,
        "mode": "stream", "afk": False, "heartbeat_seq": 7,
    }))
    health = build_capture_health(settings=allow_settings, generated_at=1000.0)
    assert health["daemon"]["state"] == "ok"
    assert health["daemon"]["mode"] == "stream"
    assert health["daemon"]["heartbeat_seq"] == 7

    # Old sidecar -> stale, never ok off a stale timestamp.
    path.write_text(json.dumps({"updated_at": 100.0}))
    health = build_capture_health(settings=allow_settings, generated_at=1000.0)
    assert health["daemon"]["state"] == "stale"

    # Malformed sidecar -> unknown.
    path.write_text("{not json")
    health = build_capture_health(settings=allow_settings, generated_at=1000.0)
    assert health["daemon"] == {"state": "unknown"}

    # NaN updated_at must not become "ok" (finite guard).
    path.write_text('{"updated_at": NaN}')
    health = build_capture_health(settings=allow_settings, generated_at=1000.0)
    assert health["daemon"] == {"state": "unknown"}


# ---------------------------------------------------------------------------
# app_changed boundary markers (Phase B)
# ---------------------------------------------------------------------------


def test_parse_app_changed_carries_bundle_only():
    from openbird.capture.daemon import parse_event

    e = parse_event(json.dumps({"type": "app_changed", "ts": 5.0, "app": "com.x.y"}))
    assert e is not None
    assert e["type"] == "app_changed"
    assert e["app"] == "com.x.y"
    # Never content-bearing fields.
    for forbidden in ("window", "url", "text"):
        assert forbidden not in e


def test_parse_system_accepts_mic_kinds_and_rejects_unknown():
    from openbird.capture.daemon import parse_event

    for kind in ("mic_started", "mic_stopped"):
        e = parse_event(json.dumps({"type": "system", "ts": 5.0, "kind": kind}))
        assert e is not None and e["type"] == "system"
        assert e["kind"] == kind
    # Closed vocabulary: an unknown kind still coerces to None (the old-daemon
    # back-compat contract — on_system(None, ts) is a no-op).
    e = parse_event(
        json.dumps({"type": "system", "ts": 5.0, "kind": "mic_exploded"})
    )
    assert e is not None and e["kind"] is None


def test_app_changed_counted_not_ingested(allow_settings):
    daemon, store = _daemon(allow_settings, "pass")
    stats = daemon.run_lines(
        [json.dumps({"type": "app_changed", "ts": 1.0, "app": "com.x.y"})]
    )
    assert stats.span_markers == 1
    assert stats.ingested == 0
    assert store.calls == []


# ---------------------------------------------------------------------------
# OCR fallback plumbing (Phase C2): system kinds -> liveness ocr_state
# ---------------------------------------------------------------------------


def test_parse_system_accepts_ocr_kinds():
    from openbird.capture.daemon import parse_event

    for kind in ("ocr_available", "ocr_unavailable"):
        e = parse_event(json.dumps({"type": "system", "ts": 5.0, "kind": kind}))
        assert e is not None and e["type"] == "system"
        assert e["kind"] == kind


def test_ocr_system_events_record_liveness_ocr_state(allow_settings, tmp_path):
    daemon, store = _daemon(allow_settings, "pass")

    stats = daemon.run_lines(
        [json.dumps({"type": "system", "ts": 1.0, "kind": "ocr_available"})]
    )
    assert stats.heartbeats == 1  # liveness credit, nothing ingested
    assert store.calls == []
    daemon._write_liveness(
        mode="stream", last_event_ts=1.0, last_capture_ts=None, heartbeat_seq=0
    )
    payload = json.loads((tmp_path / LIVENESS_FILENAME).read_text())
    assert payload["ocr_state"] == "available"

    # A revocation flip overwrites it.
    daemon.run_lines(
        [json.dumps({"type": "system", "ts": 2.0, "kind": "ocr_unavailable"})]
    )
    daemon._write_liveness(
        mode="stream", last_event_ts=2.0, last_capture_ts=None, heartbeat_seq=1
    )
    payload = json.loads((tmp_path / LIVENESS_FILENAME).read_text())
    assert payload["ocr_state"] == "unavailable"


def test_health_daemon_block_passes_through_ocr_state(allow_settings, tmp_path):
    path = tmp_path / LIVENESS_FILENAME
    path.write_text(json.dumps({
        "updated_at": 995.0, "mode": "stream", "ocr_state": "available",
    }))
    health = build_capture_health(settings=allow_settings, generated_at=1000.0)
    assert health["daemon"]["ocr_state"] == "available"

    # Closed pair: junk values are sanitized to None (never rendered raw).
    path.write_text(json.dumps({
        "updated_at": 995.0, "mode": "stream", "ocr_state": "definitely-on",
    }))
    health = build_capture_health(settings=allow_settings, generated_at=1000.0)
    assert health["daemon"]["ocr_state"] is None


# ---------------------------------------------------------------------------
# Spans end-to-end through the daemon (Phase B)
# ---------------------------------------------------------------------------


class SpanCapableStore(FakeStore):
    """FakeStore + the span API, so the real SpanTracker engages."""

    def __init__(self) -> None:
        super().__init__()
        self.spans: dict[str, dict] = {}
        self._n = 0

    def add_observation(self, text, *, span_id=None, **kw):
        obs = super().add_observation(text, **kw)
        self.calls[-1]["span_id"] = span_id
        return obs

    def open_span(self, **kw) -> str:
        self._n += 1
        sid = f"span{self._n}"
        self.spans[sid] = dict(kw)
        return sid

    def extend_span(self, span_id, end_ts):
        row = self.spans[span_id]
        row["end_ts"] = max(row["end_ts"], end_ts)

    def close_span(self, span_id, end_ts):
        row = self.spans[span_id]
        row["end_ts"] = max(row["start_ts"], end_ts)


def _span_daemon(settings):
    store = SpanCapableStore()
    daemon = CaptureDaemon(
        store, settings=settings,
        helper_cmd=("true",), require_signed_helper=False,
    )
    return daemon, store


def test_blocked_app_frames_yield_coarse_span_no_content(allow_settings):
    daemon, store = _span_daemon(allow_settings)
    # Metadata-only frames as the helper emits for non-allowlisted apps —
    # plus a hostile window title that must NOT reach the span.
    lines = [
        json.dumps({"app": "com.other.app", "window": "SECRET", "url": None,
                    "text": "", "ts": 100.0, "incognito": False}),
        json.dumps({"app": "com.other.app", "window": "SECRET", "url": None,
                    "text": "", "ts": 102.0, "incognito": False}),
    ]
    stats = daemon.run_lines(lines)
    assert stats.ingested == 0  # no content stored (empty text rejected)
    assert len(store.spans) == 1  # but the TIME is tracked
    row = next(iter(store.spans.values()))
    assert row["detail_tier"] == 0
    assert row["reason"] == "not_allowlisted"
    assert row["window"] is None and row["url_host"] is None
    assert "SECRET" not in json.dumps(row)


def test_observation_carries_its_frames_span_id(allow_settings):
    daemon, store = _span_daemon(allow_settings)
    daemon.run_lines([
        json.dumps({"app": "com.apple.mail", "window": "Inbox", "url": None,
                    "text": "hello world", "ts": 100.0, "incognito": False}),
    ])
    assert len(store.calls) == 1
    span_id = store.calls[0]["span_id"]
    assert span_id in store.spans
    assert store.spans[span_id]["detail_tier"] == 1


def test_allowlisted_empty_ax_yields_tier1_span_without_observation(allow_settings):
    daemon, store = _span_daemon(allow_settings)
    stats = daemon.run_lines([
        json.dumps({"app": "com.apple.mail", "window": "Inbox", "url": None,
                    "text": "", "ts": 100.0, "incognito": False}),
    ])
    assert stats.ingested == 0 and stats.rejected == 1
    assert len(store.spans) == 1
    assert next(iter(store.spans.values()))["detail_tier"] == 1


def test_mic_system_events_drive_the_meeting_flag_end_to_end(allow_settings):
    # mic_started -> Zoom frame -> mic_stopped, through the real parse/dispatch
    # path: the span carries meeting=1 and splits at the mic_stopped edge.
    daemon, store = _span_daemon(allow_settings)
    daemon.run_lines([
        json.dumps({"type": "system", "ts": 99.0, "kind": "mic_started"}),
        json.dumps({"app": "us.zoom.xos", "window": None, "url": None,
                    "text": "", "ts": 100.0, "incognito": False}),
        json.dumps({"type": "system", "ts": 105.0, "kind": "mic_stopped"}),
    ])
    rows = sorted(store.spans.values(), key=lambda r: r["start_ts"])
    assert [bool(r["meeting"]) for r in rows] == [True, False]
    assert rows[0]["end_ts"] == 105.0  # split at the exact edge
    assert rows[1]["start_ts"] == 105.0
