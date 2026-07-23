from __future__ import annotations

import io
import os
import threading
import time
from array import array

import pytest

from openbird.meetings.audio import AudioFrame, Track, encode_frame
from openbird.meetings.pipeline import MeetingPipeline, VADConfig
from openbird.meetings.record import (
    HELPER_MAX_SECONDS,
    MeetingRecorder,
    SupervisorNotArmed,
    arm_supervisor,
    recover_pending,
)
from openbird.meetings.transcribe import TranscriptSegment
from openbird.memory.store import MemoryStore


@pytest.fixture
def store(mem_settings, fake_provider):
    value = MemoryStore(db_path=":memory:", settings=mem_settings, provider=fake_provider)
    yield value
    value.close()


def test_pending_meeting_commit_is_idempotent_and_content_is_searchable(store):
    store.checkpoint_meeting(
        "meeting-1",
        started_ts=100.0,
        transcript="[0.00-1.00] others: release readiness confirmed",
        segments=[{"track": "system", "start_seconds": 0.0, "end_seconds": 1.0}],
        backend="parakeet",
    )

    first = store.commit_pending_meeting("meeting-1")
    second = store.commit_pending_meeting("meeting-1")

    assert first is not None
    assert second is not None
    assert first.id == second.id
    assert store.pending_meetings() == []
    row = store.conn.execute(
        "SELECT source, session_id FROM observations WHERE id = ?", (first.id,)
    ).fetchone()
    assert row == {"source": "meeting", "session_id": "meeting-1"}


def test_add_observation_embeddings_are_batched_at_32(
    mem_settings, fake_provider, monkeypatch
):
    calls: list[int] = []
    original = fake_provider.embed

    def recording_embed(texts):
        calls.append(len(texts))
        return original(texts)

    fake_provider.embed = recording_embed
    chunks = [((index, index + 1), f"unique meeting chunk {index}") for index in range(65)]
    monkeypatch.setattr("openbird.memory.store.ingest.chunk", lambda _text: chunks)
    value = MemoryStore(db_path=":memory:", settings=mem_settings, provider=fake_provider)
    try:
        value.add_observation("large meeting", source="meeting", session_id="batched")
    finally:
        value.close()
    assert calls == [32, 32, 1]


def test_pending_meetings_follow_purge_boundaries(store):
    store.checkpoint_meeting("old", started_ts=10.0, transcript="old transcript")
    store.checkpoint_meeting("new", started_ts=20.0, transcript="new transcript")

    store.delete(since_ts=20.0)
    assert [row["meeting_id"] for row in store.pending_meetings()] == ["old"]

    store.prune(older_than_ts=11.0)
    assert store.pending_meetings() == []


def test_recover_pending_reports_metadata_only(store):
    store.checkpoint_meeting("recover-me", started_ts=1.0, transcript="private words")
    result = recover_pending(store)
    assert result == {
        "pending_count": 1,
        "recovered_count": 1,
        "empty_count": 0,
        "failed_count": 0,
    }
    assert "private words" not in str(result)


def test_supervisor_fails_closed_on_wrong_token():
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"wrong\n")
        with pytest.raises(SupervisorNotArmed):
            arm_supervisor("expected", fd=read_fd, stop=threading.Event(), timeout=0.1)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_supervisor_arms_then_eof_requests_stop():
    read_fd, write_fd = os.pipe()
    stop = threading.Event()
    try:
        os.write(write_fd, b"expected\n")
        watcher = arm_supervisor("expected", fd=read_fd, stop=stop, timeout=0.1)
        os.close(write_fd)
        write_fd = -1
        watcher.join(timeout=1)
        assert stop.is_set()
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


class _FakeTranscriber:
    active_backend = "parakeet"

    def __init__(self):
        self.prepare_thread: int | None = None

    def prepare(self):
        self.prepare_thread = threading.get_ident()
        return "parakeet"

    def transcribe_segment(self, segment):
        assert self.prepare_thread == threading.get_ident()
        return TranscriptSegment(
            track=segment.track,
            start_ts=segment.start_ts,
            end_ts=segment.end_ts,
            text="decision recorded",
        )


class _FakeProcess:
    pid = 4242

    def __init__(self, payload: bytes):
        self.stdout = io.BytesIO(payload)
        self.terminated = False
        self.killed = False

    def poll(self):
        return 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


def test_recorder_drains_audio_and_persists_without_emitting_transcript(store, tmp_path):
    helper = tmp_path / "audio-helper"
    helper.write_text("test")
    helper.chmod(0o700)
    frame = AudioFrame(
        samples=array("f", [0.5] * 4_800),
        sample_rate=16_000,
        host_ts=10.0,
        track=Track.SYSTEM,
    )
    process = _FakeProcess(encode_frame(frame))
    calls = []
    events: list[dict] = []

    def popen(args, **kwargs):
        calls.append((args, kwargs))
        return process

    stop = threading.Event()

    def emit(event):
        events.append(event)
        if event["event"] == "recording_started":
            stop.set()

    result = MeetingRecorder(
        store,
        helper_path=helper,
        transcriber=_FakeTranscriber(),  # type: ignore[arg-type]
        pipeline=MeetingPipeline(VADConfig(min_speech=0.1, min_silence=0.1)),
        stop=stop,
        emit=emit,
        popen=popen,
        finalize_timeout=2,
    ).run()

    assert result.reason == "completed"
    assert result.observation_id is not None
    assert calls[0][0][-2:] == ["--max-seconds", str(HELPER_MAX_SECONDS)]
    assert any(event["event"] == "recording_started" for event in events)
    assert "decision recorded" not in str(events)
    stored = store.conn.execute(
        "SELECT b.text FROM observations o JOIN content_blobs b USING(content_hash) "
        "WHERE o.id = ?",
        (result.observation_id,),
    ).fetchone()["text"]
    assert "others: decision recorded" in stored


def test_recorder_reports_inference_failure_instead_of_no_speech(store, tmp_path):
    class FailingTranscriber(_FakeTranscriber):
        def transcribe_segment(self, segment):
            assert self.prepare_thread == threading.get_ident()
            raise RuntimeError("inference failed")

    helper = tmp_path / "audio-helper"
    helper.write_text("test")
    helper.chmod(0o700)
    frame = AudioFrame(
        samples=array("f", [0.5] * 4_800),
        sample_rate=16_000,
        host_ts=10.0,
        track=Track.SYSTEM,
    )
    stop = threading.Event()

    def emit(event):
        if event["event"] == "recording_started":
            stop.set()

    result = MeetingRecorder(
        store,
        helper_path=helper,
        transcriber=FailingTranscriber(),  # type: ignore[arg-type]
        pipeline=MeetingPipeline(VADConfig(min_speech=0.1, min_silence=0.1)),
        stop=stop,
        emit=emit,
        popen=lambda *args, **kwargs: _FakeProcess(encode_frame(frame)),
        finalize_timeout=2,
    ).run()

    assert result.reason == "transcription_failed"
    assert result.failed_windows == 1
    assert result.partial is True
    assert result.observation_id is None


def test_recorder_stop_before_helper_launch_removes_empty_checkpoint(store, tmp_path):
    helper = tmp_path / "audio-helper"
    helper.write_text("test")
    helper.chmod(0o700)
    stop = threading.Event()
    stop.set()
    calls = []

    result = MeetingRecorder(
        store,
        helper_path=helper,
        transcriber=_FakeTranscriber(),  # type: ignore[arg-type]
        stop=stop,
        emit=lambda _event: None,
        popen=lambda *args, **kwargs: calls.append((args, kwargs)),
    ).run()

    assert result.reason == "stop_requested"
    assert calls == []
    assert store.pending_meetings() == []


def test_recorder_helper_launch_failure_removes_empty_checkpoint(store, tmp_path):
    helper = tmp_path / "audio-helper"
    helper.write_text("test")
    helper.chmod(0o700)

    def fail_launch(*_args, **_kwargs):
        raise OSError("exec failed")

    result = MeetingRecorder(
        store,
        helper_path=helper,
        transcriber=_FakeTranscriber(),  # type: ignore[arg-type]
        emit=lambda _event: None,
        popen=fail_launch,
    ).run()

    assert result.reason == "helper_start_failed"
    assert result.partial is True
    assert store.pending_meetings() == []


def test_strict_truncation_saves_partial_result(store, tmp_path):
    helper = tmp_path / "audio-helper"
    helper.write_text("test")
    helper.chmod(0o700)
    frame = AudioFrame(
        samples=array("f", [0.5] * 4_800),
        sample_rate=16_000,
        host_ts=10.0,
        track=Track.MIC,
    )
    process = _FakeProcess(encode_frame(frame) + b"\x01\x02")
    result = MeetingRecorder(
        store,
        helper_path=helper,
        transcriber=_FakeTranscriber(),  # type: ignore[arg-type]
        pipeline=MeetingPipeline(VADConfig(min_speech=0.1, min_silence=0.1)),
        stop=threading.Event(),
        emit=lambda _event: None,
        popen=lambda *args, **kwargs: process,
        finalize_timeout=2,
    ).run()
    assert result.partial is True
    assert result.reason == "truncated_header"
    assert result.observation_id is not None
