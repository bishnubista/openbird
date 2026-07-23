"""Manual meeting recording controller with metadata-only process IPC.

Raw PCM crosses only the signed helper's private stdout pipe and in-process
queues. Transcript text is checkpointed in the SQLCipher-gated memory store and
is never emitted through JSONL, logs, argv, or environment variables.
"""

from __future__ import annotations

import json
import os
import queue
import select
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Callable

from openbird.meetings.audio import AudioIPCError, BinaryFrameAudioSource
from openbird.meetings.pipeline import MeetingPipeline
from openbird.meetings.transcribe import (
    MeetingsExtraNotInstalled,
    TranscriptSegment,
    Transcriber,
    configure_model_cache,
    stitch_transcript,
)
from openbird.memory.store import MemoryStore

SUPERVISOR_TOKEN_ENV = "OPENBIRD_SUPERVISOR_TOKEN"
AUDIO_HELPER_ENV = "OPENBIRD_AUDIO_HELPER"
HELPER_MAX_SECONDS = 14_400
SUPERVISOR_ARM_TIMEOUT = 5.0
FINALIZE_TIMEOUT = 180.0
WINDOW_QUEUE_CAPACITY = 8
MAX_TRANSCRIPT_BYTES = 8 * 1024 * 1024


class SupervisorNotArmed(RuntimeError):
    """The app-owned death pipe failed closed before microphone activation."""


@dataclass(frozen=True)
class MeetingResult:
    meeting_id: str
    duration_seconds: float
    segment_count: int
    window_count: int
    system_frame_count: int
    mic_frame_count: int
    observation_id: str | None
    backend: str | None
    clock_event_count: int
    dropped_windows: int
    failed_windows: int
    truncated_bytes: int
    partial: bool
    reason: str

    def metadata(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _WindowTask:
    index: int
    segment: object


@dataclass(frozen=True)
class _WindowResult:
    index: int
    segment: TranscriptSegment | None
    failed: bool = False


def emit_jsonl(payload: dict) -> None:
    """Write one metadata-only controller event."""
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def arm_supervisor(
    token: str,
    *,
    fd: int = 0,
    stop: threading.Event,
    timeout: float = SUPERVISOR_ARM_TIMEOUT,
) -> threading.Thread:
    """Fail closed unless the exact token arrives, then watch for app EOF."""
    if not token or len(token.encode("utf-8")) > 512:
        raise SupervisorNotArmed("supervisor_not_armed")
    deadline = time.monotonic() + max(0.0, timeout)
    line = bytearray()
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SupervisorNotArmed("supervisor_not_armed")
            readable, _, _ = select.select([fd], [], [], remaining)
            if not readable:
                raise SupervisorNotArmed("supervisor_not_armed")
            chunk = os.read(fd, 1)
            if not chunk:
                raise SupervisorNotArmed("supervisor_not_armed")
            if chunk == b"\n":
                break
            line.extend(chunk)
            if len(line) > 512:
                raise SupervisorNotArmed("supervisor_not_armed")
    except (OSError, ValueError) as exc:
        raise SupervisorNotArmed("supervisor_not_armed") from exc
    if bytes(line) != token.encode("utf-8"):
        raise SupervisorNotArmed("supervisor_not_armed")

    def _watch() -> None:
        try:
            while not stop.is_set():
                readable, _, _ = select.select([fd], [], [], 0.25)
                if not readable:
                    continue
                if not os.read(fd, 1):
                    stop.set()
                    return
        except (OSError, ValueError):
            # Once armed, a broken liveness channel is equivalent to app death.
            stop.set()

    watcher = threading.Thread(target=_watch, name="meeting-supervisor", daemon=True)
    watcher.start()
    return watcher


def resolve_audio_helper(override: str | None = None) -> Path:
    """Resolve only the app-provided signed helper (or hidden test override)."""
    raw = (override or os.environ.get(AUDIO_HELPER_ENV) or "").strip()
    if not raw:
        raise FileNotFoundError("audio_helper_missing")
    path = Path(raw).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError("audio_helper_missing")
    return path


def _format_transcript(
    segments: list[TranscriptSegment], origin_host_ts: float
) -> tuple[str, list[dict], int]:
    """Render literal relative timestamps while enforcing the UTF-8 cap."""
    lines: list[str] = []
    metadata: list[dict] = []
    used = 0
    truncated = 0
    overflowed = False
    for segment in stitch_transcript(segments):
        text = " ".join(segment.text.split()).strip()
        if not text:
            continue
        start = max(0.0, segment.start_ts - origin_host_ts)
        end = max(start, segment.end_ts - origin_host_ts)
        line = f"[{start:.2f}-{end:.2f}] {segment.speaker}: {text}"
        encoded = (("\n" if lines else "") + line).encode("utf-8")
        if overflowed or used + len(encoded) > MAX_TRANSCRIPT_BYTES:
            overflowed = True
            truncated += len(encoded)
            continue
        lines.append(line)
        used += len(encoded)
        metadata.append(
            {
                "track": segment.track.value,
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "text_bytes": len(text.encode("utf-8")),
            }
        )
    return "\n".join(lines), metadata, truncated


class MeetingRecorder:
    """Own one helper, streaming pipeline, ASR worker, and durable checkpoint."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        helper_path: Path,
        transcriber: Transcriber | None = None,
        pipeline: MeetingPipeline | None = None,
        stop: threading.Event | None = None,
        emit: Callable[[dict], None] = emit_jsonl,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        finalize_timeout: float = FINALIZE_TIMEOUT,
    ) -> None:
        self.store = store
        self.helper_path = helper_path
        self.transcriber = transcriber or Transcriber()
        self.pipeline = pipeline or MeetingPipeline()
        self.stop = stop or threading.Event()
        self.emit = emit
        self.popen = popen
        self.finalize_timeout = max(0.0, finalize_timeout)

    def run(self) -> MeetingResult:
        meeting_id = str(uuid.uuid4())
        attempt_started = time.time()
        self.emit({"event": "preparing", "meeting_id": meeting_id})
        configure_model_cache(offline=True)
        # MLX compute streams are thread-affine. Keep model preparation and every
        # subsequent inference call on this one persistent daemon worker rather
        # than loading on the recorder thread and generating on another thread.
        affinity_jobs: queue.Queue[Callable[[], None] | None] = queue.Queue()

        def _run_affinity_jobs() -> None:
            while True:
                job = affinity_jobs.get()
                if job is None:
                    return
                job()

        affinity_thread = threading.Thread(
            target=_run_affinity_jobs, name="meeting-asr", daemon=True
        )
        affinity_thread.start()
        prepare_done = threading.Event()
        prepare_result: list[str | BaseException] = []

        def _prepare() -> None:
            try:
                prepare_result.append(self.transcriber.prepare())
            except BaseException as exc:  # re-raised on the recorder thread below
                prepare_result.append(exc)
            finally:
                prepare_done.set()

        affinity_jobs.put(_prepare)
        prepare_done.wait()
        prepared = prepare_result[0]
        if isinstance(prepared, (MeetingsExtraNotInstalled, RuntimeError)):
            affinity_jobs.put(None)
            return self._terminal(
                meeting_id,
                attempt_started,
                reason="model_not_prepared",
                backend=None,
            )
        if isinstance(prepared, BaseException):
            affinity_jobs.put(None)
            raise prepared
        backend = prepared

        started_wall = time.time()
        self.store.checkpoint_meeting(meeting_id, started_ts=started_wall, backend=backend)
        if self.stop.is_set():
            affinity_jobs.put(None)
            self.store.commit_pending_meeting(meeting_id)
            return self._terminal(
                meeting_id,
                started_wall,
                reason="stop_requested",
                backend=backend,
            )
        try:
            process = self.popen(
                [str(self.helper_path), "--max-seconds", str(HELPER_MAX_SECONDS)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except OSError:
            affinity_jobs.put(None)
            self.store.commit_pending_meeting(meeting_id)
            return self._terminal(
                meeting_id,
                started_wall,
                reason="helper_start_failed",
                backend=backend,
                partial=True,
            )
        reader = process.stdout
        if reader is None:
            affinity_jobs.put(None)
            self.stop.set()
            if process.poll() is None:
                process.terminate()
            self.store.commit_pending_meeting(meeting_id)
            return self._terminal(
                meeting_id,
                started_wall,
                reason="helper_start_failed",
                backend=backend,
                partial=True,
            )

        self.emit(
            {
                "event": "recording_started",
                "meeting_id": meeting_id,
                "backend": backend,
                "helper_pid": int(process.pid),
                "max_seconds": HELPER_MAX_SECONDS,
            }
        )

        tasks: queue.Queue[object] = queue.Queue(maxsize=WINDOW_QUEUE_CAPACITY)
        results: queue.Queue[_WindowResult] = queue.Queue(maxsize=WINDOW_QUEUE_CAPACITY)
        sentinel = object()
        reader_done = threading.Event()
        worker_done = threading.Event()
        state_lock = threading.Lock()
        origin_host_ts: list[float | None] = [None]
        counts = {"windows": 0, "dropped": 0, "failed": 0}
        frame_counts = {"system": 0, "mic": 0}
        reader_reason: list[str | None] = [None]

        def _enqueue(segment: object) -> None:
            with state_lock:
                index = counts["windows"]
                counts["windows"] += 1
            try:
                tasks.put_nowait(_WindowTask(index=index, segment=segment))
            except queue.Full:
                with state_lock:
                    counts["dropped"] += 1

        def _read_audio() -> None:
            try:
                source = BinaryFrameAudioSource(reader, strict=True)
                for frame in source:
                    frame_counts[frame.track.value] += 1
                    if origin_host_ts[0] is None:
                        origin_host_ts[0] = frame.host_ts
                    for segment in self.pipeline.push(frame):
                        _enqueue(segment)
                for segment in self.pipeline.flush():
                    _enqueue(segment)
            except AudioIPCError as exc:
                reader_reason[0] = exc.reason
                for segment in self.pipeline.flush():
                    _enqueue(segment)
            finally:
                # At EOF inference may still be draining; the sentinel is allowed
                # to wait because the audio pipe has already been fully consumed.
                tasks.put(sentinel)
                reader_done.set()

        def _transcribe() -> None:
            try:
                while True:
                    task = tasks.get()
                    if task is sentinel:
                        return
                    assert isinstance(task, _WindowTask)
                    try:
                        segment = self.transcriber.transcribe_segment(task.segment)  # type: ignore[arg-type]
                        results.put(_WindowResult(task.index, segment))
                    except Exception:  # noqa: BLE001 - emit only typed metadata
                        results.put(_WindowResult(task.index, None, failed=True))
            finally:
                worker_done.set()

        read_thread = threading.Thread(target=_read_audio, name="meeting-audio-reader", daemon=True)
        read_thread.start()
        affinity_jobs.put(_transcribe)

        def _stop_helper() -> None:
            self.stop.wait()
            if process.poll() is None:
                process.terminate()

        threading.Thread(target=_stop_helper, name="meeting-helper-stop", daemon=True).start()

        pieces: dict[int, TranscriptSegment] = {}
        truncated_bytes = 0
        persistence_failed = False
        finalizing = False
        deadline: float | None = None

        while True:
            try:
                item = results.get(timeout=0.1)
            except queue.Empty:
                item = None
            if item is not None:
                if item.failed or item.segment is None:
                    with state_lock:
                        counts["failed"] += 1
                else:
                    pieces[item.index] = item.segment
                transcript, segment_meta, extra_truncated = _format_transcript(
                    [pieces[index] for index in sorted(pieces)],
                    origin_host_ts[0] or 0.0,
                )
                truncated_bytes = max(truncated_bytes, extra_truncated)
                try:
                    self.store.checkpoint_meeting(
                        meeting_id,
                        started_ts=started_wall,
                        transcript=transcript,
                        segments=segment_meta,
                        backend=self.transcriber.active_backend or backend,
                        dropped_windows=counts["dropped"],
                        failed_windows=counts["failed"],
                        truncated_bytes=truncated_bytes,
                    )
                except Exception:  # noqa: BLE001 - content-free terminal reason
                    persistence_failed = True
                    self.stop.set()

            if reader_done.is_set() and not finalizing:
                finalizing = True
                deadline = time.monotonic() + self.finalize_timeout
                if not self.stop.is_set():
                    reader_reason[0] = reader_reason[0] or "helper_ended_early"
                self.emit(
                    {
                        "event": "finalizing",
                        "meeting_id": meeting_id,
                        "completed_windows": len(pieces),
                        "remaining_windows": max(
                            0,
                            counts["windows"]
                            - len(pieces)
                            - counts["dropped"]
                            - counts["failed"],
                        ),
                        "dropped_windows": counts["dropped"],
                        "failed_windows": counts["failed"],
                    }
                )

            if finalizing and worker_done.is_set() and results.empty():
                break
            if finalizing and deadline is not None and time.monotonic() >= deadline:
                with state_lock:
                    outstanding = max(
                        0,
                        counts["windows"]
                        - len(pieces)
                        - counts["dropped"]
                        - counts["failed"],
                    )
                    counts["dropped"] += outstanding
                reader_reason[0] = "finalization_timeout"
                break

        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

        ended_wall = time.time()
        transcript, segment_meta, extra_truncated = _format_transcript(
            [pieces[index] for index in sorted(pieces)], origin_host_ts[0] or 0.0
        )
        truncated_bytes = max(truncated_bytes, extra_truncated)
        partial_reason = reader_reason[0]
        if counts["failed"]:
            partial_reason = partial_reason or "transcription_failed"
        if counts["dropped"]:
            partial_reason = partial_reason or "queue_overflow"
        if truncated_bytes:
            partial_reason = partial_reason or "transcript_truncated"
        try:
            self.store.checkpoint_meeting(
                meeting_id,
                started_ts=started_wall,
                ended_ts=ended_wall,
                transcript=transcript,
                segments=segment_meta,
                backend=self.transcriber.active_backend or backend,
                partial_reason=partial_reason,
                dropped_windows=counts["dropped"],
                failed_windows=counts["failed"],
                truncated_bytes=truncated_bytes,
            )
            observation = self.store.commit_pending_meeting(meeting_id)
        except Exception:  # noqa: BLE001 - checkpoint remains encrypted in store
            observation = None
            persistence_failed = True

        affinity_jobs.put(None)
        if persistence_failed:
            reason = "persistence_pending"
        elif not transcript:
            reason = partial_reason or "no_speech"
        else:
            reason = partial_reason or "completed"
        result = MeetingResult(
            meeting_id=meeting_id,
            duration_seconds=max(0.0, ended_wall - started_wall),
            segment_count=len(segment_meta),
            window_count=counts["windows"],
            system_frame_count=frame_counts["system"],
            mic_frame_count=frame_counts["mic"],
            observation_id=getattr(observation, "id", None),
            backend=self.transcriber.active_backend or backend,
            clock_event_count=len(self.pipeline.clock_events),
            dropped_windows=counts["dropped"],
            failed_windows=counts["failed"],
            truncated_bytes=truncated_bytes,
            partial=reason not in {"completed", "no_speech"},
            reason=reason,
        )
        self.emit({"event": "result", **result.metadata()})
        return result

    def _terminal(
        self,
        meeting_id: str,
        started_wall: float,
        *,
        reason: str,
        backend: str | None,
        partial: bool = False,
    ) -> MeetingResult:
        result = MeetingResult(
            meeting_id=meeting_id,
            duration_seconds=max(0.0, time.time() - started_wall),
            segment_count=0,
            window_count=0,
            system_frame_count=0,
            mic_frame_count=0,
            observation_id=None,
            backend=backend,
            clock_event_count=0,
            dropped_windows=0,
            failed_windows=0,
            truncated_bytes=0,
            partial=partial,
            reason=reason,
        )
        self.emit({"event": "result", **result.metadata()})
        return result


def install_signal_stop(stop: threading.Event) -> None:
    """Translate SIGINT/SIGTERM into the same owned-helper stop path."""

    def _handler(_signum, _frame) -> None:  # type: ignore[no-untyped-def]
        stop.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def recover_pending(store: MemoryStore, meeting_id: str | None = None) -> dict:
    """Retry encrypted pending rows idempotently, returning metadata counts only."""
    rows = store.pending_meetings(meeting_id=meeting_id)
    recovered = 0
    failed = 0
    empty = 0
    for row in rows:
        try:
            observation = store.commit_pending_meeting(str(row["meeting_id"]))
            if observation is None:
                empty += 1
            else:
                recovered += 1
        except Exception:  # noqa: BLE001 - no content in recovery response
            failed += 1
    return {
        "pending_count": len(rows),
        "recovered_count": recovered,
        "empty_count": empty,
        "failed_count": failed,
    }


__all__ = [
    "AUDIO_HELPER_ENV",
    "FINALIZE_TIMEOUT",
    "HELPER_MAX_SECONDS",
    "MeetingRecorder",
    "MeetingResult",
    "SupervisorNotArmed",
    "arm_supervisor",
    "emit_jsonl",
    "install_signal_stop",
    "recover_pending",
    "resolve_audio_helper",
]
