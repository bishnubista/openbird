"""OpenBird meetings subsystem: audio capture, VAD/stitch pipeline, transcription.

The signed Swift ``audio-helper`` captures ScreenCaptureKit system audio and mic
as separate synchronized tracks. The Python side keeps an injectable audio source
so the controller remains unit-testable without activating real audio.

Modules:
  * :mod:`openbird.meetings.audio` — audio-source abstraction + clock-sync notes.
  * :mod:`openbird.meetings.pipeline` — VAD / windowing / transcript stitching.
  * :mod:`openbird.meetings.transcribe` — faster-whisper wrapper + LLM summary.
  * :mod:`openbird.meetings.record` — supervised recording + encrypted checkpoint.
"""

from __future__ import annotations

from openbird.meetings.audio import (
    AudioFrame,
    AudioSource,
    CallbackAudioSource,
    ClockSync,
    InMemoryAudioSource,
    Track,
)
from openbird.meetings.pipeline import (
    MeetingPipeline,
    SpeechSegment,
    TranscriptPiece,
    VADConfig,
    frame_energy,
    stitch_segments,
)
from openbird.meetings.transcribe import (
    ACTION_ITEMS_SCHEMA,
    MeetingsAudioTooLong,
    MeetingsExtraNotInstalled,
    Transcriber,
    TranscriptSegment,
    format_transcript,
    stitch_transcript,
    summarize_transcript,
    whisper_available,
)

__all__ = [
    "AudioFrame",
    "AudioSource",
    "CallbackAudioSource",
    "ClockSync",
    "InMemoryAudioSource",
    "Track",
    "MeetingPipeline",
    "SpeechSegment",
    "TranscriptPiece",
    "VADConfig",
    "frame_energy",
    "stitch_segments",
    "ACTION_ITEMS_SCHEMA",
    "MeetingsAudioTooLong",
    "MeetingsExtraNotInstalled",
    "Transcriber",
    "TranscriptSegment",
    "format_transcript",
    "stitch_transcript",
    "summarize_transcript",
    "whisper_available",
]
