"""OpenBird meetings subsystem: audio capture, VAD/stitch pipeline, transcription.

The Swift ``audio-helper`` (ScreenCaptureKit system audio + mic kept as separate
synchronized tracks) is built later; the Python side here is designed around an
*injectable* audio source so it can be unit-tested without real audio or a
faster-whisper install.

Modules:
  * :mod:`openbird.meetings.audio` — audio-source abstraction + clock-sync notes.
  * :mod:`openbird.meetings.pipeline` — VAD / windowing / transcript stitching.
  * :mod:`openbird.meetings.transcribe` — faster-whisper wrapper + LLM summary.
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
