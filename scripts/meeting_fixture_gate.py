#!/usr/bin/env python3
"""Metadata-only real-backend gate for manual meeting transcription.

The caller supplies a locally generated mono 16-bit WAV. The script never prints
transcript text; it reports only durations, counts, text byte lengths, and RTF.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import wave
from array import array
from pathlib import Path

from openbird.meetings.audio import AudioFrame, Track
from openbird.meetings.pipeline import MeetingPipeline
from openbird.meetings.transcribe import Transcriber, configure_model_cache

FRAME_SECONDS = 0.1


def _frames(samples: array, sample_rate: int) -> list[AudioFrame]:
    width = max(1, int(sample_rate * FRAME_SECONDS))
    return [
        AudioFrame(
            samples=samples[offset : offset + width],
            sample_rate=sample_rate,
            host_ts=offset / sample_rate,
            track=Track.SYSTEM,
            seq=offset // width,
        )
        for offset in range(0, len(samples), width)
    ]


def _read_wav(path: Path) -> tuple[array, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("fixture must be mono 16-bit PCM")
        sample_rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    ints = array("h")
    ints.frombytes(raw)
    return array("f", (value / 32768.0 for value in ints)), sample_rate


def _tone(sample_rate: int, seconds: float, frequency: float, amplitude: float) -> array:
    return array(
        "f",
        (
            amplitude * math.sin(2 * math.pi * frequency * index / sample_rate)
            for index in range(int(sample_rate * seconds))
        ),
    )


def _transcribe(transcriber: Transcriber, samples: array, sample_rate: int) -> dict:
    segments = MeetingPipeline().process(_frames(samples, sample_rate))
    audio_seconds = sum(segment.duration for segment in segments)
    started = time.perf_counter()
    transcript = transcriber.transcribe_all(segments) if segments else []
    elapsed = time.perf_counter() - started
    return {
        "vad_segments": len(segments),
        "audio_seconds": round(audio_seconds, 3),
        "elapsed_seconds": round(elapsed, 3),
        "rtf": round(elapsed / audio_seconds, 4) if audio_seconds else 0.0,
        "transcript_segments": len(transcript),
        "transcript_bytes": sum(len(item.text.encode("utf-8")) for item in transcript),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--backend", default="parakeet")
    args = parser.parse_args()

    configure_model_cache(offline=True)
    transcriber = Transcriber(backend=args.backend)
    transcriber.prepare()
    speech, sample_rate = _read_wav(args.wav)
    silence = array("f", [0.0]) * (sample_rate * 5)
    music_tone = _tone(sample_rate, 5.0, 440.0, 0.2)
    notification = _tone(sample_rate, 0.25, 880.0, 0.5)
    notification.extend(array("f", [0.0]) * sample_rate)

    result = {
        "backend": transcriber.active_backend,
        "speech": _transcribe(transcriber, speech, sample_rate),
        "silence": _transcribe(transcriber, silence, sample_rate),
        "music_tone": _transcribe(transcriber, music_tone, sample_rate),
        "notification_tone": _transcribe(transcriber, notification, sample_rate),
    }
    print(json.dumps(result, sort_keys=True))
    speech_result = result["speech"]
    if speech_result["transcript_bytes"] <= 0:
        return 1
    if speech_result["rtf"] > 0.20:
        return 1
    if result["silence"]["vad_segments"] != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
