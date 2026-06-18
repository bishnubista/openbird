"""VAD / windowing / transcript-stitching pipeline over PCM frames.

This is a **pure-Python stub** that operates entirely on provided
:class:`~openbird.meetings.audio.AudioFrame` objects. It does NOT require real
audio, a microphone, or any native VAD library — the voice-activity decision is a
simple, deterministic energy threshold so the windowing/stitching logic can be
unit-tested with canned frames. A production build can swap in a real VAD
(WebRTC/Silero) behind the same :class:`MeetingPipeline.is_speech` seam.

Responsibilities:
  * **VAD:** classify each frame as speech/silence (energy-based here).
  * **Windowing:** group contiguous speech frames into :class:`SpeechSegment`
    windows with a max duration and a small overlap, so a downstream
    sliding-window transcriber (faster-whisper) gets bounded, overlapping audio.
  * **Stitching:** merge overlapping/adjacent transcript segments back into a
    de-duplicated transcript, correcting timestamps and preserving per-track
    ("me vs others") attribution.

All timing is on the shared ``host_ts`` clock carried by each frame, so mic and
system tracks stay aligned (see :mod:`openbird.meetings.audio`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from openbird.meetings.audio import (
    AudioFrame,
    ClockEvent,
    ClockEventKind,
    ClockSync,
    Track,
)


@dataclass(frozen=True)
class VADConfig:
    """Tunables for voice-activity detection and windowing.

    Attributes:
        energy_threshold: RMS amplitude above which a frame is "speech".
        max_window: Max duration (s) of a speech window before it is flushed.
        overlap: Trailing audio (s) of one window re-included at the start of the
            next, so the sliding-window transcriber doesn't clip word boundaries.
        min_silence: Silence (s) that must elapse to close an open speech window.
        min_speech: Minimum speech duration (s) for a window to be emitted
            (filters out blips/clicks).
    """

    energy_threshold: float = 0.01
    max_window: float = 25.0
    overlap: float = 1.0
    min_silence: float = 0.6
    min_speech: float = 0.2


@dataclass
class SpeechSegment:
    """A contiguous speech window on a single track, in host-clock seconds."""

    track: Track
    start_ts: float
    end_ts: float
    frames: list[AudioFrame] = field(default_factory=list, repr=False)

    def __repr__(self) -> str:
        """Metadata-only repr — never dumps the captured PCM frames [R4/R5]."""
        return (
            f"SpeechSegment(track={self.track.value!r}, start_ts={self.start_ts}, "
            f"end_ts={self.end_ts}, n_frames={len(self.frames)})"
        )

    @property
    def duration(self) -> float:
        return max(0.0, self.end_ts - self.start_ts)


@dataclass
class TranscriptPiece:
    """A transcribed span tied back to its track and host-clock timestamps.

    Mirrors :class:`~openbird.meetings.transcribe.TranscriptSegment` but is
    defined here too so stitching has no import cycle with the transcriber.
    """

    track: Track
    start_ts: float
    end_ts: float
    text: str = field(repr=False)

    def __repr__(self) -> str:
        """Metadata-only repr — never dumps the transcript text [R4/R5]."""
        return (
            f"TranscriptPiece(track={self.track.value!r}, start_ts={self.start_ts}, "
            f"end_ts={self.end_ts}, text_len={len(self.text)})"
        )


def frame_energy(frame: AudioFrame) -> float:
    """Return the RMS amplitude of a frame's samples (0.0 for empty)."""
    n = len(frame.samples)
    if n == 0:
        return 0.0
    return math.sqrt(sum(s * s for s in frame.samples) / n)


class MeetingPipeline:
    """Group PCM frames into per-track speech windows using energy VAD.

    Frames may arrive interleaved across tracks; each track is windowed
    independently so the two synchronized tracks never bleed into one window.
    """

    def __init__(self, config: VADConfig | None = None) -> None:
        self.config = config or VADConfig()
        self.clock = ClockSync()
        # Per-track open window state.
        self._open: dict[Track, SpeechSegment] = {}
        self._silence_since: dict[Track, float | None] = {}
        # Clock-sync anomalies observed so far (metadata only — no audio/text).
        self.clock_events: list[ClockEvent] = []

    # -- VAD seam ------------------------------------------------------------

    def is_speech(self, frame: AudioFrame) -> bool:
        """Classify a frame as speech. Override to plug in a real VAD."""
        return frame_energy(frame) >= self.config.energy_threshold

    # -- windowing -----------------------------------------------------------

    def process(self, frames: Iterable[AudioFrame]) -> list[SpeechSegment]:
        """Window an entire (finite) frame stream into speech segments.

        Convenience wrapper around :meth:`push`/:meth:`flush` for batch/test use.
        """
        out: list[SpeechSegment] = []
        for frame in frames:
            out.extend(self.push(frame))
        out.extend(self.flush())
        return out

    def push(self, frame: AudioFrame) -> list[SpeechSegment]:
        """Feed one frame; return any windows that closed as a result.

        Streaming-friendly: call repeatedly as frames arrive, then :meth:`flush`
        at end-of-stream to emit the final open windows.
        """
        self.clock.observe(frame)
        track = frame.track
        emitted: list[SpeechSegment] = []

        # React to clock-sync anomalies before VAD/windowing this frame.
        # A device switch (abrupt host-clock jump) or large cross/intra gap means
        # the current open window is no longer time-continuous, so we realign by
        # closing it (emitting if valid) and starting fresh from this frame —
        # rather than stitching pre- and post-jump audio into one bogus window.
        new_events = self.clock.drain_events()
        if new_events:
            self.clock_events.extend(new_events)
            realign_tracks = {
                ev.track
                for ev in new_events
                if ev.kind
                in (ClockEventKind.DEVICE_SWITCH, ClockEventKind.INTRA_GAP)
            }
            for ev_track in realign_tracks:
                # Only realign the track that jumped; its open window predates the
                # discontinuity. carry_overlap=False so no stale audio leaks in.
                if self._open.get(ev_track) is not None:
                    closed = self._close(ev_track, carry_overlap=False)
                    if closed is not None:
                        emitted.append(closed)
                else:
                    # Clear any stale silence bookkeeping for a now-realigned track.
                    self._silence_since[ev_track] = None

        speech = self.is_speech(frame)
        open_seg = self._open.get(track)

        if speech:
            self._silence_since[track] = None
            if open_seg is None:
                open_seg = SpeechSegment(
                    track=track, start_ts=frame.host_ts, end_ts=frame.end_ts
                )
                self._open[track] = open_seg
            open_seg.frames.append(frame)
            open_seg.end_ts = frame.end_ts
            # Flush if the window grew past max_window.
            if open_seg.duration >= self.config.max_window:
                emitted.append(self._close(track, carry_overlap=True))
        else:
            if open_seg is not None:
                started = self._silence_since.get(track)
                if started is None:
                    self._silence_since[track] = frame.host_ts
                    started = frame.host_ts
                if (frame.end_ts - started) >= self.config.min_silence:
                    closed = self._close(track, carry_overlap=False)
                    if closed is not None:
                        emitted.append(closed)
        return emitted

    def flush(self) -> list[SpeechSegment]:
        """Close all open windows at end-of-stream."""
        emitted: list[SpeechSegment] = []
        for track in list(self._open.keys()):
            closed = self._close(track, carry_overlap=False)
            if closed is not None:
                emitted.append(closed)
        return emitted

    def _close(self, track: Track, *, carry_overlap: bool) -> SpeechSegment | None:
        """Finalize the open window for ``track``; optionally seed an overlap."""
        seg = self._open.pop(track, None)
        self._silence_since[track] = None
        if seg is None:
            return None
        if seg.duration < self.config.min_speech:
            return None
        if carry_overlap and self.config.overlap > 0:
            # Re-seed a fresh window with the tail frames so the next window
            # overlaps this one (helps the sliding-window transcriber).
            cutoff = seg.end_ts - self.config.overlap
            tail = [f for f in seg.frames if f.end_ts > cutoff]
            if tail:
                carry = SpeechSegment(
                    track=track,
                    start_ts=tail[0].host_ts,
                    end_ts=tail[-1].end_ts,
                    frames=list(tail),
                )
                self._open[track] = carry
        return seg


def _overlaps(a: TranscriptPiece, b: TranscriptPiece, tol: float) -> bool:
    """Whether two same-track pieces overlap (within ``tol`` seconds)."""
    return a.track == b.track and b.start_ts <= a.end_ts + tol


def _dedup_join(left: str, right: str) -> str:
    """Join two transcript texts, dropping a repeated overlap suffix/prefix.

    The sliding window means ``right`` often re-states the tail of ``left``. We
    find the longest suffix of ``left`` that is a prefix of ``right`` (word-wise)
    and drop it from ``right`` before concatenating.
    """
    left = left.strip()
    right = right.strip()
    if not left:
        return right
    if not right:
        return left
    lw = left.split()
    rw = right.split()
    max_k = min(len(lw), len(rw))
    best = 0
    for k in range(max_k, 0, -1):
        if [w.lower() for w in lw[-k:]] == [w.lower() for w in rw[:k]]:
            best = k
            break
    joined = lw + rw[best:]
    return " ".join(joined)


def stitch_segments(
    pieces: Iterable[TranscriptPiece], *, overlap_tol: float = 1.0
) -> list[TranscriptPiece]:
    """Merge overlapping/adjacent transcript pieces into a clean transcript.

    Pieces are sorted by ``(start_ts, track)``; same-track pieces that overlap
    (within ``overlap_tol``) are merged with overlap-aware text de-duplication
    and corrected timestamps. Different tracks are never merged, preserving
    "me vs. others" attribution. The result is sorted by ``start_ts``.
    """
    ordered = sorted(pieces, key=lambda p: (p.start_ts, p.track.value))
    merged: list[TranscriptPiece] = []
    for piece in ordered:
        placed = False
        # Try to merge into an existing same-track tail piece it overlaps.
        for i in range(len(merged) - 1, -1, -1):
            prev = merged[i]
            if prev.track != piece.track:
                continue
            if _overlaps(prev, piece, overlap_tol):
                merged[i] = TranscriptPiece(
                    track=prev.track,
                    start_ts=min(prev.start_ts, piece.start_ts),
                    end_ts=max(prev.end_ts, piece.end_ts),
                    text=_dedup_join(prev.text, piece.text),
                )
                placed = True
                break
            # Stop scanning back once we pass a non-overlapping same-track piece.
            break
        if not placed:
            merged.append(
                TranscriptPiece(
                    track=piece.track,
                    start_ts=piece.start_ts,
                    end_ts=piece.end_ts,
                    text=piece.text.strip(),
                )
            )
    merged.sort(key=lambda p: p.start_ts)
    return merged


__all__ = [
    "VADConfig",
    "SpeechSegment",
    "TranscriptPiece",
    "MeetingPipeline",
    "frame_energy",
    "stitch_segments",
]
