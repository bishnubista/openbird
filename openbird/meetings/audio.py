"""Audio-source abstraction for the meetings subsystem.

The real audio capture is performed by a separate **Swift `audio-helper`** that
will be built later. This module deliberately wraps an *injectable* audio source
so the Python pipeline can be developed and unit-tested with canned PCM frames —
no microphone, no system-audio tap, no ffmpeg.

Why a Swift helper (and why not ffmpeg)
------------------------------------------------------------------
* **System output capture requires ScreenCaptureKit.** macOS does not expose the
  system *output* mix as a normal input device, so ``ffmpeg -f avfoundation`` is
  **NOT** a viable fallback for system audio — it can only read *input* devices
  (mics, BlackHole/aggregate devices). System output must come from an
  ``SCStream`` (or a native CoreAudio process tap), captured by the signed helper
  that holds Screen-Recording TCC.
* **Mic is a separate track.** The microphone is a distinct Core Audio stream
  from the ScreenCaptureKit system-audio stream. We keep mic and system audio as
  **two separate, synchronized tracks** all the way through transcription, and
  merge only at the *transcript-segment* level so "me (mic) vs. others (system)"
  speaker attribution stays meaningful.
* **Clock sync is mandatory.** Because mic and SCK system audio are independent
  streams, their sample clocks drift. Each frame therefore carries a
  ``host_ts`` (a common monotonic host timestamp, e.g. mach_absolute_time
  converted to seconds) so the pipeline can align tracks on a shared clock and
  flag drift, rather than assuming sample-aligned buffers.
* **Privacy:** PCM never lands in argv/env/log lines; raw frames stay in memory
  or in ``0600`` scratch files that are securely deleted (handled by the helper
  and the pipeline, not here).

Fallbacks (documented, not implemented in v1): BlackHole / aggregate output
devices, or a native CoreAudio process tap. ``ffmpeg avfoundation`` is explicitly
excluded for system output.

The contract the Swift helper must satisfy is captured by :class:`AudioSource`:
it yields :class:`AudioFrame` objects (mono float32 PCM samples + host timestamp
+ track id). Everything downstream operates purely on these frames.
"""

from __future__ import annotations

import enum
import math
import struct
from array import array
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import BinaryIO

# The IPC wire format is little-endian f32 PCM (see ``encode_frame``). We store
# samples as a stdlib ``array('f')`` (4 bytes/sample) rather than a Python tuple
# of floats (~28 bytes/sample) so a single 25 s @ 48 kHz frame is ~4.8 MB instead
# of ~33 MB. Pin the item size so a platform where ``float`` isn't 32-bit can't
# silently violate the wire contract. This is a real runtime invariant, so raise
# explicitly rather than ``assert`` (which is stripped under ``python -O``).
if array("f").itemsize != 4:  # pragma: no cover - CPython guarantees f32
    raise RuntimeError("array('f') must be 32-bit to match the f32 IPC wire format")


class Track(str, enum.Enum):
    """Logical audio track.

    ``MIC`` is the local microphone ("me"); ``SYSTEM`` is the ScreenCaptureKit
    system-output mix ("others"). Kept separate through transcription so speaker
    attribution survives.
    """

    MIC = "mic"
    SYSTEM = "system"


@dataclass(frozen=True)
class AudioFrame:
    """A single block of mono PCM audio from one track.

    Attributes:
        samples: Mono PCM samples as float32, stored in a stdlib ``array('f')``
            (~4 bytes/sample) rather than a tuple of Python floats (~28 bytes
            each) so large windows don't blow up memory [M6]. Any input sequence
            (tuple/list/array) is coerced to ``array('f')`` at construction;
            values land in roughly ``[-1.0, 1.0]``. The array is technically
            mutable, but the contract is that ``samples`` is **immutable** — never
            mutate it in place; build a new frame instead. (``AudioFrame`` is
            therefore unhashable; see ``__hash__``.)
        sample_rate: Sample rate in Hz (e.g. 16000 for whisper-friendly audio).
        host_ts: Common monotonic host timestamp (seconds) of the *first* sample
            in this frame. This is the shared clock used to align the mic and
            system tracks; it is NOT the per-stream sample index.
        track: Which logical track this frame belongs to.
        seq: Monotonic per-source sequence number, for gap/drop detection.
    """

    samples: "array[float]" = field(repr=False)
    sample_rate: int
    host_ts: float
    track: Track = Track.SYSTEM
    seq: int = 0

    # ``samples`` is a mutable ``array`` whose contents are part of ``__eq__``, so
    # the frame is not safely hashable. ``frozen=True`` would otherwise synthesize
    # a ``__hash__`` that calls ``hash(self.samples)`` and raises ``TypeError`` at
    # use; make the intent explicit and unsurprising instead.
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Coerce any input sequence (tuple/list/array) to a float32 ``array('f')``
        # so storage is compact and equality between two frames is array-vs-array
        # (by value). Always COPY — even an existing array('f') is duplicated so a
        # caller mutating their original buffer after construction cannot silently
        # change this frozen frame's PCM/equality. Frozen => object.__setattr__.
        object.__setattr__(self, "samples", array("f", self.samples))

    def __repr__(self) -> str:
        """Metadata-only repr — never dumps raw PCM samples.

        Captured content (``samples``) must never reach a traceback, log line,
        or macOS crash report, so the repr emits only counts/timestamps.
        """
        return (
            f"AudioFrame(track={self.track.value!r}, sample_rate={self.sample_rate}, "
            f"host_ts={self.host_ts}, seq={self.seq}, n_samples={len(self.samples)})"
        )

    @property
    def duration(self) -> float:
        """Frame duration in seconds (``len(samples) / sample_rate``)."""
        if self.sample_rate <= 0:
            return 0.0
        return len(self.samples) / float(self.sample_rate)

    @property
    def end_ts(self) -> float:
        """Host timestamp just past the last sample (``host_ts + duration``)."""
        return self.host_ts + self.duration


class AudioSource:
    """Abstract, injectable source of :class:`AudioFrame` objects.

    Implementations either wrap the Swift helper's IPC stream or replay canned
    frames in tests. The pipeline depends only on this interface, so it never
    needs real audio.
    """

    def frames(self) -> Iterator[AudioFrame]:
        """Yield audio frames until the source is exhausted/stopped.

        Subclasses must override. Frames may interleave tracks; the consumer is
        responsible for routing by :attr:`AudioFrame.track`.
        """
        raise NotImplementedError

    def __iter__(self) -> Iterator[AudioFrame]:
        return self.frames()


class InMemoryAudioSource(AudioSource):
    """Replay a fixed list of frames — the canonical test/double source."""

    def __init__(self, frames: Iterable[AudioFrame]) -> None:
        self._frames: list[AudioFrame] = list(frames)

    def frames(self) -> Iterator[AudioFrame]:
        yield from self._frames


class CallbackAudioSource(AudioSource):
    """Adapt a pull-style callback (e.g. the Swift helper's IPC reader).

    ``read`` returns the next :class:`AudioFrame` or ``None`` at end-of-stream.
    This is the shape the real ScreenCaptureKit/mic helper bridge will expose.
    """

    def __init__(self, read: Callable[[], AudioFrame | None]) -> None:
        self._read = read

    def frames(self) -> Iterator[AudioFrame]:
        while True:
            frame = self._read()
            if frame is None:
                return
            yield frame


# -- Binary IPC framing (the Swift audio-helper's wire format) ----------------
#
# Each record is little-endian: a fixed header <u8 track><f64 host_ts>
# <f64 sample_rate><u32 count> followed by a count-prefixed <count * f32 samples>
# array. This MUST stay in lockstep with
# ``audio-helper/Sources/AudioHelper/main.swift`` (FrameWriter).
_FRAME_HEADER = struct.Struct("<BddI")
_TRACK_BY_CODE: dict[int, Track] = {0: Track.SYSTEM, 1: Track.MIC}
_CODE_BY_TRACK: dict[Track, int] = {Track.SYSTEM: 0, Track.MIC: 1}

# Upper bound on samples per frame. The helper emits short buffers (SCK frames
# and 4096-sample mic taps at <=48 kHz); ~10s at 48 kHz is a generous ceiling.
# A frame claiming more than this is treated as a corrupt/hostile stream and
# stops decoding, so a bogus uint32 count can't force a huge read/allocation.
_MAX_SAMPLES_PER_FRAME = 480_000

# Sane bounds for a decoded sample rate (Hz). The header carries it as f64, so a
# crashed/corrupt helper can emit a negative, absurdly large, or sub-1 Hz value.
# A rate below 1.0 would also round to 0 in ``int(round(...))`` and silently
# disable timing math, so we require ``1.0 <= sample_rate <= _MAX_SAMPLE_RATE``
# and treat anything else as a corrupt stream [L4].
_MIN_SAMPLE_RATE = 1.0
_MAX_SAMPLE_RATE = 768_000.0


def _read_exactly(reader: BinaryIO, n: int) -> bytes | None:
    """Read exactly ``n`` bytes, or ``None`` at clean/truncated EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = reader.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def encode_frame(frame: AudioFrame) -> bytes:
    """Encode an :class:`AudioFrame` into the helper's binary record format.

    Symmetric with :func:`decode_frames`; used by bridges and tests so the wire
    contract is verifiable without real audio.
    """
    code = _CODE_BY_TRACK.get(frame.track, 0)
    out = _FRAME_HEADER.pack(code, frame.host_ts, float(frame.sample_rate), len(frame.samples))
    if frame.samples:
        out += struct.pack(f"<{len(frame.samples)}f", *frame.samples)
    return out


def decode_frames(reader: BinaryIO) -> Iterator[AudioFrame]:
    """Parse the audio-helper's length-prefixed PCM records from a binary stream.

    Yields :class:`AudioFrame` objects with a monotonically increasing ``seq``.
    A truncated trailing record (helper stopped mid-write) ends iteration cleanly
    rather than raising, since the stream is an external process pipe.
    """
    seq = 0
    while True:
        header = _read_exactly(reader, _FRAME_HEADER.size)
        if header is None:
            return
        track_code, host_ts, sample_rate, count = _FRAME_HEADER.unpack(header)
        if not math.isfinite(host_ts):
            # Corrupt/hostile stream: a NaN/inf host_ts would poison clock
            # alignment. Stop cleanly, mirroring the count/truncated-tail paths.
            return
        if not (math.isfinite(sample_rate) and _MIN_SAMPLE_RATE <= sample_rate <= _MAX_SAMPLE_RATE):
            # NaN/inf, negative, sub-1 Hz (would round to 0 and disable timing),
            # or absurdly large rates are all corrupt — clean-stop, don't crash on
            # ``int(round(sample_rate))`` or carry a nonsense rate downstream [L4].
            return
        track = _TRACK_BY_CODE.get(track_code)
        if track is None:
            # The wire protocol is a fixed two-track (SYSTEM=0, MIC=1) contract.
            # An unknown code is a corrupt/incompatible stream; silently mapping it
            # to SYSTEM would misattribute "me vs. others". Clean-stop instead [L4].
            return
        if count > _MAX_SAMPLES_PER_FRAME:
            # Corrupt/hostile stream: don't read/allocate `count * 4` bytes.
            return
        body = _read_exactly(reader, count * 4) if count else b""
        if body is None:
            return  # truncated tail
        samples = struct.unpack(f"<{count}f", body) if count else ()
        yield AudioFrame(
            samples=samples,  # AudioFrame.__post_init__ coerces to array('f') [M6]
            sample_rate=int(round(sample_rate)),
            host_ts=host_ts,
            track=track,
            seq=seq,
        )
        seq += 1


class BinaryFrameAudioSource(AudioSource):
    """:class:`AudioSource` reading the Swift audio-helper's binary IPC stream.

    Wraps any binary reader — the helper's subprocess stdout pipe, or an opened
    ``0600`` fifo/file it writes via ``--out``. Mic and system frames interleave
    on one stream and are routed by :attr:`AudioFrame.track` downstream.
    """

    def __init__(self, reader: BinaryIO) -> None:
        self._reader = reader

    def frames(self) -> Iterator[AudioFrame]:
        yield from decode_frames(self._reader)


@enum.unique
class ClockEventKind(str, enum.Enum):
    """Kinds of clock-sync anomaly the pipeline can react to."""

    INTRA_GAP = "intra_gap"  # gap/overlap within a single track's own timeline
    CROSS_DRIFT = "cross_drift"  # MIC vs SYSTEM baseline offset exceeded tolerance
    DEVICE_SWITCH = "device_switch"  # abrupt timestamp jump (e.g. BT device switch)


@dataclass(frozen=True)
class ClockEvent:
    """A single clock-sync anomaly emitted by :meth:`ClockSync.observe`.

    Carries only metadata (track, kind, magnitude, timestamps) — never any
    captured audio — so it is safe to log/raise.
    """

    kind: ClockEventKind
    track: Track
    magnitude: float  # seconds; signed where meaningful (gap>0, overlap<0)
    host_ts: float


@dataclass
class ClockSync:
    """Track-alignment helper over a shared host clock.

    Mic and system audio are independent Core Audio streams; we align them on
    :attr:`AudioFrame.host_ts` rather than assuming sample alignment. Two things
    are tracked:

    * **Intra-track continuity** — ``drift[track]`` is the gap/overlap between a
      track's previous frame end and the next frame's ``host_ts`` (a sudden large
      jump here signals a *device switch*).
    * **Cross-stream offset** — ``cross_offset`` is the MIC-vs-SYSTEM baseline
      skew measured on the *shared* host clock. Two streams that are each
      internally continuous but shifted by e.g. 500 ms relative to one another
      have ~0 intra-track drift yet a non-zero cross offset; only this catches
      the "continuous but offset" case the intra-track measure misses.
    """

    tolerance: float = 0.10  # seconds of drift tolerated before flagging
    device_switch_tolerance: float = 1.0  # abrupt jump (s) flagged as a switch
    _last_end: dict[Track, float] = field(default_factory=dict)
    _last_start: dict[Track, float] = field(default_factory=dict)
    drift: dict[Track, float] = field(default_factory=dict)
    # Cross-stream baseline: the offset between where each track's clock sits on
    # the shared host timeline. ``cross_offset`` = MIC position - SYSTEM position.
    cross_offset: float = 0.0
    events: list[ClockEvent] = field(default_factory=list)

    def observe(self, frame: AudioFrame) -> float:
        """Record a frame; return the intra-track drift (seconds) for its track.

        Drift is ``host_ts - expected_continuation`` where the expectation is the
        end timestamp of the previous frame on that track. Positive => a gap;
        negative => overlap. The first frame of a track has zero drift.

        Side effects: updates :attr:`cross_offset` (MIC↔SYSTEM baseline) and
        appends any :class:`ClockEvent` anomalies (intra gap, cross drift, or a
        device-switch jump) to :attr:`events`.
        """
        prev_end = self._last_end.get(frame.track)
        delta = 0.0 if prev_end is None else (frame.host_ts - prev_end)
        self._last_end[frame.track] = frame.end_ts
        self._last_start[frame.track] = frame.host_ts
        self.drift[frame.track] = delta

        # Device switch: an abrupt timestamp jump on the same track (the host
        # clock leapt because the input device changed mid-stream).
        if abs(delta) > self.device_switch_tolerance:
            self.events.append(
                ClockEvent(
                    kind=ClockEventKind.DEVICE_SWITCH,
                    track=frame.track,
                    magnitude=delta,
                    host_ts=frame.host_ts,
                )
            )
        elif abs(delta) > self.tolerance:
            self.events.append(
                ClockEvent(
                    kind=ClockEventKind.INTRA_GAP,
                    track=frame.track,
                    magnitude=delta,
                    host_ts=frame.host_ts,
                )
            )

        self._update_cross_offset(frame)
        return delta

    def _update_cross_offset(self, frame: AudioFrame) -> None:
        """Recompute the MIC↔SYSTEM baseline offset on the shared host clock.

        With both tracks active we compare the most recent start timestamp of
        each track. Two continuous-but-offset streams (same cadence, shifted by
        a constant) produce a stable non-zero offset that intra-track drift can
        never see, so this is what flags cross-stream skew.
        """
        mic = self._last_start.get(Track.MIC)
        system = self._last_start.get(Track.SYSTEM)
        if mic is None or system is None:
            return
        offset = mic - system
        self.cross_offset = offset
        if abs(offset) > self.tolerance:
            self.events.append(
                ClockEvent(
                    kind=ClockEventKind.CROSS_DRIFT,
                    track=frame.track,
                    magnitude=offset,
                    host_ts=frame.host_ts,
                )
            )

    def is_drifting(self, track: Track) -> bool:
        """Whether the last observed intra-track drift exceeds the tolerance."""
        return abs(self.drift.get(track, 0.0)) > self.tolerance

    def is_cross_drifting(self) -> bool:
        """Whether the MIC↔SYSTEM baseline offset exceeds the tolerance."""
        return abs(self.cross_offset) > self.tolerance

    def drain_events(self) -> list[ClockEvent]:
        """Return and clear the accumulated anomaly events."""
        events = self.events
        self.events = []
        return events


__all__ = [
    "Track",
    "AudioFrame",
    "AudioSource",
    "InMemoryAudioSource",
    "CallbackAudioSource",
    "BinaryFrameAudioSource",
    "encode_frame",
    "decode_frames",
    "ClockSync",
    "ClockEvent",
    "ClockEventKind",
]
