"""The Swift audio-helper's binary PCM IPC format, round-tripped in Python.

These tests pin the wire contract in ``openbird/meetings/audio.py`` against the
record layout written by ``audio-helper/Sources/AudioHelper/main.swift`` —
``<u8 track><f64 host_ts><f64 sample_rate><u32 count><count*f32 samples>`` LE.
No audio hardware involved.
"""

from __future__ import annotations

import io
import math
import struct
from array import array

import pytest

from openbird.meetings.audio import (
    _FRAME_HEADER,
    AudioFrame,
    BinaryFrameAudioSource,
    Track,
    decode_frames,
    encode_frame,
)


def _frame(track: Track, host_ts: float, samples: tuple[float, ...]) -> AudioFrame:
    return AudioFrame(samples=samples, sample_rate=48_000, host_ts=host_ts, track=track)


def test_roundtrip_preserves_frames():
    # f32-exact sample values so the comparison is exact after the f64->f32 hop.
    frames = [
        _frame(Track.SYSTEM, 1.0, (0.0, 0.5, -0.25, 0.75)),
        _frame(Track.MIC, 1.02, (-1.0, 0.125)),
        _frame(Track.SYSTEM, 1.04, ()),  # empty frame is legal
    ]
    blob = b"".join(encode_frame(f) for f in frames)

    decoded = list(decode_frames(io.BytesIO(blob)))
    assert len(decoded) == 3
    for original, got in zip(frames, decoded):
        assert got.track == original.track
        assert got.sample_rate == original.sample_rate
        assert got.host_ts == pytest.approx(original.host_ts)
        assert got.samples == original.samples
    # seq is assigned monotonically by the decoder.
    assert [f.seq for f in decoded] == [0, 1, 2]


def test_binary_source_routes_tracks():
    blob = encode_frame(_frame(Track.MIC, 2.0, (0.5,))) + encode_frame(
        _frame(Track.SYSTEM, 2.0, (0.5,))
    )
    src = BinaryFrameAudioSource(io.BytesIO(blob))
    tracks = [f.track for f in src]
    assert tracks == [Track.MIC, Track.SYSTEM]


def test_truncated_tail_is_ignored_not_raised():
    good = encode_frame(_frame(Track.SYSTEM, 1.0, (0.5, 0.5)))
    # A header claiming 4 samples but only 2 samples' worth of bytes present.
    truncated = encode_frame(_frame(Track.SYSTEM, 2.0, (0.1, 0.2, 0.3, 0.4)))[:-8]
    decoded = list(decode_frames(io.BytesIO(good + truncated)))
    assert len(decoded) == 1  # the complete frame survives; the partial is dropped
    assert decoded[0].host_ts == pytest.approx(1.0)


def test_samples_stored_as_float32_array_not_python_tuple():
    # [M6] A tuple of Python floats is ~28 bytes/sample; we store array('f') (4B).
    f = _frame(Track.SYSTEM, 1.0, (0.0, 0.5, -0.25))
    assert isinstance(f.samples, array)
    assert f.samples.typecode == "f"
    # Frame is intentionally unhashable (mutable array contents are part of ==).
    with pytest.raises(TypeError):
        hash(f)


@pytest.mark.parametrize(
    "bad_host_ts,bad_rate",
    [
        (float("nan"), 48_000.0),
        (1.0, float("nan")),
        (1.0, float("inf")),
        (float("-inf"), 48_000.0),
    ],
)
def test_non_finite_header_stops_cleanly_not_raises(bad_host_ts, bad_rate):
    # [L4] A corrupt/hostile header with NaN/inf host_ts or sample_rate must end
    # iteration cleanly (like the count guard and truncated tail), NOT raise a
    # ValueError/OverflowError out of int(round(...)).
    good = encode_frame(_frame(Track.SYSTEM, 1.0, (0.5, 0.5)))
    # Hand-pack a poisoned header (SYSTEM track code 0, count 1) + 1 sample body.
    poisoned = _FRAME_HEADER.pack(0, bad_host_ts, bad_rate, 1) + struct.pack("<f", 0.25)
    decoded = list(decode_frames(io.BytesIO(good + poisoned)))
    assert len(decoded) == 1  # the good frame survives; decode stops at the poison
    assert math.isfinite(decoded[0].host_ts)
