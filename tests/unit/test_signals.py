"""Tests for the signal-first briefing classifier."""

from __future__ import annotations

from openbird.signals import (
    EvaluationCase,
    SignalClassifier,
    SignalLabel,
    evaluate_signal_predictions,
    render_signal_brief,
)
from openbird.types import Observation


def _obs(idx: int, *, content_hash: str | None = None, app: str = "Code") -> Observation:
    """Build a minimal capture observation for classifier tests."""
    return Observation(
        id=f"obs-{idx}",
        content_hash=content_hash or f"hash-{idx}",
        ts=float(idx),
        app=app,
        window=None,
        url=None,
        session_id="s1",
        source="capture",
    )


class _BoomProvider:
    """Provider stub that proves model failures degrade without leaking text."""

    def complete(self, messages, *, json_schema=None):  # pragma: no cover - raises
        """Raise from completion to exercise per-item fallback."""
        raise RuntimeError("captured body must not leak")


class _DictProvider:
    """Provider stub returning a fixed structured completion payload."""

    def __init__(self, payload):
        """Store the payload and captured calls for assertions."""
        self.payload = payload
        self.calls = []

    def complete(self, messages, *, json_schema=None):
        """Return the configured payload while recording prompt metadata."""
        self.calls.append((messages, json_schema))
        return dict(self.payload)


def test_high_value_marker_survives_volatile_and_duplicate_penalties():
    """Blockers stay surfaceable despite duplicate and volatile UI penalties."""
    rows = [
        (_obs(1, content_hash="same"), "Build failed with timeout at 40% loading"),
        (_obs(2, content_hash="same"), "Build failed with timeout at 40% loading"),
    ]

    result = SignalClassifier(_BoomProvider()).classify_window(
        rows, start_ts=0.0, end_ts=10.0, local_model_status="unavailable"
    )

    assert result.grouped_duplicates_count == 1
    assert result.deterministic_fallback_count == 1
    assert result.signals[0].label == SignalLabel.BLOCKER
    assert "Build failed" not in render_signal_brief(result)


def test_sensitive_candidates_are_quarantined_and_never_sent_to_model():
    """Secret-pattern candidates are withheld before any provider call."""
    provider = _DictProvider(
        {
            "category": "open_loop",
            "confidence": 1,
            "user_value": 1,
            "evidence_observation_ids": ["obs-1"],
            "short_label": "secret",
            "why_surface": "secret",
            "why_hide": "",
        }
    )
    rows = [(_obs(1), "password: hunter2longer")]

    result = SignalClassifier(provider).classify_window(
        rows, start_ts=0.0, end_ts=10.0, local_model_status="available"
    )

    assert result.signals == ()
    assert result.sensitive_quarantine_count == 1
    assert provider.calls == []


def test_valid_model_output_is_grounded_by_evidence_id_and_text():
    """Grounded model output can surface as a non-fallback signal."""
    provider = _DictProvider(
        {
            "category": "open_loop",
            "confidence": 0.9,
            "user_value": 0.8,
            "evidence_observation_ids": ["obs-1"],
            "short_label": "Follow up on PR #53",
            "why_surface": "PR #53 follow up is explicit",
            "why_hide": "",
        }
    )
    rows = [(_obs(1), "Need to follow up on PR #53 after review.")]

    result = SignalClassifier(provider).classify_window(
        rows, start_ts=0.0, end_ts=10.0, local_model_status="available"
    )

    assert result.deterministic_fallback_count == 0
    assert result.signals[0].label == SignalLabel.OPEN_LOOP
    assert result.signals[0].short_label == "Follow up on PR #53"


def test_ungrounded_model_output_degrades_to_deterministic_fallback():
    """Ungrounded model claims are rejected in favor of deterministic fallback."""
    provider = _DictProvider(
        {
            "category": "commitment",
            "confidence": 0.99,
            "user_value": 0.99,
            "evidence_observation_ids": ["obs-1"],
            "short_label": "Call Alice by Friday",
            "why_surface": "Alice is waiting",
            "why_hide": "",
        }
    )
    rows = [(_obs(1), "Need to send the release note by tomorrow.")]

    result = SignalClassifier(provider).classify_window(
        rows, start_ts=0.0, end_ts=10.0, local_model_status="available"
    )

    assert result.deterministic_fallback_count == 1
    assert result.signals[0].deterministic_fallback is True
    assert result.signals[0].label == SignalLabel.COMMITMENT


def test_hidden_model_output_cannot_suppress_deterministic_blocker():
    """A model hide vote cannot suppress deterministic blocker evidence."""
    provider = _DictProvider(
        {
            "category": "unknown",
            "confidence": 0.9,
            "user_value": 0.1,
            "evidence_observation_ids": ["obs-1"],
            "short_label": "Build failed",
            "why_surface": "Build failed",
            "why_hide": "not useful",
        }
    )
    rows = [(_obs(1), "Build failed with timeout.")]

    result = SignalClassifier(provider).classify_window(
        rows, start_ts=0.0, end_ts=10.0, local_model_status="available"
    )

    assert result.deterministic_fallback_count == 1
    assert result.signals[0].label == SignalLabel.BLOCKER


def test_blocker_keeps_precedence_over_commitment_marker():
    """Blocker labels outrank commitment labels when both markers appear."""
    rows = [(_obs(1), "Build failed and needs to ship by tomorrow.")]

    result = SignalClassifier(None).classify_window(
        rows, start_ts=0.0, end_ts=10.0, local_model_status="unavailable"
    )

    assert result.signals[0].label == SignalLabel.BLOCKER


def test_zero_snippet_budget_does_not_retain_text():
    """A zero snippet budget must not retain captured body text."""
    rows = [(_obs(1), "Need to follow up on PR #53 after review.")]

    packets, _ = SignalClassifier(max_snippet_chars=0).build_packets(rows)

    assert packets[0].snippets == ("",)


def test_evaluation_gate_blocks_silence_and_sensitive_leaks():
    """Evaluation fails on missed must-surface items or sensitive leaks."""
    result = evaluate_signal_predictions(
        [
            EvaluationCase("a", "must_surface", None),
            EvaluationCase("b", "sensitive_never_surface", SignalLabel.OPEN_LOOP),
        ]
    )

    assert result.passed is False
    assert result.missed_important_count == 1
    assert result.sensitive_leak_count == 1


def test_evaluation_gate_passes_balanced_signal_set():
    """Evaluation passes when signal, noise, and quarantine expectations hold."""
    result = evaluate_signal_predictions(
        [
            EvaluationCase("a", "must_surface", SignalLabel.OPEN_LOOP),
            EvaluationCase("b", "useful", SignalLabel.BLOCKER),
            EvaluationCase("c", "noise", SignalLabel.NOISE),
            EvaluationCase("d", "sensitive_never_surface", SignalLabel.SENSITIVE_QUARANTINE),
        ]
    )

    assert result.passed is True
    assert result.precision_at_5 == 1.0
    assert result.sensitive_leak_count == 0
