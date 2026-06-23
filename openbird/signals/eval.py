"""Offline eval helpers for the signal-first briefing classifier.

The eval runner is deterministic by default: it uses one synthetic observation
per fixture row and no model provider, so it is safe for CI and never reads the
user's real capture store.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from openbird.signals.classifier import (
    BriefingSignals,
    EvaluationCase,
    EvaluationResult,
    SignalClassifier,
    SignalLabel,
    evaluate_signal_predictions,
)
from openbird.types import Observation

EXPECTED_LABELS: frozenset[str] = frozenset(
    {"must_surface", "useful", "maybe", "noise", "sensitive_never_surface"}
)


@dataclass(frozen=True)
class SignalEvalInput:
    """One labeled JSONL fixture row for signal evals."""

    id: str
    text: str
    app: str
    expected: str
    window: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class SignalEvalPrediction:
    """One prediction record without fixture text or notes."""

    id: str
    expected: str
    predicted: str
    reason_codes: tuple[str, ...]
    deterministic_fallback: bool
    sensitive_quarantined: bool


@dataclass(frozen=True)
class SignalEvalReport:
    """Aggregate signal eval metrics and predictions."""

    result: EvaluationResult
    total_cases: int
    predictions: tuple[SignalEvalPrediction, ...]
    expected_counts: dict[str, int]
    sensitive_quarantine_miss_count: int

    @property
    def passed(self) -> bool:
        """Return the full pass condition including positive quarantine checks."""
        return self.result.passed and self.sensitive_quarantine_miss_count == 0


def load_signal_eval_jsonl(path: str | Path) -> list[SignalEvalInput]:
    """Load and validate signal eval fixtures from UTF-8 JSONL."""
    fixture = Path(path)
    cases: list[SignalEvalInput] = []
    for line_no, raw in enumerate(fixture.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{fixture}:{line_no}: invalid JSON") from exc
        cases.append(_case_from_obj(obj, fixture=fixture, line_no=line_no))
    if not cases:
        raise ValueError(f"{fixture}: no eval cases found")
    return cases


def run_signal_eval(
    cases: Iterable[SignalEvalInput],
    *,
    classifier: SignalClassifier | None = None,
) -> SignalEvalReport:
    """Run deterministic one-row signal evals and compute gate metrics.

    Each fixture row is classified in isolation, so precision_at_5 reflects the
    first five surfaced fixture rows rather than cross-window ranking quality.
    """
    runner = classifier or SignalClassifier(provider=None)
    inputs = list(cases)
    predictions: list[SignalEvalPrediction] = []
    metric_cases: list[EvaluationCase] = []

    for index, case in enumerate(inputs):
        obs = Observation(
            id=f"eval-{case.id}",
            content_hash=f"eval-{case.id}",
            ts=float(index + 1),
            app=case.app,
            window=case.window,
            url=None,
            session_id=f"eval-{case.id}",
            source="capture",
        )
        packets, grouped_duplicates = runner.build_packets([(obs, case.text)])
        sensitive_reason_codes = next(
            (packet.reason_codes for packet in packets if packet.sensitive),
            (),
        )
        result = runner.classify_packets(
            packets,
            start_ts=obs.ts,
            end_ts=obs.ts,
            local_model_status="eval",
            grouped_duplicates_count=grouped_duplicates,
        )
        prediction = _prediction_for_result(
            case,
            result,
            sensitive_reason_codes=sensitive_reason_codes,
        )
        predictions.append(prediction)
        metric_cases.append(
            EvaluationCase(
                case_id=case.id,
                expected=case.expected,
                predicted=prediction.predicted,
            )
        )

    eval_result = evaluate_signal_predictions(metric_cases)
    quarantine_misses = sum(
        1
        for pred in predictions
        if pred.expected == "sensitive_never_surface"
        and (pred.predicted != SignalLabel.SENSITIVE_QUARANTINE or not pred.sensitive_quarantined)
    )
    return SignalEvalReport(
        result=eval_result,
        total_cases=len(inputs),
        predictions=tuple(predictions),
        expected_counts=dict(Counter(case.expected for case in inputs)),
        sensitive_quarantine_miss_count=quarantine_misses,
    )


def signal_eval_report_payload(report: SignalEvalReport) -> dict[str, object]:
    """Return a JSON-serializable report without fixture text or notes."""
    return {
        "passed": report.passed,
        "total_cases": report.total_cases,
        "precision_at_5": report.result.precision_at_5,
        "must_surface_recall": report.result.must_surface_recall,
        "missed_important_count": report.result.missed_important_count,
        "noise_rate": report.result.noise_rate,
        "sensitive_leak_count": report.result.sensitive_leak_count,
        "sensitive_quarantine_miss_count": report.sensitive_quarantine_miss_count,
        "expected_counts": report.expected_counts,
        "predictions": [
            {
                "id": pred.id,
                "expected": pred.expected,
                "predicted": pred.predicted,
                "reason_codes": list(pred.reason_codes),
                "deterministic_fallback": pred.deterministic_fallback,
                "sensitive_quarantined": pred.sensitive_quarantined,
            }
            for pred in report.predictions
        ],
    }


def _case_from_obj(obj: object, *, fixture: Path, line_no: int) -> SignalEvalInput:
    """Validate one JSON object and build an eval input."""
    if not isinstance(obj, dict):
        raise ValueError(f"{fixture}:{line_no}: expected a JSON object")
    missing = [name for name in ("id", "text", "app", "expected") if name not in obj]
    if missing:
        raise ValueError(f"{fixture}:{line_no}: missing required field(s): {', '.join(missing)}")

    case_id = _string_field(obj, "id", fixture=fixture, line_no=line_no)
    text = _string_field(obj, "text", fixture=fixture, line_no=line_no)
    app = _string_field(obj, "app", fixture=fixture, line_no=line_no)
    expected = _string_field(obj, "expected", fixture=fixture, line_no=line_no)
    if expected not in EXPECTED_LABELS:
        allowed = ", ".join(sorted(EXPECTED_LABELS))
        raise ValueError(f"{fixture}:{line_no}: expected must be one of: {allowed}")

    window = obj.get("window")
    notes = obj.get("notes")
    return SignalEvalInput(
        id=case_id,
        text=text,
        app=app,
        expected=expected,
        window=str(window) if window is not None else None,
        notes=str(notes) if notes is not None else None,
    )


def _prediction_for_result(
    case: SignalEvalInput,
    result: BriefingSignals,
    *,
    sensitive_reason_codes: tuple[str, ...],
) -> SignalEvalPrediction:
    """Map classifier output into the expected/predicted eval label space."""
    sensitive = result.sensitive_quarantine_count >= 1
    if sensitive:
        predicted = SignalLabel.SENSITIVE_QUARANTINE
        reason_codes = sensitive_reason_codes
        fallback = False
    elif result.signals:
        signal = result.signals[0]
        predicted = signal.label
        reason_codes = signal.reason_codes
        fallback = signal.deterministic_fallback
    else:
        predicted = SignalLabel.NOISE
        reason_codes = ()
        fallback = False

    return SignalEvalPrediction(
        id=case.id,
        expected=case.expected,
        predicted=str(predicted),
        reason_codes=tuple(reason_codes),
        deterministic_fallback=fallback,
        sensitive_quarantined=sensitive,
    )


def _string_field(obj: dict, name: str, *, fixture: Path, line_no: int) -> str:
    """Return a non-empty string field or raise a content-free error."""
    value = obj.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{fixture}:{line_no}: {name!r} must be a non-empty string")
    return value.strip()


__all__ = [
    "EXPECTED_LABELS",
    "SignalEvalInput",
    "SignalEvalPrediction",
    "SignalEvalReport",
    "load_signal_eval_jsonl",
    "run_signal_eval",
    "signal_eval_report_payload",
]
