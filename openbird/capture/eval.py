"""Deterministic, content-safe evaluation for capture extractor results.

The evaluator consumes explicit synthetic JSONL fixtures. It never reads the
OpenBird capture store and never calls a model provider. Reports deliberately
omit captured, reference, and forbidden fixture text.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Iterable


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

MIN_PRECISION = 0.75
MIN_RECALL = 0.90
MIN_REPEAT_SIMILARITY = 0.95
MAX_CHANGE_LATENCY_MS = 3_000.0
MAX_P95_EXTRACTION_MS = 1_000.0
MAX_EXTRACTION_BUDGET_MS = 2_000.0
MAX_TARGET_REGRESSION = 0.05
LOW_BASELINE_F1 = 0.70
MIN_LOW_BASELINE_F1_GAIN = 0.20

_REQUIRED_FIELDS = frozenset(
    {
        "case_id",
        "target",
        "state",
        "strategy",
        "captured_text",
        "reference_text",
        "forbidden_text",
        "extraction_ms",
    }
)
_OPTIONAL_FIELDS = frozenset(
    {"changed_at_ms", "captured_at_ms", "repeat_group", "budget_exceeded"}
)


@dataclass(frozen=True)
class CaptureEvalInput:
    """One strategy result for one controlled capture case."""

    case_id: str
    target: str
    state: str
    strategy: str
    captured_text: str
    reference_text: str
    forbidden_text: tuple[str, ...]
    extraction_ms: float
    changed_at_ms: float | None = None
    captured_at_ms: float | None = None
    repeat_group: str | None = None
    budget_exceeded: bool = False


@dataclass(frozen=True)
class CaptureTargetMetrics:
    """Macro extraction metrics for one target surface."""

    total_cases: int
    extraction_cases: int
    exclusion_cases: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class CaptureStrategyMetrics:
    """Aggregate metrics for one extractor strategy."""

    strategy: str
    total_cases: int
    extraction_cases: int
    exclusion_cases: int
    precision: float | None
    recall: float | None
    f1: float | None
    exclusion_accuracy: float | None
    repeat_group_count: int
    minimum_repeat_similarity: float | None
    changed_case_count: int
    changed_within_3s: float | None
    p95_extraction_ms: float
    budget_breach_count: int
    forbidden_leak_count: int
    targets: dict[str, CaptureTargetMetrics]
    absolute_gate_failures: tuple[str, ...]


@dataclass(frozen=True)
class CaptureEvalReport:
    """Baseline diagnostic or candidate promotion report."""

    mode: str
    passed: bool
    baseline: CaptureStrategyMetrics
    candidate: CaptureStrategyMetrics | None
    reason_codes: tuple[str, ...]


def load_capture_eval_jsonl(path: str | Path) -> list[CaptureEvalInput]:
    """Load and validate a synthetic capture evaluation fixture."""
    fixture = Path(path)
    cases: list[CaptureEvalInput] = []
    seen: set[tuple[str, str]] = set()

    with fixture.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{fixture}:{line_no}: invalid JSON") from exc
            case = _case_from_obj(obj, fixture=fixture, line_no=line_no)
            key = (case.strategy, case.case_id)
            if key in seen:
                raise ValueError(f"{fixture}:{line_no}: duplicate strategy/case pair")
            seen.add(key)
            cases.append(case)

    if not cases:
        raise ValueError(f"{fixture}: no eval cases found")
    return cases


def run_capture_eval(
    cases: Iterable[CaptureEvalInput],
    *,
    baseline_strategy: str = "current_helper",
    candidate_strategy: str | None = None,
) -> CaptureEvalReport:
    """Compute a baseline diagnostic or a candidate promotion gate."""
    inputs = list(cases)
    if not inputs:
        raise ValueError("capture eval requires at least one case")

    by_strategy: dict[str, list[CaptureEvalInput]] = defaultdict(list)
    for case in inputs:
        by_strategy[case.strategy].append(case)

    if baseline_strategy not in by_strategy:
        raise ValueError("baseline strategy is not present in the fixture")
    _validate_repeat_groups(by_strategy)

    baseline = _strategy_metrics(baseline_strategy, by_strategy[baseline_strategy])
    if candidate_strategy is None:
        return CaptureEvalReport(
            mode="baseline",
            passed=True,
            baseline=baseline,
            candidate=None,
            reason_codes=("baseline_diagnostic_only",),
        )

    if candidate_strategy == baseline_strategy:
        raise ValueError("candidate strategy must differ from the baseline strategy")
    if candidate_strategy not in by_strategy:
        raise ValueError("candidate strategy is not present in the fixture")

    _validate_candidate_pairing(
        by_strategy[baseline_strategy],
        by_strategy[candidate_strategy],
    )
    candidate = _strategy_metrics(candidate_strategy, by_strategy[candidate_strategy])
    reason_codes = list(candidate.absolute_gate_failures)
    reason_codes.extend(_relative_gate_failures(baseline, candidate))
    return CaptureEvalReport(
        mode="candidate",
        passed=not reason_codes,
        baseline=baseline,
        candidate=candidate,
        reason_codes=tuple(reason_codes),
    )


def capture_eval_report_payload(report: CaptureEvalReport) -> dict[str, object]:
    """Build a deterministic JSON payload containing no fixture text."""
    return {
        "schema_version": 1,
        "mode": report.mode,
        "passed": report.passed,
        "baseline_strategy": report.baseline.strategy,
        "candidate_strategy": report.candidate.strategy if report.candidate else None,
        "reason_codes": list(report.reason_codes),
        "thresholds": {
            "minimum_precision": MIN_PRECISION,
            "minimum_recall": MIN_RECALL,
            "minimum_repeat_similarity": MIN_REPEAT_SIMILARITY,
            "maximum_change_latency_ms": MAX_CHANGE_LATENCY_MS,
            "maximum_p95_extraction_ms": MAX_P95_EXTRACTION_MS,
            "maximum_extraction_budget_ms": MAX_EXTRACTION_BUDGET_MS,
            "maximum_target_regression": MAX_TARGET_REGRESSION,
            "low_baseline_f1": LOW_BASELINE_F1,
            "minimum_low_baseline_f1_gain": MIN_LOW_BASELINE_F1_GAIN,
        },
        "baseline": _strategy_payload(report.baseline),
        "candidate": _strategy_payload(report.candidate) if report.candidate else None,
    }


def _case_from_obj(obj: object, *, fixture: Path, line_no: int) -> CaptureEvalInput:
    if not isinstance(obj, dict):
        raise ValueError(f"{fixture}:{line_no}: expected a JSON object")
    missing = sorted(_REQUIRED_FIELDS.difference(obj))
    if missing:
        raise ValueError(
            f"{fixture}:{line_no}: missing required field(s): {', '.join(missing)}"
        )
    if set(obj).difference(_REQUIRED_FIELDS | _OPTIONAL_FIELDS):
        raise ValueError(f"{fixture}:{line_no}: unexpected field(s)")

    case_id = _non_empty_string(obj, "case_id", fixture=fixture, line_no=line_no)
    target = _non_empty_string(obj, "target", fixture=fixture, line_no=line_no)
    state = _non_empty_string(obj, "state", fixture=fixture, line_no=line_no)
    strategy = _non_empty_string(obj, "strategy", fixture=fixture, line_no=line_no)
    captured_text = _string(obj, "captured_text", fixture=fixture, line_no=line_no)
    reference_text = _string(obj, "reference_text", fixture=fixture, line_no=line_no)
    forbidden_text = _string_list(obj, "forbidden_text", fixture=fixture, line_no=line_no)
    extraction_ms = _non_negative_number(
        obj, "extraction_ms", fixture=fixture, line_no=line_no
    )

    changed_present = "changed_at_ms" in obj
    captured_present = "captured_at_ms" in obj
    if changed_present != captured_present:
        raise ValueError(
            f"{fixture}:{line_no}: changed_at_ms and captured_at_ms must appear together"
        )
    changed_at_ms = (
        _non_negative_number(obj, "changed_at_ms", fixture=fixture, line_no=line_no)
        if changed_present
        else None
    )
    captured_at_ms = (
        _non_negative_number(obj, "captured_at_ms", fixture=fixture, line_no=line_no)
        if captured_present
        else None
    )
    if changed_at_ms is not None and captured_at_ms is not None:
        if captured_at_ms < changed_at_ms:
            raise ValueError(f"{fixture}:{line_no}: captured_at_ms precedes changed_at_ms")

    repeat_group = obj.get("repeat_group")
    if repeat_group is not None:
        if not isinstance(repeat_group, str) or not repeat_group.strip():
            raise ValueError(
                f"{fixture}:{line_no}: 'repeat_group' must be a non-empty string"
            )
        repeat_group = repeat_group.strip()

    budget_exceeded = obj.get("budget_exceeded", False)
    if not isinstance(budget_exceeded, bool):
        raise ValueError(f"{fixture}:{line_no}: 'budget_exceeded' must be a boolean")

    return CaptureEvalInput(
        case_id=case_id,
        target=target,
        state=state,
        strategy=strategy,
        captured_text=captured_text,
        reference_text=reference_text,
        forbidden_text=forbidden_text,
        extraction_ms=extraction_ms,
        changed_at_ms=changed_at_ms,
        captured_at_ms=captured_at_ms,
        repeat_group=repeat_group,
        budget_exceeded=budget_exceeded,
    )


def _strategy_metrics(
    strategy: str,
    cases: list[CaptureEvalInput],
) -> CaptureStrategyMetrics:
    extraction_scores: list[tuple[float, float, float]] = []
    exclusions: list[bool] = []
    by_target: dict[str, list[CaptureEvalInput]] = defaultdict(list)
    forbidden_leaks = 0

    for case in cases:
        by_target[case.target].append(case)
        reference_tokens = _tokens(case.reference_text)
        captured_tokens = _tokens(case.captured_text)
        if reference_tokens:
            extraction_scores.append(_extraction_score(captured_tokens, reference_tokens))
        else:
            exclusions.append(not captured_tokens)
        normalized_capture = _normalize(case.captured_text)
        if any(_normalize(value) in normalized_capture for value in case.forbidden_text):
            forbidden_leaks += 1

    precision, recall, f1 = _macro_scores(extraction_scores)
    repeat_scores = _repeat_similarities(cases)
    changed_latencies = [
        case.captured_at_ms - case.changed_at_ms
        for case in cases
        if case.changed_at_ms is not None and case.captured_at_ms is not None
    ]
    budget_breaches = sum(
        1
        for case in cases
        if case.budget_exceeded or case.extraction_ms > MAX_EXTRACTION_BUDGET_MS
    )
    target_metrics = {
        target: _target_metrics(target_cases)
        for target, target_cases in sorted(by_target.items())
    }
    exclusion_accuracy = (
        sum(1 for passed in exclusions if passed) / len(exclusions) if exclusions else None
    )
    changed_within = (
        sum(1 for latency in changed_latencies if latency <= MAX_CHANGE_LATENCY_MS)
        / len(changed_latencies)
        if changed_latencies
        else None
    )

    provisional = CaptureStrategyMetrics(
        strategy=strategy,
        total_cases=len(cases),
        extraction_cases=len(extraction_scores),
        exclusion_cases=len(exclusions),
        precision=precision,
        recall=recall,
        f1=f1,
        exclusion_accuracy=exclusion_accuracy,
        repeat_group_count=len(repeat_scores),
        minimum_repeat_similarity=min(repeat_scores) if repeat_scores else None,
        changed_case_count=len(changed_latencies),
        changed_within_3s=changed_within,
        p95_extraction_ms=_nearest_rank_p95([case.extraction_ms for case in cases]),
        budget_breach_count=budget_breaches,
        forbidden_leak_count=forbidden_leaks,
        targets=target_metrics,
        absolute_gate_failures=(),
    )
    return replace(
        provisional,
        absolute_gate_failures=_absolute_gate_failures(provisional),
    )


def _target_metrics(cases: list[CaptureEvalInput]) -> CaptureTargetMetrics:
    scores: list[tuple[float, float, float]] = []
    exclusion_count = 0
    for case in cases:
        reference_tokens = _tokens(case.reference_text)
        if reference_tokens:
            scores.append(_extraction_score(_tokens(case.captured_text), reference_tokens))
        else:
            exclusion_count += 1
    precision, recall, f1 = _macro_scores(scores)
    return CaptureTargetMetrics(
        total_cases=len(cases),
        extraction_cases=len(scores),
        exclusion_cases=exclusion_count,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _absolute_gate_failures(metrics: CaptureStrategyMetrics) -> tuple[str, ...]:
    failures: list[str] = []
    if metrics.extraction_cases == 0:
        failures.append("extraction_cases_missing")
    else:
        if metrics.precision is not None and metrics.precision < MIN_PRECISION:
            failures.append("precision_below_minimum")
        if metrics.recall is not None and metrics.recall < MIN_RECALL:
            failures.append("recall_below_minimum")
    if (
        metrics.minimum_repeat_similarity is not None
        and metrics.minimum_repeat_similarity < MIN_REPEAT_SIMILARITY
    ):
        failures.append("repeat_similarity_below_minimum")
    if metrics.changed_within_3s is not None and metrics.changed_within_3s < 1.0:
        failures.append("change_latency_above_maximum")
    if metrics.p95_extraction_ms > MAX_P95_EXTRACTION_MS:
        failures.append("p95_extraction_above_maximum")
    if metrics.budget_breach_count:
        failures.append("extraction_budget_breach")
    if metrics.exclusion_cases == 0:
        failures.append("exclusion_cases_missing")
    elif metrics.exclusion_accuracy is not None and metrics.exclusion_accuracy < 1.0:
        failures.append("exclusion_accuracy_below_minimum")
    if metrics.forbidden_leak_count:
        failures.append("forbidden_text_leak")
    return tuple(failures)


def _relative_gate_failures(
    baseline: CaptureStrategyMetrics,
    candidate: CaptureStrategyMetrics,
) -> tuple[str, ...]:
    precision_regressed = False
    recall_regressed = False
    low_baseline_gain_missed = False

    for target, baseline_target in baseline.targets.items():
        if baseline_target.extraction_cases == 0:
            continue
        candidate_target = candidate.targets[target]
        if (
            baseline_target.precision is not None
            and candidate_target.precision is not None
            and candidate_target.precision < baseline_target.precision - MAX_TARGET_REGRESSION
        ):
            precision_regressed = True
        if (
            baseline_target.recall is not None
            and candidate_target.recall is not None
            and candidate_target.recall < baseline_target.recall - MAX_TARGET_REGRESSION
        ):
            recall_regressed = True
        if (
            baseline_target.f1 is not None
            and candidate_target.f1 is not None
            and baseline_target.f1 < LOW_BASELINE_F1
            and candidate_target.f1 < baseline_target.f1 + MIN_LOW_BASELINE_F1_GAIN
        ):
            low_baseline_gain_missed = True

    failures: list[str] = []
    if precision_regressed:
        failures.append("target_precision_regression")
    if recall_regressed:
        failures.append("target_recall_regression")
    if low_baseline_gain_missed:
        failures.append("low_baseline_f1_gain_below_minimum")
    return tuple(failures)


def _validate_candidate_pairing(
    baseline_cases: list[CaptureEvalInput],
    candidate_cases: list[CaptureEvalInput],
) -> None:
    baseline = {case.case_id: case for case in baseline_cases}
    candidate = {case.case_id: case for case in candidate_cases}
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate must contain identical case sets")

    for case_id, baseline_case in baseline.items():
        candidate_case = candidate[case_id]
        if (
            baseline_case.target != candidate_case.target
            or baseline_case.state != candidate_case.state
            or baseline_case.repeat_group != candidate_case.repeat_group
            or baseline_case.reference_text != candidate_case.reference_text
            or baseline_case.forbidden_text != candidate_case.forbidden_text
            or baseline_case.changed_at_ms != candidate_case.changed_at_ms
        ):
            raise ValueError("baseline and candidate case definitions must match")


def _validate_repeat_groups(by_strategy: dict[str, list[CaptureEvalInput]]) -> None:
    for cases in by_strategy.values():
        counts = Counter(
            (case.target, case.repeat_group)
            for case in cases
            if case.repeat_group is not None
        )
        if any(count < 2 for count in counts.values()):
            raise ValueError("each repeat group must contain at least two cases per strategy")


def _repeat_similarities(cases: list[CaptureEvalInput]) -> list[float]:
    groups: dict[tuple[str, str], list[CaptureEvalInput]] = defaultdict(list)
    for case in cases:
        if case.repeat_group is not None:
            groups[(case.target, case.repeat_group)].append(case)

    scores: list[float] = []
    for group_cases in groups.values():
        pair_scores = [
            _multiset_jaccard(_tokens(left.captured_text), _tokens(right.captured_text))
            for left, right in combinations(group_cases, 2)
        ]
        scores.append(min(pair_scores))
    return scores


def _extraction_score(
    captured_tokens: Counter[str],
    reference_tokens: Counter[str],
) -> tuple[float, float, float]:
    if not captured_tokens:
        return 0.0, 0.0, 0.0
    overlap = sum((captured_tokens & reference_tokens).values())
    precision = overlap / sum(captured_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def _macro_scores(
    scores: list[tuple[float, float, float]],
) -> tuple[float | None, float | None, float | None]:
    if not scores:
        return None, None, None
    count = len(scores)
    return (
        sum(score[0] for score in scores) / count,
        sum(score[1] for score in scores) / count,
        sum(score[2] for score in scores) / count,
    )


def _multiset_jaccard(left: Counter[str], right: Counter[str]) -> float:
    if not left and not right:
        return 1.0
    intersection = sum((left & right).values())
    union = sum((left | right).values())
    return intersection / union if union else 0.0


def _nearest_rank_p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _tokens(text: str) -> Counter[str]:
    return Counter(TOKEN_PATTERN.findall(_normalize(text)))


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _strategy_payload(metrics: CaptureStrategyMetrics) -> dict[str, object]:
    return {
        "strategy": metrics.strategy,
        "total_cases": metrics.total_cases,
        "extraction_cases": metrics.extraction_cases,
        "exclusion_cases": metrics.exclusion_cases,
        "precision": _rounded(metrics.precision),
        "recall": _rounded(metrics.recall),
        "f1": _rounded(metrics.f1),
        "exclusion_accuracy": _rounded(metrics.exclusion_accuracy),
        "repeat_group_count": metrics.repeat_group_count,
        "minimum_repeat_similarity": _rounded(metrics.minimum_repeat_similarity),
        "changed_case_count": metrics.changed_case_count,
        "changed_within_3s": _rounded(metrics.changed_within_3s),
        "p95_extraction_ms": _rounded(metrics.p95_extraction_ms),
        "budget_breach_count": metrics.budget_breach_count,
        "forbidden_leak_count": metrics.forbidden_leak_count,
        "absolute_gate_failures": list(metrics.absolute_gate_failures),
        "targets": {
            target: {
                "total_cases": values.total_cases,
                "extraction_cases": values.extraction_cases,
                "exclusion_cases": values.exclusion_cases,
                "precision": _rounded(values.precision),
                "recall": _rounded(values.recall),
                "f1": _rounded(values.f1),
            }
            for target, values in metrics.targets.items()
        },
    }


def _rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _string(obj: dict, name: str, *, fixture: Path, line_no: int) -> str:
    value = obj.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{fixture}:{line_no}: {name!r} must be a string")
    return value


def _non_empty_string(obj: dict, name: str, *, fixture: Path, line_no: int) -> str:
    value = _string(obj, name, fixture=fixture, line_no=line_no)
    if not value.strip():
        raise ValueError(f"{fixture}:{line_no}: {name!r} must be a non-empty string")
    return value.strip()


def _string_list(
    obj: dict,
    name: str,
    *,
    fixture: Path,
    line_no: int,
) -> tuple[str, ...]:
    value = obj.get(name)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(
            f"{fixture}:{line_no}: {name!r} must be a list of non-empty strings"
        )
    return tuple(value)


def _non_negative_number(
    obj: dict,
    name: str,
    *,
    fixture: Path,
    line_no: int,
) -> float:
    value = obj.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{fixture}:{line_no}: {name!r} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(
            f"{fixture}:{line_no}: {name!r} must be finite and non-negative"
        )
    return number


__all__ = [
    "CaptureEvalInput",
    "CaptureEvalReport",
    "CaptureStrategyMetrics",
    "CaptureTargetMetrics",
    "capture_eval_report_payload",
    "load_capture_eval_jsonl",
    "run_capture_eval",
]
