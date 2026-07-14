"""Tests for deterministic, content-safe capture evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from openbird import cli
from openbird.capture import (
    CaptureEvalInput,
    capture_eval_report_payload,
    load_capture_eval_jsonl,
    run_capture_eval,
)


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "capture" / "synthetic.jsonl"


def _case(
    case_id: str,
    *,
    strategy: str,
    captured: str,
    reference: str,
    target: str = "SyntheticApp",
    state: str = "controlled",
    forbidden: tuple[str, ...] = (),
    extraction_ms: float = 10.0,
    repeat_group: str | None = None,
    changed_at_ms: float | None = None,
    captured_at_ms: float | None = None,
    budget_exceeded: bool = False,
) -> CaptureEvalInput:
    return CaptureEvalInput(
        case_id=case_id,
        target=target,
        state=state,
        strategy=strategy,
        captured_text=captured,
        reference_text=reference,
        forbidden_text=forbidden,
        extraction_ms=extraction_ms,
        repeat_group=repeat_group,
        changed_at_ms=changed_at_ms,
        captured_at_ms=captured_at_ms,
        budget_exceeded=budget_exceeded,
    )


def _passing_pair() -> list[CaptureEvalInput]:
    return [
        _case(
            "extract",
            strategy="current_helper",
            captured="alpha",
            reference="alpha beta",
        ),
        _case(
            "exclude",
            strategy="current_helper",
            captured="",
            reference="",
            forbidden=("PRIVATE_SENTINEL",),
        ),
        _case(
            "extract",
            strategy="candidate",
            captured="alpha beta",
            reference="alpha beta",
        ),
        _case(
            "exclude",
            strategy="candidate",
            captured="",
            reference="",
            forbidden=("PRIVATE_SENTINEL",),
        ),
    ]


def test_loader_ignores_comments_and_rejects_duplicate_pairs_without_body(tmp_path):
    fixture = tmp_path / "fixture.jsonl"
    row = {
        "case_id": "same",
        "target": "App",
        "state": "state",
        "strategy": "current_helper",
        "captured_text": "PRIVATE DUPLICATE BODY",
        "reference_text": "reference",
        "forbidden_text": [],
        "extraction_ms": 1,
    }
    fixture.write_text(
        "# synthetic\n" + json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )

    try:
        load_capture_eval_jsonl(fixture)
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("duplicate fixture should fail")

    assert "duplicate strategy/case pair" in message
    assert "PRIVATE DUPLICATE BODY" not in message


def test_loader_rejects_malformed_fields_without_echoing_values(tmp_path):
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        json.dumps(
            {
                "case_id": "bad",
                "target": "App",
                "state": "state",
                "strategy": "current_helper",
                "captured_text": "PRIVATE MALFORMED BODY",
                "reference_text": "reference",
                "forbidden_text": [],
                "extraction_ms": -1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        load_capture_eval_jsonl(fixture)
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("negative timing should fail")

    assert "finite and non-negative" in message
    assert "PRIVATE MALFORMED BODY" not in message


def test_unicode_normalization_and_code_punctuation_are_exact():
    report = run_capture_eval(
        [
            _case(
                "unicode-code",
                strategy="current_helper",
                captured="CAFE\N{COMBINING ACUTE ACCENT} foo.bar()",
                reference="café foo.bar()",
            ),
            _case(
                "exclude",
                strategy="current_helper",
                captured="",
                reference="",
            ),
        ]
    )

    assert report.baseline.precision == 1.0
    assert report.baseline.recall == 1.0
    assert report.baseline.f1 == 1.0


def test_extraction_metrics_are_macro_averaged_not_token_weighted():
    long_reference = " ".join(f"token{index}" for index in range(100))
    report = run_capture_eval(
        [
            _case(
                "long",
                strategy="current_helper",
                captured=long_reference,
                reference=long_reference,
            ),
            _case(
                "short",
                strategy="current_helper",
                captured="",
                reference="missed",
            ),
            _case(
                "exclude",
                strategy="current_helper",
                captured="",
                reference="",
            ),
        ]
    )

    assert report.baseline.precision == 0.5
    assert report.baseline.recall == 0.5
    assert report.baseline.f1 == 0.5


def test_exclusions_do_not_inflate_extraction_metrics():
    report = run_capture_eval(
        [
            _case(
                "extract",
                strategy="current_helper",
                captured="one",
                reference="one two",
            ),
            _case(
                "excluded-empty",
                strategy="current_helper",
                captured="",
                reference="",
            ),
            _case(
                "excluded-leak",
                strategy="current_helper",
                captured="PRIVATE_SENTINEL",
                reference="",
                forbidden=("PRIVATE_SENTINEL",),
            ),
        ]
    )

    assert report.baseline.recall == 0.5
    assert report.baseline.exclusion_accuracy == 0.5
    assert report.baseline.forbidden_leak_count == 1
    assert "exclusion_accuracy_below_minimum" in report.baseline.absolute_gate_failures
    assert "forbidden_text_leak" in report.baseline.absolute_gate_failures


def test_repeat_similarity_uses_worst_pair():
    report = run_capture_eval(
        [
            _case(
                "a",
                strategy="current_helper",
                captured="stable anchor",
                reference="stable anchor",
                repeat_group="stable",
            ),
            _case(
                "b",
                strategy="current_helper",
                captured="stable anchor",
                reference="stable anchor",
                repeat_group="stable",
            ),
            _case(
                "c",
                strategy="current_helper",
                captured="different content",
                reference="stable anchor",
                repeat_group="stable",
            ),
            _case(
                "exclude",
                strategy="current_helper",
                captured="",
                reference="",
            ),
        ]
    )

    assert report.baseline.minimum_repeat_similarity == 0.0
    assert "repeat_similarity_below_minimum" in report.baseline.absolute_gate_failures


def test_candidate_can_pass_absolute_and_low_baseline_gain_gates():
    report = run_capture_eval(
        _passing_pair(),
        candidate_strategy="candidate",
    )

    assert report.mode == "candidate"
    assert report.passed is True
    assert report.reason_codes == ()
    assert report.baseline.f1 < 0.70
    assert report.candidate is not None
    assert report.candidate.f1 == 1.0


def test_candidate_reports_low_baseline_gain_and_target_regression():
    cases = _passing_pair()
    cases[2] = _case(
        "extract",
        strategy="candidate",
        captured="alpha",
        reference="alpha beta",
    )

    report = run_capture_eval(cases, candidate_strategy="candidate")

    assert report.passed is False
    assert "low_baseline_f1_gain_below_minimum" in report.reason_codes

    strong_baseline = [
        _case(
            "strong",
            strategy="current_helper",
            captured="one two",
            reference="one two",
        ),
        _case("exclude", strategy="current_helper", captured="", reference=""),
        _case(
            "strong",
            strategy="candidate",
            captured="one",
            reference="one two",
        ),
        _case("exclude", strategy="candidate", captured="", reference=""),
    ]

    regression = run_capture_eval(strong_baseline, candidate_strategy="candidate")

    assert "target_recall_regression" in regression.reason_codes


def test_candidate_fails_privacy_latency_budget_and_stability_gates():
    cases: list[CaptureEvalInput] = []
    for strategy in ("current_helper", "candidate"):
        cases.extend(
            [
                _case(
                    "repeat-a",
                    strategy=strategy,
                    captured="stable anchor",
                    reference="stable anchor",
                    repeat_group="repeat",
                    extraction_ms=10,
                ),
                _case(
                    "repeat-b",
                    strategy=strategy,
                    captured=("different" if strategy == "candidate" else "stable anchor"),
                    reference="stable anchor",
                    repeat_group="repeat",
                    extraction_ms=(2_100 if strategy == "candidate" else 10),
                    budget_exceeded=strategy == "candidate",
                ),
                _case(
                    "changed",
                    strategy=strategy,
                    captured="changed anchor",
                    reference="changed anchor",
                    changed_at_ms=1_000,
                    captured_at_ms=(4_500 if strategy == "candidate" else 1_500),
                ),
                _case(
                    "exclude",
                    strategy=strategy,
                    captured=("PRIVATE_SENTINEL" if strategy == "candidate" else ""),
                    reference="",
                    forbidden=("PRIVATE_SENTINEL",),
                ),
            ]
        )

    report = run_capture_eval(cases, candidate_strategy="candidate")

    assert report.passed is False
    assert "repeat_similarity_below_minimum" in report.reason_codes
    assert "change_latency_above_maximum" in report.reason_codes
    assert "p95_extraction_above_maximum" in report.reason_codes
    assert "extraction_budget_breach" in report.reason_codes
    assert "exclusion_accuracy_below_minimum" in report.reason_codes
    assert "forbidden_text_leak" in report.reason_codes


def test_candidate_pairing_fails_closed_on_reference_or_repeat_change():
    cases = _passing_pair()
    candidate = cases[2]
    cases[2] = _case(
        candidate.case_id,
        strategy="candidate",
        captured="alpha beta",
        reference="different reference",
    )

    try:
        run_capture_eval(cases, candidate_strategy="candidate")
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("mismatched case definitions should fail")

    assert "case definitions must match" in message
    assert "different reference" not in message


def test_checked_in_baseline_is_diagnostic_and_calibrated():
    report = run_capture_eval(load_capture_eval_jsonl(FIXTURE))
    payload = capture_eval_report_payload(report)

    assert report.mode == "baseline"
    assert report.passed is True
    assert report.reason_codes == ("baseline_diagnostic_only",)
    assert payload["baseline"] == {
        "strategy": "current_helper",
        "total_cases": 26,
        "extraction_cases": 24,
        "exclusion_cases": 2,
        "precision": 0.582198,
        "recall": 0.664931,
        "f1": 0.613476,
        "exclusion_accuracy": 0.5,
        "repeat_group_count": 4,
        "minimum_repeat_similarity": 1.0,
        "changed_case_count": 4,
        "changed_within_3s": 0.75,
        "p95_extraction_ms": 1150.0,
        "budget_breach_count": 0,
        "forbidden_leak_count": 1,
        "absolute_gate_failures": [
            "precision_below_minimum",
            "recall_below_minimum",
            "change_latency_above_maximum",
            "p95_extraction_above_maximum",
            "exclusion_accuracy_below_minimum",
            "forbidden_text_leak",
        ],
        "targets": {
            "ChatGPT": {
                "total_cases": 6,
                "extraction_cases": 6,
                "exclusion_cases": 0,
                "precision": 0.111111,
                "recall": 0.166667,
                "f1": 0.133333,
            },
            "Chrome": {
                "total_cases": 6,
                "extraction_cases": 6,
                "exclusion_cases": 0,
                "precision": 0.486111,
                "recall": 0.493056,
                "f1": 0.479293,
            },
            "Codex": {
                "total_cases": 5,
                "extraction_cases": 5,
                "exclusion_cases": 0,
                "precision": 0.79956,
                "recall": 1.0,
                "f1": 0.885024,
            },
            "Generic AX": {
                "total_cases": 9,
                "extraction_cases": 7,
                "exclusion_cases": 2,
                "precision": 0.913087,
                "recall": 1.0,
                "f1": 0.946078,
            },
        },
    }


def test_cli_baseline_json_is_deterministic_and_omits_fixture_text():
    runner = CliRunner()
    first = runner.invoke(cli.app, ["eval", "capture", str(FIXTURE), "--json"])
    second = runner.invoke(cli.app, ["eval", "capture", str(FIXTURE), "--json"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert first.stdout == second.stdout
    assert "SYNTHETIC_FORM_SENTINEL" not in first.stdout
    assert "Alpha note ready to review" not in first.stdout
    payload = json.loads(first.stdout)
    assert payload["mode"] == "baseline"
    assert payload["passed"] is True
    assert payload["baseline"]["absolute_gate_failures"]


def test_cli_human_table_omits_fixture_text():
    result = CliRunner().invoke(cli.app, ["eval", "capture", str(FIXTURE)])

    assert result.exit_code == 0, result.output
    assert "SYNTHETIC_FORM_SENTINEL" not in result.stdout
    assert "Alpha note ready to review" not in result.stdout
    assert "Capture eval" in result.stdout
    assert "Baseline gate failures: precision_below_minimum" in result.stdout


def test_cli_candidate_gate_failure_uses_exit_one_without_leaking_text(tmp_path):
    fixture = tmp_path / "candidate.jsonl"
    rows = [
        {
            "case_id": "extract",
            "target": "App",
            "state": "state",
            "strategy": "current_helper",
            "captured_text": "alpha",
            "reference_text": "alpha beta",
            "forbidden_text": [],
            "extraction_ms": 10,
        },
        {
            "case_id": "exclude",
            "target": "App",
            "state": "state",
            "strategy": "current_helper",
            "captured_text": "",
            "reference_text": "",
            "forbidden_text": ["PRIVATE_CLI_SENTINEL"],
            "extraction_ms": 10,
        },
        {
            "case_id": "extract",
            "target": "App",
            "state": "state",
            "strategy": "candidate",
            "captured_text": "alpha",
            "reference_text": "alpha beta",
            "forbidden_text": [],
            "extraction_ms": 10,
        },
        {
            "case_id": "exclude",
            "target": "App",
            "state": "state",
            "strategy": "candidate",
            "captured_text": "",
            "reference_text": "",
            "forbidden_text": ["PRIVATE_CLI_SENTINEL"],
            "extraction_ms": 10,
        },
    ]
    fixture.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli.app,
        ["eval", "capture", str(fixture), "--candidate", "candidate", "--json"],
    )

    assert result.exit_code == 1
    assert "PRIVATE_CLI_SENTINEL" not in result.stdout
    payload = json.loads(result.stdout)
    assert payload["passed"] is False
    assert "low_baseline_f1_gain_below_minimum" in payload["reason_codes"]

    human = CliRunner().invoke(
        cli.app,
        ["eval", "capture", str(fixture), "--candidate", "candidate"],
    )

    assert human.exit_code == 1
    assert "PRIVATE_CLI_SENTINEL" not in human.stdout
    assert "Candidate gate failures: recall_below_minimum" in human.stdout
    assert "Reason codes:" in human.stdout


def test_cli_bad_fixture_uses_exit_two_without_leaking_body(tmp_path):
    fixture = tmp_path / "bad.jsonl"
    fixture.write_text(
        '{"captured_text":"PRIVATE CLI BODY"}\n',
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli.app, ["eval", "capture", str(fixture), "--json"])

    assert result.exit_code == 2
    assert "Invalid capture eval fixture" in result.output
    assert "PRIVATE CLI BODY" not in result.output


def test_report_payload_never_contains_fixture_bodies():
    payload = capture_eval_report_payload(
        run_capture_eval(_passing_pair(), candidate_strategy="candidate")
    )
    encoded = json.dumps(payload)

    assert "PRIVATE_SENTINEL" not in encoded
    assert "alpha beta" not in encoded
    assert "captured_text" not in encoded
    assert "reference_text" not in encoded
    assert "forbidden_text" not in encoded
