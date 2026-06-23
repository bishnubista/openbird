"""Tests for deterministic signal eval fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from openbird import cli
from openbird.signals import (
    EXPECTED_LABELS,
    SignalLabel,
    load_signal_eval_jsonl,
    run_signal_eval,
    signal_eval_report_payload,
)


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "signals" / "synthetic.jsonl"


def test_load_signal_eval_jsonl_ignores_blank_and_comment_lines(tmp_path):
    """Blank lines and hash comments are skipped before JSON parsing."""
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        "\n"
        "# comment\n"
        '{"id":"a","text":"Build failed","app":"Code","expected":"must_surface"}\n',
        encoding="utf-8",
    )

    cases = load_signal_eval_jsonl(fixture)

    assert len(cases) == 1
    assert cases[0].id == "a"
    assert cases[0].expected in EXPECTED_LABELS


def test_load_signal_eval_jsonl_rejects_invalid_expected(tmp_path):
    """Invalid labels fail with row metadata, not fixture text."""
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        '{"id":"a","text":"SECRET BODY","app":"Code","expected":"surprise"}\n',
        encoding="utf-8",
    )

    try:
        load_signal_eval_jsonl(fixture)
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("fixture should fail")

    assert "expected must be one of" in message
    assert "SECRET BODY" not in message


def test_load_signal_eval_jsonl_rejects_empty_fixture(tmp_path):
    """An empty or comment-only fixture is a usage error, not a failed gate."""
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text("\n# no cases\n", encoding="utf-8")

    try:
        load_signal_eval_jsonl(fixture)
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("fixture should fail")

    assert "no eval cases found" in message


def test_synthetic_signal_eval_fixture_passes_and_quarantines_sensitive_rows():
    """The checked-in fixture passes all eval gates, including positive quarantine."""
    report = run_signal_eval(load_signal_eval_jsonl(FIXTURE))

    assert report.passed is True
    assert report.result.precision_at_5 >= 0.6
    assert report.result.must_surface_recall == 1.0
    assert report.result.sensitive_leak_count == 0
    assert report.sensitive_quarantine_miss_count == 0
    sensitive = [p for p in report.predictions if p.expected == "sensitive_never_surface"]
    assert sensitive
    assert all(p.predicted == SignalLabel.SENSITIVE_QUARANTINE for p in sensitive)
    assert all(p.sensitive_quarantined for p in sensitive)


def test_signal_eval_cli_json_omits_fixture_text():
    """Machine output includes metrics and predictions but never fixture bodies."""
    res = CliRunner().invoke(cli.app, ["eval", "signals", str(FIXTURE), "--json"])

    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert payload["passed"] is True
    assert payload["total_cases"] == 40
    assert "password: hunter2longer" not in res.stdout
    assert "Build failed with timeout" not in res.stdout
    assert all("text" not in pred for pred in payload["predictions"])
    assert all("notes" not in pred for pred in payload["predictions"])


def test_signal_eval_cli_table_omits_fixture_text():
    """Human table output stays limited to aggregate metrics."""
    res = CliRunner().invoke(cli.app, ["eval", "signals", str(FIXTURE)])

    assert res.exit_code == 0, res.output
    assert "password: hunter2longer" not in res.stdout
    assert "Build failed with timeout" not in res.stdout
    assert "must-001" not in res.stdout


def test_signal_eval_cli_rejects_invalid_fixture_without_leaking_text(tmp_path):
    """Bad fixture labels are a usage error and do not echo fixture text."""
    fixture = tmp_path / "bad.jsonl"
    fixture.write_text(
        '{"id":"bad","text":"PRIVATE BAD BODY","app":"Code","expected":"bad"}\n',
        encoding="utf-8",
    )

    res = CliRunner().invoke(cli.app, ["eval", "signals", str(fixture), "--json"])

    assert res.exit_code == 2
    assert "Invalid signal eval fixture" in res.output
    assert "PRIVATE BAD BODY" not in res.output


def test_signal_eval_cli_rejects_malformed_json_without_leaking_text(tmp_path):
    """Malformed JSON reports row metadata, not the raw line."""
    fixture = tmp_path / "bad.jsonl"
    fixture.write_text(
        '{"id":"bad","text":"PRIVATE MALFORMED BODY","app":"Code"\n',
        encoding="utf-8",
    )

    res = CliRunner().invoke(cli.app, ["eval", "signals", str(fixture), "--json"])

    assert res.exit_code == 2
    assert "invalid JSON" in res.output
    assert "PRIVATE MALFORMED BODY" not in res.output


def test_signal_eval_cli_fails_when_gate_fails(tmp_path):
    """A missed must-surface row exits non-zero and reports the failed metrics."""
    fixture = tmp_path / "fail.jsonl"
    fixture.write_text(
        '{"id":"missed","text":"A plain sentence without markers.",'
        '"app":"Safari","expected":"must_surface"}\n',
        encoding="utf-8",
    )

    res = CliRunner().invoke(cli.app, ["eval", "signals", str(fixture), "--json"])

    assert res.exit_code == 1
    payload = json.loads(res.stdout)
    assert payload["passed"] is False
    assert payload["missed_important_count"] == 1
    assert "A plain sentence" not in res.stdout


def test_signal_eval_report_payload_omits_notes(tmp_path):
    """Payloads never include fixture notes even when the fixture supplies them."""
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        '{"id":"a","text":"Build failed","app":"Code","expected":"must_surface","notes":"PRIVATE NOTE"}\n',
        encoding="utf-8",
    )

    payload = signal_eval_report_payload(run_signal_eval(load_signal_eval_jsonl(fixture)))

    assert "PRIVATE NOTE" not in json.dumps(payload)
    assert "notes" not in payload["predictions"][0]
