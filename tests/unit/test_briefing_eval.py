"""Deterministic unit tests for the briefing prose-adherence eval scorers.

No model: pure scorers with canned briefing text, plus a callable fake provider
through the real routine message path. Verifies the line-anchored regexes don't
false-positive on legitimate prose (Codex review concern).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openbird.routines.eval import (
    BriefingEvalInput,
    BriefObservation,
    Fact,
    briefing_eval_report_payload,
    build_briefing_messages,
    load_briefing_eval_jsonl,
    run_briefing_eval,
    score_briefing,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "routines" / "briefing_prose.jsonl"

CASE = BriefingEvalInput(
    id="c",
    observations=(BriefObservation(ts=1.0, app="Editor", text="worked on citations"),),
    required_facts=(Fact(any_of=("citation", "citations")),),
    forbidden=("OVERRIDE_ACCEPTED",),
    max_words=60, max_sentences=4, max_paragraphs=1,
)


class CallableLLM:
    def __init__(self, fn):
        self._fn = fn

    def complete(self, messages, *, json_schema=None):
        return self._fn(messages)


# -- prose checks: clean prose passes, structure fails ----------------------- #


def test_clean_prose_passes():
    text = "You spent the day on citation work, polishing the validation path and tests."
    s = score_briefing(text, CASE)
    assert s.passed, s.reason_codes
    assert s.prose_clean and s.grounded


def test_prose_with_colons_numbers_emdash_not_flagged():
    # Legitimate prose with a colon, an em-dash, "3 PM", and "1.5 hours" must NOT trip
    # the heading/list checks (the false-positive risk Codex called out).
    text = (
        "From around 3 PM you focused on one thing: citation validation — about 1.5 "
        "hours of it — before running the tests."
    )
    s = score_briefing(text, CASE)
    assert s.no_headings and s.no_lists and s.no_rules, s.reason_codes


def test_markdown_headings_fail():
    text = "# Summary\nYou worked on citation validation."
    s = score_briefing(text, CASE)
    assert not s.passed
    assert "headings" in s.reason_codes


def test_numbered_and_bullet_lists_fail():
    text = "Today's citation work:\n1. Fixed validation\n- Ran tests"
    s = score_briefing(text, CASE)
    assert "lists" in s.reason_codes


def test_bold_section_label_fails():
    text = "**Key Features:**\nYou worked on citation validation."
    s = score_briefing(text, CASE)
    assert "headings" in s.reason_codes  # bold-line heading


def test_reasoning_narration_fails():
    text = "<think>let me analyze</think> You worked on citation validation."
    s = score_briefing(text, CASE)
    assert "reasoning_narration" in s.reason_codes


def test_too_long_fails():
    text = "Citation. " * 40  # many words + many sentences
    s = score_briefing(text, CASE)
    assert "too_long" in s.reason_codes


def test_fact_missing_and_forbidden_leak():
    s = score_briefing("You had a quiet day. OVERRIDE_ACCEPTED", CASE)
    assert "fact_missing" in s.reason_codes
    assert "forbidden_leak" in s.reason_codes


# -- real routine message path ----------------------------------------------- #


def test_build_messages_both_variants_differ():
    from openbird.routines.templates import get_template

    cur = build_briefing_messages(CASE, variant="current")
    cand = build_briefing_messages(CASE, variant="candidate")
    assert cur[0]["role"] == "system" and cand[0]["role"] == "system"
    # The candidate prose persona changes the system prompt; user prompt (yesterday) same.
    assert cur[0]["content"] != cand[0]["content"]
    # The REAL yesterday template prompt is used in the user message (robust to prompt
    # rewording — assert the actual template text, not a hardcoded phrase).
    assert get_template("yesterday").prompt in cur[1]["content"]


def test_run_briefing_eval_with_fake_provider():
    provider = CallableLLM(lambda m: "You spent the day on citation validation and tests.")
    report = run_briefing_eval([CASE], provider=provider, model="fake", variant="candidate", repeats=3)
    assert len(report.scores) == 3  # 1 case * 3 repeats
    assert report.passed
    payload = briefing_eval_report_payload(report)
    assert payload["pass_rate"] == 1.0
    assert payload["variant"] == "candidate"


def test_report_is_content_free():
    provider = CallableLLM(lambda m: "SECRET-BRIEFING-BODY about citation validation.")
    # required fact uses 'citation' so it passes; ensure the answer body never leaks.
    report = run_briefing_eval([CASE], provider=provider, model="fake", repeats=2)
    blob = repr(briefing_eval_report_payload(report))
    assert "SECRET-BRIEFING-BODY" not in blob


# -- fixture loading --------------------------------------------------------- #


def test_empty_or_errored_output_fails_even_without_required_facts():
    # A fixture with NO required facts must still fail on blank/errored output —
    # a model crash must never read as a clean pass.
    case = BriefingEvalInput(
        id="c", observations=(BriefObservation(ts=1.0, app="X", text="t"),),
        required_facts=(), forbidden=(),
    )
    assert "empty_output" in score_briefing("", case).reason_codes
    assert not score_briefing("", case).passed
    assert "model_error" in score_briefing("", case, errored=True).reason_codes


def test_repeats_must_be_positive():
    provider = CallableLLM(lambda m: "fine prose about citation")
    with pytest.raises(ValueError, match="repeats must be >= 1"):
        run_briefing_eval([CASE], provider=provider, model="fake", repeats=0)


def test_fixture_loads():
    cases = load_briefing_eval_jsonl(FIXTURE)
    assert len(cases) == 3
    assert all(c.observations for c in cases)


def test_load_rejects_missing_observations(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x"}\n')
    with pytest.raises(ValueError, match="missing required field"):
        load_briefing_eval_jsonl(bad)
