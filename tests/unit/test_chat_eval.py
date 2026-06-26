"""Deterministic unit tests for the chat answer-quality eval scorers.

These run with NO model: they exercise the pure scorers with canned answers and a
callable fake provider through the real RAG path. The live-model runner is covered
separately (integration / opt-in).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openbird.chat.eval import (
    ChatEvalInput,
    GoldFact,
    GoldSource,
    chat_eval_report_payload,
    load_chat_eval_jsonl,
    normalize,
    run_chat_eval,
    score_case,
)
from openbird.types import Citation

FIXTURE = Path(__file__).parents[1] / "fixtures" / "chat" / "answer_quality.jsonl"


class CallableLLM:
    """A fake provider whose answer is computed from the prompt by ``fn``."""

    def __init__(self, fn):
        self._fn = fn

    def complete(self, messages, *, json_schema=None):
        return self._fn(messages)


def _result(answer, citation_sids, grounded, *, ts_by_sid=None):
    """Build an AnswerResult-like object with citations mapped to eval source ids."""
    from openbird.chat.eval import AnswerResult  # re-exported via rag

    ts_by_sid = ts_by_sid or {}
    cites = [
        Citation(
            observation_id=f"eval:{sid}",
            chunk_id=f"eval:{sid}:chunk",
            app=None,
            window=None,
            ts=ts_by_sid.get(sid, 0.0),
            snippet="",
        )
        for sid in citation_sids
    ]
    return AnswerResult(answer=answer, citations=cites, grounded=grounded)


# -- pure scorer ------------------------------------------------------------- #


def test_score_grounded_pass():
    case = ChatEvalInput(
        id="c", question="q", mode="search",
        sources=(GoldSource("S1", "revenue grew twelve percent", ts=1.0),),
        facts=(GoldFact(any_of=("12%", "twelve percent"), required_source_ids=("S1",)),),
        forbidden=("sushi",),
    )
    res = _result("Revenue grew twelve percent.", ["S1"], grounded=True)
    score = score_case(case, res, json_valid=True)
    assert score.passed
    assert score.reason_codes == ()


def test_score_fact_stated_but_uncited_fails():
    # The reliability bar: right fact, wrong/zero citation => FAIL.
    case = ChatEvalInput(
        id="c", question="q", mode="search",
        sources=(GoldSource("S1", "x", ts=1.0), GoldSource("S2", "y", ts=2.0)),
        facts=(GoldFact(any_of=("twelve percent",), required_source_ids=("S1",)),),
    )
    res = _result("Revenue grew twelve percent.", ["S2"], grounded=True)
    score = score_case(case, res, json_valid=True)
    assert not score.passed
    assert "citation_incomplete" in score.reason_codes


def test_score_hallucination_on_unanswerable_fails():
    case = ChatEvalInput(
        id="c", question="q", mode="search",
        sources=(GoldSource("S1", "buy milk", ts=1.0),),
        expect_grounded=False, forbidden=("milk",),
    )
    # Model answered (grounded) when it should have refused, and leaked forbidden.
    res = _result("Your note mentions milk.", ["S1"], grounded=True)
    score = score_case(case, res, json_valid=True)
    assert not score.passed
    assert "hallucinated" in score.reason_codes
    assert "forbidden_leak" in score.reason_codes


def test_precision_uses_raw_claimed_ids():
    # A model that CLAIMS a hallucinated id must fail precision, scored from the raw
    # claims (RAG drops them before AnswerResult, so resolved citations can't show it).
    case = ChatEvalInput(
        id="c", question="q", mode="search",
        sources=(GoldSource("S1", "logs content-free", ts=1.0),),
        facts=(GoldFact(any_of=("content-free",), required_source_ids=("S1",)),),
    )
    res = _result("content-free", ["S1"], grounded=True)  # post-RAG: FAKE9 dropped
    score = score_case(case, res, json_valid=True, claimed_ids=["S1", "FAKE9"])
    assert "citation_imprecise" in score.reason_codes


def test_hallucinated_id_caught_on_real_rag_path():
    # End-to-end: provider claims S1 + FAKE9. RAG drops FAKE9 from citations, but the
    # recorder captures the raw claim, so precision still fails.
    case = ChatEvalInput(
        id="c", question="how did revenue do?", mode="search",
        sources=(GoldSource("S1", "Revenue grew twelve percent.", ts=1.0),),
        facts=(GoldFact(any_of=("twelve percent",), required_source_ids=("S1",)),),
    )
    provider = CallableLLM(lambda m: {"answer": "Revenue grew twelve percent.", "citations": ["S1", "FAKE9"]})
    report = run_chat_eval([case], provider=provider, model="fake")
    assert "citation_imprecise" in report.scores[0].reason_codes


def test_score_window_violation():
    case = ChatEvalInput(
        id="c", question="q", mode="explicit-window",
        sources=(GoldSource("S1", "in window", ts=1000.0),),
        facts=(GoldFact(any_of=("in window",), required_source_ids=("S1",)),),
        window=(900.0, 1100.0),
    )
    res = _result("in window", ["S1"], grounded=True, ts_by_sid={"S1": 5000.0})
    score = score_case(case, res, json_valid=True)
    assert "window_violation" in score.reason_codes


def test_normalize_folds_percent():
    assert normalize("Twelve  Percent") == "twelve %"


# -- end-to-end through real RAG with a fake provider ------------------------ #


def test_empty_structured_output_scores_json_invalid():
    # Reproduces the qwen3:4b failure: complete(json_schema=...) returns "" (a str,
    # not a parsed dict). The recorder must flag json_valid=False, not silently pass.
    case = ChatEvalInput(
        id="c", question="how did revenue do?", mode="search",
        sources=(GoldSource("S1", "Revenue grew twelve percent.", ts=1.0),),
        facts=(GoldFact(any_of=("twelve percent",), required_source_ids=("S1",)),),
    )
    provider = CallableLLM(lambda m: "")  # empty best-effort fallback (no dict)
    report = run_chat_eval([case], provider=provider, model="fake")
    s = report.scores[0]
    assert s.json_valid is False
    assert "json_invalid" in s.reason_codes


def test_schema_invalid_dict_scores_json_invalid():
    # complete() can return a parsed dict on the best-effort fallback AFTER schema
    # validation fails. A dict lacking the {answer:str, citations:list} shape must be
    # json_invalid, not a free pass for "it's a dict".
    case = ChatEvalInput(
        id="c", question="q", mode="search",
        sources=(GoldSource("S1", "Revenue grew twelve percent.", ts=1.0),),
        facts=(GoldFact(any_of=("twelve percent",), required_source_ids=("S1",)),),
    )
    provider = CallableLLM(lambda m: {"foo": "bar"})  # parseable but wrong shape
    report = run_chat_eval([case], provider=provider, model="fake")
    assert report.scores[0].json_valid is False
    assert "json_invalid" in report.scores[0].reason_codes

    # citations list with non-string items is also schema-invalid.
    prov2 = CallableLLM(lambda m: {"answer": "x", "citations": [1, 2]})
    rep2 = run_chat_eval([case], provider=prov2, model="fake")
    assert rep2.scores[0].json_valid is False


def test_run_chat_eval_through_real_rag():
    case = ChatEvalInput(
        id="c", question="how did revenue do?", mode="search",
        sources=(
            GoldSource("S1", "The quarterly revenue grew by twelve percent.", ts=1.0),
            GoldSource("S2", "Lunch options: sushi and tacos.", ts=2.0),
        ),
        facts=(GoldFact(any_of=("twelve percent",), required_source_ids=("S1",)),),
        forbidden=("sushi",),
    )
    # A provider that cites S1 correctly (RAG assigns S1.. in fixture order).
    provider = CallableLLM(lambda m: {"answer": "Revenue grew twelve percent.", "citations": ["S1"]})
    report = run_chat_eval([case], provider=provider, model="fake")
    assert report.total == 1
    assert report.passed


# -- fixture loading + content-free report ----------------------------------- #


def test_fixture_loads_and_is_valid():
    cases = load_chat_eval_jsonl(FIXTURE)
    assert len(cases) == 6
    assert {c.mode for c in cases} == {"search", "temporal-phrase", "explicit-window"}


def test_temporal_phrase_runs_deterministically():
    # The phrase router resolves "yesterday" against a fixed eval clock; sources
    # are stamped into that window so a correct model can ground + cite S1.
    case = ChatEvalInput(
        id="c", question="What did I do yesterday?", mode="temporal-phrase",
        sources=(
            GoldSource("S1", "Ran the unit test suite.", ts=0.0),
            GoldSource("S2", "Reviewed the checklist.", ts=0.0),
        ),
        facts=(GoldFact(any_of=("unit test",), required_source_ids=("S1",)),),
    )
    provider = CallableLLM(lambda m: {"answer": "You ran the unit test suite.", "citations": ["S1"]})
    report = run_chat_eval([case], provider=provider, model="fake")
    assert report.total == 1
    assert report.passed


def test_load_rejects_bad_mode(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x", "question": "q", "mode": "nope", "sources": [{"sid":"S1","text":"t"}]}\n')
    with pytest.raises(ValueError, match="mode must be one of"):
        load_chat_eval_jsonl(bad)


def test_report_payload_is_content_free():
    case = ChatEvalInput(
        id="c", question="SECRET-QUESTION", mode="search",
        sources=(GoldSource("S1", "SECRET-SOURCE-TEXT", ts=1.0),),
        facts=(GoldFact(any_of=("SECRET-SOURCE-TEXT",), required_source_ids=("S1",)),),
    )
    provider = CallableLLM(lambda m: {"answer": "SECRET-ANSWER-BODY", "citations": ["S1"]})
    report = run_chat_eval([case], provider=provider, model="fake")
    blob = repr(chat_eval_report_payload(report))
    for leak in ("SECRET-QUESTION", "SECRET-SOURCE-TEXT", "SECRET-ANSWER-BODY"):
        assert leak not in blob
