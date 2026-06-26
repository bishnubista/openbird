"""Unit tests for the quality-eval gate logic (deterministic; no live model).

These exercise :class:`CheckResult`'s majority + grounded-rate-floor pass logic
and the ``--json`` payload. The live, model-calling path (``run_quality_eval``)
is intentionally NOT exercised here — this file pins the pure gate arithmetic.
"""
from __future__ import annotations

from openbird.routines.quality_eval import (
    GROUNDED_RATE_FLOOR,
    CheckResult,
    QualityReport,
    quality_eval_payload,
)


def _ask_run(*, ok: bool, grounded: bool) -> dict:
    """A synthetic ask run (carries the ``grounded`` flag the floor reads)."""
    return {"ok": ok, "grounded": grounded, "citations": 1 if grounded else 0,
            "self_capture": False}


def _briefing_run(*, ok: bool) -> dict:
    """A synthetic briefing run — NO ``grounded`` key, so the floor is vacuous."""
    return {"ok": ok, "ungrounded_refs": 0, "self_capture_source": False}


def test_no_runs_does_not_pass():
    assert CheckResult(label="ask:x").passed is False


def test_grounded_rate_none_for_briefing_runs():
    """Briefing runs omit the grounded flag → grounded_rate is None (not 0.0)."""
    c = CheckResult(label="briefing:day0",
                    runs=[_briefing_run(ok=True), _briefing_run(ok=True)])
    assert c.grounded_rate is None


def test_grounded_rate_computed_over_ask_runs():
    c = CheckResult(label="ask:x", runs=[
        _ask_run(ok=True, grounded=True),
        _ask_run(ok=True, grounded=True),
        _ask_run(ok=False, grounded=False),
    ])
    assert c.grounded_rate == 2 / 3


def test_briefing_majority_passes_despite_no_grounded_flag():
    """The floor must NOT fail a briefing check that lacks a grounded flag."""
    c = CheckResult(label="briefing:day0", runs=[
        _briefing_run(ok=True), _briefing_run(ok=True), _briefing_run(ok=False),
    ])
    assert c.passed is True


def test_majority_pass_but_below_floor_fails():
    """Regression guard: a strict ok-majority can still hide flaky grounding —
    3/5 grounded (0.6) is below the floor, so the check fails even though ok is a
    majority. This is the exact slip the synthesis-persona fix prevents."""
    runs = [_ask_run(ok=True, grounded=True)] * 3 + \
           [_ask_run(ok=False, grounded=False)] * 2
    c = CheckResult(label="ask:Summarize my day", runs=runs)
    assert sum(r["ok"] for r in c.runs) * 2 > len(c.runs)  # majority holds
    assert c.grounded_rate == 0.6
    assert c.passed is False  # ...but the floor fails it


def test_at_floor_passes():
    """Exactly at the floor (0.8) is a pass — the boundary is inclusive."""
    runs = [_ask_run(ok=True, grounded=True)] * 4 + \
           [_ask_run(ok=False, grounded=False)]
    c = CheckResult(label="ask:x", runs=runs)
    assert c.grounded_rate == GROUNDED_RATE_FLOOR
    assert c.passed is True


def test_all_grounded_passes():
    c = CheckResult(label="ask:x", runs=[_ask_run(ok=True, grounded=True)] * 5)
    assert c.passed is True


def test_grounded_but_no_majority_ok_fails():
    """High grounded rate cannot rescue a check that lost the ok majority (e.g.
    every run grounded but self-captured, so ok=False)."""
    runs = [{"ok": False, "grounded": True, "citations": 1, "self_capture": True}] * 3
    c = CheckResult(label="ask:x", runs=runs)
    assert c.grounded_rate == 1.0
    assert c.passed is False  # majority ok check fails first


def test_payload_surfaces_grounded_rate():
    report = QualityReport(checks=[
        CheckResult(label="ask:x", runs=[_ask_run(ok=True, grounded=True)] * 2),
        CheckResult(label="briefing:day0", runs=[_briefing_run(ok=True)] * 2),
    ])
    payload = quality_eval_payload(report)
    by_label = {c["label"]: c for c in payload["checks"]}
    assert by_label["ask:x"]["grounded_rate"] == 1.0
    assert by_label["briefing:day0"]["grounded_rate"] is None
    assert payload["passed"] is True
