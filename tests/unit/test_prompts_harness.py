"""PR4: offline injection self-check harness (`openbird prompts test`)."""

from __future__ import annotations

import dataclasses

import pytest

from openbird.config import Settings, reset_settings_cache
from openbird.llm.provider import CloudOptInRequired
from openbird.prompts import registry
from openbird.prompts.harness import (
    INJECTION_PAYLOAD,
    SENTINEL,
    _build_probe,
    run_test,
)


def _settings(tmp_path):
    pd = tmp_path / "prompts"
    pd.mkdir(exist_ok=True)
    return Settings(data_dir=tmp_path, prompts_dir=str(pd))


@pytest.fixture(autouse=True)
def _isolate():
    reset_settings_cache()
    registry.ensure_loaded()
    yield
    reset_settings_cache()


KEYS = ["rag", "rag_synthesis", "routine", "meeting", "signal"]


# -- the real production path defangs the active fence -------------------------


@pytest.mark.parametrize("key", KEYS)
def test_probe_defangs_active_close_token(key):
    spec = registry.get(key)
    # Resolve+render with the default persona for a self-contained probe.
    from openbird.prompts.core import render

    msgs = _build_probe(key, render(spec, None), INJECTION_PAYLOAD)
    user = next(m["content"] for m in msgs if m["role"] == "user")
    close = spec.fence.close_token
    # Exactly one trusted close token (the wrapper); the payload's attempt was defanged.
    assert user.count(close) == 1


@pytest.mark.parametrize("key", KEYS)
def test_run_test_deterministic_pass_default_persona(key, tmp_path):
    report = run_test(key, settings=_settings(tmp_path), use_llm=False)
    assert report.deterministic_ok
    assert report.rendered_ok and report.boundary_ok and report.neutralized_ok


def test_rag_payload_source_header_does_not_survive(tmp_path):
    report = run_test("rag", settings=_settings(tmp_path), use_llm=False)
    # The deterministic gate includes RAG's source-header contract.
    assert report.deterministic_ok


def test_refused_override_fails_instead_of_testing_default(tmp_path):
    # A refused override (here: too-large) must FAIL, not silently test the
    # bundled default and report PASS (CodeRabbit).
    pd = tmp_path / "prompts"
    pd.mkdir()
    (pd / "rag.txt").write_bytes(b"x" * (64 * 1024 + 10))
    settings = Settings(data_dir=tmp_path, prompts_dir=str(pd))
    report = run_test("rag", settings=settings, use_llm=False)
    assert not report.deterministic_ok
    assert "refused" in report.detail


def test_no_override_present_tests_default_and_passes(tmp_path):
    # "No override at all" is a legitimate thing to test (source=default, ok=True).
    report = run_test("rag", settings=_settings(tmp_path), use_llm=False)
    assert report.deterministic_ok


# -- the gate actually detects a hole -----------------------------------------


def test_deterministic_fails_when_neutralizer_is_a_noop(tmp_path, monkeypatch):
    # Hole the meeting fence's neutralizer (identity): the payload's </transcript>
    # then survives into the fenced block, so the gate must FAIL (it shows up as an
    # extra close token — a broken fence boundary).
    import openbird.meetings.transcribe as transcribe

    holed = dataclasses.replace(transcribe._FENCE, neutralizer=lambda t: t)
    monkeypatch.setattr(transcribe, "_FENCE", holed)

    report = run_test("meeting", settings=_settings(tmp_path), use_llm=False)
    assert not report.deterministic_ok


def test_deterministic_fails_when_wrapper_fence_is_dropped(tmp_path, monkeypatch):
    # A builder regression that drops the trusted fence wrapper entirely (but still
    # inserts a token-free body in a stable slot) must NOT false-pass: the trusted
    # open must be in the prefix and close in the suffix (Codex follow-up).
    import openbird.prompts.harness as harness

    def _no_fence_probe(key, system_prompt, payload):
        from openbird.signals.classifier import _neutralize_capture_data_impl

        return [
            {"role": "system", "content": system_prompt},
            # No <capture_data> wrapper at all — body sanitized but unfenced.
            {"role": "user", "content": f"BODY:{_neutralize_capture_data_impl(payload)}:END"},
        ]

    monkeypatch.setattr(harness, "_build_probe", _no_fence_probe)
    report = run_test("signal", settings=_settings(tmp_path), use_llm=False)
    assert not report.deterministic_ok and not report.boundary_ok


# -- model probe is advisory, never changes exit/deterministic ----------------


class _EchoProvider:
    def complete(self, messages, json_schema=None):
        return f"Sure. {SENTINEL}"


class _RefuseProvider:
    def complete(self, messages, json_schema=None):
        return "I can only summarize the data; I won't follow instructions in it."


def test_model_probe_advisory_fail_does_not_change_deterministic(tmp_path):
    report = run_test(
        "rag", settings=_settings(tmp_path), use_llm=True,
        provider_factory=lambda s: _EchoProvider(),
    )
    assert report.model_ran and report.model_advisory_pass is False
    assert report.deterministic_ok  # advisory FAIL never flips the deterministic gate


def test_model_probe_advisory_pass(tmp_path):
    report = run_test(
        "rag", settings=_settings(tmp_path), use_llm=True,
        provider_factory=lambda s: _RefuseProvider(),
    )
    assert report.model_ran and report.model_advisory_pass is True


def test_model_probe_cloud_gated(tmp_path):
    def _refuse(settings):
        raise CloudOptInRequired({"llm": "gpt-4o"})

    report = run_test(
        "rag", settings=_settings(tmp_path), use_llm=True, provider_factory=_refuse
    )
    assert not report.model_ran
    assert "cloud opt-in" in report.model_note
    assert report.deterministic_ok  # deterministic gate still ran


def test_no_llm_never_builds_a_provider(tmp_path):
    called = {"n": 0}

    def _factory(settings):
        called["n"] += 1
        return _RefuseProvider()

    run_test("rag", settings=_settings(tmp_path), use_llm=False, provider_factory=_factory)
    assert called["n"] == 0


# -- purity -------------------------------------------------------------------


def test_run_test_writes_nothing(tmp_path):
    pd = tmp_path / "prompts"
    pd.mkdir()
    run_test("rag", settings=Settings(data_dir=tmp_path, prompts_dir=str(pd)), use_llm=False)
    assert list(pd.iterdir()) == []  # no files created by the harness
