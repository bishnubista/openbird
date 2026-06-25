"""Tests for persona override resolution (loader) + RAG runtime wiring (PR2)."""

from __future__ import annotations

import os

import pytest

import openbird.chat.rag as rag
from openbird.config import reset_settings_cache
from openbird.prompts.loader import MAX_PERSONA_BYTES, resolve_persona


@pytest.fixture
def prompts_dir(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    return d


# -- precedence ---------------------------------------------------------------


def test_default_when_no_override(prompts_dir):
    res = resolve_persona("rag", prompts_dir=prompts_dir)
    assert res.source == "default" and res.persona is None and res.ok


def test_file_override(prompts_dir):
    (prompts_dir / "rag.txt").write_text("ANSWER IN HAIKU.", encoding="utf-8")
    res = resolve_persona("rag", prompts_dir=prompts_dir)
    assert res.source == "file" and res.persona == "ANSWER IN HAIKU." and res.ok


def test_env_beats_file(prompts_dir, monkeypatch):
    (prompts_dir / "rag.txt").write_text("FROM FILE.", encoding="utf-8")
    monkeypatch.setenv("OPENBIRD_PROMPT_RAG", "FROM ENV.")
    res = resolve_persona("rag", prompts_dir=prompts_dir)
    assert res.source == "env" and res.persona == "FROM ENV."


def test_empty_env_is_no_override(prompts_dir, monkeypatch):
    (prompts_dir / "rag.txt").write_text("FROM FILE.", encoding="utf-8")
    monkeypatch.setenv("OPENBIRD_PROMPT_RAG", "")  # empty == unset
    res = resolve_persona("rag", prompts_dir=prompts_dir)
    assert res.source == "file" and res.persona == "FROM FILE."


def test_whitespace_env_refused(prompts_dir, monkeypatch):
    monkeypatch.setenv("OPENBIRD_PROMPT_RAG", "   ")
    res = resolve_persona("rag", prompts_dir=prompts_dir)
    assert res.source == "env" and res.persona is None and not res.ok
    assert res.reason == "empty"


# -- file guards (TOCTOU-safe read) -------------------------------------------


def test_symlink_refused(prompts_dir, tmp_path):
    target = tmp_path / "elsewhere.txt"
    target.write_text("SNEAKY", encoding="utf-8")
    (prompts_dir / "rag.txt").symlink_to(target)
    res = resolve_persona("rag", prompts_dir=prompts_dir)
    assert res.persona is None and not res.ok and res.reason == "symlink"


def test_oversized_refused(prompts_dir):
    (prompts_dir / "rag.txt").write_bytes(b"x" * (MAX_PERSONA_BYTES + 10))
    res = resolve_persona("rag", prompts_dir=prompts_dir)
    assert res.persona is None and not res.ok and res.reason == "too-large"


def test_at_size_cap_accepted(prompts_dir):
    body = "a" * MAX_PERSONA_BYTES
    (prompts_dir / "rag.txt").write_bytes(body.encode("utf-8"))
    res = resolve_persona("rag", prompts_dir=prompts_dir)
    assert res.ok and res.persona == body


def test_invalid_utf8_refused(prompts_dir):
    (prompts_dir / "rag.txt").write_bytes(b"\xff\xfe not utf8")
    res = resolve_persona("rag", prompts_dir=prompts_dir)
    assert res.persona is None and not res.ok and res.reason == "utf8"


def test_whitespace_file_refused(prompts_dir):
    (prompts_dir / "rag.txt").write_text("   \n\t", encoding="utf-8")
    res = resolve_persona("rag", prompts_dir=prompts_dir)
    assert res.persona is None and not res.ok and res.reason == "empty"


def test_fifo_refused(prompts_dir):
    fifo = prompts_dir / "rag.txt"
    os.mkfifo(fifo)
    res = resolve_persona("rag", prompts_dir=prompts_dir)
    assert res.persona is None and not res.ok and res.reason == "not-regular"


# -- RAG runtime wiring -------------------------------------------------------


def test_rag_applies_file_override(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    reset_settings_cache()
    pd = tmp_path / "prompts"
    pd.mkdir()
    (pd / "rag.txt").write_text("CUSTOM PERSONA LINE.", encoding="utf-8")
    try:
        r = rag.RAG(store=object(), provider=object())
        assert "CUSTOM PERSONA LINE." in r._system_prompt
        # Locked scaffold must still be present even with a custom persona.
        assert rag._DATA_OPEN in r._system_prompt
        assert "SECURITY REMINDER" in r._system_prompt
    finally:
        reset_settings_cache()


def test_rag_falls_back_to_default_on_bad_override(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    reset_settings_cache()
    pd = tmp_path / "prompts"
    pd.mkdir()
    (pd / "rag.txt").write_bytes(b"x" * (MAX_PERSONA_BYTES + 10))  # too large
    try:
        r = rag.RAG(store=object(), provider=object())
        # Degrades to the default prompt (still valid), never raises.
        assert r._system_prompt == rag._SYSTEM_PROMPT
    finally:
        reset_settings_cache()
