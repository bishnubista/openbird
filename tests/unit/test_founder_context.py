"""Deterministic coverage for the on-demand founder-context recap."""

from __future__ import annotations

import datetime as dt
import re

import pytest

from openbird.chat.founder_context import (
    FOUNDER_CONTEXT_QUERY,
    is_founder_context_query,
    parse_founder_context_response,
    select_founder_context_rows,
)
from openbird.chat.rag import RAG
from openbird.config import Settings, reset_settings_cache
from openbird.llm.provider import LLMTimeoutError
from openbird.memory.store import MemoryStore
from openbird.types import Observation
from tests.unit.conftest import FakeProvider


def _obs(
    obs_id: str,
    text: str,
    *,
    ts: float,
    app: str,
    source: str = "capture",
    session: str | None = None,
    content_hash: str | None = None,
) -> tuple[Observation, str]:
    return (
        Observation(
            id=obs_id,
            content_hash=content_hash or f"h-{obs_id}",
            ts=ts,
            app=app,
            window=f"Work item {obs_id}",
            session_id=session,
            source=source,
        ),
        text,
    )


class _PagedStore:
    def __init__(self, rows: list[tuple[Observation, str]]) -> None:
        self.rows = rows
        self.page_calls = 0
        self.time_calls = 0
        self.search_calls = 0

    def founder_context_page(self, start, end, *, limit, before=None):
        self.page_calls += 1
        rows = [
            row
            for row in self.rows
            if start <= row[0].ts <= end
            and (
                before is None
                or row[0].ts < before[0]
                or (row[0].ts == before[0] and row[0].id < before[1])
            )
        ]
        rows.sort(key=lambda row: (row[0].ts, row[0].id), reverse=True)
        return rows[:limit]

    def time_range_text(self, *_args, **_kwargs):
        self.time_calls += 1
        return []

    def search(self, *_args, **_kwargs):
        self.search_calls += 1
        return []

    def block_summaries_for_range(self, *_args, **_kwargs):
        raise AssertionError("founder context must bypass summary-first retrieval")

    def week_memories_overlapping(self, *_args, **_kwargs):
        raise AssertionError("founder context must bypass cached week prose")


class _FounderProvider:
    llm_model = ""

    def __init__(self, settings: Settings, *, timeout: bool = False) -> None:
        self.settings = settings
        self.timeout = timeout
        self.calls: list[dict] = []

    def complete(
        self, messages, *, json_schema=None, max_attempts=None, timeout=None
    ):
        self.calls.append(
            {
                "messages": messages,
                "schema": json_schema,
                "max_attempts": max_attempts,
                "timeout": timeout,
            }
        )
        if self.timeout:
            raise LLMTimeoutError("bounded test timeout")
        ids = re.findall(r"\[source_id: (S\d+)\]", messages[-1]["content"])
        assert len(ids) >= 3
        return {
            "likely_focus": {
                "text": "The active thread is the OpenBird founder recap.",
                "citations": ids[:2],
            },
            "recent_activity": [
                {"text": "Implementation moved forward.", "citations": [ids[-1]]}
            ],
            "decisions_progress": [
                {"text": "The local launchd approach was chosen.", "citations": [ids[0]]}
            ],
            "open_loops": [
                {"text": "Validation remains.", "citations": [ids[1]]}
            ],
        }


def _settings(monkeypatch, tmp_path) -> Settings:
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBIRD_DISABLE_KEYRING", "1")
    reset_settings_cache()
    return Settings(data_dir=tmp_path)


def test_founder_intent_is_narrow_and_nearby_synthesis_is_unchanged():
    for query in (
        FOUNDER_CONTEXT_QUERY,
        "Bring me back up to speed",
        "Where did I leave off?",
        "Catch me back up on my work",
        "Get me back up to speed on my work.",
    ):
        assert is_founder_context_query(query)
    for query in (
        "Catch me up",
        "Recap my day",
        "What did I work on?",
        "Bring me up to speed on the auth design",
        "Where did I leave off yesterday?",
    ):
        assert not is_founder_context_query(query)


def test_claim_parser_drops_malformed_uncited_and_hallucinated_fields():
    answer, citations = parse_founder_context_response(
        {
            # One valid source is insufficient for likely focus.
            "likely_focus": {"text": "Too weak", "citations": ["S1", "S9"]},
            "recent_activity": [
                {"text": "Valid activity", "citations": ["S2"]},
                {"text": "Hallucinated", "citations": ["S99"]},
                {"text": 42, "citations": ["S1"]},
            ],
            "decisions_progress": "not-a-list",
            "open_loops": [{"text": "Valid loop", "citations": ["S1", "S1"]}],
        },
        valid_source_ids={"S1", "S2"},
    )
    assert "Likely focus" not in answer
    assert "Valid activity" in answer
    assert "Valid loop" in answer
    assert "Hallucinated" not in answer
    assert citations == ["S2", "S1"]


def test_selector_reserves_at_most_one_cue_row_per_origin():
    now = 10_000.0
    rows = [
        _obs(
            "noise-decision",
            "decided " * 100,
            ts=9_990,
            app="com.example.one",
        ),
        _obs(
            "noise-progress",
            "implemented " * 100,
            ts=9_989,
            app="com.example.one",
        ),
        _obs(
            "real-progress",
            "Implemented the bounded recent-work scan and verified progress.",
            ts=9_980,
            app="com.example.two",
        ),
        _obs(
            "real-loop",
            "Next step is to run the full validation suite.",
            ts=9_970,
            app="com.example.three",
            source="mcp",
        ),
    ]
    chosen = select_founder_context_rows(
        rows,
        now=now,
        signal_score=lambda _obs, text: min(len(text), 400) + 100,
        max_sources=3,
    )
    origins = {(row[0].app, row[0].source) for row in chosen}
    assert len(chosen) == len(origins) == 3
    assert {row[0].id for row in chosen} >= {"real-progress", "real-loop"}


def test_founder_route_reconstructs_recent_work_under_bounded_contract(
    monkeypatch, tmp_path
):
    settings = _settings(monkeypatch, tmp_path)
    now = dt.datetime(2026, 7, 29, 12, 0).timestamp()
    rows = [
        _obs(
            "active",
            "Building the OpenBird founder recap with source-grounded recent work.",
            ts=now - 60,
            app="com.openai.codex",
            session="s-active",
        ),
        _obs(
            "decision",
            "Decided to use a six-hour local LaunchAgent instead of cron.",
            ts=now - 300,
            app="com.apple.Notes",
            session="s-decision",
        ),
        _obs(
            "progress",
            "Implemented keyset pagination and the occurrence citation path.",
            ts=now - 600,
            app="com.apple.Terminal",
            session="s-progress",
        ),
        _obs(
            "loop",
            "Next step is full tests and adversarial review.",
            ts=now - 900,
            app="com.apple.mail",
            source="mcp",
            session="s-loop",
        ),
        _obs(
            "duplicate",
            "Building the OpenBird founder recap with source-grounded recent work.",
            ts=now - 30,
            app="com.openai.codex",
            session="s-active",
            content_hash="h-active",
        ),
        _obs(
            "self",
            "OpenBird UI contains the same prompt and must not cite itself.",
            ts=now - 10,
            app="ai.openbird.OpenBird",
        ),
        _obs(
            "older",
            "Older competing project research with substantial context.",
            ts=now - 4 * 86_400,
            app="com.apple.Safari",
            session="s-old",
        ),
        _obs(
            "too-old",
            "A stale project outside the five local-day window.",
            ts=now - 7 * 86_400,
            app="com.apple.Safari",
        ),
    ]
    store = _PagedStore(rows)
    provider = _FounderProvider(settings)
    rag = RAG(store, provider)
    rag._now = lambda: now

    result = rag.answer(FOUNDER_CONTEXT_QUERY)

    assert result.grounded is True
    assert result.grounding == "occurrence"
    assert len(result.citations) >= 3
    assert provider.calls[0]["max_attempts"] == 2
    assert provider.calls[0]["timeout"] == 20.0
    prompt = provider.calls[0]["messages"][-1]["content"]
    assert "six-hour local LaunchAgent" in prompt
    assert "keyset pagination" in prompt
    assert "Next step is full tests" in prompt
    assert "OpenBird UI contains" not in prompt
    assert "outside the five local-day window" not in prompt
    assert store.search_calls == 0
    assert result.memory_context["duplicate_rows"] == 1
    assert result.memory_context["self_capture_rows"] == 1


def test_explicit_scope_and_temporal_phrase_override_founder_default(
    monkeypatch, tmp_path
):
    settings = _settings(monkeypatch, tmp_path)
    store = _PagedStore([])
    rag = RAG(store, _FounderProvider(settings))

    explicit = rag.answer(FOUNDER_CONTEXT_QUERY, window=(1.0, 2.0))
    assert explicit.grounding == "empty"
    assert store.page_calls == 0
    assert store.time_calls == 1

    temporal = rag.answer(
        "Bring me back up to speed on what I was working on yesterday."
    )
    assert temporal.grounding == "empty"
    assert store.page_calls == 0
    assert store.time_calls == 2


def test_founder_timeout_is_explicit_and_grounded_false(monkeypatch, tmp_path):
    settings = _settings(monkeypatch, tmp_path)
    now = dt.datetime(2026, 7, 29, 12, 0).timestamp()
    rows = [
        _obs(
            f"o{i}",
            f"Substantial current founder work source {i} with implementation detail.",
            ts=now - i,
            app=f"com.example.{i}",
        )
        for i in range(3)
    ]
    provider = _FounderProvider(settings, timeout=True)
    rag = RAG(_PagedStore(rows), provider)
    rag._now = lambda: now

    result = rag.answer(FOUNDER_CONTEXT_QUERY)

    assert result.grounded is False
    assert result.grounding == "timeout"
    assert "timed out" in result.answer
    assert len(provider.calls) == 1


def test_malformed_app_exclusion_fails_closed_before_read(monkeypatch, tmp_path):
    settings = _settings(monkeypatch, tmp_path)
    settings.deep_brain_excluded_apps = ["re:["]
    store = _PagedStore([])
    rag = RAG(store, _FounderProvider(settings))

    with pytest.raises(ValueError, match="excluded-app pattern"):
        rag.answer(FOUNDER_CONTEXT_QUERY)
    assert store.page_calls == 0


def test_store_founder_page_is_keyset_bounded_and_source_filtered(
    monkeypatch, tmp_path
):
    settings = _settings(monkeypatch, tmp_path)
    provider = FakeProvider(settings.embed_dim)
    store = MemoryStore(settings=settings, provider=provider)
    try:
        rows = [
            store.add_observation(
                f"work-{source}",
                source=source,
                ts=100.0,
                app=f"com.example.{source}",
            )
            for source in ("capture", "meeting", "ingest", "mcp", "other")
        ]
        page = store.founder_context_page(0, 200, limit=2)
        assert len(page) == 2
        assert all(item[0].source in {"capture", "meeting", "ingest", "mcp"} for item in page)
        boundary = (page[-1][0].ts, page[-1][0].id)
        second = store.founder_context_page(0, 200, limit=10, before=boundary)
        assert not {item[0].id for item in page} & {item[0].id for item in second}
        assert rows[-1].id not in {item[0].id for item in page + second}
    finally:
        store.close()
