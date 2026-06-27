from __future__ import annotations

import datetime as dt
import json

import pytest
from typer.testing import CliRunner

from openbird import cli
from openbird.config import Settings, reset_settings_cache
from openbird.day_memory import (
    EXTRACTOR_VERSION,
    answer_productivity_coach,
    build_day_memory,
    build_productivity_coach_packet,
    build_productivity_coach_report,
    build_productivity_report,
    classify_observation,
    productivity_exclusions_block,
    productivity_coach_packet_json_for_model,
)
from openbird.memory.store import MemoryStore
from openbird.types import Observation


@pytest.fixture(autouse=True)
def _clean_settings_cache():
    yield
    reset_settings_cache()


def _ts(y: int, mo: int, d: int, h: int, mi: int = 0) -> float:
    return dt.datetime(y, mo, d, h, mi).timestamp()


def _obs(
    oid: str,
    *,
    ts: float,
    app: str = "com.google.Chrome",
    window: str | None = None,
    url: str | None = None,
    session_id: str | None = None,
    source: str = "capture",
) -> Observation:
    return Observation(
        id=oid,
        content_hash=f"h-{oid}",
        ts=ts,
        app=app,
        window=window,
        url=url,
        session_id=session_id,
        source=source,
    )


class _Provider:
    llm_model = "stub-model"

    def __init__(self, response):
        self.response = response
        self.messages = None
        self.schema = None

    def complete(self, messages, *, json_schema=None):
        self.messages = messages
        self.schema = json_schema
        return self.response


def _saved_day_memory(rows, *, start: float, end: float, day_offset: int = 0):
    built = build_day_memory(
        rows,
        start_ts=start,
        end_ts=end,
        day_offset=day_offset,
        gap_seconds=300,
    )
    return {
        "payload": built.payload,
        "local_date": built.payload["local_date"],
        "source_scope": "capture",
        "source_count": len(built.source_ids),
        "generated_at": end,
        "extractor_version": EXTRACTOR_VERSION,
    }


def _coach_report(
    rows, *, start: float, end: float, settings: Settings | None = None
) -> dict:
    return build_productivity_coach_report(
        rows,
        start_ts=start,
        end_ts=end,
        day_offset=0,
        source_scope="capture",
        settings=settings or Settings(),
    )


def test_classify_observation_uses_descriptive_categories():
    obs = _obs(
        "yt",
        ts=1.0,
        window="How to debug launchd - YouTube",
        url="https://www.youtube.com/watch?v=abc",
    )
    assert classify_observation(obs)[0] == "browser_media"

    code = _obs("code", ts=2.0, app="com.mitchellh.ghostty", window="openbird")
    assert classify_observation(code)[0] == "coding"


def test_build_day_memory_payload_has_metrics_sources_and_no_narrative_text():
    start = _ts(2026, 6, 12, 9)
    rows = [
        (
            _obs(
                "o1",
                ts=start,
                app="com.mitchellh.ghostty",
                window="openbird git status",
                session_id="s1",
            ),
            "ran tests in openbird",
        ),
        (
            _obs(
                "o2",
                ts=start + 60,
                app="com.google.Chrome",
                window="bishnubista/openbird: Local-first memory - Google Chrome",
                url="https://github.com/bishnubista/openbird",
                session_id="s2",
            ),
            "github repo",
        ),
        (
            _obs(
                "o3",
                ts=start + 360,
                app="com.google.Chrome",
                window="Debugging SwiftUI - YouTube",
                url="https://youtube.com/watch?v=debug",
                session_id="s3",
            ),
            "video title",
        ),
    ]

    built = build_day_memory(
        rows,
        start_ts=start,
        end_ts=start + 3600,
        day_offset=0,
        gap_seconds=300,
    )

    payload = built.payload
    assert built.source_ids == ["o1", "o2", "o3"]
    assert payload["narrative_status"] == "not_persisted"
    assert "narrative" not in payload
    assert payload["coverage"]["observations"] == 3
    assert payload["metrics"]["context_switch_count"] == 2
    assert payload["metrics"]["time_by_category"]["coding"] == 60
    assert payload["metrics"]["time_by_category"]["browser_research"] == 300
    assert any(d["value"] == "github.com" for d in payload["entities"]["domains"])
    assert any(d["value"] == "youtube.com" for d in payload["entities"]["domains"])
    assert payload["workstreams"]
    assert payload["workstreams"][0]["source_ids"]
    assert payload["focus_blocks"]
    assert payload["source_fingerprint"]["count"] == 3


def test_build_day_memory_sessions_include_grounded_bounded_cues():
    start = _ts(2026, 6, 12, 9)
    rows = [
        (
            _obs(
                "repo-source-id",
                ts=start,
                app="com.google.Chrome",
                window="Review github.com/bishnubista/openbird/pull/180",
                url="https://github.com/bishnubista/openbird/pull/180",
                session_id="s1",
            ),
            "todo review route labels in github.com/bishnubista/openbird/pull/180",
        ),
        (
            _obs(
                "docs-source-id",
                ts=start + 60,
                app="com.google.Chrome",
                window="OpenBird local memory notes",
                url="https://docs.example.com/openbird",
                session_id="s1",
            ),
            "follow up on local memory notes",
        ),
    ]

    built = build_day_memory(
        rows,
        start_ts=start,
        end_ts=start + 120,
        day_offset=0,
        gap_seconds=300,
    )
    rebuilt = build_day_memory(
        list(reversed(rows)),
        start_ts=start,
        end_ts=start + 120,
        day_offset=0,
        gap_seconds=300,
    )

    cues = built.payload["sessions"][0]["cues"]
    assert built.payload["sessions"][0]["session_id"] == "s1"
    assert cues == rebuilt.payload["sessions"][0]["cues"]
    assert {
        "value": "github.com",
        "count": 1,
        "source_ids": ["repo-source-id"],
    } in cues["domains"]
    assert any(item["value"] == "bishnubista/openbird" for item in cues["repos"])
    assert any(
        item["kind"] == "github_pr" and item["source_ids"] == ["repo-source-id"]
        for item in cues["open_loops"]
    )
    assert any(
        item["token"] == "memory"
        and item["source_ids"] == ["docs-source-id"]
        for item in cues["title_tokens"]
    )
    assert all(
        len(cues[key]) <= 5 for key in ("domains", "repos", "title_tokens", "open_loops")
    )


def test_build_day_memory_times_terminal_observation_against_window_end():
    start = _ts(2026, 6, 12, 9)
    rows = [
        (
            _obs(
                "o1",
                ts=start,
                app="com.mitchellh.ghostty",
                window="openbird",
                session_id="s1",
            ),
            "coding",
        )
    ]

    built = build_day_memory(
        rows,
        start_ts=start,
        end_ts=start + 120,
        day_offset=0,
        gap_seconds=300,
    )

    metrics = built.payload["metrics"]
    assert metrics["active_seconds"] == 120
    assert metrics["time_by_category"]["coding"] == 120
    assert metrics["longest_same_category_streak"] == {"category": "coding", "seconds": 120}


def test_productivity_focus_blocks_share_canonical_category_metrics():
    start = _ts(2026, 6, 12, 9)
    rows = [
        (
            _obs(
                "c1",
                ts=start,
                app="com.mitchellh.ghostty",
                window="SECRET_CODE_WINDOW",
                session_id="s1",
            ),
            "SECRET_CODE_TEXT",
        ),
        (
            _obs(
                "c2",
                ts=start + 60,
                app="com.mitchellh.ghostty",
                window="more code",
                session_id="s1",
            ),
            "coding",
        ),
        (
            _obs("m1", ts=start + 120, app="Slack", window="SECRET_SLACK", session_id="s2"),
            "follow up",
        ),
        (
            _obs(
                "c3",
                ts=start + 180,
                app="com.mitchellh.ghostty",
                window="return code",
                session_id="s3",
            ),
            "coding",
        ),
    ]

    built = build_day_memory(
        rows,
        start_ts=start,
        end_ts=start + 240,
        day_offset=0,
        gap_seconds=300,
    )
    saved = {
        "payload": built.payload,
        "local_date": built.payload["local_date"],
        "source_scope": "capture",
        "source_count": len(built.source_ids),
        "generated_at": start + 300,
        "extractor_version": EXTRACTOR_VERSION,
    }
    report = build_productivity_report(saved)

    metrics = built.payload["metrics"]
    blocks = built.payload["focus_blocks"]
    assert blocks == [
        {
            "category": "coding",
            "start": start,
            "end": start + 120,
            "seconds": 120,
            "source_ids": ["c1", "c2"],
            "session_count": 1,
        },
        {
            "category": "communication",
            "start": start + 120,
            "end": start + 180,
            "seconds": 60,
            "source_ids": ["m1"],
            "session_count": 1,
        },
        {
            "category": "coding",
            "start": start + 180,
            "end": start + 240,
            "seconds": 60,
            "source_ids": ["c3"],
            "session_count": 1,
        },
    ]
    assert metrics["longest_same_category_streak"] == {
        "category": "coding",
        "seconds": 120,
    }

    productivity = report["productivity"]
    facts = productivity["facts"]
    assert facts["active_seconds"] == metrics["active_seconds"]
    assert facts["context_switch_count"] == metrics["context_switch_count"]
    assert facts["context_switches_per_active_hour"] == 30.0
    assert facts["top_category"]["category"] == "coding"
    assert facts["top_category"]["seconds"] == metrics["time_by_category"]["coding"]
    assert facts["top_category"]["source_ids"] == ["c1", "c2", "c3"]
    assert [ref["session_id"] for ref in facts["top_category"]["session_refs"]] == [
        "s1",
        "s3",
    ]
    assert facts["longest_focus_block"]["session_refs"] == [
        {
            "session_id": "s1",
            "app": "com.mitchellh.ghostty",
            "category": "coding",
            "start": start,
            "end": start + 60,
            "source_count": 2,
        }
    ]

    by_category = {item["category"]: item for item in productivity["category_sources"]}
    for category, seconds in metrics["time_by_category"].items():
        assert by_category[category]["active_seconds"] == seconds
    assert by_category["coding"]["source_ids"] == ["c1", "c2", "c3"]
    assert [ref["session_id"] for ref in by_category["coding"]["session_refs"]] == [
        "s1",
        "s3",
    ]


def test_productivity_report_keeps_raw_text_out_of_output():
    start = _ts(2026, 6, 12, 9)
    rows = [
        (
            _obs(
                "secret",
                ts=start,
                app="com.mitchellh.ghostty",
                window="ULTRA_SECRET_WINDOW",
                url="https://example.com/ULTRA_SECRET_URL",
                session_id="s1",
            ),
            "ULTRA_SECRET_TEXT",
        )
    ]
    built = build_day_memory(rows, start_ts=start, end_ts=start + 60, day_offset=0)
    saved = {
        "payload": built.payload,
        "local_date": built.payload["local_date"],
        "source_scope": "capture",
        "source_count": 1,
        "generated_at": start + 60,
        "extractor_version": EXTRACTOR_VERSION,
    }

    report = build_productivity_report(saved)
    serialized = json.dumps(report, sort_keys=True)

    assert "ULTRA_SECRET_WINDOW" not in serialized
    assert "ULTRA_SECRET_URL" not in serialized
    assert "ULTRA_SECRET_TEXT" not in serialized
    coach_packet = report["productivity"]["coach_ready_packet"]
    assert "source_ids" not in json.dumps(coach_packet, sort_keys=True)
    assert report["memory_context"]["coverage"]["observations"] == 1


def test_productivity_coach_packet_uses_synthetic_ids_and_local_citation_map():
    start = _ts(2026, 6, 12, 9)
    rows = [
        (
            _obs(
                "SRC_SECRET",
                ts=start,
                app="com.mitchellh.ghostty",
                window="ULTRA_SECRET_WINDOW",
                url="https://example.com/ULTRA_SECRET_URL",
                session_id="s1",
            ),
            "ULTRA_SECRET_TEXT",
        )
    ]
    report = build_productivity_report(
        _saved_day_memory(rows, start=start, end=start + 60)
    )

    packet = build_productivity_coach_packet(report)
    serialized = productivity_coach_packet_json_for_model(packet)

    assert packet["citation_count"] > 0
    assert "category:coding" in packet["citation_map"]
    assert "block:1" in packet["citation_map"]
    assert "hour:09:00" in packet["citation_map"]
    assert packet["citation_map"]["category:coding"]["source_ids"] == ["SRC_SECRET"]
    assert packet["citation_map"]["category:coding"]["session_refs"] == [
        {
            "session_id": "s1",
            "app": "com.mitchellh.ghostty",
            "category": "coding",
            "start": start,
            "end": start,
            "source_count": 1,
        }
    ]
    assert "source_ids" not in serialized
    assert "session_refs" not in serialized
    assert "session_id" not in serialized
    assert "s1" not in serialized
    assert "SRC_SECRET" not in serialized
    assert "ULTRA_SECRET_WINDOW" not in serialized
    assert "ULTRA_SECRET_URL" not in serialized
    assert "ULTRA_SECRET_TEXT" not in serialized
    assert packet["model_packet"]["category_sources"][0]["citation_id"] == "category:coding"
    assert packet["model_packet"]["focus_blocks"][0]["citation_id"] == "block:1"
    assert packet["model_packet"]["facts"]["top_hour"]["citation_id"] == "hour:09:00"


def test_productivity_exclusions_block_matches_deep_brain_shape():
    from openbird.deep_brain import filter_rows_for_deep_brain

    rows = [(_obs("c1", ts=_ts(2026, 6, 12, 9)), "coding")]
    _filtered, exclusions = filter_rows_for_deep_brain(rows, settings=Settings())

    assert set(productivity_exclusions_block(exclusions)) == set(exclusions)


def test_productivity_coach_filters_excluded_rows_before_prompt_and_citations():
    start = _ts(2026, 6, 12, 9)
    settings = Settings(
        deep_brain_enabled=True,
        deep_brain_excluded_apps=["com.mitchellh.ghostty"],
        deep_brain_excluded_sources=["private"],
        deep_brain_excluded_observation_ids=["secret-id"],
    )
    rows = [
        (
            _obs(
                "secret-app",
                ts=start,
                app="com.mitchellh.ghostty",
                window="SECRET_APP_WINDOW",
            ),
            "SECRET_APP_TEXT",
        ),
        (
            _obs(
                "secret-source",
                ts=start + 10,
                app="com.apple.dt.Xcode",
                source="private",
                window="SECRET_SOURCE_WINDOW",
            ),
            "SECRET_SOURCE_TEXT",
        ),
        (
            _obs(
                "secret-id",
                ts=start + 20,
                app="com.apple.dt.Xcode",
                window="SECRET_ID_WINDOW",
            ),
            "SECRET_ID_TEXT",
        ),
        (
            _obs(
                "kept",
                ts=start + 30,
                app="com.apple.dt.Xcode",
                window="openbird coding",
                session_id="s1",
            ),
            "coding openbird",
        ),
    ]
    report = _coach_report(rows, start=start, end=start + 90, settings=settings)
    packet = build_productivity_coach_packet(report)
    provider = _Provider(
        {
            "answer": "Your coding block was grounded.",
            "citation_ids": ["category:coding"],
            "confidence": "medium",
        }
    )

    result = answer_productivity_coach("coach me", report, provider, settings=settings)
    rendered = json.dumps(
        {
            "model_packet": packet["model_packet"],
            "citation_map": packet["citation_map"],
            "result": result,
            "prompt": provider.messages[1]["content"],
        },
        sort_keys=True,
    )

    assert report["productivity"]["coach_ready_packet"]["source_count"] == 1
    assert packet["exclusions"]["kept_observations"] == 1
    assert packet["exclusions"]["excluded_by"] == {
        "app": 1,
        "observation_id": 1,
        "source": 1,
    }
    assert packet["citation_count"] > 0
    assert result["citations"][0]["source_ids"] == ["kept"]
    assert result["exclusions"]["excluded_observations"] == 3
    assert "secret-app" not in rendered
    assert "secret-source" not in rendered
    assert "secret-id" not in rendered
    assert "SECRET_APP" not in rendered
    assert "SECRET_SOURCE" not in rendered
    assert "SECRET_ID" not in rendered
    assert "com.mitchellh.ghostty" not in provider.messages[1]["content"]


def test_productivity_coach_all_excluded_rows_never_call_model():
    start = _ts(2026, 6, 12, 9)
    settings = Settings(
        deep_brain_enabled=True,
        deep_brain_excluded_sources=["capture"],
    )
    report = _coach_report(
        [(_obs("c1", ts=start, app="com.apple.dt.Xcode"), "coding")],
        start=start,
        end=start + 60,
        settings=settings,
    )
    provider = _Provider(
        {
            "answer": "should not run",
            "citation_ids": ["category:coding"],
            "confidence": "high",
        }
    )

    result = answer_productivity_coach("coach me", report, provider, settings=settings)

    assert result["confidence"] == "insufficient_evidence"
    assert result["citations"] == []
    assert result["exclusions"]["kept_observations"] == 0
    assert result["exclusions"]["excluded_by"] == {"source": 1}
    assert provider.messages is None


def test_productivity_coach_refuses_report_without_exclusions_sidecar():
    start = _ts(2026, 6, 12, 9)
    report = build_productivity_report(
        _saved_day_memory(
            [(_obs("c1", ts=start, app="com.apple.dt.Xcode"), "coding")],
            start=start,
            end=start + 60,
        )
    )
    provider = _Provider(
        {
            "answer": "should not run",
            "citation_ids": ["category:coding"],
            "confidence": "medium",
        }
    )

    result = answer_productivity_coach(
        "coach me", report, provider, settings=Settings(deep_brain_enabled=True)
    )

    assert result["ok"] is False
    assert "not prepared with exclusions" in " ".join(result["blocked_reasons"])
    assert provider.messages is None


def test_productivity_coach_local_model_needs_feature_gate_only():
    start = _ts(2026, 6, 12, 9)
    report = _coach_report(
        [
            (
                _obs(
                    "c1",
                    ts=start,
                    app="com.mitchellh.ghostty",
                    session_id="s1",
                ),
                "coding",
            )
        ],
        start=start,
        end=start + 60,
    )
    provider = _Provider(
        {
            "answer": "You had a focused coding block.",
            "citation_ids": ["category:coding"],
            "confidence": "medium",
        }
    )

    result = answer_productivity_coach(
        "how did I do?",
        report,
        provider,
        settings=Settings(deep_brain_enabled=True),
    )

    assert result["ok"] is True
    assert result["reasoning_route"] == "local_model"
    assert result["egress"] == "none"
    assert result["citations"][0]["citation_id"] == "category:coding"
    assert result["citations"][0]["source_ids"] == ["c1"]
    assert result["citations"][0]["session_refs"] == [
        {
            "session_id": "s1",
            "app": "com.mitchellh.ghostty",
            "category": "coding",
            "start": start,
            "end": start,
            "source_count": 1,
        }
    ]
    assert "source_ids" not in provider.messages[1]["content"]
    assert "session_refs" not in provider.messages[1]["content"]
    assert "session_id" not in provider.messages[1]["content"]


def test_productivity_coach_refuses_remote_without_cloud_optin():
    start = _ts(2026, 6, 12, 9)
    report = _coach_report(
        [(_obs("c1", ts=start, app="com.mitchellh.ghostty"), "coding")],
        start=start,
        end=start + 60,
    )
    provider = _Provider(
        {"answer": "should not run", "citation_ids": ["category:coding"], "confidence": "low"}
    )

    result = answer_productivity_coach(
        "coach me",
        report,
        provider,
        settings=Settings(deep_brain_enabled=True, llm_model="gpt-4o-mini"),
    )

    assert result["ok"] is False
    assert "OPENBIRD_ALLOW_CLOUD" in " ".join(result["blocked_reasons"])
    assert provider.messages is None


def test_productivity_coach_remote_route_truth_with_both_gates():
    start = _ts(2026, 6, 12, 9)
    report = _coach_report(
        [(_obs("c1", ts=start, app="com.mitchellh.ghostty"), "coding")],
        start=start,
        end=start + 60,
    )
    provider = _Provider(
        {
            "answer": "Your coding block was a strength.",
            "citation_ids": ["block:1"],
            "confidence": "medium",
        }
    )

    result = answer_productivity_coach(
        "coach me",
        report,
        provider,
        settings=Settings(
            deep_brain_enabled=True,
            allow_cloud=True,
            llm_model="gpt-4o-mini",
        ),
    )

    assert result["reasoning_route"] == "cloud_reasoning_active"
    assert result["egress"] == "active_model_route"
    assert result["model"] == "stub-model"
    assert result["citations"][0]["citation_id"] == "block:1"


def test_productivity_coach_drops_hallucinated_citations():
    start = _ts(2026, 6, 12, 9)
    report = _coach_report(
        [(_obs("c1", ts=start, app="com.mitchellh.ghostty"), "coding")],
        start=start,
        end=start + 60,
    )
    provider = _Provider(
        {"answer": "Unsupported coaching.", "citation_ids": ["block:99"], "confidence": "high"}
    )

    result = answer_productivity_coach(
        "coach me",
        report,
        provider,
        settings=Settings(deep_brain_enabled=True),
    )

    assert result["answer"] == "I could not ground productivity coaching in the local facts packet."
    assert result["confidence"] == "insufficient_evidence"
    assert result["citations"] == []


def test_productivity_coach_empty_packet_never_calls_model():
    start = _ts(2026, 6, 12, 0)
    report = _coach_report([], start=start, end=start + 3600)
    provider = _Provider(
        {"answer": "Unsupported coaching.", "citation_ids": [], "confidence": "high"}
    )

    result = answer_productivity_coach(
        "coach me",
        report,
        provider,
        settings=Settings(deep_brain_enabled=True),
    )

    assert result["answer"] == "I do not have enough cited productivity evidence to coach on that."
    assert result["confidence"] == "insufficient_evidence"
    assert result["grounded"] is False
    assert result["egress"] == "none"
    assert provider.messages is None


def test_productivity_coach_uncitable_facts_never_call_model():
    start = _ts(2026, 6, 12, 9)
    report = _coach_report(
        [(_obs("c1", ts=start, app="com.mitchellh.ghostty"), "coding")],
        start=start,
        end=start + 60,
    )
    report["productivity"]["coach_ready_packet"]["category_sources"] = []
    report["productivity"]["coach_ready_packet"]["focus_blocks"] = []
    report["productivity"]["coach_ready_packet"]["facts"]["top_hour"] = None
    provider = _Provider(
        {"answer": "Unsupported coaching.", "citation_ids": [], "confidence": "high"}
    )

    result = answer_productivity_coach(
        "coach me",
        report,
        provider,
        settings=Settings(deep_brain_enabled=True),
    )

    assert result["reasoning_route"] == "local_deterministic"
    assert result["citations"] == []
    assert provider.messages is None


def test_productivity_top_hour_sources_use_observation_hour_bucket():
    start = _ts(2026, 6, 12, 9, 59)
    rows = [
        (
            _obs(
                "o9",
                ts=start,
                app="com.mitchellh.ghostty",
                session_id="s1",
            ),
            "coding",
        ),
        (
            _obs(
                "o10a",
                ts=start + 60,
                app="com.mitchellh.ghostty",
                session_id="s1",
            ),
            "coding",
        ),
        (
            _obs(
                "o10b",
                ts=start + 360,
                app="com.mitchellh.ghostty",
                session_id="s1",
            ),
            "coding",
        ),
    ]
    built = build_day_memory(
        rows,
        start_ts=_ts(2026, 6, 12, 0),
        end_ts=start + 660,
        day_offset=0,
        gap_seconds=300,
    )
    saved = {
        "payload": built.payload,
        "local_date": built.payload["local_date"],
        "source_scope": "capture",
        "source_count": len(built.source_ids),
        "generated_at": start + 660,
        "extractor_version": EXTRACTOR_VERSION,
    }

    top_hour = build_productivity_report(saved)["productivity"]["facts"]["top_hour"]

    assert built.payload["metrics"]["time_by_hour"] == {
        "09:00": 60,
        "10:00": 600,
    }
    assert top_hour["hour"] == "10:00"
    assert top_hour["seconds"] == 600
    assert top_hour["minutes"] == 10.0
    assert top_hour["source_ids"] == ["o10a", "o10b"]
    assert top_hour["source_count"] == 2
    assert top_hour["session_refs"] == [
        {
            "session_id": "s1",
            "app": "com.mitchellh.ghostty",
            "category": "coding",
            "start": start,
            "end": start + 360,
            "source_count": 3,
        }
    ]


def test_productivity_empty_day_is_zero_safe():
    built = build_day_memory(
        [],
        start_ts=_ts(2026, 6, 12, 0),
        end_ts=_ts(2026, 6, 13, 0),
        day_offset=0,
    )
    saved = {
        "payload": built.payload,
        "local_date": built.payload["local_date"],
        "source_scope": "capture",
        "source_count": 0,
        "generated_at": _ts(2026, 6, 12, 1),
        "extractor_version": EXTRACTOR_VERSION,
    }

    report = build_productivity_report(saved)

    assert report["route"] == "productivity.local_facts"
    assert report["egress"] == "none"
    facts = report["productivity"]["facts"]
    assert facts["active_seconds"] == 0
    assert facts["context_switches_per_active_hour"] == 0.0
    assert facts["top_category"] is None
    assert report["productivity"]["focus_blocks"] == []


def test_build_day_memory_uses_stable_tie_breakers():
    start = _ts(2026, 6, 12, 9)
    rows = [
        (
            _obs(
                "b",
                ts=start,
                app="com.google.Chrome",
                window="Beta Alpha",
                url="https://example.org/b",
                session_id="s",
            ),
            "Beta token",
        ),
        (
            _obs(
                "a",
                ts=start,
                app="com.google.Chrome",
                window="Alpha Beta",
                url="https://example.org/a",
                session_id="s",
            ),
            "Alpha token",
        ),
    ]

    built = build_day_memory(
        rows,
        start_ts=start,
        end_ts=start + 60,
        day_offset=0,
        gap_seconds=300,
    )

    assert built.source_ids == ["a", "b"]
    assert built.payload["sessions"][0]["session_id"] == "s"
    assert built.payload["sessions"][0]["source_ids"] == ["a", "b"]
    assert [token["token"] for token in built.payload["entities"]["title_tokens"][:2]] == [
        "alpha",
        "beta",
    ]


def test_build_day_memory_legacy_null_session_id_stays_none():
    start = _ts(2026, 6, 12, 9)
    built = build_day_memory(
        [(_obs("legacy", ts=start, app="com.apple.finder"), "legacy row")],
        start_ts=start,
        end_ts=start + 60,
        day_offset=0,
        gap_seconds=300,
    )

    assert built.payload["sessions"][0]["session_id"] is None
    assert built.payload["sessions"][0]["source_ids"] == ["legacy"]


def test_build_day_memory_namespaces_legacy_bucket_from_real_session_id():
    start = _ts(2026, 6, 12, 9)
    built = build_day_memory(
        [
            (
                _obs(
                    "shared-id",
                    ts=start,
                    app="com.google.Chrome",
                    session_id=None,
                ),
                "legacy observation",
            ),
            (
                _obs(
                    "real-source",
                    ts=start + 1,
                    app="com.google.Chrome",
                    session_id="shared-id",
                ),
                "real session observation",
            ),
        ],
        start_ts=start,
        end_ts=start + 60,
        day_offset=0,
        gap_seconds=300,
    )

    sessions = sorted(
        built.payload["sessions"], key=lambda item: item["source_ids"][0]
    )
    assert [
        (session["session_id"], session["source_ids"])
        for session in sessions
    ] == [
        ("shared-id", ["real-source"]),
        (None, ["shared-id"]),
    ]


def test_build_day_memory_extracts_open_loop_cues_without_narrative():
    start = _ts(2026, 6, 12, 9)
    rows = [
        (
            _obs(
                "issue",
                ts=start,
                app="com.google.Chrome",
                window="Fix capture race · Issue #42 · bishnubista/openbird",
                url="https://github.com/bishnubista/openbird/issues/42",
            ),
            "TODO follow up on regression test",
        )
    ]

    built = build_day_memory(
        rows,
        start_ts=start,
        end_ts=start + 60,
        day_offset=0,
    )

    payload = built.payload
    assert "narrative" not in payload
    assert payload["open_loops"][0]["kind"] == "github_issue"
    assert payload["open_loops"][0]["source_ids"] == ["issue"]


def test_build_day_memory_dedupes_github_open_loop_by_canonical_cue():
    start = _ts(2026, 6, 12, 9)
    rows = [
        (
            _obs(
                "a",
                ts=start,
                window="Old title · Issue #42",
                url="https://github.com/bishnubista/openbird/issues/42",
            ),
            "review",
        ),
        (
            _obs(
                "b",
                ts=start + 10,
                window="New title · Issue #42",
                url="https://github.com/bishnubista/openbird/issues/42",
            ),
            "fix",
        ),
    ]

    built = build_day_memory(rows, start_ts=start, end_ts=start + 60, day_offset=0)

    assert len(built.payload["open_loops"]) == 1
    assert built.payload["open_loops"][0]["source_count"] == 2
    assert built.payload["open_loops"][0]["source_ids"] == ["a", "b"]


def test_ensure_day_memory_rebuilds_v2_payload_for_productivity(
    mem_settings, fake_provider
):
    store = MemoryStore(db_path=":memory:", settings=mem_settings, provider=fake_provider)
    try:
        start = _ts(2026, 6, 12, 9)
        obs = store.add_observation(
            "coding",
            source="capture",
            app="com.mitchellh.ghostty",
            session_id="s1",
            ts=start,
        )
        rows = store.time_range_text(start, start + 60, source="capture")
        fingerprint = store.day_memory_source_fingerprint_from_rows(rows)
        store.save_day_memory(
            local_date="2026-06-12",
            source_scope="capture",
            extractor_version="day-memory-v2",
            payload={
                "local_date": "2026-06-12",
                "source_scope": "capture",
                "source_fingerprint": fingerprint,
                "metrics": {},
            },
            source_ids=[obs.id],
            generated_at=start,
        )

        saved = store.ensure_day_memory(
            local_date="2026-06-12",
            start_ts=start,
            end_ts=start + 60,
            day_offset=0,
            source_scope="capture",
            force=False,
        )

        assert saved["extractor_version"] == EXTRACTOR_VERSION
        assert saved["payload"]["focus_blocks"]
        assert saved["payload"]["sessions"][0]["cues"]
        assert saved["payload"]["sessions"][0]["session_id"] == "s1"
    finally:
        store.close()


class _DayMemoryStoreStub:
    def __init__(self, rows):
        self.rows = rows
        self.saved = None

    def time_range_text(self, start, end, *, source=None):
        return [(o, t) for o, t in self.rows if start <= o.ts <= end]

    def save_day_memory(self, **kwargs):
        self.saved = kwargs
        return {
            "id": "dm1",
            "local_date": kwargs["local_date"],
            "source_scope": kwargs["source_scope"],
            "extractor_version": kwargs["extractor_version"],
            "generated_at": 123.0,
            "source_count": len(kwargs["source_ids"]),
            "source_ids": kwargs["source_ids"],
            "payload": kwargs["payload"],
        }

    def get_day_memory(self, *, local_date, source_scope="capture"):
        if self.saved is None:
            return None
        return self.save_day_memory(**self.saved)

    def ensure_day_memory(self, **kwargs):
        from openbird.day_memory import EXTRACTOR_VERSION, build_day_memory

        rows = self.time_range_text(
            kwargs["start_ts"], kwargs["end_ts"], source=kwargs.get("source_scope")
        )
        built = build_day_memory(
            rows,
            start_ts=kwargs["start_ts"],
            end_ts=kwargs["end_ts"],
            day_offset=kwargs["day_offset"],
            source_scope=kwargs.get("source_scope", "capture"),
        )
        return self.save_day_memory(
            local_date=kwargs["local_date"],
            source_scope=kwargs.get("source_scope", "capture"),
            extractor_version=EXTRACTOR_VERSION,
            payload=built.payload,
            source_ids=built.source_ids,
        )

    def close(self):
        pass


def test_day_memory_build_cli_is_no_model_and_json(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    reset_settings_cache()
    today = dt.datetime.now().replace(hour=9, minute=0, second=0, microsecond=0).timestamp()
    rows = [
        (_obs("o1", ts=today, app="com.mitchellh.ghostty", window="openbird"), "coding"),
    ]
    stub = _DayMemoryStoreStub(rows)
    monkeypatch.setattr(cli, "_store_maintenance", lambda: stub)
    monkeypatch.setattr(
        cli,
        "_provider",
        lambda: (_ for _ in ()).throw(AssertionError("day-memory must not build provider")),
    )

    res = CliRunner().invoke(cli.app, ["day-memory", "build", "--day", "0", "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert payload["built"] is True
    assert payload["day_memory"]["source_ids"] == ["o1"]
    assert payload["day_memory"]["payload"]["narrative_status"] == "not_persisted"


def test_day_memory_show_json_ensures_empty_day(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    reset_settings_cache()
    stub = _DayMemoryStoreStub([])
    monkeypatch.setattr(cli, "_store_maintenance", lambda: stub)

    try:
        res = CliRunner().invoke(cli.app, ["day-memory", "show", "--day", "0", "--json"])
    finally:
        reset_settings_cache()

    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert payload["built"] is True
    assert payload["day_memory"]["payload"]["coverage"]["observations"] == 0


def test_productivity_cli_json_uses_local_route(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    reset_settings_cache()
    today = (
        dt.datetime.now()
        .replace(hour=9, minute=0, second=0, microsecond=0)
        .timestamp()
    )
    rows = [
        (_obs("o1", ts=today, app="com.mitchellh.ghostty", window="openbird"), "coding"),
    ]
    stub = _DayMemoryStoreStub(rows)
    monkeypatch.setattr(cli, "_store_maintenance", lambda: stub)
    monkeypatch.setattr(
        cli,
        "_provider",
        lambda: (_ for _ in ()).throw(AssertionError("productivity must not build provider")),
    )

    try:
        res = CliRunner().invoke(cli.app, ["productivity", "--day", "0", "--json"])
    finally:
        reset_settings_cache()

    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert payload["route"] == "productivity.local_facts"
    assert payload["egress"] == "none"
    assert payload["productivity_status"] == "local_facts_only"
    assert payload["memory_context"]["route"] == "local_deterministic"
    assert "source_ids" not in payload["memory_context"]["coverage"]
    assert payload["productivity"]["focus_blocks"][0]["source_ids"] == ["o1"]
    assert "source_ids" not in json.dumps(
        payload["productivity"]["coach_ready_packet"],
        sort_keys=True,
    )


def test_productivity_coach_cli_refuses_before_provider_without_feature_gate(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBIRD_DEEP_BRAIN_EXCLUDED_APPS", "com.mitchellh.ghostty")
    reset_settings_cache()
    today = (
        dt.datetime.now()
        .replace(hour=9, minute=0, second=0, microsecond=0)
        .timestamp()
    )
    rows = [
        (_obs("o1", ts=today, app="com.mitchellh.ghostty", window="openbird"), "coding"),
    ]
    monkeypatch.setattr(cli, "_store_maintenance", lambda: _DayMemoryStoreStub(rows))
    monkeypatch.setattr(
        cli,
        "_completion_provider",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("refusal must not use provider")
        ),
    )

    try:
        res = CliRunner().invoke(
            cli.app,
            ["productivity-coach", "coach me", "--day", "0", "--json"],
        )
    finally:
        reset_settings_cache()

    assert res.exit_code == 2
    payload = json.loads(res.stdout)
    assert payload["ok"] is False
    assert "OPENBIRD_DEEP_BRAIN_ENABLED" in " ".join(payload["blocked_reasons"])
    assert payload["packet"]["citation_count"] == 0
    assert payload["packet"]["exclusions"]["kept_observations"] == 0
    assert payload["packet"]["exclusions"]["excluded_by"] == {"app": 1}


def test_productivity_coach_cli_uses_packet_label_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBIRD_DEEP_BRAIN_ENABLED", "1")
    reset_settings_cache()
    today = (
        dt.datetime.now()
        .replace(hour=9, minute=0, second=0, microsecond=0)
        .timestamp()
    )
    rows = [
        (_obs("o1", ts=today, app="com.mitchellh.ghostty", window="openbird"), "coding"),
    ]
    provider = _Provider(
        {
            "answer": "Your coding block was a useful focus anchor.",
            "citation_ids": ["category:coding"],
            "confidence": "medium",
        }
    )
    packet_labels: list[str] = []

    def _fake_completion_provider(*, packet_label: str = "Deep Brain packet"):
        packet_labels.append(packet_label)
        return provider

    monkeypatch.setattr(cli, "_store_maintenance", lambda: _DayMemoryStoreStub(rows))
    monkeypatch.setattr(cli, "_completion_provider", _fake_completion_provider)

    try:
        res = CliRunner().invoke(
            cli.app,
            ["productivity-coach", "--day", "0", "--json", "--stdin"],
            input="how did I do?",
        )
    finally:
        reset_settings_cache()

    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert packet_labels == ["productivity coaching packet"]
    assert payload["ok"] is True
    assert payload["packet_route"] == "productivity.coach_packet"
    assert payload["reasoning_route"] == "local_model"
    assert payload["egress"] == "none"
    assert payload["citations"][0]["citation_id"] == "category:coding"
    assert payload["citations"][0]["source_ids"] == ["o1"]
    assert "source_ids" not in provider.messages[1]["content"]


def test_productivity_cli_rejects_negative_day(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    reset_settings_cache()

    try:
        res = CliRunner().invoke(cli.app, ["productivity", "--day=-1", "--json"])
    finally:
        reset_settings_cache()

    assert res.exit_code == 2
