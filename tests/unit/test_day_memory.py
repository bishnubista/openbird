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
    saved_day_memory_with_day_offset,
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


def test_build_day_memory_terminal_observation_has_no_inferred_active_time():
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
    assert metrics["active_seconds"] == 0
    assert metrics["time_by_category"] == {}
    assert metrics["longest_same_category_streak"] is None
    assert built.payload["focus_blocks"] == []


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
    ]
    assert metrics["longest_same_category_streak"] == {
        "category": "coding",
        "seconds": 120,
    }

    productivity = report["productivity"]
    facts = productivity["facts"]
    assert facts["active_seconds"] == metrics["active_seconds"]
    assert facts["context_switch_count"] == metrics["context_switch_count"]
    assert facts["context_switches_per_active_hour"] == 40.0
    assert facts["top_category"]["category"] == "coding"
    assert facts["top_category"]["seconds"] == metrics["time_by_category"]["coding"]
    assert facts["top_category"]["source_ids"] == ["c1", "c2"]
    assert [ref["session_id"] for ref in facts["top_category"]["session_refs"]] == [
        "s1",
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
    assert by_category["coding"]["source_ids"] == ["c1", "c2"]
    assert [ref["session_id"] for ref in by_category["coding"]["session_refs"]] == [
        "s1",
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
        ),
        (
            _obs(
                "SRC_SECRET_2",
                ts=start + 60,
                app="com.mitchellh.ghostty",
                window="ULTRA_SECRET_WINDOW",
                url="https://example.com/ULTRA_SECRET_URL",
                session_id="s1",
            ),
            "ULTRA_SECRET_TEXT",
        ),
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
            "end": start + 60,
            "source_count": 2,
        }
    ]
    assert "source_ids" not in serialized
    assert "session_refs" not in serialized
    assert "session_id" not in serialized
    assert "s1" not in serialized
    assert "SRC_SECRET" not in serialized
    assert "SRC_SECRET_2" not in serialized
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
        (
            _obs(
                "kept-2",
                ts=start + 60,
                app="com.apple.dt.Xcode",
                window="openbird coding continued",
                session_id="s1",
            ),
            "coding openbird continued",
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

    assert report["productivity"]["coach_ready_packet"]["source_count"] == 2
    assert packet["exclusions"]["kept_observations"] == 2
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
            ),
            (
                _obs(
                    "c2",
                    ts=start + 60,
                    app="com.mitchellh.ghostty",
                    session_id="s1",
                ),
                "coding",
            ),
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
            "end": start + 60,
            "source_count": 2,
        }
    ]
    assert "source_ids" not in provider.messages[1]["content"]
    assert "session_refs" not in provider.messages[1]["content"]
    assert "session_id" not in provider.messages[1]["content"]


def test_productivity_coach_refuses_remote_without_cloud_optin():
    start = _ts(2026, 6, 12, 9)
    report = _coach_report(
        [
            (_obs("c1", ts=start, app="com.mitchellh.ghostty"), "coding"),
            (_obs("c2", ts=start + 60, app="com.mitchellh.ghostty"), "coding"),
        ],
        start=start,
        end=start + 120,
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
        [
            (_obs("c1", ts=start, app="com.mitchellh.ghostty"), "coding"),
            (_obs("c2", ts=start + 60, app="com.mitchellh.ghostty"), "coding"),
        ],
        start=start,
        end=start + 120,
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
        [
            (_obs("c1", ts=start, app="com.mitchellh.ghostty"), "coding"),
            (_obs("c2", ts=start + 60, app="com.mitchellh.ghostty"), "coding"),
        ],
        start=start,
        end=start + 120,
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
        "10:00": 300,
    }
    assert top_hour["hour"] == "10:00"
    assert top_hour["seconds"] == 300
    assert top_hour["minutes"] == 5.0
    assert top_hour["source_ids"] == ["o10a"]
    assert top_hour["source_count"] == 1
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


def test_day_memory_active_seconds_matches_timeline_and_productivity(
    mem_settings, fake_provider
):
    store = MemoryStore(db_path=":memory:", settings=mem_settings, provider=fake_provider)
    try:
        start = _ts(2026, 6, 12, 9)
        store.add_observation(
            "coding one",
            source="capture",
            app="com.mitchellh.ghostty",
            session_id="s1",
            ts=start,
        )
        store.add_observation(
            "coding two",
            source="capture",
            app="com.mitchellh.ghostty",
            session_id="s1",
            ts=start + 60,
        )
        store.add_observation(
            "browser research",
            source="capture",
            app="com.google.Chrome",
            session_id="s2",
            ts=start + 460,
        )

        timeline_active = store.active_seconds(start, start + 900, 300.0)
        saved = store.ensure_day_memory(
            local_date="2026-06-12",
            start_ts=start,
            end_ts=start + 900,
            day_offset=0,
            source_scope="capture",
        )
        productivity_active = build_productivity_report(saved)["productivity"]["facts"][
            "active_seconds"
        ]

        assert timeline_active == 360.0
        assert saved["payload"]["metrics"]["active_seconds"] == timeline_active
        assert productivity_active == timeline_active
    finally:
        store.close()


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
        obs2 = store.add_observation(
            "coding continued",
            source="capture",
            app="com.mitchellh.ghostty",
            session_id="s1",
            ts=start + 60,
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
            source_ids=[obs.id, obs2.id],
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


def test_ensure_day_memory_rebuilds_stale_v5_active_time(
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
        obs2 = store.add_observation(
            "coding continued",
            source="capture",
            app="com.mitchellh.ghostty",
            session_id="s1",
            ts=start + 60,
        )
        rows = store.time_range_text(start, start + 3600, source="capture")
        fingerprint = store.day_memory_source_fingerprint_from_rows(rows)
        store.save_day_memory(
            local_date="2026-06-12",
            source_scope="capture",
            extractor_version="day-memory-v5",
            payload={
                "local_date": "2026-06-12",
                "source_scope": "capture",
                "source_fingerprint": fingerprint,
                "coverage": {
                    "observations": 2,
                    "source_ids": [obs.id, obs2.id],
                },
                "metrics": {"active_seconds": 9999},
            },
            source_ids=[obs.id, obs2.id],
            generated_at=start,
        )

        saved = store.ensure_day_memory(
            local_date="2026-06-12",
            start_ts=start,
            end_ts=start + 3600,
            day_offset=0,
            source_scope="capture",
            force=False,
        )

        assert saved["extractor_version"] == EXTRACTOR_VERSION
        assert saved["payload"]["metrics"]["active_seconds"] == 60
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


class _ReusedStaleOffsetDayMemoryStoreStub(_DayMemoryStoreStub):
    def __init__(self, rows, *, stale_day_offset: int):
        super().__init__(rows)
        self.stale_day_offset = stale_day_offset

    def ensure_day_memory(self, **kwargs):
        from openbird.day_memory import EXTRACTOR_VERSION, build_day_memory

        rows = self.time_range_text(
            kwargs["start_ts"], kwargs["end_ts"], source=kwargs.get("source_scope")
        )
        built = build_day_memory(
            rows,
            start_ts=kwargs["start_ts"],
            end_ts=kwargs["end_ts"],
            day_offset=self.stale_day_offset,
            source_scope=kwargs.get("source_scope", "capture"),
        )
        return self.save_day_memory(
            local_date=kwargs["local_date"],
            source_scope=kwargs.get("source_scope", "capture"),
            extractor_version=EXTRACTOR_VERSION,
            payload=built.payload,
            source_ids=built.source_ids,
        )


def test_productivity_report_day_offset_override_does_not_mutate_saved_payload():
    start = _ts(2026, 6, 12, 9)
    built = build_day_memory(
        [(_obs("o1", ts=start, app="com.mitchellh.ghostty"), "coding")],
        start_ts=start,
        end_ts=start + 60,
        day_offset=0,
    )
    saved = {
        "payload": built.payload,
        "local_date": built.payload["local_date"],
        "source_scope": "capture",
        "source_count": len(built.source_ids),
        "source_ids": built.source_ids,
        "generated_at": start + 120,
        "extractor_version": EXTRACTOR_VERSION,
    }

    display_saved = saved_day_memory_with_day_offset(saved, 1)
    report = build_productivity_report(saved, day_offset=1)

    assert saved["payload"]["day_offset"] == 0
    assert display_saved["payload"]["day_offset"] == 1
    assert report["day_offset"] == 1


def test_day_memory_build_cli_is_no_model_and_json(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    reset_settings_cache()
    today = dt.datetime.now().replace(hour=9, minute=0, second=0, microsecond=0).timestamp()
    rows = [
        (_obs("o1", ts=today, app="com.mitchellh.ghostty", window="openbird"), "coding"),
        (
            _obs(
                "o2",
                ts=today + 60,
                app="com.mitchellh.ghostty",
                window="openbird",
            ),
            "coding continued",
        ),
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
    assert payload["day_memory"]["source_ids"] == ["o1", "o2"]
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


def test_day_memory_show_json_projects_requested_day_offset(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    reset_settings_cache()
    yesterday = (
        dt.datetime.now()
        .replace(hour=9, minute=0, second=0, microsecond=0)
        - dt.timedelta(days=1)
    ).timestamp()
    rows = [
        (_obs("o1", ts=yesterday, app="com.mitchellh.ghostty", window="openbird"), "coding")
    ]
    stub = _ReusedStaleOffsetDayMemoryStoreStub(rows, stale_day_offset=0)
    monkeypatch.setattr(cli, "_store_maintenance", lambda: stub)

    try:
        res = CliRunner().invoke(cli.app, ["day-memory", "show", "--day", "1", "--json"])
    finally:
        reset_settings_cache()

    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert payload["day_memory"]["payload"]["day_offset"] == 1
    assert stub.saved["payload"]["day_offset"] == 0


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
        (
            _obs(
                "o2",
                ts=today + 60,
                app="com.mitchellh.ghostty",
                window="openbird",
            ),
            "coding continued",
        ),
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


def test_productivity_cli_json_projects_requested_day_offset(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    reset_settings_cache()
    yesterday = (
        dt.datetime.now()
        .replace(hour=9, minute=0, second=0, microsecond=0)
        - dt.timedelta(days=1)
    ).timestamp()
    rows = [
        (_obs("o1", ts=yesterday, app="com.mitchellh.ghostty", window="openbird"), "coding")
    ]
    stub = _ReusedStaleOffsetDayMemoryStoreStub(rows, stale_day_offset=0)
    monkeypatch.setattr(cli, "_store_maintenance", lambda: stub)
    monkeypatch.setattr(
        cli,
        "_provider",
        lambda: (_ for _ in ()).throw(AssertionError("productivity must not build provider")),
    )

    try:
        res = CliRunner().invoke(cli.app, ["productivity", "--day", "1", "--json"])
    finally:
        reset_settings_cache()

    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert payload["route"] == "productivity.local_facts"
    assert payload["egress"] == "none"
    assert payload["day_offset"] == 1
    assert stub.saved["payload"]["day_offset"] == 0


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
        (
            _obs(
                "o2",
                ts=today + 60,
                app="com.mitchellh.ghostty",
                window="openbird",
            ),
            "coding continued",
        ),
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
    assert payload["packet"]["exclusions"]["excluded_by"] == {"app": 2}


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
        (
            _obs(
                "o2",
                ts=today + 60,
                app="com.mitchellh.ghostty",
                window="openbird",
            ),
            "coding continued",
        ),
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


def test_productivity_coach_cli_one_off_exclusion_skips_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBIRD_DEEP_BRAIN_ENABLED", "1")
    reset_settings_cache()
    day_start = _ts(2026, 6, 27, 0)
    monkeypatch.setattr(
        cli,
        "_day_window",
        lambda _day_offset: (day_start, day_start + 86400 - 0.000001),
    )
    today = _ts(2026, 6, 27, 9)
    rows = [
        (_obs("o1", ts=today, app="com.openai.codex", source="capture"), "coding"),
    ]
    monkeypatch.setattr(cli, "_store_maintenance", lambda: _DayMemoryStoreStub(rows))
    monkeypatch.setattr(
        cli,
        "_completion_provider",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("excluded productivity coaching must not build provider")
        ),
    )

    try:
        res = CliRunner().invoke(
            cli.app,
            [
                "productivity-coach",
                "coach me",
                "--day",
                "0",
                "--json",
                "--exclude-source",
                "capture",
            ],
        )
    finally:
        reset_settings_cache()

    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert payload["ok"] is True
    assert payload["reasoning_route"] == "local_deterministic"
    assert payload["egress"] == "none"
    assert payload["citations"] == []
    assert payload["exclusions"]["kept_observations"] == 0
    assert payload["exclusions"]["excluded_by"] == {"source": 1}
    assert payload["exclusions"]["excluded_sources_configured"] == ["capture"]


def test_productivity_cli_rejects_negative_day(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    reset_settings_cache()

    try:
        res = CliRunner().invoke(cli.app, ["productivity", "--day=-1", "--json"])
    finally:
        reset_settings_cache()

    assert res.exit_code == 2


# ---------------------------------------------------------------------------
# span_metrics (Phase B): measured time from activity spans
# ---------------------------------------------------------------------------


def _span(sid, start, end, *, bundle="com.apple.mail", afk=0, reason=None, meeting=0):
    return {
        "span_id": sid, "start_ts": start, "end_ts": end,
        "bundle_id": bundle, "afk": afk, "reason": reason, "detail_tier": 1,
        "meeting": meeting,
    }


def test_span_metrics_time_accounting_and_afk():
    from openbird.day_memory import build_day_memory

    spans = [
        _span("s1", 1000.0, 1600.0),                       # 600s mail
        _span("s2", 1600.0, 1900.0, bundle="com.apple.Terminal",
              reason="blocklisted"),                        # 300s coarse
        _span("s3", 1900.0, 2500.0, afk=1),                # 600s AFK
    ]
    built = build_day_memory(
        [], start_ts=0.0, end_ts=100_000.0, day_offset=0, spans=spans
    )
    m = built.payload["span_metrics"]
    assert m["span_time_by_app"]["com.apple.mail"] == 600.0
    assert m["span_time_by_app"]["com.apple.Terminal"] == 300.0
    assert m["span_time_by_reason"]["blocklisted"] == 300.0
    assert m["afk_seconds"] == 600.0
    assert m["active_span_seconds"] == 900.0
    assert m["meeting_seconds"] == 0.0 and m["meeting_count"] == 0
    assert built.span_ids == ["s1", "s2", "s3"]
    assert built.payload["span_fingerprint"]["span_count"] == 3


def test_span_metrics_clip_to_window():
    from openbird.day_memory import build_day_memory

    spans = [_span("s1", 0.0, 200.0)]  # only [100, 200] is inside the window
    built = build_day_memory(
        [], start_ts=100.0, end_ts=1000.0, day_offset=0, spans=spans
    )
    assert built.payload["span_metrics"]["span_time_by_app"]["com.apple.mail"] == 100.0


def test_span_metrics_hour_splitting():
    import datetime as dt

    from openbird.day_memory import build_day_memory

    # A span straddling a local hour boundary splits between the two hours.
    base = dt.datetime(2026, 1, 5, 9, 50, 0)
    start = base.timestamp()
    end = (base + dt.timedelta(minutes=20)).timestamp()
    built = build_day_memory(
        [], start_ts=start - 100, end_ts=end + 100, day_offset=0,
        spans=[_span("s1", start, end)],
    )
    by_hour = built.payload["span_metrics"]["span_time_by_hour"]
    assert by_hour["09:00"] == 600.0
    assert by_hour["10:00"] == 600.0


def test_span_focus_blocks_rules():
    from openbird.day_memory import build_day_memory

    spans = [
        # A 15-minute two-app run with small gaps -> one block.
        _span("s1", 1000.0, 1400.0),
        _span("s2", 1430.0, 1900.0, bundle="com.apple.notes"),
        # A >60s gap breaks the run; the next span alone is too short.
        _span("s3", 2200.0, 2400.0),
    ]
    built = build_day_memory(
        [], start_ts=0.0, end_ts=100_000.0, day_offset=0, spans=spans
    )
    blocks = built.payload["span_metrics"]["span_focus_blocks"]
    assert len(blocks) == 1
    assert blocks[0]["span_ids"] == ["s1", "s2"]
    assert blocks[0]["seconds"] == 900.0


def test_span_fingerprint_changes_on_extension():
    from openbird.day_memory import span_fingerprint_for_spans

    a = [_span("s1", 0.0, 100.0)]
    b = [_span("s1", 0.0, 150.0)]  # same span, extended
    assert span_fingerprint_for_spans(a) != span_fingerprint_for_spans(b)


def test_no_spans_omits_block():
    from openbird.day_memory import build_day_memory

    built = build_day_memory([], start_ts=0.0, end_ts=1.0, day_offset=0)
    assert "span_metrics" not in built.payload
    assert built.span_ids == []


def test_paused_spans_are_not_active_time():
    from openbird.day_memory import build_day_memory

    spans = [
        _span("s1", 1000.0, 1600.0),  # 600s real work
        {"span_id": "s2", "start_ts": 1600.0, "end_ts": 2600.0,
         "bundle_id": None, "afk": 0, "reason": "paused", "detail_tier": 0},
    ]
    built = build_day_memory(
        [], start_ts=0.0, end_ts=100_000.0, day_offset=0, spans=spans
    )
    m = built.payload["span_metrics"]
    assert m["active_span_seconds"] == 600.0  # paused 1000s NOT counted
    assert m["paused_seconds"] == 1000.0
    assert m["span_time_by_reason"]["paused"] == 1000.0
    assert "(untracked)" not in m["span_time_by_app"]
    # Paused time can never form/extend a focus block.
    for block in m["span_focus_blocks"]:
        assert "s2" not in block["span_ids"]
    # Hour buckets carry only the active 600s.
    assert sum(m["span_time_by_hour"].values()) == 600.0


def test_paused_afk_span_counts_as_paused_not_afk():
    from openbird.day_memory import build_day_memory

    spans = [
        {"span_id": "s1", "start_ts": 1000.0, "end_ts": 1500.0,
         "bundle_id": None, "afk": 1, "reason": "paused", "detail_tier": 0},
    ]
    m = build_day_memory(
        [], start_ts=0.0, end_ts=100_000.0, day_offset=0, spans=spans
    ).payload["span_metrics"]
    # Paused dominates: 500s is PAUSED time, never AFK or active.
    assert m["paused_seconds"] == 500.0
    assert m["afk_seconds"] == 0.0
    assert m["active_span_seconds"] == 0.0
    assert m["span_time_by_reason"]["paused"] == 500.0


def test_productivity_report_prefers_span_ground_truth():
    from openbird.day_memory import build_day_memory, build_productivity_report

    spans = [
        _span("s1", 1000.0, 1600.0),
        _span("s2", 1700.0, 2000.0, afk=1),
    ]
    built = build_day_memory(
        [], start_ts=0.0, end_ts=100_000.0, day_offset=0, spans=spans
    )
    saved = {
        "payload": built.payload,
        "local_date": built.payload.get("local_date"),
        "source_scope": "capture",
        "source_count": 0,
        "generated_at": 0.0,
    }
    report = build_productivity_report(saved)
    facts = report["productivity"]["facts"]
    assert facts["duration_basis"] == "spans"
    assert facts["active_seconds"] == 600.0
    assert facts["afk_minutes"] == 5.0
    # Coach packet never carries local span ids.
    packet = report["productivity"]["coach_ready_packet"]
    assert "span_ids" not in json_dumps_all_keys(packet)


def json_dumps_all_keys(value) -> set:
    keys: set = set()

    def walk(v):
        if isinstance(v, dict):
            for k, item in v.items():
                keys.add(k)
                walk(item)
        elif isinstance(v, list):
            for item in v:
                walk(item)

    walk(value)
    return keys


def test_productivity_report_legacy_fallback_without_spans():
    from openbird.day_memory import build_day_memory, build_productivity_report

    built = build_day_memory([], start_ts=0.0, end_ts=1.0, day_offset=0)
    saved = {"payload": built.payload, "local_date": "2026-01-01",
             "source_scope": "capture", "source_count": 0, "generated_at": 0.0}
    facts = build_productivity_report(saved)["productivity"]["facts"]
    assert facts["duration_basis"] == "observations"
    assert "afk_minutes" not in facts


# -- Phase C1: meeting time metrics (day-memory-v9) -----------------------------


def test_span_metrics_meeting_seconds_includes_afk_meeting_spans():
    from openbird.day_memory import build_day_memory

    spans = [
        _span("s1", 1000.0, 1600.0, bundle="us.zoom.xos", meeting=1),
        # Mid-call AFK (on a call you don't type): MUST count as meeting time.
        _span("s2", 1600.0, 2200.0, bundle="us.zoom.xos", afk=1, meeting=1),
        _span("s3", 2200.0, 2500.0),
    ]
    built = build_day_memory(
        [], start_ts=0.0, end_ts=100_000.0, day_offset=0, spans=spans
    )
    m = built.payload["span_metrics"]
    assert m["meeting_seconds"] == 1200.0
    assert m["meeting_count"] == 1  # contiguous split spans are ONE meeting
    assert m["afk_seconds"] == 600.0  # AFK accounting unchanged
    assert m["span_time_by_app"].get("us.zoom.xos") == 600.0  # active only


def test_span_metrics_meeting_runs_merge_across_small_gaps():
    from openbird.day_memory import build_day_memory

    spans = [
        _span("s1", 1000.0, 1600.0, bundle="us.zoom.xos", meeting=1),
        # 120s gap (== the bound): the SAME meeting.
        _span("s2", 1720.0, 2000.0, bundle="us.zoom.xos", meeting=1),
        # 121s gap (> the bound): a NEW meeting.
        _span("s3", 2121.0, 2400.0, bundle="us.zoom.xos", meeting=1),
    ]
    built = build_day_memory(
        [], start_ts=0.0, end_ts=100_000.0, day_offset=0, spans=spans
    )
    m = built.payload["span_metrics"]
    assert m["meeting_count"] == 2
    assert m["meeting_seconds"] == 600.0 + 280.0 + 279.0


def test_v8_cached_day_memory_is_rebuilt_under_v9(mem_settings, fake_provider):
    # The reuse gate compares extractor_version, so the v9 bump (required: a
    # metrics-shape change alone never invalidates cached days) rebuilds.
    store = MemoryStore(
        db_path=":memory:", settings=mem_settings, provider=fake_provider
    )
    try:
        start = _ts(2026, 6, 12, 9)
        obs = store.add_observation(
            "coding",
            source="capture",
            app="com.mitchellh.ghostty",
            session_id="s1",
            ts=start,
        )
        rows = store.time_range_text(start, start + 3600, source="capture")
        fingerprint = store.day_memory_source_fingerprint_from_rows(rows)
        store.save_day_memory(
            local_date="2026-06-12",
            source_scope="capture",
            extractor_version="day-memory-v8",
            payload={
                "local_date": "2026-06-12",
                "source_scope": "capture",
                "source_fingerprint": fingerprint,
                "coverage": {"observations": 1, "source_ids": [obs.id]},
                "metrics": {"active_seconds": 9999},
            },
            source_ids=[obs.id],
            generated_at=start,
        )
        saved = store.ensure_day_memory(
            local_date="2026-06-12",
            start_ts=start,
            end_ts=start + 3600,
            day_offset=0,
            source_scope="capture",
            force=False,
        )
        assert saved["extractor_version"] == "day-memory-v9"
        assert saved["payload"]["metrics"]["active_seconds"] != 9999
    finally:
        store.close()


# -- Phase D: taxonomy-level measured time (introduced in day-memory-v8) --------


def _tspan(sid, start, end, *, bundle="com.apple.mail", host=None, afk=0, reason=None):
    span = _span(sid, start, end, bundle=bundle, afk=afk, reason=reason)
    span["url_host"] = host
    return span


def test_extractor_version_is_v9():
    assert EXTRACTOR_VERSION == "day-memory-v9"


def test_span_time_by_level_host_over_bundle_with_afk_and_paused_excluded():
    from openbird.day_memory import build_day_memory

    taxonomy = {
        "bundle:com.apple.mail": "other_work",
        "bundle:com.google.chrome": "personal",  # host must outrank this
        "host:github.com": "focus_work",
    }
    spans = [
        _tspan("s1", 1000.0, 1600.0),  # 600s mail -> other_work
        _tspan("s2", 1600.0, 1900.0, bundle="com.google.chrome",
               host="github.com"),  # 300s -> focus_work (host over bundle)
        _tspan("s3", 1900.0, 2500.0, afk=1),  # AFK: excluded from level time
        _tspan("s4", 2500.0, 2800.0, bundle=None, reason="paused"),  # excluded
        _tspan("s5", 2800.0, 2900.0, bundle="com.unknown.app"),  # 100s uncategorized
    ]
    built = build_day_memory(
        [], start_ts=0.0, end_ts=100_000.0, day_offset=0, spans=spans,
        taxonomy=taxonomy, taxonomy_fingerprint="fp-1",
    )
    m = built.payload["span_metrics"]
    assert m["span_time_by_level"] == {
        "other_work": 600.0,
        "focus_work": 300.0,
        "uncategorized": 100.0,
    }
    assert built.payload["taxonomy_fingerprint"] == "fp-1"


def test_no_taxonomy_omits_level_block_and_fingerprint():
    from openbird.day_memory import build_day_memory

    built = build_day_memory(
        [], start_ts=0.0, end_ts=10_000.0, day_offset=0,
        spans=[_tspan("s1", 1000.0, 1600.0)],
    )
    assert "span_time_by_level" not in built.payload["span_metrics"]
    assert "uncategorized_identity_seconds" not in built.payload["span_metrics"]
    assert "taxonomy_fingerprint" not in built.payload


def test_uncategorized_identity_seconds_threshold_and_ordering():
    from openbird.day_memory import build_day_memory

    taxonomy = {"bundle:com.apple.mail": "other_work"}
    spans = [
        _tspan("s1", 0.0, 600.0),  # mail: categorized, never queued
        _tspan("s2", 600.0, 800.0, bundle="com.unknown.big"),  # 200s >= 120 queued
        _tspan("s3", 800.0, 900.0, bundle="com.unknown.small"),  # 100s < 120 skipped
        _tspan("s4", 900.0, 1100.0, bundle="com.google.chrome",
               host="mystery.example"),  # both identities 200s, uncategorized
    ]
    built = build_day_memory(
        [], start_ts=0.0, end_ts=10_000.0, day_offset=0, spans=spans,
        taxonomy=taxonomy, taxonomy_fingerprint="fp",
    )
    pending = built.payload["span_metrics"]["uncategorized_identity_seconds"]
    assert pending == {
        "bundle:com.google.chrome": 200.0,
        "bundle:com.unknown.big": 200.0,
        "host:mystery.example": 200.0,
    }


def test_render_prose_includes_descriptive_level_sentence():
    from openbird.day_memory import render_day_memory_prose

    payload = {
        "local_date": "2026-06-13",
        "coverage": {"observations": 4, "sessions": 1},
        "metrics": {"active_seconds": 1200.0},
        "span_metrics": {
            "span_time_by_level": {
                "focus_work": 3600.0,
                "other_work": 600.0,
                "uncategorized": 90.0,
            }
        },
    }
    prose = render_day_memory_prose(payload)
    assert "Measured span time leaned toward focus work (60m), other work (10m)." in prose
    assert "uncategorized" not in prose
    # No score/judgment words — descriptive framing only.
    assert "productivity" not in prose.lower()
    assert "score" not in prose.lower()


def test_render_prose_unchanged_without_level_block():
    from openbird.day_memory import render_day_memory_prose

    payload = {
        "local_date": "2026-06-13",
        "coverage": {"observations": 4, "sessions": 1},
        "metrics": {"active_seconds": 1200.0},
    }
    assert "Measured span time" not in render_day_memory_prose(payload)


def test_ensure_day_memory_taxonomy_fingerprint_freshness(tmp_path):
    """Editing taxonomy.json rebuilds the cached day memory (fingerprint miss)."""
    import json as _json

    from tests.unit.conftest import FakeProvider

    settings = Settings(data_dir=tmp_path, embed_dim=64)
    store = MemoryStore(db_path=":memory:", settings=settings,
                        provider=FakeProvider(embed_dim=64))
    try:
        start = dt.datetime(2026, 6, 13, 0, 0, 0).timestamp()
        end = dt.datetime(2026, 6, 13, 23, 59, 59).timestamp()
        store.open_span(
            epoch_id="e", start_ts=start + 3600, end_ts=start + 4200,
            bundle_id="com.unknown.app", detail_tier=1,
        )
        first = store.ensure_day_memory(
            local_date="2026-06-13", start_ts=start, end_ts=end, day_offset=0
        )
        assert first["extractor_version"] == "day-memory-v9"
        assert first["payload"]["span_metrics"]["span_time_by_level"] == {
            "uncategorized": 600.0
        }
        # Unchanged sources + unchanged taxonomy -> the cached row is reused.
        again = store.ensure_day_memory(
            local_date="2026-06-13", start_ts=start, end_ts=end, day_offset=0
        )
        assert again["id"] == first["id"]

        (tmp_path / "taxonomy.json").write_text(
            _json.dumps({"bundle:com.unknown.app": "personal"})
        )
        rebuilt = store.ensure_day_memory(
            local_date="2026-06-13", start_ts=start, end_ts=end, day_offset=0
        )
        assert rebuilt["id"] != first["id"]
        assert rebuilt["payload"]["span_metrics"]["span_time_by_level"] == {
            "personal": 600.0
        }
        assert (
            rebuilt["payload"]["taxonomy_fingerprint"]
            != first["payload"]["taxonomy_fingerprint"]
        )
    finally:
        store.close()


def test_ensure_day_memory_cache_hit_updates_after_llm_assignment(tmp_path):
    """A newly cached LLM level also invalidates the cached day memory."""
    from tests.unit.conftest import FakeProvider

    settings = Settings(data_dir=tmp_path, embed_dim=64)
    store = MemoryStore(db_path=":memory:", settings=settings,
                        provider=FakeProvider(embed_dim=64))
    try:
        start = dt.datetime(2026, 6, 13, 0, 0, 0).timestamp()
        end = dt.datetime(2026, 6, 13, 23, 59, 59).timestamp()
        store.open_span(
            epoch_id="e", start_ts=start + 3600, end_ts=start + 4200,
            bundle_id="com.unknown.app", detail_tier=1,
        )
        first = store.ensure_day_memory(
            local_date="2026-06-13", start_ts=start, end_ts=end, day_offset=0
        )
        store.save_category_assignment("bundle:com.unknown.app", "focus_work", "m")
        rebuilt = store.ensure_day_memory(
            local_date="2026-06-13", start_ts=start, end_ts=end, day_offset=0
        )
        assert rebuilt["id"] != first["id"]
        assert rebuilt["payload"]["span_metrics"]["span_time_by_level"] == {
            "focus_work": 600.0
        }
    finally:
        store.close()
