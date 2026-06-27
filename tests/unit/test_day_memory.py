from __future__ import annotations

import datetime as dt
import json

import pytest
from typer.testing import CliRunner

from openbird import cli
from openbird.config import reset_settings_cache
from openbird.day_memory import (
    EXTRACTOR_VERSION,
    build_day_memory,
    build_productivity_report,
    classify_observation,
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
) -> Observation:
    return Observation(
        id=oid,
        content_hash=f"h-{oid}",
        ts=ts,
        app=app,
        window=window,
        url=url,
        session_id=session_id,
        source="capture",
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

    by_category = {item["category"]: item for item in productivity["category_sources"]}
    for category, seconds in metrics["time_by_category"].items():
        assert by_category[category]["active_seconds"] == seconds
    assert by_category["coding"]["source_ids"] == ["c1", "c2", "c3"]


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
    assert built.payload["sessions"][0]["source_ids"] == ["a", "b"]
    assert [token["token"] for token in built.payload["entities"]["title_tokens"][:2]] == [
        "alpha",
        "beta",
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


def test_productivity_cli_rejects_negative_day(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    reset_settings_cache()

    try:
        res = CliRunner().invoke(cli.app, ["productivity", "--day=-1", "--json"])
    finally:
        reset_settings_cache()

    assert res.exit_code == 2
