from __future__ import annotations

import datetime as dt
import json
import logging

from typer.testing import CliRunner

from openbird import cli
from openbird.config import Settings, reset_settings_cache
from openbird.deep_brain import (
    PACKET_BUILD_ROUTE_DETERMINISTIC,
    answer_deep_brain,
    build_deep_brain_messages,
    build_deep_brain_period_preview,
    build_deep_brain_preview,
    packet_json_for_model,
)
from openbird.prompts import registry as _prompt_registry
from openbird.types import Observation


def _day(y: int, mo: int, d: int, h: int = 0, mi: int = 0) -> float:
    return dt.datetime(y, mo, d, h, mi, 0).timestamp()


def _obs(
    id_: str,
    *,
    h: str,
    ts: float,
    app: str | None = "Code",
    window: str | None = None,
    url: str | None = None,
    source: str = "capture",
) -> Observation:
    return Observation(
        id=id_,
        content_hash=h,
        ts=ts,
        app=app,
        window=window,
        url=url,
        session_id=id_,
        source=source,
    )


def _has_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_has_key(v, key) for v in value.values())
    if isinstance(value, list):
        return any(_has_key(v, key) for v in value)
    return False


def _window(y: int, mo: int, d: int, *, offset: int) -> dict:
    start = _day(y, mo, d)
    return {
        "start": start,
        "end": start + 86400 - 0.000001,
        "day_offset": offset,
        "local_date": dt.date(y, mo, d).isoformat(),
    }


def test_preview_builds_locally_with_cloud_off_and_no_source_ids(tmp_path):
    start = _day(2026, 6, 12, 9)
    rows = [
        (
            _obs(
                "leaky-session-source-id",
                h="h1",
                ts=start,
                window="Review github.com/bishnubista/openbird/pull/180",
                url="https://github.com/bishnubista/openbird/pull/180",
            ),
            "todo review github.com/bishnubista/openbird/pull/180",
        )
    ]
    settings = Settings(data_dir=tmp_path)

    packet = build_deep_brain_preview(
        rows,
        start_ts=start,
        end_ts=start + 3600,
        day_offset=0,
        source_scope="capture",
        settings=settings,
    )

    assert packet["route"] == "deep_brain.preview"
    assert packet["packet_build_route"] == PACKET_BUILD_ROUTE_DETERMINISTIC
    assert packet["egress"] == "none_preview"
    assert packet["cloud_ready"] is False
    assert "OPENBIRD_ALLOW_CLOUD" in " ".join(packet["blocked_reasons"])
    assert "OPENBIRD_DEEP_BRAIN_ENABLED" in " ".join(packet["blocked_reasons"])
    assert _has_key(packet["memory_summary"], "source_ids") is False
    rendered_summary = json.dumps(packet["memory_summary"], sort_keys=True)
    assert "leaky-session-source-id" not in rendered_summary
    assert "source_fingerprint" not in packet["memory_summary"]
    cues = packet["memory_summary"]["sessions"][0]["cues"]
    assert {"value": "github.com", "count": 1} in cues["domains"]
    assert {"value": "bishnubista/openbird", "count": 1} in cues["repos"]
    assert any(item == {"token": "bishnubista", "count": 3} for item in cues["title_tokens"])
    assert cues["open_loops"][0]["kind"] == "github_pr"
    assert "source_count" in cues["open_loops"][0]


def test_period_preview_days_one_matches_existing_preview(tmp_path):
    start = _day(2026, 6, 12, 9)
    rows = [(_obs("o1", h="h1", ts=start, window="openbird issue"), "fix issue")]
    settings = Settings(data_dir=tmp_path)

    existing = build_deep_brain_preview(
        rows,
        start_ts=_day(2026, 6, 12),
        end_ts=_day(2026, 6, 13) - 0.000001,
        day_offset=0,
        source_scope="capture",
        settings=settings,
    )
    period = build_deep_brain_period_preview(
        [rows],
        day_windows=[_window(2026, 6, 12, offset=0)],
        day_offset=0,
        days=1,
        source_scope="capture",
        settings=settings,
    )

    assert period == existing
    assert period["packet_build_route"] == PACKET_BUILD_ROUTE_DETERMINISTIC


def test_period_preview_compacts_days_and_keeps_oldest_day_grounded(tmp_path):
    old = _day(2026, 6, 10, 9)
    new = _day(2026, 6, 11, 9)
    old_rows = [(_obs("old-anchor", h="old", ts=old, window="old notes"), "old notes")]
    new_rows = [
        (
            _obs(f"new-{idx}", h=f"new-{idx}", ts=new + idx, window=f"new {idx}"),
            f"new notes {idx}",
        )
        for idx in range(20)
    ]
    settings = Settings(data_dir=tmp_path, deep_brain_enabled=True)

    packet = build_deep_brain_period_preview(
        [old_rows, new_rows],
        day_windows=[
            _window(2026, 6, 10, offset=1),
            _window(2026, 6, 11, offset=0),
        ],
        day_offset=0,
        days=2,
        source_scope="capture",
        settings=settings,
    )

    assert packet["period"]["days"] == 2
    assert packet["packet_build_route"] == PACKET_BUILD_ROUTE_DETERMINISTIC
    assert [item["local_date"] for item in packet["memory_summaries"]] == [
        "2026-06-10",
        "2026-06-11",
    ]
    rendered = json.dumps(packet, sort_keys=True)
    assert '"source_ids"' not in rendered
    assert "source_fingerprint" not in rendered
    assert all("sessions" not in item for item in packet["memory_summaries"])
    assert "time_by_hour" not in json.dumps(packet["memory_summaries"], sort_keys=True)
    assert "time_by_hour" not in json.dumps(packet["memory_summary"], sort_keys=True)
    assert packet["memory_summary"]["coverage"]["observations"] == 21
    assert packet["memory_summary"]["coverage"]["sessions"] == 21
    assert packet["memory_summary"]["coverage"]["active_day_count"] == 2
    assert packet["memory_summary"]["coverage"]["apps_active_day_sum"] == 2
    assert packet["selected_sources"][0]["observation_id"] == "old-anchor"
    assert packet["selected_sources"][0]["local_date"] == "2026-06-10"
    assert len([s for s in packet["selected_sources"] if s["local_date"] == "2026-06-11"]) == 12

    provider = _Provider(
        {
            "answer": "The older day and newer day both have anchors.",
            "citation_ids": ["old-anchor", "new-19"],
            "confidence": "medium",
        }
    )
    result = answer_deep_brain("what changed?", packet, provider, settings=settings)
    assert result["grounded"] is True
    assert [item["observation_id"] for item in result["citations"]] == [
        "old-anchor",
        "new-19",
    ]


def test_period_preview_applies_exclusions_per_day_without_config_duplication(tmp_path):
    old = _day(2026, 6, 10, 9)
    new = _day(2026, 6, 11, 9)
    settings = Settings(
        data_dir=tmp_path,
        deep_brain_excluded_apps=["com.mitchellh.ghostty"],
        deep_brain_excluded_sources=["private"],
    )

    packet = build_deep_brain_period_preview(
        [
            [
                (
                    _obs(
                        "old-secret",
                        h="s1",
                        ts=old,
                        app="com.mitchellh.ghostty",
                        window="SECRET_OLD_WINDOW",
                    ),
                    "SECRET_OLD_TEXT",
                ),
                (_obs("old-kept", h="k1", ts=old + 1, window="old kept"), "old kept"),
            ],
            [
                (
                    _obs(
                        "new-secret",
                        h="s2",
                        ts=new,
                        source="private",
                        window="SECRET_NEW_WINDOW",
                    ),
                    "SECRET_NEW_TEXT",
                ),
                (_obs("new-kept", h="k2", ts=new + 1, window="new kept"), "new kept"),
            ],
        ],
        day_windows=[
            _window(2026, 6, 10, offset=1),
            _window(2026, 6, 11, offset=0),
        ],
        day_offset=0,
        days=2,
        source_scope="capture",
        settings=settings,
    )

    rendered = json.dumps(packet, sort_keys=True)
    assert "SECRET_OLD_WINDOW" not in rendered
    assert "SECRET_OLD_TEXT" not in rendered
    assert "SECRET_NEW_WINDOW" not in rendered
    assert "SECRET_NEW_TEXT" not in rendered
    assert packet["exclusions"]["excluded_by"] == {"app": 1, "source": 1}
    assert packet["exclusions"]["excluded_apps_configured"] == ["com.mitchellh.ghostty"]
    assert packet["exclusions"]["excluded_sources_configured"] == ["private"]


def test_period_empty_packet_never_calls_model(tmp_path):
    settings = Settings(data_dir=tmp_path, deep_brain_enabled=True)
    packet = build_deep_brain_period_preview(
        [[], []],
        day_windows=[
            _window(2026, 6, 10, offset=1),
            _window(2026, 6, 11, offset=0),
        ],
        day_offset=0,
        days=2,
        source_scope="capture",
        settings=settings,
    )
    provider = _Provider({"answer": "unsupported", "citation_ids": [], "confidence": "high"})

    result = answer_deep_brain("summarize", packet, provider, settings=settings)

    assert packet["memory_summary"]["coverage"]["observations"] == 0
    assert packet["memory_summary"]["metrics"]["first_seen"] is None
    assert packet["memory_summary"]["metrics"]["last_seen"] is None
    assert result["confidence"] == "insufficient_evidence"
    assert provider.messages is None


def test_preview_cloud_ready_requires_both_opt_ins(tmp_path):
    start = _day(2026, 6, 12, 9)
    rows = [(_obs("o1", h="h1", ts=start), "notes")]

    packet = build_deep_brain_preview(
        rows,
        start_ts=start,
        end_ts=start + 60,
        day_offset=0,
        source_scope="capture",
        settings=Settings(data_dir=tmp_path, allow_cloud=True, deep_brain_enabled=True),
    )

    assert packet["cloud_ready"] is True
    assert packet["blocked_reasons"] == []


def test_exclusions_apply_before_distillation_and_sources(tmp_path):
    start = _day(2026, 6, 12, 9)
    rows = [
        (
            _obs(
                "secret-terminal",
                h="h1",
                ts=start,
                app="com.mitchellh.ghostty",
                window="SECRET_TERMINAL_FOLLOWUP github.com/acme/private-repo issue #7",
            ),
            "todo SECRET_TERMINAL_FOLLOWUP github.com/acme/private-repo issue #7",
        ),
        (_obs("private-source", h="h2", ts=start + 30, source="private"), "PRIVATE_SOURCE_TOKEN"),
        (_obs("private-id", h="h3", ts=start + 40, window="PRIVATE_ID_WINDOW"), "PRIVATE_ID_TOKEN"),
        (_obs("kept", h="h4", ts=start + 60, window="public notes"), "public notes"),
    ]
    settings = Settings(
        data_dir=tmp_path,
        deep_brain_excluded_apps=["com.mitchellh.ghostty"],
        deep_brain_excluded_sources=["private"],
        deep_brain_excluded_observation_ids=["private-id"],
    )

    packet = build_deep_brain_preview(
        rows,
        start_ts=start,
        end_ts=start + 3600,
        day_offset=0,
        source_scope="capture",
        settings=settings,
    )
    rendered = json.dumps(packet["memory_summary"], sort_keys=True)
    rendered_sources = json.dumps(packet["selected_sources"], sort_keys=True)

    assert "SECRET_TERMINAL_FOLLOWUP" not in rendered
    assert "private-repo" not in rendered
    assert "PRIVATE_SOURCE_TOKEN" not in rendered
    assert "PRIVATE_ID_WINDOW" not in rendered
    assert "secret-terminal" not in rendered_sources
    assert "private-source" not in rendered_sources
    assert "private-id" not in rendered_sources
    assert packet["exclusions"]["excluded_by"] == {
        "app": 1,
        "observation_id": 1,
        "source": 1,
    }
    assert packet["exclusions"]["kept_observations"] == 1


def test_unknown_app_kept_is_counted_when_app_exclusion_configured(tmp_path):
    start = _day(2026, 6, 12, 9)
    rows = [(_obs("unknown", h="h1", ts=start, app=None), "unknown app row")]
    packet = build_deep_brain_preview(
        rows,
        start_ts=start,
        end_ts=start + 60,
        day_offset=0,
        source_scope="capture",
        settings=Settings(data_dir=tmp_path, deep_brain_excluded_apps=["Code"]),
    )

    assert packet["exclusions"]["unknown_app_kept"] == 1
    assert packet["exclusions"]["kept_observations"] == 1


def test_selected_source_metadata_is_minimized_for_egress(tmp_path):
    start = _day(2026, 6, 12, 9)
    rows = [
        (
            _obs(
                "url-source",
                h="h1",
                ts=start,
                app="Browser",
                window=None,
                url="https://example.com/private/path?token=secret#fragment",
            ),
            "public source text",
        ),
        (
            _obs(
                "title-source",
                h="h2",
                ts=start + 1,
                window=(
                    "OPENAI_API_KEY=sk-proj-ABCDEFGHIJKLMNOP secret window "
                    "https://example.org/private/path?token=secret"
                ),
            ),
            "title source text",
        ),
    ]

    packet = build_deep_brain_preview(
        rows,
        start_ts=start,
        end_ts=start + 60,
        day_offset=0,
        source_scope="capture",
        settings=Settings(data_dir=tmp_path),
    )
    by_id = {source["observation_id"]: source for source in packet["selected_sources"]}

    assert by_id["url-source"]["window_or_url"] == "https://example.com"
    assert "private/path" not in json.dumps(packet["selected_sources"])
    assert "token=secret" not in json.dumps(packet["selected_sources"])
    assert "[REDACTED]" in by_id["title-source"]["window_or_url"]
    assert "sk-proj-" not in by_id["title-source"]["window_or_url"]
    assert "example.org/private/path" not in by_id["title-source"]["window_or_url"]
    assert "https://example.org" in by_id["title-source"]["window_or_url"]
    assert "window" not in by_id["url-source"]


def test_preview_builder_does_not_log_content(tmp_path, caplog):
    start = _day(2026, 6, 12, 9)
    rows = [(_obs("o1", h="h1", ts=start, window="PRIVATE_WINDOW"), "PRIVATE_TEXT")]
    with caplog.at_level(logging.INFO):
        build_deep_brain_preview(
            rows,
            start_ts=start,
            end_ts=start + 60,
            day_offset=0,
            source_scope="capture",
            settings=Settings(data_dir=tmp_path),
        )

    assert caplog.records == []


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


def test_deep_brain_ask_local_model_needs_feature_gate_only(tmp_path):
    start = _day(2026, 6, 12, 9)
    rows = [(_obs("o1", h="h1", ts=start, window="notes"), "useful notes")]
    packet = build_deep_brain_preview(
        rows,
        start_ts=start,
        end_ts=start + 60,
        day_offset=0,
        source_scope="capture",
        settings=Settings(data_dir=tmp_path, deep_brain_enabled=True),
    )
    provider = _Provider(
        {"answer": "The day was notes-heavy.", "citation_ids": ["o1"], "confidence": "high"}
    )

    result = answer_deep_brain(
        "what happened?",
        packet,
        provider,
        settings=Settings(data_dir=tmp_path, deep_brain_enabled=True),
    )

    assert result["ok"] is True
    assert result["reasoning_route"] == "local_model"
    assert result["egress"] == "none"
    assert result["citations"][0]["observation_id"] == "o1"
    assert packet_json_for_model(packet) in provider.messages[1]["content"]


def test_deep_brain_ask_refuses_remote_without_cloud_optin(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        deep_brain_enabled=True,
        llm_model="gpt-4o-mini",
    )
    packet = build_deep_brain_preview(
        [],
        start_ts=_day(2026, 6, 12),
        end_ts=_day(2026, 6, 12, 1),
        day_offset=0,
        source_scope="capture",
        settings=settings,
    )
    provider = _Provider({"answer": "should not run", "citation_ids": [], "confidence": "low"})

    result = answer_deep_brain("q", packet, provider, settings=settings)

    assert result["ok"] is False
    assert "OPENBIRD_ALLOW_CLOUD" in " ".join(result["blocked_reasons"])
    assert provider.messages is None


def test_deep_brain_ask_empty_packet_never_calls_model(tmp_path):
    settings = Settings(data_dir=tmp_path, deep_brain_enabled=True)
    packet = build_deep_brain_preview(
        [],
        start_ts=_day(2026, 6, 12),
        end_ts=_day(2026, 6, 12, 1),
        day_offset=0,
        source_scope="capture",
        settings=settings,
    )
    provider = _Provider(
        {"answer": "Unsupported empty-packet claim.", "citation_ids": [], "confidence": "high"}
    )

    result = answer_deep_brain("q", packet, provider, settings=settings)

    assert result["ok"] is True
    assert result["answer"] == "I do not have enough Deep Brain packet evidence to answer that."
    assert result["confidence"] == "insufficient_evidence"
    assert result["grounded"] is False
    assert packet["packet_build_route"] == PACKET_BUILD_ROUTE_DETERMINISTIC
    assert packet["packet_build_route"] != result["reasoning_route"]
    assert result["egress"] == "none"
    assert result["citations"] == []
    assert provider.messages is None


def test_deep_brain_ask_remote_route_truth_with_both_gates(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        deep_brain_enabled=True,
        allow_cloud=True,
        llm_model="gpt-4o-mini",
    )
    start = _day(2026, 6, 12, 9)
    packet = build_deep_brain_preview(
        [(_obs("o1", h="h1", ts=start), "coding block")],
        start_ts=start,
        end_ts=start + 60,
        day_offset=0,
        source_scope="capture",
        settings=settings,
    )
    provider = _Provider(
        {"answer": "You had a coding block.", "citation_ids": ["o1"], "confidence": "medium"}
    )

    result = answer_deep_brain("productivity?", packet, provider, settings=settings)

    assert packet["packet_build_route"] == PACKET_BUILD_ROUTE_DETERMINISTIC
    assert packet["packet_build_route"] != result["reasoning_route"]
    assert result["reasoning_route"] == "cloud_reasoning_active"
    assert result["egress"] == "active_model_route"
    assert result["model"] == "stub-model"
    assert f'"packet_build_route":"{PACKET_BUILD_ROUTE_DETERMINISTIC}"' in provider.messages[1]["content"]


def test_deep_brain_ask_drops_hallucinated_citations_and_gates_answer(tmp_path):
    settings = Settings(data_dir=tmp_path, deep_brain_enabled=True)
    start = _day(2026, 6, 12, 9)
    packet = build_deep_brain_preview(
        [(_obs("real", h="h1", ts=start), "real notes")],
        start_ts=start,
        end_ts=start + 60,
        day_offset=0,
        source_scope="capture",
        settings=settings,
    )
    provider = _Provider(
        {"answer": "Unsupported claim.", "citation_ids": ["fake"], "confidence": "high"}
    )

    result = answer_deep_brain("q", packet, provider, settings=settings)

    assert result["answer"] == "I could not ground that answer in the Deep Brain packet."
    assert result["confidence"] == "insufficient_evidence"
    assert result["citations"] == []


def test_deep_brain_prompt_uses_exact_preview_packet_and_no_excluded_content(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        deep_brain_enabled=True,
        deep_brain_excluded_apps=["com.mitchellh.ghostty"],
    )
    start = _day(2026, 6, 12, 9)
    packet = build_deep_brain_preview(
        [
            (
                _obs(
                    "secret",
                    h="h1",
                    ts=start,
                    app="com.mitchellh.ghostty",
                    window="SECRET_WINDOW",
                ),
                "SECRET_TEXT",
            ),
            (_obs("kept", h="h2", ts=start + 1, window="kept"), "kept text"),
        ],
        start_ts=start,
        end_ts=start + 60,
        day_offset=0,
        source_scope="capture",
        settings=settings,
    )

    messages = build_deep_brain_messages("what matters?", packet)
    prompt = messages[1]["content"]
    _prompt_registry.ensure_loaded()
    neutralized_packet = _prompt_registry.get("rag").fence.neutralize(
        packet_json_for_model(packet)
    )

    assert neutralized_packet in prompt
    assert "SECRET_WINDOW" not in prompt
    assert "SECRET_TEXT" not in prompt
    assert "<<<END_OPENBIRD_UNTRUSTED_CONTEXT>>>" not in packet_json_for_model(packet)


class _PreviewStore:
    def __init__(self, rows):
        self.rows = rows

    def time_range_text(self, start, end, source="capture"):
        return [(obs, text) for obs, text in self.rows if source is None or obs.source == source]

    def close(self):
        pass


def test_deep_brain_preview_cli_uses_maintenance_store_not_model_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    reset_settings_cache()
    start = _day(*dt.datetime.now().timetuple()[:3], 9)
    rows = [(_obs("o1", h="h1", ts=start, window="public notes"), "public notes")]
    used_maintenance_store = False

    def _fake_store_maintenance():
        nonlocal used_maintenance_store
        used_maintenance_store = True
        return _PreviewStore(rows)

    monkeypatch.setattr(cli, "_store_maintenance", _fake_store_maintenance)
    monkeypatch.setattr(
        cli,
        "_provider",
        lambda: (_ for _ in ()).throw(
            AssertionError("preview must not use configured model provider")
        ),
    )

    res = CliRunner().invoke(cli.app, ["deep-brain", "preview", "--day", "0", "--json"])

    assert res.exit_code == 0, res.output
    assert used_maintenance_store is True
    packet = json.loads(res.stdout)
    assert packet["route"] == "deep_brain.preview"
    assert packet["packet_build_route"] == PACKET_BUILD_ROUTE_DETERMINISTIC
    assert packet["egress"] == "none_preview"
    assert packet["selected_sources"][0]["observation_id"] == "o1"


def test_deep_brain_ask_cli_refuses_before_provider_without_feature_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    reset_settings_cache()
    start = _day(*dt.datetime.now().timetuple()[:3], 9)
    rows = [(_obs("o1", h="h1", ts=start, window="public notes"), "public notes")]
    monkeypatch.setattr(cli, "_store_maintenance", lambda: _PreviewStore(rows))
    monkeypatch.setattr(
        cli,
        "_completion_provider",
        lambda: (_ for _ in ()).throw(AssertionError("refusal must not use provider")),
    )

    try:
        res = CliRunner().invoke(
            cli.app,
            ["deep-brain", "ask", "what did I do?", "--day", "0", "--json"],
        )
    finally:
        reset_settings_cache()

    assert res.exit_code == 2
    payload = json.loads(res.stdout)
    assert payload["ok"] is False
    assert "OPENBIRD_DEEP_BRAIN_ENABLED" in " ".join(payload["blocked_reasons"])


def test_deep_brain_ask_cli_uses_provider_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBIRD_DEEP_BRAIN_ENABLED", "1")
    reset_settings_cache()
    start = _day(*dt.datetime.now().timetuple()[:3], 9)
    rows = [(_obs("o1", h="h1", ts=start, window="public notes"), "public notes")]
    monkeypatch.setattr(cli, "_store_maintenance", lambda: _PreviewStore(rows))
    provider = _Provider(
        {"answer": "You worked in notes.", "citation_ids": ["o1"], "confidence": "high"}
    )
    monkeypatch.setattr(cli, "_completion_provider", lambda: provider)

    try:
        res = CliRunner().invoke(
            cli.app,
            ["deep-brain", "ask", "--day", "0", "--json", "--stdin"],
            input="what did I do?",
        )
    finally:
        reset_settings_cache()

    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert payload["ok"] is True
    assert payload["packet_route"] == "deep_brain.preview"
    assert payload["reasoning_route"] == "local_model"
    assert payload["citations"][0]["observation_id"] == "o1"


def test_deep_brain_cli_rejects_out_of_range_days(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    reset_settings_cache()

    try:
        too_low = CliRunner().invoke(
            cli.app,
            ["deep-brain", "preview", "--days", "0", "--json"],
        )
        too_high = CliRunner().invoke(
            cli.app,
            ["deep-brain", "ask", "what happened?", "--days", "8", "--json"],
        )
    finally:
        reset_settings_cache()

    assert too_low.exit_code == 2
    assert too_high.exit_code == 2


def test_deep_brain_ask_cli_days_refuses_before_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    reset_settings_cache()
    today = dt.datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    rows = [
        (_obs("old", h="h-old", ts=(today - dt.timedelta(days=1)).timestamp()), "old notes"),
        (_obs("new", h="h-new", ts=today.timestamp()), "new notes"),
    ]
    monkeypatch.setattr(cli, "_store_maintenance", lambda: _PreviewStore(rows))
    monkeypatch.setattr(
        cli,
        "_completion_provider",
        lambda: (_ for _ in ()).throw(AssertionError("refusal must not use provider")),
    )

    try:
        res = CliRunner().invoke(
            cli.app,
            ["deep-brain", "ask", "what happened?", "--days", "2", "--json"],
        )
    finally:
        reset_settings_cache()

    assert res.exit_code == 2
    payload = json.loads(res.stdout)
    assert payload["packet"]["period"]["days"] == 2
    assert "OPENBIRD_DEEP_BRAIN_ENABLED" in " ".join(payload["blocked_reasons"])


def test_deep_brain_ask_cli_days_uses_provider_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBIRD_DEEP_BRAIN_ENABLED", "1")
    reset_settings_cache()
    today = dt.datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    rows = [
        (_obs("old", h="h-old", ts=(today - dt.timedelta(days=1)).timestamp()), "old notes"),
        (_obs("new", h="h-new", ts=today.timestamp()), "new notes"),
    ]
    monkeypatch.setattr(cli, "_store_maintenance", lambda: _PreviewStore(rows))
    provider = _Provider(
        {"answer": "The older day is grounded.", "citation_ids": ["old"], "confidence": "high"}
    )
    monkeypatch.setattr(cli, "_completion_provider", lambda: provider)

    try:
        res = CliRunner().invoke(
            cli.app,
            ["deep-brain", "ask", "--days", "2", "--json", "--stdin"],
            input="what happened?",
        )
    finally:
        reset_settings_cache()

    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert payload["ok"] is True
    assert payload["packet_route"] == "deep_brain.preview"
    assert payload["citations"][0]["observation_id"] == "old"
    assert payload["citations"][0]["local_date"]
    assert '"source_ids"' not in provider.messages[1]["content"]
