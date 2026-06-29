from __future__ import annotations

import json

from typer.testing import CliRunner

from openbird import cli
from openbird.capture.health import build_capture_health
from openbird.config import Settings, reset_settings_cache
from openbird.memory.store import MemoryStore


def test_capture_health_uses_capture_policy_and_blocklist_wins(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        allowlist=["com.example.Editor"],
        blocklist=["com.example.Editor"],
    )

    report = build_capture_health(
        settings=settings,
        activity_by_app={
            "com.example.Editor": {
                "total_observations": 3,
                "recent_observations": 2,
                "last_captured_ts": 123.0,
            }
        },
        generated_at=200.0,
        recent_window_seconds=60,
        paused=False,
    )

    row = report["apps"][0]
    assert row["bundle_id"] == "com.example.Editor"
    assert row["policy"] == {"capture": False, "reason": "blocklisted"}
    assert row["effective_state"] == "blocked"
    assert row["quality"] == "blocked"
    assert row["recent_observations"] == 2
    assert row["last_captured_ts"] == 123.0


def test_capture_health_reports_allowed_recent_stale_and_no_recent(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        allowlist=["com.apple.mail", "com.apple.Notes", "com.example.Unknown"],
        blocklist=[],
    )

    report = build_capture_health(
        settings=settings,
        activity_by_app={
            "com.apple.mail": {
                "total_observations": 5,
                "recent_observations": 2,
                "last_captured_ts": 190.0,
            },
            "com.apple.Notes": {
                "total_observations": 4,
                "recent_observations": 0,
                "last_captured_ts": 80.0,
            },
        },
        generated_at=200.0,
        recent_window_seconds=60,
        paused=False,
    )

    by_app = {row["bundle_id"]: row for row in report["apps"]}
    assert by_app["com.apple.mail"]["effective_state"] == "allowed_recent"
    assert by_app["com.apple.mail"]["quality"] == "good"
    assert by_app["com.apple.Notes"]["effective_state"] == "allowed_stale"
    assert by_app["com.apple.Notes"]["quality"] == "no_recent"
    assert by_app["com.example.Unknown"]["effective_state"] == "allowed_no_recent"
    assert by_app["com.example.Unknown"]["quality"] == "no_recent"
    assert by_app["com.example.Unknown"]["coverage"] == "unknown"


def test_capture_health_resolves_pattern_allowlist_entries_from_activity(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        allowlist=["glob:com.acme.*"],
        blocklist=[],
    )

    report = build_capture_health(
        settings=settings,
        activity_by_app={
            "com.acme.Editor": {
                "total_observations": 2,
                "recent_observations": 1,
                "last_captured_ts": 190.0,
            },
            "com.other.App": {
                "total_observations": 2,
                "recent_observations": 1,
                "last_captured_ts": 190.0,
            },
        },
        generated_at=200.0,
        paused=False,
    )

    assert [row["bundle_id"] for row in report["apps"]] == ["com.acme.Editor"]
    row = report["apps"][0]
    assert row["policy"] == {"capture": True, "reason": "allowlisted"}
    assert row["effective_state"] == "allowed_recent"


def test_capture_health_marks_dangerous_and_self_capture_blocked(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        allowlist=["com.1password.1password", "ai.openbird.openbird"],
        blocklist=[],
    )

    report = build_capture_health(
        settings=settings,
        activity_by_app={},
        generated_at=200.0,
        paused=False,
    )

    by_app = {row["bundle_id"]: row for row in report["apps"]}
    assert by_app["com.1password.1password"]["policy"] == {
        "capture": False,
        "reason": "dangerous_app",
    }
    assert by_app["ai.openbird.openbird"]["policy"] == {
        "capture": False,
        "reason": "self_capture",
    }


def test_capture_app_activity_reads_only_metadata(tmp_path, fake_provider):
    settings = Settings(data_dir=tmp_path, embed_dim=768)
    store = MemoryStore(settings=settings, provider=fake_provider)
    try:
        store.add_observation(
            "captured text that must not appear",
            app="com.example.Editor",
            window="Sensitive Window",
            url="https://example.invalid/private",
            source="capture",
            ts=100.0,
        )
        store.add_observation(
            "older captured text that must not appear",
            app="com.example.Editor",
            window="Older Sensitive Window",
            url="https://example.invalid/older",
            source="capture",
            ts=10.0,
        )
        store.add_observation(
            "manual note ignored by capture health",
            app="com.example.Editor",
            source="manual",
            ts=120.0,
        )

        activity = store.capture_app_activity(recent_since_ts=50.0)
    finally:
        store.close()

    assert activity == {
        "com.example.Editor": {
            "total_observations": 2,
            "recent_observations": 1,
            "last_captured_ts": 100.0,
        }
    }
    leaked = json.dumps(activity)
    assert "captured text" not in leaked
    assert "Sensitive Window" not in leaked
    assert "example.invalid" not in leaked


def test_capture_health_cli_json_is_metadata_only(tmp_path, fake_provider, monkeypatch):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBIRD_ALLOWLIST", "com.example.Editor")
    monkeypatch.setenv("OPENBIRD_BLOCKLIST", "")
    monkeypatch.setenv("OPENBIRD_DISABLE_KEYRING", "1")
    reset_settings_cache()
    settings = Settings(
        data_dir=tmp_path,
        allowlist=["com.example.Editor"],
        blocklist=[],
        embed_dim=768,
    )
    store = MemoryStore(settings=settings, provider=fake_provider)
    try:
        store.add_observation(
            "secret capture text should stay out of health",
            app="com.example.Editor",
            window="Private planning doc",
            url="https://example.invalid/secret",
            source="capture",
            ts=100.0,
        )
    finally:
        store.close()

    try:
        result = CliRunner().invoke(
            cli.app,
            ["data", "capture-health", "--json", "--recent-window-seconds", "10"],
        )
    finally:
        reset_settings_cache()

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["allowlist_count"] == 1
    row = payload["apps"][0]
    assert row["bundle_id"] == "com.example.Editor"
    assert row["policy"] == {"capture": True, "reason": "allowlisted"}
    rendered = json.dumps(payload)
    assert "secret capture text" not in rendered
    assert "Private planning doc" not in rendered
    assert "example.invalid" not in rendered
