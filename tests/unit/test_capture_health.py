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


def test_capture_health_marks_opted_in_ocr_rows(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        allowlist=["com.microsoft.teams2", "com.apple.mail"],
        blocklist=[],
        capture_ocr_apps=["com.microsoft.teams2"],
    )

    report = build_capture_health(
        settings=settings,
        activity_by_app={},
        generated_at=200.0,
        paused=False,
    )

    by_app = {row["bundle_id"]: row for row in report["apps"]}
    # Opted-in rows carry the additive marker; others carry NO ocr key at all.
    assert by_app["com.microsoft.teams2"]["ocr"] == "opted_in"
    assert "ocr" not in by_app["com.apple.mail"]
    assert report["ocr_apps_count"] == 1
    # No sidecar in this test dir -> daemon availability is honestly unknown
    # (the CLI table renders opted-in rows as "unknown" in that case).
    assert report["daemon"] == {"state": "unknown"}


def test_capture_health_reports_detailed_capture_availability_and_state(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        allowlist=["com.mitchellh.ghostty", "com.apple.Terminal", "com.apple.mail"],
        detailed_capture_apps=["com.mitchellh.ghostty"],
    )

    report = build_capture_health(
        settings=settings,
        activity_by_app={},
        generated_at=200.0,
        paused=False,
    )

    by_app = {row["bundle_id"]: row for row in report["apps"]}
    assert by_app["com.mitchellh.ghostty"]["detailed_capture"] == "enabled"
    assert by_app["com.mitchellh.ghostty"]["policy"]["capture"] is True
    assert by_app["com.apple.Terminal"]["detailed_capture"] == "available"
    assert by_app["com.apple.Terminal"]["policy"] == {
        "capture": False,
        "reason": "blocklisted",
    }
    assert "detailed_capture" not in by_app["com.apple.mail"]
    assert report["detailed_capture_apps_count"] == 1


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
    # Deliberately extended for Phase A: the additive `daemon` liveness block
    # (metadata only — state/timestamps/mode/afk/seq, never content).
    assert set(payload) == {
        "generated_at",
        "recent_window_seconds",
        "paused",
        "allowlist_count",
        "blocklist_count",
        "detailed_capture_apps_count",
        "ocr_apps_count",
        "daemon",
        "apps",
    }
    assert payload["ocr_apps_count"] == 0
    # No daemon running in this test -> no sidecar -> unknown, never "ok".
    assert payload["daemon"] == {"state": "unknown"}
    assert payload["allowlist_count"] == 1
    row = payload["apps"][0]
    assert set(row) == {
        "bundle_id",
        "policy",
        "effective_state",
        "quality",
        "coverage",
        "total_observations",
        "recent_observations",
        "last_captured_ts",
    }
    assert row["bundle_id"] == "com.example.Editor"
    assert row["policy"] == {"capture": True, "reason": "allowlisted"}
    rendered = json.dumps(payload)
    assert "secret capture text" not in rendered
    assert "Private planning doc" not in rendered
    assert "example.invalid" not in rendered
