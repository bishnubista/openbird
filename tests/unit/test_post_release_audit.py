from __future__ import annotations

import importlib.util
import json
import os
import stat
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "script" / "post_release_audit.py"
SPEC = importlib.util.spec_from_file_location("post_release_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
post_release_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(post_release_audit)


def _app(bundle_id: str, quality: str, samples: int = 10) -> dict:
    return {
        "bundle_id": bundle_id,
        "context_quality": quality,
        "sample_count": samples,
    }


def test_compare_audits_requires_sample_floor_and_ignores_repetition():
    previous = {
        "schema_version": 1,
        "capture_audit": {
            "apps": [
                {**_app("improved", "low_context"), "distinct_ratio": 1.0},
                _app("too-small", "low_context", 2),
            ]
        },
    }
    current = {
        "apps": [
            {**_app("improved", "rich_context"), "distinct_ratio": 0.01},
            _app("too-small", "rich_context", 10),
        ]
    }

    comparison = post_release_audit.compare_audits(current, previous, minimum_samples=5)

    assert comparison == [
        {
            "bundle_id": "improved",
            "change": "improved",
            "previous_quality": "low_context",
            "current_quality": "rich_context",
        }
    ]


def test_validate_daemon_requires_rotated_installed_app_identity(monkeypatch):
    monkeypatch.setattr(post_release_audit, "_is_app_process", lambda pid: pid == 321)
    audit = {
        "daemon": {
            "state": "ok",
            "instance_uuid": "00000000-0000-4000-8000-000000000002",
            "pid": 321,
            "runtime_version": "1.2.3",
        }
    }

    post_release_audit._validate_daemon(
        audit,
        expected_version="1.2.3",
        old_instance="00000000-0000-4000-8000-000000000001",
    )

    audit["daemon"]["instance_uuid"] = "00000000-0000-4000-8000-000000000001"
    with pytest.raises(post_release_audit.Blocked, match="identity_not_rotated"):
        post_release_audit._validate_daemon(
            audit,
            expected_version="1.2.3",
            old_instance="00000000-0000-4000-8000-000000000001",
        )


def test_write_report_is_private_and_atomic(tmp_path, monkeypatch):
    report_dir = tmp_path / "audits"
    monkeypatch.setattr(post_release_audit, "REPORT_DIR", report_dir)
    report = {
        "version": "1.2.3",
        "capture_audit": {"apps": [], "overall_state": "pass"},
    }

    path = post_release_audit._write_report(report, "1.2.3")

    assert path.parent == report_dir
    assert json.loads(path.read_text()) == report
    assert stat.S_IMODE(report_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert list(report_dir.glob(".post-release-*")) == []


def test_latest_report_uses_write_time_not_lexical_version(tmp_path, monkeypatch):
    monkeypatch.setattr(post_release_audit, "REPORT_DIR", tmp_path)
    lexical_newer = tmp_path / "post-release-9.0.0-older.json"
    chronological_newer = tmp_path / "post-release-10.0.0-newer.json"
    lexical_newer.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "version": "9.0.0",
                "capture_audit": {"apps": []},
            }
        )
    )
    chronological_newer.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "version": "10.0.0",
                "capture_audit": {"apps": []},
            }
        )
    )
    os.utime(lexical_newer, ns=(1, 1))
    os.utime(chronological_newer, ns=(2, 2))

    assert post_release_audit._latest_report()["version"] == "10.0.0"


def test_malformed_latest_report_blocks_instead_of_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(post_release_audit, "REPORT_DIR", tmp_path)
    (tmp_path / "post-release-1.2.3-bad.json").write_text(
        json.dumps({"schema_version": 1, "capture_audit": {"apps": [None]}})
    )

    with pytest.raises(post_release_audit.Blocked, match="prior_report_malformed"):
        post_release_audit._latest_report()


def test_missing_command_is_blocked():
    with pytest.raises(post_release_audit.Blocked, match="command_unavailable"):
        post_release_audit._run(["/definitely/missing/openbird-command"], timeout=0.1)


def test_process_discovery_error_status_is_blocked(monkeypatch):
    result = post_release_audit.subprocess.CompletedProcess(
        args=["pgrep"], returncode=2, stdout="", stderr="pgrep metadata error"
    )
    monkeypatch.setattr(
        post_release_audit.subprocess, "run", lambda *args, **kwargs: result
    )

    with pytest.raises(
        post_release_audit.Blocked, match="process_discovery_failed:rc_2"
    ):
        post_release_audit._app_processes()
