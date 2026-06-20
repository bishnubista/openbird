"""Unit tests for `openbird doctor` — content-safety + never-crash contracts."""

from __future__ import annotations

import json

import openbird.doctor as doc

_SYS = {
    "openbird_version": "0.1.0",
    "os": "Darwin",
    "os_version": "15.0",
    "arch": "arm64",
    "python": "3.13.0",
}


def _preflight(*, ok=True, allowlist=(), blocklist=(), extra=None):
    def _run(settings=None, *, probe_ollama=True):
        report = {
            "privacy": {
                "allowlist": list(allowlist),
                "blocklist": list(blocklist),
                "ocr_enabled": False,
            },
            "ollama": {"reachable": True},
            "runtime_ok": ok,
        }
        if extra:
            report.update(extra)
        return report

    return _run


def test_home_path_redacted_in_helper_path():
    report = doc.build_doctor_report(
        preflight_runner=_preflight(),
        system_info=lambda: _SYS,
        helper_path="/Users/me/proj/OpenBird.app/Contents/MacOS/capture-helper",
        command_runner=lambda argv, t: None,
        home="/Users/me",
        is_macos=True,
    )
    assert "/Users/me" not in json.dumps(report)
    assert report["helper_path"].startswith("~")


def test_allowlist_projected_to_counts_no_values():
    report = doc.build_doctor_report(
        preflight_runner=_preflight(allowlist=["com.apple.Safari", "com.acme.App"]),
        system_info=lambda: _SYS,
        command_runner=lambda argv, t: None,
        home="/h",
        is_macos=True,
    )
    priv = report["preflight"]["privacy"]
    assert priv["allowlist_count"] == 2
    assert priv["allowlist_empty"] is False
    assert "allowlist" not in priv and "blocklist" not in priv
    # no captured bundle id may appear anywhere in the shareable report
    assert "com.apple.Safari" not in json.dumps(report)


def test_empty_allowlist_flagged():
    report = doc.build_doctor_report(
        preflight_runner=_preflight(allowlist=[]),
        system_info=lambda: _SYS,
        command_runner=lambda argv, t: None,
        home="/h",
        is_macos=True,
    )
    assert report["preflight"]["privacy"]["allowlist_empty"] is True
    assert "EMPTY" in doc.render(report)


def test_token_shaped_secret_scrubbed():
    report = doc.build_doctor_report(
        preflight_runner=_preflight(extra={"note": "leaked sk-abcd1234efgh5678 token"}),
        system_info=lambda: _SYS,
        command_runner=lambda argv, t: None,
        home="/h",
        is_macos=True,
    )
    blob = json.dumps(report)
    assert "sk-abcd1234efgh5678" not in blob
    assert "<redacted>" in blob


def test_signing_parsed_from_fake_codesign(tmp_path):
    helper = tmp_path / "OpenBird.app" / "Contents" / "MacOS" / "capture-helper"
    helper.parent.mkdir(parents=True)
    helper.write_text("#!/bin/sh\n")

    def runner(argv, timeout):
        if list(argv[:2]) == ["codesign", "-dvv"]:
            # codesign writes these to stderr on success
            return (0, "", "Authority=Developer ID Application: bishnu bista (SB26BAMXJM)\nTeamIdentifier=SB26BAMXJM\n")
        if list(argv[:3]) == ["codesign", "-dr", "-"]:
            return (0, "designated => identifier \"dev.openbird.capture-helper\"\n", "")
        if list(argv[:2]) == ["xattr", "-p"]:
            return (1, "", "No such xattr: com.apple.quarantine")  # attr not set
        return None

    report = doc.build_doctor_report(
        preflight_runner=_preflight(),
        system_info=lambda: _SYS,
        helper_path=str(helper),
        command_runner=runner,
        home="/h",
        is_macos=True,
    )
    sign = report["signing"]
    assert sign["signed"] is True
    assert sign["team_identifier"] == "SB26BAMXJM"
    assert "Developer ID Application" in sign["authority"]
    assert "designated_requirement" in sign
    assert report["quarantine"] == "absent"


def test_non_macos_signing_na_no_crash():
    report = doc.build_doctor_report(
        preflight_runner=_preflight(),
        system_info=lambda: {**_SYS, "os": "Linux"},
        command_runner=lambda argv, t: None,
        home="/h",
        is_macos=False,
    )
    assert report["signing"] == {"signed": "n/a"}
    assert report["quarantine"] == "n/a"
    assert report["runtime_ok"] is True


def test_never_raises_when_everything_missing():
    def boom(settings=None, *, probe_ollama=True):
        raise RuntimeError("preflight blew up")

    report = doc.build_doctor_report(
        preflight_runner=boom,
        system_info=lambda: _SYS,
        helper_path="/nope/OpenBird.app/Contents/MacOS/capture-helper",
        command_runner=lambda argv, t: None,
        home="/h",
        is_macos=True,
    )
    # preflight error captured as a type name, not a message/traceback
    assert report["preflight"] == {"error": "RuntimeError"}
    assert "blew up" not in json.dumps(report)
    assert report["runtime_ok"] is False
    assert report["signing"]["signed"] == "unknown"  # helper not found
    # render must also not raise
    assert isinstance(doc.render(report), str)


def test_dict_keys_are_scrubbed():
    # preflight keys some maps by model/host strings; a home path in a KEY must redact.
    report = doc.build_doctor_report(
        preflight_runner=_preflight(extra={"models": {"/Users/me/m.gguf": "ok"}}),
        system_info=lambda: _SYS,
        command_runner=lambda argv, t: None,
        home="/Users/me",
        is_macos=True,
    )
    assert "/Users/me" not in json.dumps(report)


def test_url_credentials_masked():
    report = doc.build_doctor_report(
        preflight_runner=_preflight(extra={"ollama": {"host": "https://user:s3cret@host:11434"}}),
        system_info=lambda: _SYS,
        command_runner=lambda argv, t: None,
        home="/h",
        is_macos=True,
    )
    blob = json.dumps(report)
    assert "s3cret" not in blob
    assert "user:s3cret" not in blob
    assert "<redacted>@" in blob


def test_quarantine_unknown_on_non_absent_error(tmp_path):
    helper = tmp_path / "OpenBird.app" / "Contents" / "MacOS" / "capture-helper"
    helper.parent.mkdir(parents=True)
    helper.write_text("#!/bin/sh\n")

    def runner(argv, timeout):
        if list(argv[:2]) == ["codesign", "-dvv"]:
            return (0, "", "Authority=X\nTeamIdentifier=T\n")
        if list(argv[:2]) == ["xattr", "-p"]:
            return (1, "", "xattr: [Errno 1] Operation not permitted")
        return None

    report = doc.build_doctor_report(
        preflight_runner=_preflight(),
        system_info=lambda: _SYS,
        helper_path=str(helper),
        command_runner=runner,
        home="/h",
        is_macos=True,
    )
    assert report["quarantine"] == "unknown"  # permission error, not "absent"


def test_settings_resolution_failure_does_not_crash(monkeypatch):
    def boom_settings():
        raise RuntimeError("bad env / unwritable data dir")

    monkeypatch.setattr(doc, "get_settings", boom_settings)
    # No settings passed -> builder resolves via get_settings() inside its catch.
    report = doc.build_doctor_report(
        system_info=lambda: _SYS,
        command_runner=lambda argv, t: None,
        home="/h",
        is_macos=True,
    )
    assert report["preflight"] == {"error": "RuntimeError"}
    assert report["runtime_ok"] is False
    assert "unwritable" not in json.dumps(report)
