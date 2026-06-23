"""Contract tests for the privacy route inventory.

The YAML file is user-facing architecture evidence, not decorative docs. These
tests keep the highest-risk privacy claims pinned to machine-checkable facts.
"""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _routes() -> dict:
    data = yaml.safe_load((ROOT / "docs/privacy-routes.yaml").read_text())
    return {route["id"]: route for route in data["routes"]}


def test_privacy_route_inventory_models_export_as_egress() -> None:
    data = yaml.safe_load((ROOT / "docs/privacy-routes.yaml").read_text())
    routes = {route["id"]: route for route in data["routes"]}

    assert data["route_classes"]["unknown"]["egress"] is True
    assert data["release_policy"]["unknown_route_release_green"] is False

    export = routes["data.export"]
    assert export["class"] == "unknown"
    assert export["status"] == "implemented_jsonl"
    assert export["egress"]["default"] == "user_selected_destination"
    assert "explicit_destination_warning" in export["controls_active"]
    assert "no_tempfile_leak" in export["controls_active"]
    assert "output_file_mode_0600" in export["controls_active"]
    assert "optional_encrypted_export" in export["controls_planned"]


def test_remote_ollama_uses_cloud_active_truth_surface() -> None:
    route = _routes()["model.remote_ollama"]

    assert route["class"] == "self-hosted-remote"
    assert route["egress"]["condition"] == "ollama_host_is_not_loopback"
    assert route["enforcement"]["requires"] == "OPENBIRD_ALLOW_CLOUD"
    assert "preflight.cloud" in route["truth_surface"]
    assert "cli.CLOUD_ACTIVE" in route["truth_surface"]
    assert "app.CLOUD_ACTIVE" in route["truth_surface"]


def test_release_policy_requires_verified_sqlcipher() -> None:
    data = yaml.safe_load((ROOT / "docs/privacy-routes.yaml").read_text())

    assert data["release_policy"]["public_beta_requires_verified_sqlcipher"] is True
    assert data["release_policy"]["plaintext_0600_release_green"] is False


def test_diagnostics_route_forbids_content_bearing_fields() -> None:
    route = _routes()["diagnostics.logs"]

    assert route["class"] == "local"
    assert route["egress"]["default"] == "none"
    assert {
        "captured_text",
        "raw_window_title",
        "raw_url",
        "provider_secret",
    }.issubset(set(route["forbidden_fields"]))


def test_capture_pause_is_enforced_at_helper_boundary() -> None:
    route = _routes()["capture.pause"]

    assert route["enforcement"]["boundary"] == "Swift helper before AX read"
    assert route["enforcement"]["failure_mode"] == "fail_closed"
