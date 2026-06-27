"""Contract tests for the privacy route inventory.

The YAML file is user-facing architecture evidence, not decorative docs. These
tests keep the highest-risk privacy claims pinned to machine-checkable facts.

Two layers of tests live here:

* *Manifest self-consistency* (ids unique, references resolve, key fields pinned).
* *Code-vs-manifest enforcement* (the ``test_egress_*`` block) — these drive the
  ACTUAL runtime classifier and gating code and cross-check it against the
  manifest, so a future undeclared egress, or a declared-but-ungated cloud route,
  FAILS the build. They do not re-read the YAML and compare it to itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from openbird.config import Settings
from openbird.llm.provider import (
    CLOUD_EGRESS_ROUTES,
    CloudOptInRequired,
    classify_models,
    create_llm_provider,
)
from openbird.llm.rerank import rerank_is_remote


ROOT = Path(__file__).resolve().parents[2]

# A loopback Ollama host pinned for classification so these tests never depend on
# (or probe) the developer's real OLLAMA_HOST. No network call is ever made.
LOOPBACK_OLLAMA = "http://127.0.0.1:11434"


def _routes() -> dict:
    data = yaml.safe_load((ROOT / "docs/privacy-routes.yaml").read_text())
    routes = data["routes"]
    ids = [route["id"] for route in routes]
    assert len(ids) == len(set(ids)), "privacy-routes.yaml contains duplicate route ids"
    return {route["id"]: route for route in routes}


def test_privacy_route_inventory_has_expected_routes() -> None:
    routes = _routes()

    assert {
        "capture.active_window",
        "capture.pause",
        "chat.local",
        "chat.remote",
        "data.export",
        "deep_brain.ask",
        "deep_brain.preview",
        "diagnostics.logs",
        "embedding.remote",
        "ingest.files",
        "meetings.audio",
        "model.local_ollama",
        "model.remote_ollama",
        "model.third_party_cloud",
        "productivity.coach",
        "productivity.local_facts",
        "rerank.remote",
        "routines.summary",
    }.issubset(routes)


def test_privacy_route_references_resolve() -> None:
    routes = _routes()
    referenced: set[str] = set()

    for route in routes.values():
        egress = route.get("egress", {})
        for key in ("inherited_by", "inherited_from", "resolved_by"):
            referenced.update(egress.get(key, []))

    assert referenced.issubset(routes)


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


def test_deep_brain_ask_inherits_active_model_route() -> None:
    route = _routes()["deep_brain.ask"]

    assert route["class"] == "unknown"
    assert route["egress"]["default"] == "inherits_active_model_route"
    assert route["egress"]["inherited_from"] == ["chat.local", "chat.remote"]
    assert "requires" not in route.get("enforcement", {})
    assert route["enforcement"]["feature_requires"] == "OPENBIRD_DEEP_BRAIN_ENABLED"
    assert {
        "distilled_day_memory_content",
        "distilled_day_or_week_memory_content",
        "period_metadata",
        "selected_citation_snippets",
        "selected_source_window_or_url",
        "generated_answer",
        "validated_citations",
    }.issubset(set(route["captured_fields"]))


def test_deep_brain_preview_declares_day_or_week_packet_metadata() -> None:
    route = _routes()["deep_brain.preview"]

    assert route["class"] == "local"
    assert route["egress"]["default"] == "none"
    assert {
        "distilled_day_or_week_memory_content",
        "period_metadata",
        "selected_citation_snippets",
        "packet_build_route",
        "exclusion_counts",
    }.issubset(set(route["captured_fields"]))


def test_productivity_local_facts_route_is_local_and_content_safe() -> None:
    route = _routes()["productivity.local_facts"]

    assert route["class"] == "local"
    assert route["egress"]["default"] == "none"
    assert route["storage"] == ["sqlite.day_memories"]
    assert {
        "active_seconds",
        "context_switch_count",
        "focus_blocks",
        "category_sources",
        "source_ids",
    }.issubset(set(route["captured_fields"]))
    assert {
        "captured_text",
        "raw_window_title",
        "raw_url",
        "raw_title",
    }.issubset(set(route["forbidden_fields"]))


def test_productivity_coach_inherits_route_and_forbids_prompt_source_ids() -> None:
    route = _routes()["productivity.coach"]

    assert route["class"] == "unknown"
    assert route["egress"]["default"] == "inherits_active_model_route"
    assert route["egress"]["inherited_from"] == ["chat.local", "chat.remote"]
    assert route["enforcement"]["feature_requires"] == "OPENBIRD_DEEP_BRAIN_ENABLED"
    assert "cli.productivity_coach" in route["truth_surface"]
    assert {
        "user_question",
        "productivity_facts",
        "synthetic_citation_ids",
        "generated_coaching",
        "validated_citations_local_source_ids",
        "exclusion_counts",
        "exclusion_metadata",
    }.issubset(set(route["captured_fields"]))
    assert {
        "captured_text",
        "raw_window_title",
        "raw_url",
        "raw_title",
        "source_ids_in_prompt",
        "observation_ids_in_prompt",
    }.issubset(set(route["forbidden_fields"]))


# --------------------------------------------------------------------------- #
# Code-vs-manifest enforcement                                                #
#                                                                             #
# These tests are the real teeth: they drive the runtime cloud-egress         #
# classifier (``classify_models``) and the per-route gating, then cross-check #
# that surface against the manifest. They do NOT re-parse the YAML and compare #
# it to itself. ``classify_models`` is the single source of truth for the     #
# runtime's cloud-egress surface: all three runtime egress call sites          #
# (``litellm.completion`` chat, ``litellm.embedding`` embeddings, and the      #
# rerank ``/v1/rerank`` POST) are gated EXCLUSIVELY through it + the cloud      #
# opt-in, so the set of roles it can emit IS the egress surface. The explicit  #
# ``CLOUD_EGRESS_ROUTES`` registry maps each such role to its manifest route.  #
#                                                                             #
# Privacy-safe: no test below makes a real network call. Cloud configs are     #
# classified/refused BEFORE any socket is opened (the opt-in gate fires in the #
# provider constructor), and loopback hosts are never actually contacted.      #
# --------------------------------------------------------------------------- #


def _all_remote_settings() -> Settings:
    """Settings whose every role resolves to a REMOTE (off-device) route.

    Exercises all three egress call sites at once: a cloud LLM, a cloud embed
    model, and a reranker pointed at a non-loopback host.
    """
    return Settings(
        llm_model="gpt-4o",
        embed_model="text-embedding-3-small",
        embed_dim=1536,
        rerank_model="bge-reranker-v2-m3",
        rerank_host="https://rerank.example.com",
    )


def _all_local_settings() -> Settings:
    """Settings whose every role is on-device (the local-first default shape)."""
    return Settings(
        llm_model="ollama/qwen3:4b",
        embed_model="ollama/embeddinggemma",
        rerank_model="bge-reranker-v2-m3",
        rerank_host="http://127.0.0.1:8080",
    )


def test_egress_registry_roles_match_classifier_roles() -> None:
    """Every role the classifier can emit is registered, and vice versa.

    If a future change adds a NEW cloud-egress role to ``classify_models``
    without registering it (and therefore without a manifest route), this fails —
    that is the undeclared-egress tripwire.
    """
    emitted = set(classify_models(_all_remote_settings(), ollama_host=LOOPBACK_OLLAMA))
    assert emitted == set(CLOUD_EGRESS_ROUTES), (
        "classify_models emitted roles that are not in CLOUD_EGRESS_ROUTES (or "
        "vice versa). A new cloud-egress role MUST be registered AND declared in "
        f"docs/privacy-routes.yaml. emitted={sorted(emitted)} "
        f"registered={sorted(CLOUD_EGRESS_ROUTES)}"
    )


def test_egress_every_registered_role_has_a_gated_egress_route() -> None:
    """Each registered egress role maps to a declared, egress-class, opt-in route.

    Pulls the route the code says egresses, finds it in the manifest, and asserts
    the manifest marks it as a real (egress=true) cloud route gated by
    OPENBIRD_ALLOW_CLOUD. A route that egressed without the opt-in gate, or whose
    route_class did not actually egress, fails here.
    """
    data = yaml.safe_load((ROOT / "docs/privacy-routes.yaml").read_text())
    routes = {r["id"]: r for r in data["routes"]}
    classes = data["route_classes"]

    for role, route_id in CLOUD_EGRESS_ROUTES.items():
        assert route_id in routes, (
            f"role {role!r} egresses to undeclared route {route_id!r}"
        )
        route = routes[route_id]
        route_class = route["class"]
        assert classes[route_class]["egress"] is True, (
            f"route {route_id!r} (role {role!r}) is a cloud-egress path but its "
            f"class {route_class!r} declares egress=false"
        )
        assert route.get("enforcement", {}).get("requires") == "OPENBIRD_ALLOW_CLOUD", (
            f"cloud-egress route {route_id!r} (role {role!r}) must be gated by "
            "OPENBIRD_ALLOW_CLOUD — a route cannot egress when the policy is "
            "local-only"
        )


def test_egress_local_config_does_not_egress() -> None:
    """Policy is local-only -> NO role egresses (the route is truly off)."""
    remote = classify_models(_all_local_settings(), ollama_host=LOOPBACK_OLLAMA)
    assert remote == {}, (
        f"a local-only config must not egress, but classify_models found: {remote}"
    )
    # And the provider builds without any cloud opt-in.
    provider = create_llm_provider(_all_local_settings(), allow_cloud=False)
    assert provider is not None


def test_egress_remote_config_egresses_and_is_refused_without_optin() -> None:
    """Policy resolves remote -> every role egresses AND the gate refuses w/o opt-in.

    Proves the gating is real: the same config that classify_models flags as
    remote is REFUSED at provider construction unless cloud is opted into. No
    network call happens — the refusal precedes any socket.
    """
    settings = _all_remote_settings()
    remote = classify_models(settings, ollama_host=LOOPBACK_OLLAMA)
    assert set(remote) == set(CLOUD_EGRESS_ROUTES)

    with pytest.raises(CloudOptInRequired):
        create_llm_provider(settings, allow_cloud=False)

    # With explicit opt-in the gate stops refusing (still no network call: the
    # constructor only classifies + stores config; nothing is sent here).
    provider = create_llm_provider(settings, allow_cloud=True)
    assert provider is not None


def test_egress_rerank_gate_follows_loopback_policy() -> None:
    """The rerank.remote route egresses iff its host is non-loopback.

    Directly exercises the gating predicate behind the rerank ``/v1/rerank`` call
    site so the manifest's ``rerank_host_is_not_loopback`` condition is enforced,
    not just asserted as a YAML string.
    """
    disabled = Settings(rerank_model="", rerank_host="https://rerank.example.com")
    assert rerank_is_remote(disabled) is False  # no model -> no route at all

    loopback = Settings(rerank_model="bge-reranker-v2-m3", rerank_host="http://127.0.0.1:8080")
    assert rerank_is_remote(loopback) is False
    assert "rerank" not in classify_models(loopback, ollama_host=LOOPBACK_OLLAMA)

    remote = Settings(rerank_model="bge-reranker-v2-m3", rerank_host="https://rerank.example.com")
    assert rerank_is_remote(remote) is True
    assert classify_models(remote, ollama_host=LOOPBACK_OLLAMA).get("rerank") == "bge-reranker-v2-m3"


def test_egress_manifest_declares_no_undeclared_optin_call_site_route() -> None:
    """The registry's mapped routes are exactly the per-call-site egress routes.

    Manifest routes gated by OPENBIRD_ALLOW_CLOUD fall into two kinds: the three
    per-call-site egress routes (chat.remote, embedding.remote, rerank.remote) and
    the resolver routes that describe WHERE egress goes (model.remote_ollama,
    model.third_party_cloud). This pins the per-call-site set to the registry, so a
    new opt-in *call-site* route added to the manifest without a backing code role
    (or a registry entry pointed at the wrong route) is caught.
    """
    data = yaml.safe_load((ROOT / "docs/privacy-routes.yaml").read_text())
    routes = {r["id"]: r for r in data["routes"]}

    # Resolver routes name themselves via these self-referential conditions.
    resolver_ids = {"model.remote_ollama", "model.third_party_cloud"}

    optin_routes = {
        rid
        for rid, r in routes.items()
        if r.get("enforcement", {}).get("requires") == "OPENBIRD_ALLOW_CLOUD"
    }
    call_site_routes = optin_routes - resolver_ids
    assert call_site_routes == set(CLOUD_EGRESS_ROUTES.values()), (
        "per-call-site cloud-egress routes in the manifest must match the code's "
        f"CLOUD_EGRESS_ROUTES registry. manifest={sorted(call_site_routes)} "
        f"registry={sorted(CLOUD_EGRESS_ROUTES.values())}"
    )
