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
        "briefing.model",
        "chat.day_facts",
        "chat.day_memory",
        "chat.local",
        "chat.remote",
        "assistant.mcp_read",
        "data.export",
        "deep_brain.ask",
        "deep_brain.preview",
        "deep_brain.status",
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
        "summaries.block",
        "summaries.week",
        "chat.day_memory_cached_summary",
        "chat.week_memory_cached_summary",
        "chat.summary_grounded",
        "entities.aggregation",
        "chat.entity_ledger",
    }.issubset(routes)


def test_assistant_mcp_route_is_bounded_explicit_egress() -> None:
    route = _routes()["assistant.mcp_read"]

    assert route["class"] == "third-party-cloud"
    assert route["egress"]["default"] == "none_until_content_tool_invocation"
    assert route["enforcement"]["transport"] == "local_stdio_only"
    assert route["enforcement"]["model_calls"] == "forbidden"
    assert route["enforcement"]["unknown_app"] == "fail_closed"
    assert route["enforcement"]["install_requires"] == (
        "explicit warning and interactive confirmation or --yes"
    )
    assert route["enforcement"]["exclusions"] == (
        "deep_brain app, source, and observation-id exclusions before serialization"
    )
    assert route["enforcement"]["bounds"] == (
        "2000 characters per excerpt and 12000 excerpt characters per call"
    )
    assert "raw_url" in route["forbidden_fields"]
    assert "raw_window_title" in route["forbidden_fields"]
    assert "unknown_app_observation" in route["forbidden_fields"]
    assert "cli.assistant_install_warning" in route["truth_surface"]
    assert "mcp.egress_notice" in route["truth_surface"]


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


def test_capture_ocr_window_route_is_local_and_pixel_free() -> None:
    route = _routes()["capture.ocr_window"]

    assert route["class"] == "local"
    assert route["derives_from"] == ["capture.active_window"]
    # The load-bearing privacy claims: text is scrubbed BEFORE storage and
    # pixels are transient (never persisted or logged).
    assert "recognized_text_scrubbed_before_storage" in route["captured_fields"]
    assert "never persists" in route["pixels"]
    assert route["egress"]["default"] == "none"
    assert route["enforcement"]["failure_mode"] == "fail_closed"
    assert "never prompts" in route["enforcement"]["boundary"]
    assert "app.deep_capture_section" in route["truth_surface"]
    # Same storage/deletion contract as the AX capture route.
    ax = _routes()["capture.active_window"]
    assert route["storage"] == ax["storage"]
    assert route["deletion"] == ax["deletion"]


def test_deep_brain_ask_inherits_active_model_route() -> None:
    route = _routes()["deep_brain.ask"]

    assert route["class"] == "unknown"
    assert route["storage"] == ["sqlite.reasoning_send_ledger_metadata"]
    assert route["egress"]["default"] == "inherits_active_model_route"
    assert route["egress"]["inherited_from"] == ["chat.local", "chat.remote"]
    assert "requires" not in route.get("enforcement", {})
    assert route["enforcement"]["feature_requires"] == "OPENBIRD_DEEP_BRAIN_ENABLED"
    assert "cli.data_reasoning_ledger" in route["truth_surface"]
    assert {
        "distilled_day_memory_content",
        "distilled_day_or_week_memory_content",
        "period_metadata",
        "selected_citation_snippets",
        "selected_source_window_or_url",
        "packet_build_route",
        "generated_answer",
        "validated_citations",
        "exclusion_counts",
        "exclusion_metadata",
    }.issubset(set(route["captured_fields"]))
    assert {
        "ledger_raw_question",
        "ledger_generated_answer",
        "ledger_packet_json",
        "ledger_observation_ids",
        "ledger_citation_ids",
        "ledger_configured_exclusion_names",
        "ledger_source_app_names",
        "ledger_url_or_window_title_metadata",
    }.issubset(set(route["forbidden_fields"]))


def test_deep_brain_preview_declares_day_or_week_packet_metadata() -> None:
    route = _routes()["deep_brain.preview"]

    assert route["class"] == "local"
    assert route["storage"] == []
    assert route["egress"]["default"] == "none"
    assert {
        "distilled_day_or_week_memory_content",
        "period_metadata",
        "selected_citation_snippets",
        "packet_build_route",
        "exclusion_counts",
        "exclusion_metadata",
    }.issubset(set(route["captured_fields"]))


def test_deep_brain_status_route_is_local_settings_only() -> None:
    route = _routes()["deep_brain.status"]

    assert route["class"] == "local"
    assert route["storage"] == []
    assert route["egress"]["default"] == "none"
    assert "no provider" in route["egress"]["note"]
    assert "cli.deep_brain_status" in route["truth_surface"]
    assert "app.deepBrainStatusRow" in route["truth_surface"]
    assert {
        "opt_in_gate_status",
        "exclusion_metadata",
        "configured_exclusion_counts",
    }.issubset(set(route["captured_fields"]))


def test_briefing_model_uses_distilled_packet_and_cloud_opt_in_only() -> None:
    route = _routes()["briefing.model"]

    assert route["class"] == "unknown"
    assert route["storage"] == ["sqlite.reasoning_send_ledger_metadata"]
    assert route["egress"]["default"] == "inherits_active_model_route"
    assert route["egress"]["inherited_from"] == ["chat.local", "chat.remote"]
    assert route["enforcement"]["requires"] == "OPENBIRD_ALLOW_CLOUD"
    assert "feature_requires" not in route["enforcement"]
    assert "cli.briefing_model" in route["truth_surface"]
    assert "cli.CLOUD_ACTIVE" in route["truth_surface"]
    assert "cli.data_reasoning_ledger" in route["truth_surface"]
    assert {
        "model_briefing_prompt",
        "distilled_day_memory_content",
        "selected_citation_snippets",
        "packet_build_route",
        "packet_preview_egress_label",
        "opt_in_gate_status",
        "generated_briefing",
        "validated_citations",
        "exclusion_counts",
        "exclusion_metadata",
    }.issubset(set(route["captured_fields"]))
    assert {
        "ledger_raw_question",
        "ledger_generated_briefing",
        "ledger_packet_json",
        "ledger_observation_ids",
        "ledger_citation_ids",
        "ledger_configured_exclusion_names",
        "ledger_source_app_names",
        "ledger_url_or_window_title_metadata",
    }.issubset(set(route["forbidden_fields"]))


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


def test_chat_day_facts_route_is_local_branch_over_day_memory() -> None:
    route = _routes()["chat.day_facts"]

    assert route["class"] == "local"
    assert route["derives_from"] == ["productivity.local_facts"]
    assert route["storage"] == ["sqlite.day_memories"]
    assert route["egress"]["default"] == "none"
    assert "_provider" in route["egress"]["note"]
    assert "_MaintenanceProvider" in route["egress"]["note"]
    assert "provider.complete" in route["egress"]["note"]
    assert "No completion or embedding request" in route["egress"]["note"]
    # The explicit --day branch instantiates a non-egress maintenance stub; the
    # route must not collapse that into a false generic provider claim.
    assert "provider is constructed" not in route["egress"]["note"]
    assert "inherited_from" not in route["egress"]
    assert "inherited_by" not in route["egress"]
    assert "resolved_by" not in route["egress"]
    assert set(route["captured_fields"]) == {
        "user_question_for_classification_only",
        "active_seconds",
        "active_minutes",
        "context_switch_count",
        "top_category",
        "top_hour",
        "longest_focus_block",
        "derived_citation_source_ids",
        "memory_context_counts",
    }
    assert set(route["forbidden_fields"]) == {
        "captured_text",
        "raw_window_title",
        "raw_url",
        "raw_title",
        "source_ids_in_memory_context",
    }
    assert set(route["truth_surface"]) == {
        "cli.chat_json.reasoning_route",
        "app.ChatResult.routeLabel",
    }


def test_chat_day_memory_route_is_local_synthesis_over_day_distillation() -> None:
    route = _routes()["chat.day_memory"]

    assert route["class"] == "local"
    assert route["derives_from"] == ["capture.active_window"]
    assert route["storage"] == ["sqlite.day_memories"]
    assert route["egress"]["default"] == "none"
    assert "_provider" in route["egress"]["note"]
    assert "_MaintenanceProvider" in route["egress"]["note"]
    assert "provider.complete" in route["egress"]["note"]
    assert "No completion or embedding request" in route["egress"]["note"]
    assert "provider is constructed" not in route["egress"]["note"]
    assert "inherited_from" not in route["egress"]
    assert "inherited_by" not in route["egress"]
    assert "resolved_by" not in route["egress"]
    assert set(route["captured_fields"]) == {
        "user_question_for_classification_only",
        "distilled_day_memory_metrics",
        "workstreams",
        "sessions",
        "open_loops",
        "domains",
        "repos",
        "derived_citation_source_ids",
        "memory_context_counts",
    }
    assert set(route["forbidden_fields"]) == {
        "captured_text",
        "raw_window_title",
        "raw_url",
        "raw_title",
        "source_ids_in_memory_context",
    }
    assert set(route["truth_surface"]) == {
        "cli.chat_json.reasoning_route",
        "app.ChatResult.routeLabel",
    }


def test_chat_day_facts_is_not_modeled_as_egress_inheritance() -> None:
    routes = _routes()
    for route in routes.values():
        egress = route.get("egress", {})
        inherited = set(egress.get("inherited_by", []))
        inherited.update(egress.get("inherited_from", []))
        inherited.update(egress.get("resolved_by", []))
        assert "chat.day_facts" not in inherited
        assert "chat.day_memory" not in inherited


def test_productivity_coach_inherits_route_and_forbids_prompt_source_ids() -> None:
    route = _routes()["productivity.coach"]

    assert route["class"] == "unknown"
    assert route["storage"] == ["sqlite.reasoning_send_ledger_metadata"]
    assert route["egress"]["default"] == "inherits_active_model_route"
    assert route["egress"]["inherited_from"] == ["chat.local", "chat.remote"]
    assert route["enforcement"]["feature_requires"] == "OPENBIRD_DEEP_BRAIN_ENABLED"
    assert "cli.productivity_coach" in route["truth_surface"]
    assert "cli.data_reasoning_ledger" in route["truth_surface"]
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
        "ledger_raw_question",
        "ledger_generated_coaching",
        "ledger_packet_json",
        "ledger_observation_ids",
        "ledger_citation_ids",
        "ledger_configured_exclusion_names",
        "ledger_source_app_names",
        "ledger_url_or_window_title_metadata",
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
    inherited_model_routes = {
        rid
        for rid, r in routes.items()
        if r.get("egress", {}).get("default") == "inherits_active_model_route"
    }
    call_site_routes = optin_routes - resolver_ids - inherited_model_routes
    assert call_site_routes == set(CLOUD_EGRESS_ROUTES.values()), (
        "per-call-site cloud-egress routes in the manifest must match the code's "
        f"CLOUD_EGRESS_ROUTES registry. manifest={sorted(call_site_routes)} "
        f"registry={sorted(CLOUD_EGRESS_ROUTES.values())}"
    )


def test_chat_day_memory_cached_summary_route_is_local_and_truthful() -> None:
    """The Phase D composed-day-answer route: local, no answer-time egress, and
    the disclosure surfaces (route field + BOTH app labels) are pinned."""
    route = _routes()["chat.day_memory_cached_summary"]

    assert route["class"] == "local"
    assert set(route["derives_from"]) == {"chat.day_memory", "summaries.block"}
    assert set(route["storage"]) == {"sqlite.day_memories", "sqlite.block_summaries"}
    assert route["egress"]["default"] == "none"
    note = route["egress"]["note"]
    assert "no provider call" in note
    assert "battery/idle gate" in note
    assert "byte-identical" in note
    assert "stored_block_summary_prose" in route["captured_fields"]
    assert "derived_citation_typed_source_refs" in route["captured_fields"]
    assert "captured_text" in route["forbidden_fields"]
    # The app must label this route on BOTH surfaces (chat + briefing).
    assert set(route["truth_surface"]) == {
        "cli.chat_json.reasoning_route",
        "app.ChatResult.routeLabel",
        "app.DayBriefing.routeLabel",
    }


def test_summaries_block_route_declares_gated_generation_and_counts_only() -> None:
    route = _routes()["summaries.block"]

    assert route["class"] == "local"
    assert route["derives_from"] == ["capture.active_window"]
    assert set(route["storage"]) == {
        "sqlite.block_summaries",
        "sqlite.category_assignments",
    }
    # Generation inherits the active model route (cloud-opted users opted in);
    # the note pins the battery/idle gate + CloudOptInRequired enforcement.
    assert route["egress"]["default"] == "inherits_active_model_route"
    note = route["egress"]["note"]
    assert "battery/idle gate" in note
    assert "CloudOptInRequired" in note
    assert "counts + reason codes only" in note
    assert "summary_text_in_logs" in route["forbidden_fields"]
    assert "summary_text_in_routine_output" in route["forbidden_fields"]
    deletion = route["deletion"]["must_remove"]
    assert "block_summaries_citing_deleted_sources" in deletion
    assert "day_memories_citing_deleted_summaries" in deletion
    assert "category_assignments_on_full_purge" in deletion


def test_summaries_week_route_mirrors_block_gating_and_deletion() -> None:
    """Phase E1 week-digest generation: same gate/egress model as summaries.block,
    with the summary-index deletion contract pinned."""
    route = _routes()["summaries.week"]

    assert route["class"] == "local"
    assert route["derives_from"] == ["summaries.block"]
    assert set(route["storage"]) == {
        "sqlite.day_memories",
        "sqlite.summary_index_entries",
        "sqlite.fts_summaries",
        "sqlite.vec_summaries",
    }
    assert route["egress"]["default"] == "inherits_active_model_route"
    note = route["egress"]["note"]
    assert "battery/idle gate" in note
    assert "CloudOptInRequired" in note
    assert "counts + reason codes only" in note
    assert "week_memory_ungrounded" in note
    assert "digest_text_in_logs" in route["forbidden_fields"]
    assert "digest_text_in_routine_output" in route["forbidden_fields"]
    deletion = route["deletion"]["must_remove"]
    assert "week_rows_citing_deleted_block_summaries" in deletion
    assert "summary_index_entries_and_fts_vec_rows" in deletion


def test_summaries_block_deletion_covers_summary_index_rows() -> None:
    """Phase E1 extends the block deletion contract with the parallel index."""
    route = _routes()["summaries.block"]
    assert (
        "summary_index_entries_and_fts_vec_rows" in route["deletion"]["must_remove"]
    )


def test_chat_week_memory_cached_summary_route_is_local_and_truthful() -> None:
    """The Phase E1 cached week answer: local composition, zero answer-time
    egress, disclosure surfaces pinned (chat + the briefing --week CLI)."""
    route = _routes()["chat.week_memory_cached_summary"]

    assert route["class"] == "local"
    assert set(route["derives_from"]) == {
        "chat.day_memory_cached_summary",
        "summaries.week",
    }
    assert set(route["storage"]) == {"sqlite.day_memories", "sqlite.block_summaries"}
    assert route["egress"]["default"] == "none"
    note = route["egress"]["note"]
    assert "no provider call" in note
    assert "no embedding request" in note
    assert "stored_week_digest_prose" in route["captured_fields"]
    assert "derived_citation_typed_source_refs" in route["captured_fields"]
    assert "captured_text" in route["forbidden_fields"]
    assert set(route["truth_surface"]) == {
        "cli.chat_json.reasoning_route",
        "app.ChatResult.routeLabel",
        "cli.briefing_week",
    }


def test_chat_summary_grounded_route_documents_summary_prose_egress() -> None:
    """Fresh completions over cached summary prose: the prose egresses ONLY
    under the active model route with cloud opt-in (same class as
    retrieved_chunks) — the route documents that honestly."""
    route = _routes()["chat.summary_grounded"]

    assert route["class"] == "unknown"
    assert set(route["derives_from"]) == {
        "summaries.block",
        "summaries.week",
        "chat.local",
        "chat.remote",
    }
    assert route["egress"]["default"] == "inherits_active_model_route"
    assert set(route["egress"]["inherited_from"]) == {"chat.local", "chat.remote"}
    warning = route["egress"]["warning"]
    assert "cloud opt-in" in warning
    assert "retrieved_chunks" in warning
    assert "stored_block_summary_prose_fenced" in route["captured_fields"]
    assert "stored_week_digest_prose_fenced" in route["captured_fields"]
    assert "cli.CLOUD_ACTIVE" in route["truth_surface"]


def test_privacy_route_entities_aggregation_is_local_and_structurally_modelless() -> None:
    route = _routes()["entities.aggregation"]
    assert route["class"] == "local"
    assert route["egress"]["default"] == "none"
    assert set(route["storage"]) == {"sqlite.entities", "sqlite.entity_evidence"}
    forbidden = set(route["forbidden_fields"])
    assert "entity_names_in_logs" in forbidden
    assert "entity_names_in_routine_output" in forbidden
    # Code-vs-manifest: the "structurally no model" claim is checkable — the
    # aggregation entrypoint accepts NO provider argument, so no completion or
    # embedding request can be issued from this route.
    import inspect

    from openbird.entities import run_entity_aggregation

    assert "provider" not in inspect.signature(run_entity_aggregation).parameters


def test_privacy_route_chat_entity_ledger_is_local_deterministic() -> None:
    route = _routes()["chat.entity_ledger"]
    assert route["class"] == "local"
    assert route["egress"]["default"] == "none"
    assert route["derives_from"] == ["entities.aggregation"]
    forbidden = set(route["forbidden_fields"])
    assert {"captured_text", "raw_window_title", "raw_url"} <= forbidden
    # Deletion lineage: the answer path stores nothing of its own — both
    # storage tables belong to the aggregation route's deletion contract.
    assert set(route["storage"]) == {"sqlite.entities", "sqlite.entity_evidence"}


def test_assistant_activity_summary_route_is_metadata_only_egress() -> None:
    route = _routes()["assistant.activity_summary"]

    assert route["class"] == "third-party-cloud"
    assert route["egress"]["default"] == "none_until_tool_invocation"
    assert route["enforcement"]["transport"] == "local_stdio_only"
    assert route["enforcement"]["model_calls"] == "forbidden"
    assert route["enforcement"]["malformed_exclusion_regex"] == "fail_closed"
    assert route["enforcement"]["bucket_precedence"] == (
        "excluded then redacted then afk then visible, strict first-match per span"
    )
    assert "captured_text" in route["forbidden_fields"]
    assert "window_title" in route["forbidden_fields"]
    assert "url_host" in route["forbidden_fields"]
    # Redacted (tier-0) time is attributed per app+reason — that is captured,
    # disclosed metadata. EXCLUDED apps remain the unnamed hiding mechanism.
    assert "excluded_bundle_id" in route["forbidden_fields"]
    assert "redacted_by_app_bundle_id_reason_seconds" in route["captured_fields"]
    assert "redacted_unattributed_seconds" in route["captured_fields"]
    assert "capture_host_label" in route["captured_fields"]
    assert "observation_derived_statistics" in route["forbidden_fields"]
    assert "cli.assistant_install_warning" in route["truth_surface"]
    assert "app.assistant_connect_confirmations" in route["truth_surface"]
    assert "mcp.activity_egress_notice" in route["truth_surface"]


def test_assistant_capture_status_route_is_declared_and_count_scoped() -> None:
    route = _routes()["assistant.capture_status"]

    assert route["class"] == "third-party-cloud"
    assert route["egress"]["default"] == "none_until_tool_invocation"
    assert route["enforcement"]["transport"] == "local_stdio_only"
    assert route["enforcement"]["model_calls"] == "forbidden"
    assert "store-lifetime" in route["enforcement"]["counts_scope"]
    assert "observation_total_count" in route["captured_fields"]
    assert "capture_host_label" in route["captured_fields"]
    assert "captured_text" in route["forbidden_fields"]
    assert "bundle_id" in route["forbidden_fields"]
    assert "cli.assistant_install_warning" in route["truth_surface"]
    assert "app.assistant_connect_confirmations" in route["truth_surface"]
    assert "mcp.status_egress_notice" in route["truth_surface"]


def test_every_assistant_mcp_tool_maps_to_a_declared_route(tmp_path) -> None:
    # The MCP server's tool surface and the privacy manifest must not drift:
    # each of the four tools is covered by exactly one assistant route.
    routes = _routes()
    assert {
        "assistant.mcp_read",
        "assistant.activity_summary",
        "assistant.capture_status",
    } <= set(routes)
    # mcp_read covers both content tools; the other two map one-to-one.
    import asyncio

    from openbird import assistant as assistant_module

    service = assistant_module.AssistantCaptureService(
        settings=Settings(data_dir=tmp_path), store_factory=lambda: None
    )
    tools = asyncio.run(assistant_module.create_mcp_server(service).list_tools())
    assert {tool.name for tool in tools} == {
        "openbird_recent_capture",   # assistant.mcp_read
        "openbird_search_capture",   # assistant.mcp_read
        "openbird_activity_summary", # assistant.activity_summary
        "openbird_capture_status",   # assistant.capture_status
    }
