import AppKit
import XCTest
@testable import OpenBirdApp

@MainActor
final class MemoryStatsTests: XCTestCase {
    func testMenuBarSymbolUsesBrandedBirdIcon() {
        let cases = [
            (actual: AppModel.menuBarSymbol(captureRunning: false, capturePaused: false), expected: "bird"),
            (actual: AppModel.menuBarSymbol(captureRunning: true, capturePaused: false), expected: "bird.fill"),
            (actual: AppModel.menuBarSymbol(captureRunning: true, capturePaused: true), expected: "pause.circle")
        ]

        for (actual, expected) in cases {
            XCTAssertEqual(actual, expected)
            XCTAssertNotNil(NSImage(systemSymbolName: actual, accessibilityDescription: nil))
        }
    }

    func testChildEnvironmentDisablesPythonBytecodeWrites() {
        let env = OpenBirdService.childEnvironment(base: ["PYTHONDONTWRITEBYTECODE": "0"])

        XCTAssertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
    }

    func testChildEnvironmentDisablesPythonKeyringFallbackWhenAppKeyMissing() {
        let env = OpenBirdService.childEnvironment(base: [:])

        XCTAssertEqual(env["OPENBIRD_DISABLE_KEYRING"], "1")
        XCTAssertEqual(env["OPENBIRD_REQUIRE_ENCRYPTION"], "1")
    }

    func testChildEnvironmentRespectsExplicitKeyringOptOutForDevSelfTest() {
        // A developer (e.g. the headless self-test) who EXPLICITLY exported
        // OPENBIRD_DISABLE_KEYRING=1 has opted into a plaintext dev DB — the forced
        // OPENBIRD_REQUIRE_ENCRYPTION guard must NOT override that, or the chat
        // subprocess refuses to open the plaintext DB. The strict default still
        // applies when the operator did NOT set the flag (asserted above).
        let env = OpenBirdService.childEnvironment(base: ["OPENBIRD_DISABLE_KEYRING": "1"])

        XCTAssertEqual(env["OPENBIRD_DISABLE_KEYRING"], "1")
        XCTAssertNil(env["OPENBIRD_REQUIRE_ENCRYPTION"])
    }

    func testChildEnvironmentKeyringOptOutMatchesPythonTruthyValues() {
        // The opt-out must recognize the SAME truthy set the Python keyring parser
        // uses (storage/crypto.py: 1/true/yes/on, case-insensitive, trimmed) — not
        // just "1" — or e.g. `=true` silently forces encryption and breaks the
        // plaintext dev path.
        for raw in ["true", "YES", " On ", "1"] {
            let env = OpenBirdService.childEnvironment(base: ["OPENBIRD_DISABLE_KEYRING": raw])
            XCTAssertNil(env["OPENBIRD_REQUIRE_ENCRYPTION"], "expected opt-out for \(raw)")
        }
        // A non-truthy value is NOT an opt-out: the strict default still applies.
        let strict = OpenBirdService.childEnvironment(base: ["OPENBIRD_DISABLE_KEYRING": "0"])
        XCTAssertEqual(strict["OPENBIRD_REQUIRE_ENCRYPTION"], "1")
    }

    func testKeyringDisabledTruthHelperMatchesChildEnvironmentOptOut() {
        for raw in ["1", "true", "YES", " On "] {
            XCTAssertTrue(
                OpenBirdService.isKeyringExplicitlyDisabled(
                    environment: ["OPENBIRD_DISABLE_KEYRING": raw]
                ),
                "expected keyring opt-out for \(raw)"
            )
        }
        for raw in ["", "0", "false", "off", "no"] {
            XCTAssertFalse(
                OpenBirdService.isKeyringExplicitlyDisabled(
                    environment: ["OPENBIRD_DISABLE_KEYRING": raw]
                ),
                "did not expect keyring opt-out for \(raw)"
            )
        }
    }

    func testSelfTestBootstrapsKeyUnlessPlaintextDevOptOutIsExplicit() {
        XCTAssertFalse(
            AppDelegate.shouldBootstrapDBKeyForSelfTest(
                environment: ["OPENBIRD_DISABLE_KEYRING": "1"]
            )
        )
        XCTAssertFalse(
            AppDelegate.shouldBootstrapDBKeyForSelfTest(
                environment: ["OPENBIRD_DISABLE_KEYRING": "true"]
            )
        )
        XCTAssertTrue(
            AppDelegate.shouldBootstrapDBKeyForSelfTest(environment: [:])
        )
        XCTAssertTrue(
            AppDelegate.shouldBootstrapDBKeyForSelfTest(
                environment: ["OPENBIRD_DISABLE_KEYRING": "0"]
            )
        )
    }

    func testSelfTestErrorSignalUsesReasonCodesOnly() {
        XCTAssertEqual(AppDelegate.selfTestErrorSignal(ChatError.cliMissing), "cli_missing")
        XCTAssertEqual(
            AppDelegate.selfTestErrorSignal(
                ChatError.failed("raw provider stderr must not be emitted")
            ),
            "chat_failed"
        )
        XCTAssertEqual(AppDelegate.selfTestErrorSignal(ChatError.decode), "decode")
        XCTAssertEqual(AppDelegate.selfTestErrorSignal(NSError(domain: "x", code: 1)), "unknown")
    }

    func testParseMemoryStatsDecodesCliJson() {
        let stats = OpenBirdService.parseMemoryStats("""
        {
          "observations": 7,
          "blobs": 3,
          "chunks": 11,
          "vectors": 11,
          "embed_dim": 768,
          "cohort_key": "ollama:ollama/nomic-embed-text:768:test",
          "encryption_enabled": false
        }
        """)

        XCTAssertEqual(stats?.observations, 7)
        XCTAssertEqual(stats?.blobs, 3)
        XCTAssertEqual(stats?.chunks, 11)
        XCTAssertEqual(stats?.vectors, 11)
    }

    func testParseMemoryStatsRejectsNonJsonOutput() {
        XCTAssertNil(OpenBirdService.parseMemoryStats("not json"))
    }

    func testParseDeepBrainStatusDecodesRouteAndCountsObservationIdExclusions() {
        let status = OpenBirdService.parseDeepBrainStatus("""
        {
          "route": "deep_brain.status",
          "egress": "none",
          "route_label": "Deep Brain local ask available · no cloud",
          "deep_brain_enabled": true,
          "cloud_opt_in": false,
          "cloud_gates_enabled": false,
          "cloud_blocked_reasons": ["OPENBIRD_ALLOW_CLOUD is not enabled"],
          "ask_available": true,
          "ask_blocked_reasons": [],
          "exclusions": {
            "excluded_apps_configured": ["Code"],
            "excluded_sources_configured": ["capture"],
            "excluded_observation_ids_configured": 2
          }
        }
        """)

        XCTAssertEqual(status?.route, "deep_brain.status")
        XCTAssertEqual(status?.egress, "none")
        XCTAssertEqual(status?.routeLabel, "Deep Brain local ask available · no cloud")
        XCTAssertEqual(status?.exclusions.excludedAppsConfigured, ["Code"])
        XCTAssertEqual(status?.exclusions.excludedSourcesConfigured, ["capture"])
        XCTAssertEqual(status?.exclusions.excludedObservationIdsConfigured, 2)
    }

    func testParseDeepBrainPreviewDecodesOnlyCountSurface() {
        let preview = OpenBirdService.parseDeepBrainPreview("""
        {
          "route": "deep_brain.preview",
          "packet_build_route": "deterministic_distillation",
          "egress": "none_preview",
          "cloud_ready": false,
          "blocked_reasons": ["OPENBIRD_ALLOW_CLOUD is not enabled"],
          "local_date": "2026-06-27",
          "day_offset": 0,
          "source_scope": "capture",
          "memory_summary": {
            "coverage": {"observations": 9},
            "workstreams": [{"label": "secret launch plan"}]
          },
          "selected_sources": [
            {
              "observation_id": "obs-secret",
              "app": "Code",
              "window_or_url": "https://example.com/private?token=secret",
              "ts": 1000,
              "snippet": "secret packet snippet"
            }
          ],
          "sources_total": 4,
          "exclusions": {
            "input_observations": 9,
            "kept_observations": 7,
            "excluded_observations": 2,
            "excluded_by": {"app": 1, "source": 1},
            "unknown_app_kept": 0,
            "excluded_apps_configured": ["Code"],
            "excluded_sources_configured": ["private"],
            "excluded_observation_ids_configured": 1
          }
        }
        """)

        XCTAssertEqual(preview?.route, "deep_brain.preview")
        XCTAssertEqual(preview?.egress, "none_preview")
        XCTAssertEqual(preview?.localDate, "2026-06-27")
        XCTAssertEqual(preview?.sourcesTotal, 4)
        XCTAssertEqual(preview?.exclusions.inputObservations, 9)
        XCTAssertEqual(preview?.exclusions.keptObservations, 7)
        XCTAssertEqual(preview?.exclusions.excludedObservations, 2)
        XCTAssertEqual(preview?.exclusions.excludedBy, ["app": 1, "source": 1])
        XCTAssertEqual(preview?.exclusions.excludedAppsConfigured, ["Code"])
        XCTAssertEqual(preview?.exclusions.excludedSourcesConfigured, ["private"])
        XCTAssertEqual(preview?.exclusions.excludedObservationIdsConfigured, 1)
    }

    func testDeepBrainPreviewSummaryLabelsUnitsAndDoesNotLeakRawPacketContent() {
        guard let preview = OpenBirdService.parseDeepBrainPreview("""
        {
          "route": "deep_brain.preview",
          "egress": "none_preview",
          "cloud_ready": true,
          "local_date": "2026-06-27",
          "day_offset": 0,
          "source_scope": "capture",
          "memory_summary": {"workstreams": [{"label": "secret launch plan"}]},
          "selected_sources": [
            {
              "observation_id": "obs-secret",
              "app": "Code",
              "window_or_url": "https://example.com/private?token=secret",
              "ts": 1000,
              "snippet": "secret packet snippet"
            }
          ],
          "sources_total": 4,
          "exclusions": {
            "input_observations": 9,
            "kept_observations": 7,
            "excluded_observations": 2,
            "excluded_by": {"app": 1, "source": 1},
            "unknown_app_kept": 0,
            "excluded_apps_configured": ["Code"],
            "excluded_sources_configured": ["private"],
            "excluded_observation_ids_configured": 1
          }
        }
        """) else {
            XCTFail("expected preview JSON to decode")
            return
        }
        let model = AppModel(service: OpenBirdService(), initialReport: PreflightReport())

        model.setDeepBrainPreviewForTesting(.loaded(preview))

        XCTAssertTrue(model.deepBrainPreviewSummary.contains("Local snapshot"))
        XCTAssertTrue(model.deepBrainPreviewSummary.contains("7 eligible observations"))
        XCTAssertTrue(model.deepBrainPreviewSummary.contains("2 excluded observations"))
        XCTAssertTrue(model.deepBrainPreviewSummary.contains("4 available source groups"))
        XCTAssertTrue(model.deepBrainPreviewSummary.contains("No provider or cloud send was used"))
        XCTAssertTrue(model.deepBrainPreviewSummary.contains("Cloud gates are ready if you ask"))
        XCTAssertFalse(model.deepBrainPreviewSummary.contains("obs-secret"))
        XCTAssertFalse(model.deepBrainPreviewSummary.contains("secret packet snippet"))
        XCTAssertFalse(model.deepBrainPreviewSummary.contains("secret launch plan"))
        XCTAssertFalse(model.deepBrainPreviewSummary.contains("token=secret"))
        assertNoAbsoluteDeviceClaim(model.deepBrainPreviewSummary)
    }

    func testDeepBrainPreviewEmptyAndFailureCopyStayCautious() {
        let model = AppModel(service: OpenBirdService(), initialReport: PreflightReport())

        XCTAssertTrue(model.deepBrainPreviewSummary.contains("user-triggered"))
        XCTAssertTrue(model.deepBrainPreviewSummary.contains("does not use a provider"))
        model.setDeepBrainPreviewForTesting(.failed)
        XCTAssertTrue(model.deepBrainPreviewSummary.contains("Could not build"))
        XCTAssertTrue(model.deepBrainPreviewSummary.contains("No provider or cloud send was used"))
    }

    func testParseDeepBrainPreviewRejectsNonJsonOutput() {
        XCTAssertNil(OpenBirdService.parseDeepBrainPreview("not json"))
    }

    func testDeepBrainLocalAskSummaryStaysLocalAndDoesNotPrintObservationIds() {
        let model = AppModel(service: OpenBirdService(), initialReport: PreflightReport())
        model.setDeepBrainStatusForTesting(.loaded(deepBrainStatus(
            routeLabel: "Deep Brain local ask available · no cloud",
            deepBrainEnabled: true,
            cloudOptIn: false,
            cloudGatesEnabled: false,
            askAvailable: true,
            apps: ["Code"],
            sources: ["capture"],
            observationIdCount: 2
        )))

        XCTAssertEqual(model.deepBrainStatusTitle, "Deep Brain local ask available · no cloud")
        XCTAssertEqual(model.deepBrainStatusBadge, "Local ask")
        XCTAssertFalse(model.deepBrainStatusNeedsAttention)
        XCTAssertTrue(model.deepBrainStatusSummary.contains("local model route without cloud"))
        XCTAssertTrue(model.deepBrainStatusSummary.contains("Status check is local"))
        XCTAssertTrue(model.deepBrainStatusSummary.contains("apps: Code"))
        XCTAssertTrue(model.deepBrainStatusSummary.contains("sources: capture"))
        XCTAssertTrue(model.deepBrainStatusSummary.contains("2 observation ids"))
        XCTAssertFalse(model.deepBrainStatusSummary.contains("obs-secret"))
        assertNoAbsoluteDeviceClaim(model.deepBrainStatusSummary)
    }

    func testDeepBrainCloudGatesSummaryDoesNotOverclaimNoEgress() {
        let model = AppModel(service: OpenBirdService(), initialReport: PreflightReport())
        model.setDeepBrainStatusForTesting(.loaded(deepBrainStatus(
            routeLabel: "Cloud reasoning gates enabled",
            deepBrainEnabled: true,
            cloudOptIn: true,
            cloudGatesEnabled: true,
            askAvailable: true
        )))

        XCTAssertEqual(model.deepBrainStatusBadge, "Cloud gates")
        XCTAssertFalse(model.deepBrainStatusNeedsAttention)
        XCTAssertTrue(model.deepBrainStatusSummary.contains("Cloud reasoning gates are enabled"))
        XCTAssertTrue(model.deepBrainStatusSummary.contains("when you ask"))
        XCTAssertTrue(model.deepBrainStatusSummary.contains("Status check is local"))
        assertNoAbsoluteDeviceClaim(model.deepBrainStatusSummary)
    }

    func testDeepBrainStatusFailureIsCautious() {
        let model = AppModel(service: OpenBirdService(), initialReport: PreflightReport())
        model.setDeepBrainStatusForTesting(.failed)

        XCTAssertEqual(model.deepBrainStatusTitle, "Deep Brain status unavailable")
        XCTAssertEqual(model.deepBrainStatusBadge, "Unavailable")
        XCTAssertTrue(model.deepBrainStatusNeedsAttention)
        XCTAssertTrue(model.deepBrainStatusSummary.contains("Could not read Deep Brain status"))
    }

    func testDeepBrainOffIsNotWarnedBecauseItIsOptional() {
        let model = AppModel(service: OpenBirdService(), initialReport: PreflightReport())
        model.setDeepBrainStatusForTesting(.loaded(deepBrainStatus(
            routeLabel: "Deep Brain off",
            deepBrainEnabled: false,
            cloudOptIn: false,
            cloudGatesEnabled: false,
            askAvailable: false
        )))

        XCTAssertEqual(model.deepBrainStatusBadge, "Off")
        XCTAssertFalse(model.deepBrainStatusNeedsAttention)
        XCTAssertTrue(model.deepBrainStatusSummary.contains("Deep Brain ask is off"))
        assertNoAbsoluteDeviceClaim(model.deepBrainStatusSummary)
    }

    func testDataDeletionSummaryNamesConfirmationCascadeAndVacuum() {
        let model = AppModel(service: OpenBirdService(), initialReport: PreflightReport())

        XCTAssertTrue(model.dataDeletionSummary.contains("ask for confirmation"))
        XCTAssertTrue(model.dataDeletionSummary.contains("cascade-deletes"))
        XCTAssertTrue(model.dataDeletionSummary.contains("blobs/chunks/FTS/vector"))
        XCTAssertTrue(model.dataDeletionSummary.contains("vacuum"))
        XCTAssertTrue(model.dataDeletionSummary.contains(AppModel.dataPurgeAllCommand))
        XCTAssertFalse(model.dataDeletionSummary.contains("--yes"))
        assertNoAbsoluteDeviceClaim(model.dataDeletionSummary)
    }

    func testCopyDataPruneCommandUsesInjectedPasteboardAndKeepsConfirmationGate() {
        final class Recorder: @unchecked Sendable {
            var copied: [String] = []
        }
        let recorder = Recorder()
        let service = OpenBirdService(pasteboardWriter: { recorder.copied.append($0) })
        let model = AppModel(service: service, initialReport: PreflightReport())

        model.copyDataPruneCommand()

        XCTAssertEqual(recorder.copied, [AppModel.dataPruneCommand])
        XCTAssertTrue(AppModel.dataPruneCommand.contains("data prune"))
        XCTAssertTrue(AppModel.dataPruneCommand.contains("--older-than 90d"))
        XCTAssertFalse(AppModel.dataPruneCommand.contains("purge --all"))
        XCTAssertFalse(AppModel.dataPruneCommand.contains("--yes"))
        XCTAssertEqual(model.lastActionMessage, "Copied confirmation-gated prune command.")
    }

    func testDeepBrainAskCommandSummaryNamesOneShotStdinAndSeparateCloudOptIn() {
        let model = AppModel(service: OpenBirdService(), initialReport: PreflightReport())

        XCTAssertTrue(model.deepBrainAskCommandSummary.contains("one-shot Terminal command"))
        XCTAssertTrue(model.deepBrainAskCommandSummary.contains("Preview the packet first"))
        XCTAssertTrue(model.deepBrainAskCommandSummary.contains("does not persist consent"))
        XCTAssertTrue(model.deepBrainAskCommandSummary.contains("type your question"))
        XCTAssertTrue(model.deepBrainAskCommandSummary.contains("press Ctrl-D"))
        XCTAssertTrue(model.deepBrainAskCommandSummary.contains("Remote LLM routes still require separate OPENBIRD_ALLOW_CLOUD=1"))
        assertNoAbsoluteDeviceClaim(model.deepBrainAskCommandSummary)
    }

    func testCopyDeepBrainAskCommandUsesInjectedPasteboardAndKeepsCloudOptInSeparate() {
        final class Recorder: @unchecked Sendable {
            var copied: [String] = []
        }
        let recorder = Recorder()
        let service = OpenBirdService(pasteboardWriter: { recorder.copied.append($0) })
        let model = AppModel(service: service, initialReport: PreflightReport())

        model.copyDeepBrainAskCommand()

        XCTAssertEqual(recorder.copied, [AppModel.deepBrainAskCommand])
        XCTAssertTrue(AppModel.deepBrainAskCommand.contains("OPENBIRD_DEEP_BRAIN_ENABLED=1"))
        XCTAssertTrue(AppModel.deepBrainAskCommand.contains("openbird deep-brain ask"))
        XCTAssertTrue(AppModel.deepBrainAskCommand.contains("--day 0"))
        XCTAssertTrue(AppModel.deepBrainAskCommand.contains("--stdin"))
        XCTAssertFalse(AppModel.deepBrainAskCommand.contains("OPENBIRD_ALLOW_CLOUD"))
        XCTAssertFalse(AppModel.deepBrainAskCommand.contains("OPENBIRD_LLM_MODEL"))
        XCTAssertFalse(AppModel.deepBrainAskCommand.contains("OPENBIRD_EMBED_MODEL"))
        XCTAssertFalse(AppModel.deepBrainAskCommand.contains("--yes"))
        XCTAssertFalse(AppModel.deepBrainAskCommand.contains("purge"))
        XCTAssertEqual(model.lastActionMessage, "Copied one-shot Deep Brain ask command.")
    }

    func testProductivityCoachCommandSummaryNamesLocalFactsAndSeparateCloudOptIn() {
        let model = AppModel(service: OpenBirdService(), initialReport: PreflightReport())

        XCTAssertTrue(model.productivityCoachCommandSummary.contains("one-shot Terminal command"))
        XCTAssertTrue(model.productivityCoachCommandSummary.contains("openbird productivity"))
        XCTAssertTrue(model.productivityCoachCommandSummary.contains("local-only, no model"))
        XCTAssertTrue(model.productivityCoachCommandSummary.contains("does not persist consent"))
        XCTAssertTrue(model.productivityCoachCommandSummary.contains("type your question"))
        XCTAssertTrue(model.productivityCoachCommandSummary.contains("press Ctrl-D"))
        XCTAssertTrue(model.productivityCoachCommandSummary.contains("Remote LLM routes still require separate OPENBIRD_ALLOW_CLOUD=1"))
        assertNoAbsoluteDeviceClaim(model.productivityCoachCommandSummary)
    }

    func testCopyProductivityCoachCommandUsesInjectedPasteboardAndKeepsCloudOptInSeparate() {
        final class Recorder: @unchecked Sendable {
            var copied: [String] = []
        }
        let recorder = Recorder()
        let service = OpenBirdService(pasteboardWriter: { recorder.copied.append($0) })
        let model = AppModel(service: service, initialReport: PreflightReport())

        model.copyProductivityCoachCommand()

        XCTAssertEqual(recorder.copied, [AppModel.productivityCoachCommand])
        XCTAssertTrue(AppModel.productivityCoachCommand.contains("OPENBIRD_DEEP_BRAIN_ENABLED=1"))
        XCTAssertTrue(AppModel.productivityCoachCommand.contains("openbird productivity-coach"))
        XCTAssertTrue(AppModel.productivityCoachCommand.contains("--day 0"))
        XCTAssertTrue(AppModel.productivityCoachCommand.contains("--stdin"))
        XCTAssertFalse(AppModel.productivityCoachCommand.contains("OPENBIRD_ALLOW_CLOUD"))
        XCTAssertFalse(AppModel.productivityCoachCommand.contains("OPENBIRD_LLM_MODEL"))
        XCTAssertFalse(AppModel.productivityCoachCommand.contains("OPENBIRD_EMBED_MODEL"))
        XCTAssertFalse(AppModel.productivityCoachCommand.contains("--yes"))
        XCTAssertFalse(AppModel.productivityCoachCommand.contains("purge"))
        XCTAssertEqual(model.lastActionMessage, "Copied one-shot productivity coach command.")
    }

    func testParsePreflightDecodesLocalModelRoute() {
        let report = OpenBirdService.parsePreflight("""
        {
          "runtime_ok": true,
          "release_gate_ok": false,
          "ollama": {
            "reachable": true,
            "host": "http://localhost:11434",
            "required_models": ["llama3.2", "nomic-embed-text"],
            "missing_models": [],
            "auto_pull_allowed": true,
            "version_ok": true
          },
          "cloud": {
            "llm_model": "ollama/llama3.2",
            "embed_model": "ollama/nomic-embed-text",
            "remote_models": {},
            "uses_local_ollama": true,
            "blocked": false
          }
        }
        """)

        XCTAssertTrue(report.ollamaReachable == true)
        XCTAssertEqual(report.ollamaHost, "http://localhost:11434")
        XCTAssertEqual(report.requiredModels, ["llama3.2", "nomic-embed-text"])
        XCTAssertEqual(report.missingModels, [])
        XCTAssertTrue(report.autoPullAllowed)
        XCTAssertEqual(report.ollamaVersionOK, true)
        XCTAssertEqual(report.llmModel, "ollama/llama3.2")
        XCTAssertEqual(report.embedModel, "ollama/nomic-embed-text")
        XCTAssertEqual(report.remoteModelRoles, [:])
        XCTAssertEqual(report.remoteModels, [])
        XCTAssertTrue(report.usesLocalOllama)
        XCTAssertFalse(report.cloudBlocked)
    }

    func testParsePreflightDecodesTooOldOllamaVersionGate() {
        let report = OpenBirdService.parsePreflight("""
        {
          "runtime_ok": false,
          "ollama": {
            "reachable": true,
            "required_models": ["qwen3:8b", "embeddinggemma"],
            "missing_models": [],
            "version_ok": false
          },
          "cloud": {
            "llm_model": "ollama/qwen3:8b",
            "embed_model": "ollama/embeddinggemma",
            "uses_local_ollama": true
          }
        }
        """)

        XCTAssertEqual(report.ollamaVersionOK, false)
        XCTAssertEqual(report.embedModel, "ollama/embeddinggemma")
    }

    func testParsePreflightLeavesMissingOllamaVersionUnknown() {
        let report = OpenBirdService.parsePreflight("""
        {
          "runtime_ok": true,
          "ollama": {
            "reachable": true,
            "required_models": ["llama3.2", "nomic-embed-text"],
            "missing_models": []
          }
        }
        """)

        XCTAssertNil(report.ollamaVersionOK)
    }

    func testParsePreflightDecodesRoleKeyedRemoteModels() {
        let report = OpenBirdService.parsePreflight("""
        {
          "runtime_ok": true,
          "cloud": {
            "llm_model": "openai/gpt-4o-mini",
            "embed_model": "openai/text-embedding-3-small",
            "remote_models": {
              "llm": "openai/gpt-4o-mini",
              "embed": "openai/text-embedding-3-small"
            },
            "uses_local_ollama": false,
            "blocked": false
          }
        }
        """)

        XCTAssertEqual(report.remoteModelRoles, [
            "embed": "openai/text-embedding-3-small",
            "llm": "openai/gpt-4o-mini"
        ])
        XCTAssertEqual(report.remoteModels, [
            "openai/text-embedding-3-small",
            "openai/gpt-4o-mini"
        ])
        XCTAssertFalse(report.usesLocalOllama)
    }

    func testParsePreflightPreservesLegacyArrayRemoteModels() {
        let report = OpenBirdService.parsePreflight("""
        {
          "runtime_ok": true,
          "cloud": {
            "llm_model": "openai/gpt-4o-mini",
            "embed_model": "openai/text-embedding-3-small",
            "remote_models": [
              "openai/gpt-4o-mini",
              "openai/text-embedding-3-small"
            ],
            "uses_local_ollama": false,
            "blocked": false
          }
        }
        """)

        XCTAssertEqual(report.remoteModelRoles, [:])
        XCTAssertEqual(report.remoteModels, [
            "openai/gpt-4o-mini",
            "openai/text-embedding-3-small"
        ])
    }

    func testParsePreflightDecodesMeetingTranscriptionReadiness() throws {
        let report = OpenBirdService.parsePreflight("""
        {
          "runtime_ok": true,
          "meetings": {
            "transcription": {
              "parakeet_mlx_available": true,
              "faster_whisper_available": false,
              "backend_available": true,
              "recommended_backend": "parakeet-mlx",
              "recommended_extra": "meetings-mlx",
              "fallback_backend": "faster-whisper",
              "fallback_extra": "meetings"
            }
          }
        }
        """)

        let readiness = try XCTUnwrap(report.meetingTranscription)
        XCTAssertTrue(readiness.parakeetMLXAvailable)
        XCTAssertFalse(readiness.fasterWhisperAvailable)
        XCTAssertTrue(readiness.backendAvailable)
        XCTAssertEqual(readiness.recommendedBackend, "parakeet-mlx")
        XCTAssertEqual(readiness.recommendedExtra, "meetings-mlx")
        XCTAssertEqual(readiness.fallbackBackend, "faster-whisper")
        XCTAssertEqual(readiness.fallbackExtra, "meetings")
    }

    func testParsePreflightLeavesMeetingTranscriptionUnknownWhenMissing() {
        let report = OpenBirdService.parsePreflight("""
        {
          "runtime_ok": true,
          "release_gate_ok": false
        }
        """)

        XCTAssertNil(report.meetingTranscription)
    }

    func testParsePreflightLeavesPartialMeetingTranscriptionPayloadUnknown() {
        let report = OpenBirdService.parsePreflight("""
        {
          "runtime_ok": true,
          "meetings": {
            "transcription": {
              "parakeet_mlx_available": false,
              "backend_available": false,
              "recommended_backend": "parakeet-mlx"
            }
          }
        }
        """)

        XCTAssertNil(report.meetingTranscription)
    }

    func testLocalRoutePrivacyCopyStaysRouteConditional() {
        let model = AppModel(service: OpenBirdService(), initialReport: localRuntimeOKReport())

        XCTAssertEqual(model.localModelStatusState, .ok)
        // Local route renders as "on-device" in the sidebar footer (handoff copy); the
        // remote/blocked/unknown labels asserted elsewhere keep their honest wording.
        XCTAssertEqual(model.modelRouteFooterLabel, "on-device")
        XCTAssertTrue(model.privacyTransmissionSummary.contains("Model requests stay on this Mac"))
        assertNoAbsoluteDeviceClaim(model.privacyTransmissionSummary)
        assertNoAbsoluteDeviceClaim(model.localModelStatusSummary)
    }

    func testDefaultRouteStateIsCautiousUntilPreflightRuns() {
        let model = AppModel(service: OpenBirdService())

        XCTAssertEqual(model.localModelStatusState, .unknown)
        XCTAssertEqual(model.modelRouteProvisioningState, .unknown)
        XCTAssertEqual(model.modelRouteFooterLabel, "model route unknown")
        XCTAssertTrue(model.privacyTransmissionSummary.contains("not verified yet"))
        assertNoAbsoluteDeviceClaim(model.privacyTransmissionSummary)
        assertNoUnknownLocalRouteClaim(model.privacyTransmissionSummary)
    }

    func testMeetingTranscriptionUnknownUntilPreflightReportsIt() {
        let model = AppModel(service: OpenBirdService())

        XCTAssertEqual(model.meetingTranscriptionState, .unknown)
        XCTAssertEqual(
            model.meetingTranscriptionSummary,
            "Re-check setup to detect parakeet-mlx or faster-whisper."
        )
    }

    func testMeetingTranscriptionPrefersParakeetWhenAvailable() {
        var report = PreflightReport()
        report.meetingTranscription = meetingReadiness(
            parakeet: true,
            whisper: true,
            available: true
        )
        let model = AppModel(service: OpenBirdService(), initialReport: report)

        XCTAssertEqual(model.meetingTranscriptionState, .ok)
        XCTAssertEqual(
            model.meetingTranscriptionSummary,
            "parakeet-mlx ready · Apple Silicon recommended"
        )
    }

    func testMeetingTranscriptionReportsWhisperFallback() {
        var report = PreflightReport()
        report.meetingTranscription = meetingReadiness(
            parakeet: false,
            whisper: true,
            available: true
        )
        let model = AppModel(service: OpenBirdService(), initialReport: report)

        XCTAssertEqual(model.meetingTranscriptionState, .ok)
        XCTAssertEqual(
            model.meetingTranscriptionSummary,
            "faster-whisper ready · portable fallback"
        )
    }

    func testMeetingTranscriptionMissingNamesCliEnvironmentInstall() {
        var report = PreflightReport()
        report.meetingTranscription = meetingReadiness(
            parakeet: false,
            whisper: false,
            available: false
        )
        let model = AppModel(service: OpenBirdService(), initialReport: report)

        XCTAssertEqual(model.meetingTranscriptionState, .attention)
        XCTAssertTrue(model.meetingTranscriptionSummary.contains("parakeet-mlx is recommended on Apple Silicon"))
        XCTAssertTrue(model.meetingTranscriptionSummary.contains("faster-whisper is the portable fallback"))
        XCTAssertTrue(model.meetingTranscriptionSummary.contains("openbird CLI environment"))
        XCTAssertFalse(model.meetingTranscriptionSummary.contains("uv sync"))
    }

    func testPreflightErrorRouteStateIsCautious() {
        let report = PreflightReport(error: "Could not parse preflight JSON.")
        let model = AppModel(service: OpenBirdService(), initialReport: report)

        XCTAssertEqual(model.localModelStatusState, .attention)
        XCTAssertEqual(model.modelRouteFooterLabel, "model route unknown")
        XCTAssertTrue(model.localModelStatusSummary.contains("Could not parse preflight JSON."))
        XCTAssertTrue(model.privacyTransmissionSummary.contains("could not be verified"))
        assertNoAbsoluteDeviceClaim(model.privacyTransmissionSummary)
        assertNoUnknownLocalRouteClaim(model.privacyTransmissionSummary)
    }

    func testCloudBlockedRouteBlocksReadinessAndNamesRemoteModel() {
        var report = remoteReport(runtimeOK: false, blocked: true, roles: [
            "llm": "openai/gpt-4o-mini"
        ])
        report.grants["accessibility"] = "passed"
        let model = AppModel(service: OpenBirdService(), initialReport: report)

        XCTAssertEqual(model.localModelStatusState, .attention)
        XCTAssertEqual(model.modelRouteProvisioningState, .remoteRoute)
        XCTAssertEqual(model.modelRouteFooterLabel, "remote blocked")
        XCTAssertFalse(model.isFullyConfigured)
        XCTAssertTrue(model.localModelStatusSummary.contains("openai/gpt-4o-mini"))
        XCTAssertTrue(model.localModelStatusSummary.contains("opt in"))
        XCTAssertTrue(model.privacyTransmissionSummary.contains("captured memory stays local until"))
        assertNoAbsoluteDeviceClaim(model.privacyTransmissionSummary)
    }

    func testRemoteActiveVerifiedRouteDisclosesTransmissionButNeedsCapturedMemoryForReadiness() {
        var report = remoteReport(runtimeOK: true, blocked: false, roles: [
            "llm": "openai/gpt-4o-mini"
        ])
        report.grants["accessibility"] = "passed"
        let model = AppModel(service: OpenBirdService(), initialReport: report)

        XCTAssertEqual(model.localModelStatusState, .ok)
        XCTAssertEqual(model.modelRouteProvisioningState, .remoteRoute)
        XCTAssertEqual(model.modelRouteFooterLabel, "remote model")
        XCTAssertFalse(model.isFullyConfigured)
        XCTAssertTrue(model.nextStepSummary.contains("capture allowlist"))
        XCTAssertTrue(model.privacyTransmissionSummary.contains("may send captured memory"))
        XCTAssertTrue(model.privacyTransmissionSummary.contains("openai/gpt-4o-mini"))
        assertNoAbsoluteDeviceClaim(model.privacyTransmissionSummary)
    }

    func testRemoteActiveVerifiedRouteWithCapturedMemoryCanBeReady() {
        var report = remoteReport(runtimeOK: true, blocked: false, roles: [
            "llm": "openai/gpt-4o-mini"
        ])
        report.grants["accessibility"] = "passed"
        let model = AppModel(service: OpenBirdService(), initialReport: report)
        model.setReadinessStateForTesting(
            allowlist: ["com.example.editor"],
            captureRunning: true,
            memoryStats: MemoryStats(observations: 1, blobs: 1, chunks: 1, vectors: 1)
        )

        XCTAssertEqual(model.localModelStatusState, .ok)
        XCTAssertEqual(model.modelRouteProvisioningState, .remoteRoute)
        XCTAssertEqual(model.modelRouteFooterLabel, "remote model")
        XCTAssertTrue(model.isFullyConfigured)
        XCTAssertEqual(model.nextStepSummary, "Ready: capture is storing memory. Ask a question when you need it.")
        XCTAssertTrue(model.privacyTransmissionSummary.contains("may send captured memory"))
        XCTAssertTrue(model.privacyTransmissionSummary.contains("openai/gpt-4o-mini"))
        assertNoAbsoluteDeviceClaim(model.privacyTransmissionSummary)
    }

    func testRemoteActiveUnverifiedRouteDisclosesTransmissionButBlocksReadiness() {
        var report = remoteReport(runtimeOK: false, blocked: false, roles: [
            "llm": "openai/gpt-4o-mini"
        ])
        report.grants["accessibility"] = "passed"
        let model = AppModel(service: OpenBirdService(), initialReport: report)

        XCTAssertEqual(model.localModelStatusState, .attention)
        XCTAssertEqual(model.modelRouteProvisioningState, .remoteRoute)
        XCTAssertEqual(model.modelRouteFooterLabel, "remote model")
        XCTAssertFalse(model.isFullyConfigured)
        XCTAssertTrue(model.localModelStatusSummary.contains("configured but not verified by preflight"))
        XCTAssertTrue(model.privacyTransmissionSummary.contains("may send captured memory"))
        assertNoAbsoluteDeviceClaim(model.privacyTransmissionSummary)
    }

    func testMixedLocalAndRemoteRouteDisclosesRemoteTransmission() {
        var report = remoteReport(runtimeOK: false, blocked: false, roles: [
            "embed": "openai/text-embedding-3-small"
        ])
        report.usesLocalOllama = true
        report.ollamaReachable = true
        report.requiredModels = ["llama3.2"]
        report.missingModels = []
        let model = AppModel(service: OpenBirdService(), initialReport: report)

        XCTAssertTrue(model.hasRemoteModelRoute)
        XCTAssertTrue(model.localModelStatusSummary.contains("embed=openai/text-embedding-3-small"))
        XCTAssertTrue(model.privacyTransmissionSummary.contains("may send captured memory"))
        assertNoAbsoluteDeviceClaim(model.privacyTransmissionSummary)
    }

    func testEmbedOnlyRemoteRoleIsEnoughToMarkRouteRemote() {
        let model = AppModel(
            service: OpenBirdService(),
            initialReport: remoteReport(runtimeOK: true, blocked: false, roles: [
                "embed": "openai/text-embedding-3-small"
            ])
        )

        XCTAssertTrue(model.hasRemoteModelRoute)
        XCTAssertEqual(model.modelRouteProvisioningState, .remoteRoute)
        XCTAssertTrue(model.privacyTransmissionSummary.contains("embed=openai/text-embedding-3-small"))
        XCTAssertTrue(model.privacyTransmissionSummary.contains("may send captured memory"))
    }

    func testLocalMissingModelsCanBePulledOnlyWhenPreflightAllowsIt() {
        var report = localRuntimeOKReport()
        report.runtimeOK = false
        report.missingModels = ["llama3.2"]
        report.autoPullAllowed = true
        let model = AppModel(service: OpenBirdService(), initialReport: report)

        XCTAssertEqual(model.modelRouteProvisioningState, .modelsMissing(canPull: true))
        XCTAssertEqual(model.modelRouteActionLabel, "Download models")
        XCTAssertTrue(model.canPullMissingModels)
    }

    func testLocalMissingModelsBlockedWhenPreflightDisallowsAutoPull() {
        var report = localRuntimeOKReport()
        report.runtimeOK = false
        report.missingModels = ["llama3.2"]
        report.autoPullAllowed = false
        let model = AppModel(service: OpenBirdService(), initialReport: report)

        XCTAssertEqual(model.modelRouteProvisioningState, .modelsMissing(canPull: false))
        XCTAssertNil(model.modelRouteActionLabel)
        XCTAssertFalse(model.canPullMissingModels)
        XCTAssertTrue(model.localModelStatusSummary.contains("automatic model download is disabled"))
    }

    func testRefreshClearsStaleProvisioningError() async {
        let service = OpenBirdService(openBirdCLIResolver: { nil })
        let model = AppModel(service: service, initialReport: localRuntimeOKReport())
        model.setProvisioningErrorForTesting("stale pull failure")

        XCTAssertEqual(model.modelRouteProvisioningState, .error("stale pull failure"))

        await model.refresh()

        XCTAssertNotEqual(model.modelRouteProvisioningState, .error("stale pull failure"))
        XCTAssertFalse(model.localModelStatusSummary.contains("stale pull failure"))
    }

    func testProvisioningErrorShowsRetryWhenModelsStillMissing() {
        var report = localRuntimeOKReport()
        report.runtimeOK = false
        report.missingModels = ["llama3.2"]
        let model = AppModel(service: OpenBirdService(), initialReport: report)
        model.setProvisioningErrorForTesting("pull failed")

        XCTAssertEqual(model.modelRouteProvisioningState, .error("pull failed"))
        XCTAssertEqual(model.modelRouteActionLabel, "Retry")
        XCTAssertTrue(model.localModelStatusSummary.contains("pull failed"))
    }

    func testOllamaUnavailableShowsGetOllamaAction() {
        var report = localRuntimeOKReport()
        report.runtimeOK = false
        report.ollamaReachable = false
        report.missingModels = ["llama3.2"]
        let model = AppModel(service: OpenBirdService(), initialReport: report)

        XCTAssertEqual(model.modelRouteProvisioningState, .ollamaUnavailable)
        XCTAssertEqual(model.modelRouteActionLabel, "Get Ollama")
    }

    func testTooOldOllamaForEmbeddingGemmaShowsSpecificSetupCopyAndAction() {
        var report = embeddingGemmaTooOldOllamaReport()
        report.missingModels = ["embeddinggemma"]
        let model = AppModel(service: OpenBirdService(), initialReport: report)

        XCTAssertEqual(model.localModelStatusState, .attention)
        XCTAssertEqual(model.modelRouteProvisioningState, .ollamaTooOldForEmbeddingGemma)
        XCTAssertEqual(model.modelRouteActionLabel, "Update Ollama")
        let minimumOllamaVersion = AppModel.embeddingGemmaMinimumOllamaVersion
        XCTAssertTrue(model.localModelStatusSummary.contains("Ollama is too old for embeddinggemma"))
        XCTAssertTrue(model.localModelStatusSummary.contains("\(minimumOllamaVersion) or newer"))
        XCTAssertTrue(model.nextStepSummary.contains("update Ollama to \(minimumOllamaVersion) or newer for embeddinggemma"))

        var openedURL: URL?
        model.performModelRouteAction { openedURL = $0 }
        XCTAssertEqual(openedURL?.absoluteString, "https://ollama.com")
    }

    func testUnknownOllamaVersionKeepsLegacyRuntimeNotReadyCopy() {
        var report = embeddingGemmaTooOldOllamaReport()
        report.ollamaVersionOK = nil
        let model = AppModel(service: OpenBirdService(), initialReport: report)

        XCTAssertEqual(model.modelRouteProvisioningState, .unknown)
        XCTAssertNil(model.modelRouteActionLabel)
        XCTAssertTrue(model.localModelStatusSummary.contains("Local model route is not runtime-ready"))
        XCTAssertFalse(model.localModelStatusSummary.contains("too old"))
    }

    func testPullProgressLineDecodesPercent() throws {
        let progress = try XCTUnwrap(OpenBirdService.parsePullProgressLine(
            #"{"status":"pulling manifest","completed":25,"total":100}"#,
            model: "llama3.2"
        ))

        XCTAssertEqual(progress.model, "llama3.2")
        XCTAssertEqual(progress.status, "pulling manifest")
        XCTAssertEqual(progress.completed, 25)
        XCTAssertEqual(progress.total, 100)
        XCTAssertEqual(progress.fraction, 0.25)
    }

    func testPullProgressLineThrowsOnOllamaError() {
        XCTAssertThrowsError(try OpenBirdService.parsePullProgressLine(
            #"{"error":"model not found"}"#,
            model: "llama3.2"
        ))
    }

    func testChatFailureSummaryClassifiesMissingModel() {
        XCTAssertEqual(
            OpenBirdService.chatFailureSummary(exitCode: 1, stderr: "model missing: llama3.2"),
            "Chat failed because a required local model is missing."
        )
    }

    func testChatFailureSummaryClassifiesCloudBlock() {
        XCTAssertEqual(
            OpenBirdService.chatFailureSummary(exitCode: 2, stderr: "Set OPENBIRD_ALLOW_CLOUD=1"),
            "Chat blocked because a cloud model is configured without opt-in."
        )
    }

    func testChatFailureSummaryClassifiesOllamaFailure() {
        XCTAssertEqual(
            OpenBirdService.chatFailureSummary(exitCode: 3, stderr: "connection refused to Ollama"),
            "Chat failed because the local Ollama model request did not complete."
        )
    }

    func testChatFailureSummaryPrefersMissingModelOverGenericOllama() {
        XCTAssertEqual(
            OpenBirdService.chatFailureSummary(exitCode: 5, stderr: "ollama error: model not found"),
            "Chat failed because a required local model is missing."
        )
    }

    func testChatFailureSummaryAvoidsRawUnknownStderr() {
        XCTAssertEqual(
            OpenBirdService.chatFailureSummary(exitCode: 4, stderr: "private traceback with captured text"),
            "Chat failed (exit 4). Run openbird doctor for details."
        )
    }

    private func localRuntimeOKReport() -> PreflightReport {
        var report = PreflightReport()
        report.runtimeOK = true
        report.ollamaReachable = true
        report.requiredModels = ["llama3.2", "nomic-embed-text"]
        report.missingModels = []
        report.llmModel = "ollama/llama3.2"
        report.embedModel = "ollama/nomic-embed-text"
        report.remoteModelRoles = [:]
        report.remoteModels = []
        report.usesLocalOllama = true
        report.autoPullAllowed = true
        return report
    }

    private func embeddingGemmaTooOldOllamaReport() -> PreflightReport {
        var report = localRuntimeOKReport()
        report.runtimeOK = false
        report.requiredModels = ["qwen3:8b", "embeddinggemma"]
        report.missingModels = []
        report.llmModel = "ollama/qwen3:8b"
        report.embedModel = "ollama/embeddinggemma"
        report.ollamaVersionOK = false
        return report
    }

    private func meetingReadiness(
        parakeet: Bool,
        whisper: Bool,
        available: Bool
    ) -> MeetingTranscriptionReadiness {
        MeetingTranscriptionReadiness(
            parakeetMLXAvailable: parakeet,
            fasterWhisperAvailable: whisper,
            backendAvailable: available,
            recommendedBackend: "parakeet-mlx",
            recommendedExtra: "meetings-mlx",
            fallbackBackend: "faster-whisper",
            fallbackExtra: "meetings"
        )
    }

    private func remoteReport(
        runtimeOK: Bool,
        blocked: Bool,
        roles: [String: String]
    ) -> PreflightReport {
        var report = PreflightReport()
        report.runtimeOK = runtimeOK
        report.ollamaReachable = nil
        report.remoteModelRoles = roles
        report.remoteModels = roles.keys.sorted().compactMap { roles[$0] }
        report.usesLocalOllama = false
        report.cloudBlocked = blocked
        report.llmModel = roles["llm"] ?? "ollama/llama3.2"
        report.embedModel = roles["embed"] ?? "ollama/nomic-embed-text"
        return report
    }

    private func deepBrainStatus(
        routeLabel: String,
        deepBrainEnabled: Bool,
        cloudOptIn: Bool,
        cloudGatesEnabled: Bool,
        askAvailable: Bool,
        apps: [String] = [],
        sources: [String] = [],
        observationIdCount: Int = 0
    ) -> DeepBrainStatus {
        DeepBrainStatus(
            route: "deep_brain.status",
            egress: "none",
            routeLabel: routeLabel,
            deepBrainEnabled: deepBrainEnabled,
            cloudOptIn: cloudOptIn,
            cloudGatesEnabled: cloudGatesEnabled,
            cloudBlockedReasons: cloudGatesEnabled ? [] : ["OPENBIRD_ALLOW_CLOUD is not enabled"],
            askAvailable: askAvailable,
            askBlockedReasons: askAvailable ? [] : ["OPENBIRD_DEEP_BRAIN_ENABLED is not enabled"],
            exclusions: DeepBrainStatus.Exclusions(
                excludedAppsConfigured: apps,
                excludedSourcesConfigured: sources,
                excludedObservationIdsConfigured: observationIdCount
            )
        )
    }

    private func assertNoAbsoluteDeviceClaim(
        _ text: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertFalse(text.contains("Nothing leaves your device"), file: file, line: line)
        XCTAssertFalse(text.contains("keeps everything on this device"), file: file, line: line)
    }

    private func assertNoUnknownLocalRouteClaim(
        _ text: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertFalse(text.contains("No remote model route is configured"), file: file, line: line)
        XCTAssertFalse(text.contains("Model requests stay on this Mac"), file: file, line: line)
    }
}
