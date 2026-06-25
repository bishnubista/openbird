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

    func testRemoteActiveVerifiedRouteDisclosesTransmissionAndCanBeReady() {
        var report = remoteReport(runtimeOK: true, blocked: false, roles: [
            "llm": "openai/gpt-4o-mini"
        ])
        report.grants["accessibility"] = "passed"
        let model = AppModel(service: OpenBirdService(), initialReport: report)

        XCTAssertEqual(model.localModelStatusState, .ok)
        XCTAssertEqual(model.modelRouteProvisioningState, .remoteRoute)
        XCTAssertEqual(model.modelRouteFooterLabel, "remote model")
        XCTAssertTrue(model.isFullyConfigured)
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
