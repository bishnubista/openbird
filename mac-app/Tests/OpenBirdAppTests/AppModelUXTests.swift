import XCTest
@testable import OpenBirdApp

@MainActor
final class AppModelUXTests: XCTestCase {
    private let allowlistKey = "openbird.captureAllowlist"

    private final class BoolProbe: @unchecked Sendable {
        var value: Bool

        init(_ value: Bool) {
            self.value = value
        }
    }

    private func withRestoredAllowlist<T>(_ body: () throws -> T) rethrows -> T {
        let defaults = UserDefaults.standard
        let old = defaults.stringArray(forKey: allowlistKey)
        defer {
            if let old {
                defaults.set(old, forKey: allowlistKey)
            } else {
                defaults.removeObject(forKey: allowlistKey)
            }
        }
        defaults.removeObject(forKey: allowlistKey)
        return try body()
    }

    private func withRestoredAllowlist<T>(_ body: () async throws -> T) async rethrows -> T {
        let defaults = UserDefaults.standard
        let old = defaults.stringArray(forKey: allowlistKey)
        defer {
            if let old {
                defaults.set(old, forKey: allowlistKey)
            } else {
                defaults.removeObject(forKey: allowlistKey)
            }
        }
        defaults.removeObject(forKey: allowlistKey)
        return try await body()
    }

    /// Build an `OpenBirdService` whose external-capture detection is stubbed to
    /// "nothing running", so `AppModel.init`'s `isCaptureRunning()` call is hermetic
    /// regardless of the host (a dev box may have a real `capture --loop` daemon).
    private func serviceWithoutExternalCapture(
        accessibilityProbe: @escaping @Sendable () -> Bool = { true },
        openBirdCLIResolver: @escaping @Sendable () -> String? = { "/tmp/openbird" }
    ) -> OpenBirdService {
        OpenBirdService(
            accessibilityProbe: accessibilityProbe,
            openBirdCLIResolver: openBirdCLIResolver,
            externalLoopDaemonProbe: { false },
            captureHelperRunningProbe: { false }
        )
    }

    private func readyReport() -> PreflightReport {
        var report = PreflightReport()
        report.ollamaReachable = true
        report.runtimeOK = true
        return report
    }

    private func onboardingModel(
        allowlist: [String] = ["com.example.editor"],
        accessibility: Bool = true,
        cli: String? = "/tmp/openbird",
        captureRunning: Bool = false,
        report: PreflightReport? = nil
    ) -> AppModel {
        let service = OpenBirdService(
            accessibilityProbe: { accessibility },
            openBirdCLIResolver: { cli },
            externalLoopDaemonProbe: { captureRunning },
            captureHelperRunningProbe: { false }
        )
        service.setAllowlist(allowlist)
        let model = AppModel(service: service, initialReport: report ?? readyReport())
        model.setOnboardingStateForTesting(lastRefresh: Date(), captureRunning: captureRunning)
        return model
    }

    func testAskUnavailableReasonPrefersModelReadiness() {
        var report = PreflightReport()
        report.ollamaReachable = false
        let service = OpenBirdService(openBirdCLIResolver: { "/tmp/openbird" })
        let model = AppModel(service: service, initialReport: report)

        XCTAssertEqual(model.askUnavailableReason, "Ollama is not reachable. Launch Ollama, then re-check.")
    }

    func testAskUnavailableReasonKeepsMemoryReadFailureDistinct() async {
        let service = OpenBirdService(openBirdCLIResolver: { nil })
        let model = AppModel(service: service, initialReport: readyReport())

        await model.refreshMemoryStats()

        XCTAssertEqual(model.askUnavailableReason, "Could not check local memory. Re-check setup before asking.")
    }

    // The single-window shell drives the active pane off `model.selection`: the sidebar
    // highlight, the menu bar, and the deep-link router all read/write it. It must
    // default to `.today` (the launch surface) and be freely settable to any destination.
    func testSelectionDefaultsToTodayAndIsSettable() {
        let service = OpenBirdService(openBirdCLIResolver: { "/tmp/openbird" })
        let model = AppModel(service: service)

        XCTAssertEqual(model.selection, .today)

        for destination in AppDestination.allCases {
            model.selection = destination
            XCTAssertEqual(model.selection, destination)
        }
    }

    func testCanStartCaptureNowRequiresAccessibilityAllowlistAndCli() {
        withRestoredAllowlist {
            // Stub external-capture detection: `canStartCaptureNow` is `&& !captureRunning`,
            // so a real `capture --loop` daemon on the host would otherwise flip it false.
            let service = serviceWithoutExternalCapture()
            let model = AppModel(service: service, initialReport: readyReport())

            XCTAssertFalse(model.canStartCaptureNow)
            model.addToAllowlist("com.example.editor")
            XCTAssertTrue(model.canStartCaptureNow)
        }
    }

    func testCanStartCaptureNowRejectsMissingAccessibilityOrCli() {
        withRestoredAllowlist {
            let noAccessibility = OpenBirdService(
                accessibilityProbe: { false },
                openBirdCLIResolver: { "/tmp/openbird" }
            )
            noAccessibility.setAllowlist(["com.example.editor"])
            let noAccessibilityModel = AppModel(service: noAccessibility, initialReport: readyReport())
            XCTAssertFalse(noAccessibilityModel.canStartCaptureNow)

            let noCLI = OpenBirdService(
                accessibilityProbe: { true },
                openBirdCLIResolver: { nil }
            )
            noCLI.setAllowlist(["com.example.editor"])
            let noCLIModel = AppModel(service: noCLI, initialReport: readyReport())
            XCTAssertFalse(noCLIModel.canStartCaptureNow)
        }
    }

    func testCaptureRowStatusUsesEffectivePolicyAndRecentCounts() {
        withRestoredAllowlist {
            let service = serviceWithoutExternalCapture()
            let model = AppModel(service: service, initialReport: readyReport())
            model.setReadinessStateForTesting(
                allowlist: ["com.example.Editor", "com.apple.Terminal"],
                captureRunning: true
            )
            model.setCaptureHealthStateForTesting(.loaded(CaptureHealthReport(
                generatedAt: 200,
                recentWindowSeconds: 86_400,
                paused: false,
                allowlistCount: 2,
                blocklistCount: 1,
                apps: [
                    CaptureHealthApp(
                        bundleID: "com.example.Editor",
                        policy: CaptureHealthPolicy(capture: true, reason: "allowlisted"),
                        effectiveState: "allowed_recent",
                        quality: "good",
                        coverage: "unknown",
                        totalObservations: 4,
                        recentObservations: 2,
                        lastCapturedTS: 123
                    ),
                    CaptureHealthApp(
                        bundleID: "com.apple.Terminal",
                        policy: CaptureHealthPolicy(capture: false, reason: "blocklisted"),
                        effectiveState: "blocked",
                        quality: "blocked",
                        coverage: "degraded",
                        totalObservations: 0,
                        recentObservations: 0,
                        lastCapturedTS: nil
                    )
                ]
            )))

            let capturing = model.captureRowStatus(for: "com.example.Editor")
            XCTAssertEqual(capturing.label, "Capturing")
            XCTAssertEqual(capturing.tone, .ok)
            XCTAssertTrue(capturing.detail.contains("Good signal"))

            let blocked = model.captureRowStatus(for: "com.apple.Terminal")
            XCTAssertEqual(blocked.label, "Blocked by safety")
            XCTAssertEqual(blocked.tone, .attention)
            XCTAssertTrue(blocked.detail.contains("overrides"))
            XCTAssertEqual(model.effectiveCaptureAllowedCount, 1)
            XCTAssertEqual(model.captureAllowedSummary, "1 app effectively allowed")
        }
    }

    func testCaptureAllowedSummaryFallsBackToNominalCopyWhenHealthUnavailable() {
        withRestoredAllowlist {
            let service = serviceWithoutExternalCapture()
            let model = AppModel(service: service, initialReport: readyReport())
            model.setReadinessStateForTesting(
                allowlist: ["com.example.Editor", "com.apple.Terminal"],
                captureRunning: false
            )
            model.setCaptureHealthStateForTesting(.failed)

            XCTAssertEqual(model.effectiveCaptureAllowedCount, 2)
            XCTAssertEqual(model.captureAllowedSummary, "2 apps allowed")
        }
    }

    func testRefreshCaptureHealthSynchronizesRuntimeFlags() async {
        await withRestoredAllowlist {
            let running = BoolProbe(false)
            let service = OpenBirdService(
                accessibilityProbe: { true },
                openBirdCLIResolver: { nil },
                externalLoopDaemonProbe: { running.value },
                captureHelperRunningProbe: { false }
            )
            service.setAllowlist(["com.example.editor"])
            let model = AppModel(service: service, initialReport: readyReport())
            model.setReadinessStateForTesting(allowlist: ["com.example.editor"], captureRunning: false)

            running.value = true
            await model.refreshCaptureHealth()

            XCTAssertTrue(model.captureRunning)
            XCTAssertEqual(model.menuBarSymbol, "bird.fill")
        }
    }

    func testRemoveFromAllowlistUpdatesLastActionMessage() {
        withRestoredAllowlist {
            let service = serviceWithoutExternalCapture()
            service.setAllowlist(["com.example.editor"])
            let model = AppModel(service: service, initialReport: readyReport())

            model.removeFromAllowlist("com.example.editor")

            XCTAssertEqual(model.lastActionMessage, "Removed com.example.editor from capture allowlist.")
        }
    }

    func testCaptureRowStatusShowsAllowedButNoRecentSignal() {
        withRestoredAllowlist {
            let service = serviceWithoutExternalCapture()
            let model = AppModel(service: service, initialReport: readyReport())
            model.setReadinessStateForTesting(allowlist: ["com.example.Editor"], captureRunning: true)
            model.setCaptureHealthStateForTesting(.loaded(CaptureHealthReport(
                generatedAt: 200,
                recentWindowSeconds: 86_400,
                paused: false,
                allowlistCount: 1,
                blocklistCount: 0,
                apps: [
                    CaptureHealthApp(
                        bundleID: "com.example.Editor",
                        policy: CaptureHealthPolicy(capture: true, reason: "allowlisted"),
                        effectiveState: "allowed_no_recent",
                        quality: "no_recent",
                        coverage: "unknown",
                        totalObservations: 0,
                        recentObservations: 0,
                        lastCapturedTS: nil
                    )
                ]
            )))

            let status = model.captureRowStatus(for: "com.example.Editor")
            XCTAssertEqual(status.label, "No recent captures")
            XCTAssertEqual(status.tone, .attention)
            XCTAssertTrue(status.detail.contains("No recent signal"))
        }
    }

    func testCaptureRowStatusSurfacesLowSignalCapture() {
        withRestoredAllowlist {
            let service = serviceWithoutExternalCapture()
            let model = AppModel(service: service, initialReport: readyReport())
            model.setReadinessStateForTesting(allowlist: ["us.zoom.xos"], captureRunning: true)
            model.setCaptureHealthStateForTesting(.loaded(CaptureHealthReport(
                generatedAt: 200,
                recentWindowSeconds: 86_400,
                paused: false,
                allowlistCount: 1,
                blocklistCount: 0,
                apps: [
                    CaptureHealthApp(
                        bundleID: "us.zoom.xos",
                        policy: CaptureHealthPolicy(capture: true, reason: "allowlisted"),
                        effectiveState: "allowed_recent",
                        quality: "low_signal",
                        coverage: "degraded",
                        totalObservations: 3,
                        recentObservations: 1,
                        lastCapturedTS: 190
                    )
                ]
            )))

            let status = model.captureRowStatus(for: "us.zoom.xos")
            XCTAssertEqual(status.label, "Low signal")
            XCTAssertEqual(status.tone, .attention)
            XCTAssertTrue(status.detail.contains("Low signal"))
        }
    }

    func testOnboardingPrimaryActionStartsOnlyWhenReady() {
        withRestoredAllowlist {
            let model = onboardingModel()

            XCTAssertEqual(model.onboardingPrimaryAction, .start)
        }
    }

    func testOnboardingPrimaryActionCheckingUntilRefreshed() {
        withRestoredAllowlist {
            let service = serviceWithoutExternalCapture()
            service.setAllowlist(["com.example.editor"])
            let model = AppModel(service: service, initialReport: readyReport())

            XCTAssertEqual(model.onboardingPrimaryAction, .checking)
        }
    }

    func testOnboardingPrimaryActionCompletesWhenAlreadyRunning() {
        withRestoredAllowlist {
            let model = onboardingModel(captureRunning: true)

            XCTAssertEqual(model.onboardingPrimaryAction, .complete)
        }
    }

    func testOnboardingPrimaryActionBlocksMissingSetup() {
        withRestoredAllowlist {
            XCTAssertBlocked(onboardingModel(allowlist: []).onboardingPrimaryAction, contains: "allowlist")
            XCTAssertBlocked(
                onboardingModel(accessibility: false).onboardingPrimaryAction,
                contains: "Accessibility"
            )

            var modelMissing = readyReport()
            modelMissing.runtimeOK = false
            modelMissing.ollamaReachable = false
            XCTAssertBlocked(
                onboardingModel(report: modelMissing).onboardingPrimaryAction,
                contains: "Ollama"
            )

            XCTAssertBlocked(onboardingModel(cli: nil).onboardingPrimaryAction, contains: "CLI")
        }
    }

    func testOnboardingPresentationDismissDoesNotComplete() {
        var state = OnboardingPresentationState(completed: false, presented: true)

        state.dismissWithoutCompleting()

        XCTAssertFalse(state.completed)
        XCTAssertFalse(state.presented)

        state.presented = true
        state.complete()

        XCTAssertTrue(state.completed)
        XCTAssertFalse(state.presented)
    }

    func testOnboardingRepairIsOneShot() {
        let repaired = AppModel.repairIncompleteOnboardingCompletionIfNeeded(
            completed: true,
            repairDone: false,
            allowlistIsEmpty: true,
            captureRunning: false
        )
        XCTAssertFalse(repaired.completed)
        XCTAssertTrue(repaired.repairDone)
        XCTAssertTrue(repaired.repaired)

        let repeated = AppModel.repairIncompleteOnboardingCompletionIfNeeded(
            completed: true,
            repairDone: true,
            allowlistIsEmpty: true,
            captureRunning: false
        )
        XCTAssertTrue(repeated.completed)
        XCTAssertTrue(repeated.repairDone)
        XCTAssertFalse(repeated.repaired)
    }

    func testOnboardingRepairInstanceWritesDefaults() {
        let defaults = UserDefaults.standard
        let oldCompleted = defaults.object(forKey: AppModel.onboardingCompletedKey)
        let oldRepair = defaults.object(forKey: AppModel.onboardingRepairV1DoneKey)
        defer {
            if let oldCompleted {
                defaults.set(oldCompleted, forKey: AppModel.onboardingCompletedKey)
            } else {
                defaults.removeObject(forKey: AppModel.onboardingCompletedKey)
            }
            if let oldRepair {
                defaults.set(oldRepair, forKey: AppModel.onboardingRepairV1DoneKey)
            } else {
                defaults.removeObject(forKey: AppModel.onboardingRepairV1DoneKey)
            }
        }

        withRestoredAllowlist {
            defaults.set(true, forKey: AppModel.onboardingCompletedKey)
            defaults.set(false, forKey: AppModel.onboardingRepairV1DoneKey)
            let model = onboardingModel(allowlist: [], captureRunning: false)

            XCTAssertTrue(model.repairIncompleteOnboardingCompletionIfNeeded())
            XCTAssertFalse(defaults.bool(forKey: AppModel.onboardingCompletedKey))
            XCTAssertTrue(defaults.bool(forKey: AppModel.onboardingRepairV1DoneKey))
        }
    }

    private func XCTAssertBlocked(
        _ action: OnboardingPrimaryAction,
        contains expected: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        guard case .blocked(let message) = action else {
            XCTFail("Expected blocked action, got \(action)", file: file, line: line)
            return
        }
        XCTAssertTrue(message.contains(expected), "Message was \(message)", file: file, line: line)
    }

    // MARK: - Capture exit-code mapping

    /// The reindex-required exit code must flip `captureNeedsReindex` and surface an
    /// actionable message — that is what drives the one-click Reindex affordance
    /// instead of the dead-end "stopped unexpectedly (exit 1)".
    func testCaptureExitReindexCodeFlagsReindexNeeded() async {
        // Stub external-capture detection so `captureRunning` starts false regardless
        // of host state (a dev box may run a real `capture --loop` daemon).
        let service = serviceWithoutExternalCapture()
        let model = AppModel(service: service, initialReport: readyReport())

        await model.handleCaptureExit(code: AppModel.captureReindexExitCode)

        XCTAssertTrue(model.captureNeedsReindex)
        XCTAssertFalse(model.captureRunning)
        XCTAssertTrue(model.lastActionMessage.lowercased().contains("reindex"))
    }

    /// A generic non-zero exit keeps the original "stopped unexpectedly" message and
    /// must NOT offer reindex — that would mis-route an unrelated crash.
    func testGenericCaptureExitDoesNotOfferReindex() async {
        let service = OpenBirdService(openBirdCLIResolver: { "/tmp/openbird" })
        let model = AppModel(service: service, initialReport: readyReport())

        await model.handleCaptureExit(code: 1)

        XCTAssertFalse(model.captureNeedsReindex)
        XCTAssertTrue(model.lastActionMessage.contains("exit 1"))
    }

    /// The capture "no-progress" failure code (Python `_CAPTURE_NO_PROGRESS_EXIT`,
    /// value 6 — a session that received events but ingested none) must NOT be
    /// mistaken for the reindex-required code (5). Pinning 6 here guards the
    /// cross-component contract: if either side ever re-collides on 5, a broken
    /// capture session would wrongly surface the one-click Reindex affordance.
    func testCaptureNoProgressExitDoesNotOfferReindex() async {
        let noProgressExitCode: Int32 = 6
        XCTAssertNotEqual(noProgressExitCode, AppModel.captureReindexExitCode)

        let service = OpenBirdService(openBirdCLIResolver: { "/tmp/openbird" })
        let model = AppModel(service: service, initialReport: readyReport())

        await model.handleCaptureExit(code: noProgressExitCode)

        XCTAssertFalse(model.captureNeedsReindex)
        XCTAssertTrue(model.lastActionMessage.contains("exit 6"))
    }

    /// A clean exit (code 0) reports a plain stop, no reindex prompt.
    func testCleanCaptureExitReportsStopped() async {
        let service = OpenBirdService(openBirdCLIResolver: { "/tmp/openbird" })
        let model = AppModel(service: service, initialReport: readyReport())

        await model.handleCaptureExit(code: 0)

        XCTAssertFalse(model.captureNeedsReindex)
        XCTAssertEqual(model.lastActionMessage, "Capture stopped.")
    }

    /// The "already running" exit code (Python `CAPTURE_EXIT_ALREADY_RUNNING`,
    /// value 7) means a daemon we optimistically spawned lost the single-instance
    /// flock race to another daemon. It is BENIGN — capture is still running — so
    /// it must NOT surface as "stopped unexpectedly (exit 7)". Pinning 7 also
    /// guards the cross-component contract with the Python CLI.
    func testCaptureAlreadyRunningExitIsBenign() async {
        XCTAssertEqual(AppModel.captureAlreadyRunningExitCode, 7)
        XCTAssertNotEqual(
            AppModel.captureAlreadyRunningExitCode, AppModel.captureReindexExitCode)

        let service = OpenBirdService(openBirdCLIResolver: { "/tmp/openbird" })
        let model = AppModel(service: service, initialReport: readyReport())

        await model.handleCaptureExit(code: AppModel.captureAlreadyRunningExitCode)

        // The contract: exit 7 is benign — never the reindex prompt, and never
        // the "stopped unexpectedly (exit N)" failure message. (Whether
        // captureRunning ends true depends on whether the lock holder is still
        // alive, which is host-dependent, so we don't pin it here.)
        XCTAssertFalse(model.captureNeedsReindex)
        XCTAssertFalse(model.lastActionMessage.contains("unexpectedly"))
        XCTAssertFalse(model.lastActionMessage.contains("exit 7"))
    }
}
