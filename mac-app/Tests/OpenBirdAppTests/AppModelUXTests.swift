import XCTest
@testable import OpenBirdApp

@MainActor
final class AppModelUXTests: XCTestCase {
    private let allowlistKey = "openbird.captureAllowlist"

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
