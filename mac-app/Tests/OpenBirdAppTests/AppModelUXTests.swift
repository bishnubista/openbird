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
            let service = OpenBirdService(
                accessibilityProbe: { true },
                openBirdCLIResolver: { "/tmp/openbird" }
            )
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
        let service = OpenBirdService(openBirdCLIResolver: { "/tmp/openbird" })
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
}
