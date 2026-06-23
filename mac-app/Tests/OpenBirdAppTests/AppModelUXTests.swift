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
}
