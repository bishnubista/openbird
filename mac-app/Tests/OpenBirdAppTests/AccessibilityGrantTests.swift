import XCTest
@testable import OpenBirdApp

final class PermissionRecorder: @unchecked Sendable {
    var trusted: Bool
    var promptCount = 0
    var openedPanes: [PrivacyPane] = []

    init(trusted: Bool) {
        self.trusted = trusted
    }
}

final class AccessibilityRequestDecisionTests: XCTestCase {
    func testAlreadyTrustedAccessibilityRequestIsAlreadyGranted() {
        XCTAssertEqual(
            OpenBirdService.decideAccessibilityRequest(isTrusted: true),
            .alreadyGranted
        )
    }

    func testUntrustedAccessibilityRequestNeedsPrompt() {
        XCTAssertEqual(
            OpenBirdService.decideAccessibilityRequest(isTrusted: false),
            .needsPrompt
        )
    }
}

final class AccessibilityGrantServiceTests: XCTestCase {
    func testRequestAccessibilityDoesNotPromptOrOpenSettingsWhenAlreadyGranted() {
        let recorder = PermissionRecorder(trusted: true)
        let service = OpenBirdService(
            accessibilityProbe: { recorder.trusted },
            accessibilityPrompter: { recorder.promptCount += 1 },
            privacyPaneOpener: { recorder.openedPanes.append($0) }
        )

        XCTAssertEqual(service.requestAccessibility(), .alreadyGranted)
        XCTAssertEqual(recorder.promptCount, 0)
        XCTAssertTrue(recorder.openedPanes.isEmpty)
    }

    func testRequestAccessibilityPromptsAndOpensSettingsWhenNotGranted() {
        let recorder = PermissionRecorder(trusted: false)
        let service = OpenBirdService(
            accessibilityProbe: { recorder.trusted },
            accessibilityPrompter: { recorder.promptCount += 1 },
            privacyPaneOpener: { recorder.openedPanes.append($0) }
        )

        XCTAssertEqual(service.requestAccessibility(), .needsPrompt)
        XCTAssertEqual(recorder.promptCount, 1)
        XCTAssertEqual(recorder.openedPanes, [.accessibility])
    }
}

@MainActor
final class AccessibilityGrantAppModelTests: XCTestCase {
    func testRequestAccessibilityRefreshesStaleDeniedStateWhenServiceAlreadyGranted() {
        let recorder = PermissionRecorder(trusted: false)
        let service = OpenBirdService(
            accessibilityProbe: { recorder.trusted },
            accessibilityPrompter: { recorder.promptCount += 1 },
            privacyPaneOpener: { recorder.openedPanes.append($0) }
        )
        let model = AppModel(service: service)
        XCTAssertFalse(model.accessibilityGranted)

        recorder.trusted = true
        model.requestAccessibility()

        XCTAssertTrue(model.accessibilityGranted)
        XCTAssertEqual(recorder.promptCount, 0)
        XCTAssertTrue(recorder.openedPanes.isEmpty)
        XCTAssertEqual(model.lastActionMessage, "Accessibility is already granted.")
    }

    func testPreflightGrantedAccessibilityKeepsStateAndRequestMessageConsistent() {
        let recorder = PermissionRecorder(trusted: false)
        let service = OpenBirdService(
            accessibilityProbe: { recorder.trusted },
            accessibilityPrompter: { recorder.promptCount += 1 },
            privacyPaneOpener: { recorder.openedPanes.append($0) }
        )
        var report = PreflightReport()
        report.grants["accessibility"] = "passed"
        let model = AppModel(service: service, initialReport: report)

        XCTAssertFalse(model.accessibilityGranted)
        XCTAssertTrue(model.accessibilityEffectivelyGranted)
        XCTAssertEqual(model.accessibilityState, .ok)

        model.requestAccessibility()

        XCTAssertEqual(model.accessibilityState, .ok)
        XCTAssertEqual(recorder.promptCount, 0)
        XCTAssertTrue(recorder.openedPanes.isEmpty)
        XCTAssertEqual(model.lastActionMessage, "Accessibility is already granted.")
    }

    func testRequestAccessibilityRefreshesBeforeCheckingCachedGrantedState() {
        let recorder = PermissionRecorder(trusted: true)
        let service = OpenBirdService(
            accessibilityProbe: { recorder.trusted },
            accessibilityPrompter: { recorder.promptCount += 1 },
            privacyPaneOpener: { recorder.openedPanes.append($0) }
        )
        let model = AppModel(service: service)
        XCTAssertTrue(model.accessibilityGranted)

        recorder.trusted = false
        model.requestAccessibility()

        XCTAssertFalse(model.accessibilityGranted)
        XCTAssertEqual(model.accessibilityState, .attention)
        XCTAssertEqual(recorder.promptCount, 1)
        XCTAssertEqual(recorder.openedPanes, [.accessibility])
        XCTAssertEqual(
            model.lastActionMessage,
            "Approve OpenBird in the prompt (or System Settings), then Re-check."
        )
    }
}
