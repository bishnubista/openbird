import XCTest
@testable import OpenBirdApp

/// "Launch at login" toggle (SMAppService). The real login item only behaves
/// correctly for a signed, installed app, so these tests inject a fake backend and
/// assert the model's status-mapping + messaging logic. End-to-end "does it actually
/// relaunch at login" is verified on the notarized build.
final class LoginItemRecorder: @unchecked Sendable {
    /// The status the fake backend reports — set to the state that would result AFTER
    /// a register/unregister so the model's post-mutation handling can be exercised.
    var state: LaunchAtLoginState
    var registerCount = 0
    var unregisterCount = 0
    var openSettingsCount = 0
    var registerError: Error?
    var unregisterError: Error?

    init(state: LaunchAtLoginState) { self.state = state }
}

private struct LoginItemTestError: Error {}

@MainActor
final class LaunchAtLoginTests: XCTestCase {
    private func makeService(_ rec: LoginItemRecorder) -> OpenBirdService {
        OpenBirdService(
            openBirdCLIResolver: { "/tmp/openbird" },
            externalLoopDaemonProbe: { false },
            captureHelperRunningProbe: { false },
            loginItemStateProbe: { rec.state },
            loginItemRegister: {
                rec.registerCount += 1
                if let error = rec.registerError { throw error }
            },
            loginItemUnregister: {
                rec.unregisterCount += 1
                if let error = rec.unregisterError { throw error }
            },
            loginItemsSettingsOpener: { rec.openSettingsCount += 1 }
        )
    }

    private func makeModel(_ rec: LoginItemRecorder) -> AppModel {
        AppModel(service: makeService(rec))
    }

    func testEnableSucceedsWhenStatusEnabled() {
        let rec = LoginItemRecorder(state: .enabled)
        let model = makeModel(rec)
        model.setLaunchAtLogin(true)
        XCTAssertEqual(rec.registerCount, 1)
        XCTAssertEqual(rec.openSettingsCount, 0)
        XCTAssertTrue(model.launchAtLogin)
        XCTAssertTrue(model.lastActionMessage.localizedCaseInsensitiveContains("launch at login"))
    }

    func testEnableRequiringApprovalOpensSettingsAndStaysOff() {
        let rec = LoginItemRecorder(state: .requiresApproval)
        let model = makeModel(rec)
        model.setLaunchAtLogin(true)
        XCTAssertEqual(rec.registerCount, 1)
        XCTAssertEqual(rec.openSettingsCount, 1)
        XCTAssertFalse(model.launchAtLogin)  // not yet eligible until approved
        XCTAssertTrue(model.lastActionMessage.localizedCaseInsensitiveContains("System Settings"))
    }

    func testEnableReportsFailureWhenStatusUnexpectedlyOff() {
        let rec = LoginItemRecorder(state: .disabled)  // register "ok" but not enabled
        let model = makeModel(rec)
        model.setLaunchAtLogin(true)
        XCTAssertFalse(model.launchAtLogin)
        XCTAssertEqual(rec.openSettingsCount, 0)
        XCTAssertTrue(model.lastActionMessage.localizedCaseInsensitiveContains("could not enable"))
    }

    func testEnableThrowReportsFailureAndDoesNotCrash() {
        let rec = LoginItemRecorder(state: .disabled)
        rec.registerError = LoginItemTestError()
        let model = makeModel(rec)
        model.setLaunchAtLogin(true)
        XCTAssertEqual(rec.registerCount, 1)
        XCTAssertFalse(model.launchAtLogin)
        XCTAssertTrue(model.lastActionMessage.localizedCaseInsensitiveContains("could not update"))
    }

    func testDisableSucceedsWhenStatusBecomesDisabled() {
        let rec = LoginItemRecorder(state: .enabled)
        let model = makeModel(rec)
        XCTAssertTrue(model.launchAtLogin)  // reflects init status
        rec.state = .disabled
        model.setLaunchAtLogin(false)
        XCTAssertEqual(rec.unregisterCount, 1)
        XCTAssertFalse(model.launchAtLogin)
        XCTAssertTrue(model.lastActionMessage.localizedCaseInsensitiveContains("will not launch"))
    }

    func testInitAndRefreshReflectLiveStatus() {
        let rec = LoginItemRecorder(state: .enabled)
        let model = makeModel(rec)
        XCTAssertTrue(model.launchAtLogin)
        // User disables it in System Settings out-of-band; refresh picks it up.
        rec.state = .disabled
        model.refreshLaunchAtLoginState()
        XCTAssertFalse(model.launchAtLogin)
    }
}
