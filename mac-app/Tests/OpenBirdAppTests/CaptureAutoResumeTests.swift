import XCTest
@testable import OpenBirdApp

/// Auto-resume capture on launch: the app kills its daemon on quit and otherwise
/// only starts capture from a UI action, so a quit→relaunch would leave capture
/// silently off. `autoResumeCaptureIfNeeded` restarts it once per launch when the
/// user is configured and hasn't paused. These tests drive the real gate via
/// injected service state and the `start` seam, so no real capture/process runs.
@MainActor
final class CaptureAutoResumeTests: XCTestCase {
    private let allowlistKey = "openbird.captureAllowlist"
    private let onboardingKey = AppModel.onboardingCompletedKey

    private func withEnvironment<T>(
        _ updates: [String: String?],
        _ body: () throws -> T
    ) rethrows -> T {
        var old: [String: String?] = [:]
        for key in updates.keys { old[key] = ProcessInfo.processInfo.environment[key] }
        defer {
            for (key, value) in old {
                if let value { setenv(key, value, 1) } else { unsetenv(key) }
            }
        }
        for (key, value) in updates {
            if let value { setenv(key, value, 1) } else { unsetenv(key) }
        }
        return try body()
    }

    private func readyReport() -> PreflightReport {
        var report = PreflightReport()
        report.ollamaReachable = true
        report.runtimeOK = true
        return report
    }

    private func makeService(accessibility: Bool, cli: String?) -> OpenBirdService {
        OpenBirdService(
            accessibilityProbe: { accessibility },
            openBirdCLIResolver: { cli },
            externalLoopDaemonProbe: { false },
            captureHelperRunningProbe: { false }
        )
    }

    /// Build an `AppModel` with fully controlled launch state, run `body`, and
    /// restore the mutated `UserDefaults` keys + temp data dir afterward.
    private func withConfiguredModel(
        allowlist: [String] = ["com.apple.Safari"],
        onboarding: Bool = true,
        paused: Bool = false,
        accessibility: Bool = true,
        cli: String? = "/tmp/openbird",
        _ body: (AppModel) -> Void
    ) {
        let dataDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try? FileManager.default.createDirectory(at: dataDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dataDir) }
        if paused {
            FileManager.default.createFile(
                atPath: dataDir.appendingPathComponent("capture.paused").path, contents: Data()
            )
        }

        let defaults = UserDefaults.standard
        let oldAllow = defaults.stringArray(forKey: allowlistKey)
        let oldOnb = defaults.object(forKey: onboardingKey)
        defer {
            if let oldAllow { defaults.set(oldAllow, forKey: allowlistKey) }
            else { defaults.removeObject(forKey: allowlistKey) }
            if let oldOnb { defaults.set(oldOnb, forKey: onboardingKey) }
            else { defaults.removeObject(forKey: onboardingKey) }
        }
        if allowlist.isEmpty { defaults.removeObject(forKey: allowlistKey) }
        else { defaults.set(allowlist, forKey: allowlistKey) }
        defaults.set(onboarding, forKey: onboardingKey)

        withEnvironment(["OPENBIRD_DATA_DIR": dataDir.path]) {
            let model = AppModel(
                service: makeService(accessibility: accessibility, cli: cli),
                initialReport: readyReport()
            )
            body(model)
        }
    }

    func testAttemptsStartOnceWhenConfigured() {
        withConfiguredModel { model in
            XCTAssertTrue(model.shouldAutoResumeCapture)
            var count = 0
            XCTAssertTrue(model.autoResumeCaptureIfNeeded(start: { count += 1 }))
            XCTAssertEqual(count, 1)
        }
    }

    func testIsIdempotentAcrossRepeatedCalls() {
        withConfiguredModel { model in
            var count = 0
            XCTAssertTrue(model.autoResumeCaptureIfNeeded(start: { count += 1 }))
            // A second invocation (e.g. SwiftUI re-running the launch .task) must
            // not attempt another start.
            XCTAssertFalse(model.autoResumeCaptureIfNeeded(start: { count += 1 }))
            XCTAssertEqual(count, 1)
        }
    }

    func testNoAttemptWhenAccessibilityDenied() {
        withConfiguredModel(accessibility: false) { model in
            XCTAssertFalse(model.shouldAutoResumeCapture)
            var count = 0
            XCTAssertFalse(model.autoResumeCaptureIfNeeded(start: { count += 1 }))
            XCTAssertEqual(count, 0)
        }
    }

    func testNoAttemptWhenAllowlistEmpty() {
        withConfiguredModel(allowlist: []) { model in
            XCTAssertFalse(model.shouldAutoResumeCapture)
            var count = 0
            XCTAssertFalse(model.autoResumeCaptureIfNeeded(start: { count += 1 }))
            XCTAssertEqual(count, 0)
        }
    }

    func testNoAttemptWhenPaused() {
        withConfiguredModel(paused: true) { model in
            XCTAssertFalse(model.shouldAutoResumeCapture)
            var count = 0
            XCTAssertFalse(model.autoResumeCaptureIfNeeded(start: { count += 1 }))
            XCTAssertEqual(count, 0)
        }
    }

    func testNoAttemptWhenOnboardingIncomplete() {
        withConfiguredModel(onboarding: false) { model in
            XCTAssertFalse(model.shouldAutoResumeCapture)
            var count = 0
            XCTAssertFalse(model.autoResumeCaptureIfNeeded(start: { count += 1 }))
            XCTAssertEqual(count, 0)
        }
    }

    /// A call that fails the readiness gate must NOT consume the one-shot flag, so a
    /// later call (once state is ready) still starts. Drives the live onboarding
    /// read from not-done → done within a single model.
    func testRetriesUntilReadyThenStartsOnce() {
        withConfiguredModel(onboarding: false) { model in
            var count = 0
            // Gate fails (onboarding incomplete): no attempt, flag not consumed.
            XCTAssertFalse(model.autoResumeCaptureIfNeeded(start: { count += 1 }))
            XCTAssertEqual(count, 0)
            // User finishes onboarding; the next launch-task invocation now starts.
            UserDefaults.standard.set(true, forKey: onboardingKey)
            XCTAssertTrue(model.autoResumeCaptureIfNeeded(start: { count += 1 }))
            XCTAssertEqual(count, 1)
            // ...and only once thereafter.
            XCTAssertFalse(model.autoResumeCaptureIfNeeded(start: { count += 1 }))
            XCTAssertEqual(count, 1)
        }
    }
}
