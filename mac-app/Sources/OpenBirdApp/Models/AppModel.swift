import AppKit
import Foundation

struct HelperStatus: Identifiable, Equatable {
    let id: String
    let label: String
    let isBundled: Bool
    let path: String
}

/// The state of one guided-setup step, used to drive the checklist UI.
enum StepState: Equatable {
    case ok          // satisfied
    case attention   // action needed
    case unknown     // cannot determine (e.g. no probe / off-mac)
    case working     // an action is in progress
}

@MainActor
final class AppModel: ObservableObject {
    @Published private(set) var report = PreflightReport()
    @Published private(set) var capturePaused = false
    @Published private(set) var captureRunning = false
    @Published private(set) var helpers: [HelperStatus] = []
    @Published private(set) var allowlist: [String] = []
    @Published private(set) var lastRefresh: Date?
    @Published private(set) var isRefreshing = false
    @Published private(set) var workingMessage: String?
    @Published var lastActionMessage = ""

    private let service: OpenBirdService

    init(service: OpenBirdService) {
        self.service = service
        self.capturePaused = service.isCapturePaused()
        self.helpers = service.helperStatuses()
        self.allowlist = service.allowlist()
        self.captureRunning = service.isCaptureRunning()
    }

    var menuBarSymbol: String {
        if captureRunning && !capturePaused { return "bird.fill" }
        return capturePaused ? "pause.circle" : "bird"
    }

    /// True when every setup step is satisfied — the app is fully configured.
    var isFullyConfigured: Bool {
        modelsState == .ok && encryptionState == .ok
            && accessibilityState == .ok && screenRecordingState == .ok
    }

    // MARK: - Derived step states

    var ollamaState: StepState {
        switch report.ollamaReachable {
        case .some(true): return .ok
        case .some(false): return .attention
        case .none: return .unknown
        }
    }

    var modelsState: StepState {
        if report.ollamaReachable != true { return .attention }
        return report.missingModels.isEmpty ? .ok : .attention
    }

    var encryptionState: StepState {
        if report.encryptionEnabled { return .ok }
        if report.encryptionStatus == "unknown" { return .unknown }
        return .attention   // plaintext-0600
    }

    func grantState(_ capability: String) -> StepState {
        switch report.grant(capability) {
        case "passed": return .ok
        case "failed": return .attention
        default: return report.helperPresent ? .attention : .unknown
        }
    }

    var accessibilityState: StepState { grantState("accessibility") }
    var screenRecordingState: StepState { grantState("screen_recording") }
    var microphoneState: StepState { grantState("microphone") }

    // MARK: - Actions

    func refresh() async {
        isRefreshing = true
        defer { isRefreshing = false }
        capturePaused = service.isCapturePaused()
        captureRunning = service.isCaptureRunning()
        helpers = service.helperStatuses()
        allowlist = service.allowlist()
        report = await service.preflightReport()
        lastRefresh = Date()
    }

    func pullMissingModels() async {
        let missing = report.missingModels
        guard !missing.isEmpty else { return }
        for model in missing {
            workingMessage = "Pulling \(model)… (this can take a few minutes)"
            let outcome = await service.pullModel(model)
            lastActionMessage = outcome.message
            if !outcome.ok { break }
        }
        workingMessage = nil
        await refresh()
    }

    func openAccessibilitySettings() { service.openPrivacyPane(.accessibility) }
    func openScreenRecordingSettings() { service.openPrivacyPane(.screenRecording) }
    func openMicrophoneSettings() { service.openPrivacyPane(.microphone) }

    func toggleCapturePause() {
        do {
            capturePaused = try service.setCapturePaused(!capturePaused)
            lastActionMessage = capturePaused ? "Capture paused." : "Capture resumed."
        } catch {
            lastActionMessage = "Could not update pause state: \(error.localizedDescription)"
        }
    }

    func startCapture() {
        guard !allowlist.isEmpty else {
            lastActionMessage = "Add at least one app to the capture allowlist first."
            return
        }
        if service.startCapture() {
            captureRunning = true
            capturePaused = false
            lastActionMessage = "Capture started for \(allowlist.count) app(s)."
        } else {
            lastActionMessage = "Could not start capture (CLI not found)."
        }
    }

    func stopCapture() {
        service.stopCapture()
        captureRunning = false
        lastActionMessage = "Capture stopped."
    }

    func stopHelpers() {
        let stopped = service.stopHelperProcesses()
        captureRunning = service.isCaptureRunning()
        lastActionMessage = stopped ? "Stopped helper processes." : "No helper processes were running."
    }

    func addToAllowlist(_ bundleID: String) {
        let trimmed = bundleID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        var updated = allowlist
        updated.append(trimmed)
        service.setAllowlist(updated)
        allowlist = service.allowlist()
        lastActionMessage = "Added \(trimmed) to capture allowlist."
    }

    func removeFromAllowlist(_ bundleID: String) {
        service.setAllowlist(allowlist.filter { $0 != bundleID })
        allowlist = service.allowlist()
    }

    func runningAppSuggestions() -> [String] {
        let current = Set(allowlist)
        return service.runningAppBundleIDs().filter { !current.contains($0) }
    }

    func openDataFolder() { service.openDataFolder() }
    func openBundleFolder() { service.openBundleFolder() }
}
