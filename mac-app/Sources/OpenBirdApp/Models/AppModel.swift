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

enum MemoryStatsState: Equatable {
    case unknown
    case loaded(MemoryStats)
    case failed
}

@MainActor
final class AppModel: ObservableObject {
    @Published private(set) var report = PreflightReport()
    @Published private(set) var capturePaused = false
    @Published private(set) var captureRunning = false
    @Published private(set) var helpers: [HelperStatus] = []
    @Published private(set) var allowlist: [String] = []
    // TCC grants are checked from the APP's own process (that is where macOS
    // records them — a nested helper's request is attributed to the app bundle).
    @Published private(set) var accessibilityGranted = false
    @Published private(set) var screenRecordingGranted = false
    @Published private(set) var microphoneGranted = false
    @Published private(set) var memoryStats = MemoryStats.empty
    @Published private(set) var memoryStatsState = MemoryStatsState.unknown
    @Published private(set) var lastMemoryRefresh: Date?
    @Published private(set) var lastRefresh: Date?
    @Published private(set) var isRefreshing = false
    @Published private(set) var workingMessage: String?
    @Published var lastActionMessage = ""

    // Quick-chat state (window chat panel).
    @Published private(set) var chatBusy = false
    @Published private(set) var chatResult: ChatResult?
    @Published private(set) var chatError: String?

    private let service: OpenBirdService
    private var captureStopRequested = false
    /// Delayed post-start check that confirms capture is actually storing memory.
    private var captureHealthCheckTask: Task<Void, Never>?

    init(service: OpenBirdService) {
        self.service = service
        self.capturePaused = service.isCapturePaused()
        self.helpers = service.helperStatuses()
        self.allowlist = service.allowlist()
        self.captureRunning = service.isCaptureRunning()
        self.accessibilityGranted = service.accessibilityGranted()
        self.screenRecordingGranted = service.screenRecordingGranted()
        self.microphoneGranted = service.microphoneGranted()

        // Safety net: if the app is quit by any path (Cmd-Q, menu, dock), stop the
        // capture daemon WE launched so it is never orphaned past app lifetime.
        // Capture the (non-actor-isolated) service directly so the synchronous
        // termination handler does not hop the main actor while the app is exiting.
        let service = self.service
        NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification,
            object: nil,
            queue: .main
        ) { _ in
            service.terminateLaunchedCapture()
        }
    }

    var menuBarSymbol: String {
        if captureRunning && !capturePaused { return "bird.fill" }
        return capturePaused ? "pause.circle" : "bird"
    }

    /// User-facing summary of how many local memory observations are currently stored.
    var memorySummary: String {
        let noun = memoryStats.observations == 1 ? "observation" : "observations"
        if let lastMemoryRefresh {
            return "\(memoryStats.observations) \(noun) stored · checked \(lastMemoryRefresh.formatted(date: .omitted, time: .shortened))"
        }
        return "\(memoryStats.observations) \(noun) stored"
    }

    var askUnavailableReason: String? {
        switch memoryStatsState {
        case .loaded(let stats) where stats.observations == 0:
            return "No memory stored yet. Start capture or ingest notes before asking."
        case .failed:
            return "Could not check local memory. Re-check setup before asking."
        case .unknown, .loaded:
            return nil
        }
    }

    /// Ask a grounded question over captured memory. Runs the (blocking) CLI off
    /// the main actor and publishes the answer + citations (or a friendly error).
    func ask(_ question: String) {
        let q = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty, !chatBusy else { return }
        if let askUnavailableReason {
            chatResult = nil
            chatError = askUnavailableReason
            return
        }
        chatBusy = true
        chatError = nil
        let service = self.service
        Task {
            do {
                let result = try await Task.detached(priority: .userInitiated) {
                    try service.askChat(q)
                }.value
                self.chatResult = result
            } catch {
                self.chatResult = nil
                self.chatError = Self.describeChatError(error)
            }
            self.chatBusy = false
        }
    }

    private static func describeChatError(_ error: Error) -> String {
        switch error {
        case ChatError.cliMissing:
            return "OpenBird CLI not found in the app bundle."
        case ChatError.failed(let message):
            return message
        case ChatError.decode:
            return "Could not read the chat response."
        default:
            return "Chat error."
        }
    }

    /// True when the CORE text-capture path is ready: local models + Accessibility.
    /// Encryption is a data-at-rest enhancement (the product runs plaintext-0600 +
    /// FileVault by design), and Screen Recording / Microphone are meetings-only —
    /// none of them gate "Ready", so the badge never contradicts an optional or
    /// informational checklist row.
    var isFullyConfigured: Bool {
        modelsState == .ok && accessibilityState == .ok
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

    var accessibilityState: StepState { accessibilityGranted ? .ok : .attention }
    var screenRecordingState: StepState { screenRecordingGranted ? .ok : .attention }
    var microphoneState: StepState { microphoneGranted ? .ok : .attention }

    // MARK: - Actions

    func refresh() async {
        isRefreshing = true
        defer { isRefreshing = false }
        capturePaused = service.isCapturePaused()
        captureRunning = service.isCaptureRunning()
        helpers = service.helperStatuses()
        allowlist = service.allowlist()
        accessibilityGranted = service.accessibilityGranted()
        screenRecordingGranted = service.screenRecordingGranted()
        microphoneGranted = service.microphoneGranted()
        report = await service.preflightReport()
        await refreshMemoryStats()
        lastRefresh = Date()
    }

    /// Refresh local DB counters so capture status is grounded in stored memory.
    func refreshMemoryStats() async {
        if let stats = await service.memoryStats() {
            memoryStats = stats
            memoryStatsState = .loaded(stats)
        } else {
            memoryStats = .empty
            memoryStatsState = .failed
        }
        lastMemoryRefresh = Date()
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

    // These trigger the native TCC prompt via the helper (registering it in the
    // relevant System Settings list) and also open the pane. After granting,
    // the user taps Re-check to refresh status.
    func requestAccessibility() {
        service.requestAccessibility()
        lastActionMessage = "Approve OpenBird in the prompt (or System Settings), then Re-check."
    }
    func requestScreenRecording() {
        service.requestScreenRecording()
        lastActionMessage = "Approve OpenBird in the prompt (or System Settings), then Re-check."
    }
    func requestMicrophone() {
        service.requestMicrophone()
        lastActionMessage = "Approve OpenBird in the prompt (or System Settings), then Re-check."
    }

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
        lastActionMessage = "Starting capture..."
        Task { await startCaptureAfterRefreshingStats() }
    }

    /// Start capture after recording the current observation count for health comparison.
    private func startCaptureAfterRefreshingStats() async {
        await refreshMemoryStats()
        captureHealthCheckTask?.cancel()
        captureStopRequested = false
        let observationsBeforeStart = memoryStats.observations
        if service.startCapture(onExit: { [weak self] code in
            Task { @MainActor in
                guard let self else { return }
                self.captureRunning = self.service.isCaptureRunning()
                self.captureHealthCheckTask?.cancel()
                self.captureHealthCheckTask = nil
                await self.refreshMemoryStats()
                if self.captureStopRequested {
                    self.captureStopRequested = false
                } else {
                    self.lastActionMessage = code == 0
                        ? "Capture stopped."
                        : "Capture stopped unexpectedly (exit \(code)). Re-check setup."
                }
            }
        }) {
            captureRunning = true
            capturePaused = false
            lastActionMessage = "Capture started for \(allowlist.count) app(s). Waiting for memory updates."
            captureHealthCheckTask = Task {
                await refreshMemoryStats()
                try? await Task.sleep(nanoseconds: 6_000_000_000)
                guard !Task.isCancelled, captureRunning else { return }
                await refreshMemoryStats()
                if memoryStats.observations <= observationsBeforeStart {
                    lastActionMessage = "Capture is running, but no new memory is stored yet. Bring an allowed app to the front or re-check setup."
                }
            }
        } else {
            lastActionMessage = "Could not start capture (CLI not found)."
        }
    }

    func stopCapture() {
        captureStopRequested = true
        captureHealthCheckTask?.cancel()
        captureHealthCheckTask = nil
        service.stopCapture()
        captureRunning = false
        lastActionMessage = "Capture stopped."
        Task { await refreshMemoryStats() }
    }

    /// Stop the app-launched capture daemon, then terminate the app. Used by the
    /// Quit menu item so an explicit quit never leaves capture orphaned.
    func quit() {
        service.terminateLaunchedCapture()
        NSApplication.shared.terminate(nil)
    }

    func stopHelpers() {
        captureStopRequested = true
        captureHealthCheckTask?.cancel()
        captureHealthCheckTask = nil
        let stopped = service.stopHelperProcesses()
        captureRunning = service.isCaptureRunning()
        lastActionMessage = stopped ? "Stopped helper processes." : "No helper processes were running."
        Task { await refreshMemoryStats() }
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
