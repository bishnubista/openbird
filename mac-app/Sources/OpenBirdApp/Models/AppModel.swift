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

enum ModelRouteProvisioningState: Equatable {
    case unknown
    case remoteRoute
    case ollamaUnavailable
    case modelsMissing(canPull: Bool)
    case pulling
    case runtimeReady
    case error(String)
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
    @Published private(set) var provisioningModel: String?
    @Published private(set) var provisioningProgress: ModelPullProgress?
    @Published private(set) var provisioningError: String?
    @Published var lastActionMessage = ""

    // Quick-chat state (window chat panel).
    @Published private(set) var chatBusy = false
    @Published private(set) var chatResult: ChatResult?
    @Published private(set) var chatError: String?

    private let service: OpenBirdService
    private var captureStopRequested = false
    /// Delayed post-start check that confirms capture is actually storing memory.
    private var captureHealthCheckTask: Task<Void, Never>?

    init(service: OpenBirdService, initialReport: PreflightReport = PreflightReport()) {
        self.service = service
        self.report = initialReport
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
        Self.menuBarSymbol(captureRunning: captureRunning, capturePaused: capturePaused)
    }

    nonisolated static func menuBarSymbol(captureRunning: Bool, capturePaused: Bool) -> String {
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
        if localModelStatusState != .ok {
            return localModelStatusSummary
        }
        switch memoryStatsState {
        case .loaded(let stats) where stats.observations == 0:
            return "No memories yet. Start capture or ingest notes to ask grounded questions."
        case .failed:
            return "Could not check local memory. Re-check setup before asking."
        case .unknown, .loaded:
            return nil
        }
    }

    var askEmptyPrompt: String {
        askUnavailableReason ?? "Ask about your work to get a grounded, cited answer."
    }

    var localModelStatusState: StepState {
        if report.error != nil { return .attention }
        if report.cloudBlocked { return .attention }
        if hasRemoteModelRoute { return report.runtimeOK ? .ok : .attention }
        if !hasDecodedModelRoute { return .unknown }
        if report.runtimeOK { return .ok }
        if report.ollamaReachable == nil { return .unknown }
        return .attention
    }

    var localModelStatusSummary: String {
        if let provisioningError {
            return provisioningError
        }
        if let provisioningModel {
            if let progress = provisioningProgress {
                return Self.pullWorkingMessage(progress)
            }
            return "Downloading \(provisioningModel)…"
        }
        if let error = report.error {
            return "Preflight could not read model route: \(error)"
        }
        if report.cloudBlocked {
            return "Remote model blocked: \(remoteModelRouteSummary). Use local Ollama or opt in."
        }
        if hasRemoteModelRoute {
            if report.runtimeOK {
                return "Remote model route verified by preflight: \(remoteModelRouteSummary)"
            }
            return "Remote model route configured but not verified by preflight: \(remoteModelRouteSummary)"
        }
        if !hasDecodedModelRoute {
            return "Model route not verified yet. Re-check setup."
        }
        if report.runtimeOK {
            return "Local Ollama route verified by preflight: \(requiredModelSummary)"
        }
        if report.ollamaReachable == true {
            if report.missingModels.isEmpty { return "Local model route is not runtime-ready. Re-check setup." }
            if !report.autoPullAllowed {
                return "Ollama connected, but automatic model download is disabled for this host."
            }
            return "Ollama connected · missing models: \(report.missingModels.joined(separator: ", "))"
        }
        if report.ollamaReachable == false {
            return "Ollama is not reachable. Launch Ollama, then re-check."
        }
        return "Local model status unknown. Re-check setup."
    }

    var privacyStorageSummary: String {
        "Captured memory is stored on this Mac."
    }

    var privacyTransmissionSummary: String {
        if report.error != nil {
            return "Model route could not be verified. Re-check setup before relying on transmission privacy."
        }
        if report.cloudBlocked {
            return "Remote model configured (\(remoteModelRouteSummary)), but cloud use is blocked; captured memory stays local until you opt in or switch to local Ollama."
        }
        if hasRemoteModelRoute {
            return "Remote model route active (\(remoteModelRouteSummary)); AI features may send captured memory to the configured provider."
        }
        if !hasDecodedModelRoute {
            return "Model route not verified yet. Re-check setup before relying on transmission privacy."
        }
        if report.runtimeOK {
            return "Model requests stay on this Mac through local Ollama."
        }
        return "No remote model route is configured; finish local model setup before using AI features."
    }

    var modelRouteFooterLabel: String {
        if report.error != nil { return "model route unknown" }
        if report.cloudBlocked { return "remote blocked" }
        if hasRemoteModelRoute { return "remote model" }
        if localModelStatusState == .unknown { return "model route unknown" }
        return "local model"
    }

    var hasRemoteModelRoute: Bool {
        !report.remoteModelRoles.isEmpty || !report.remoteModels.isEmpty
    }

    var modelRouteProvisioningState: ModelRouteProvisioningState {
        if let provisioningError { return .error(provisioningError) }
        if provisioningModel != nil { return .pulling }
        if hasRemoteModelRoute { return .remoteRoute }
        if !hasDecodedModelRoute { return .unknown }
        if report.runtimeOK { return .runtimeReady }
        if report.ollamaReachable == false { return .ollamaUnavailable }
        if !report.missingModels.isEmpty {
            return .modelsMissing(canPull: canPullMissingModels)
        }
        return .unknown
    }

    var modelRouteActionLabel: String? {
        switch modelRouteProvisioningState {
        case .ollamaUnavailable:
            return "Get Ollama"
        case .modelsMissing(let canPull):
            return canPull ? "Download models" : nil
        case .error:
            return canPullMissingModels ? "Retry" : nil
        default:
            return nil
        }
    }

    var canPullMissingModels: Bool {
        report.usesLocalOllama
            && report.ollamaReachable == true
            && report.autoPullAllowed
            && !report.missingModels.isEmpty
    }

    func performModelRouteAction(openURL: (URL) -> Void) {
        switch modelRouteProvisioningState {
        case .ollamaUnavailable:
            if let url = URL(string: "https://ollama.com") {
                openURL(url)
            }
        case .modelsMissing(let canPull) where canPull:
            Task { await pullMissingModels() }
        case .error where canPullMissingModels:
            Task { await pullMissingModels() }
        default:
            break
        }
    }

    private var hasDecodedModelRoute: Bool {
        report.llmModel != nil
            || report.embedModel != nil
            || report.ollamaReachable != nil
    }

    private var remoteModelRouteSummary: String {
        if !report.remoteModelRoles.isEmpty {
            return report.remoteModelRoles
                .keys
                .sorted()
                .compactMap { role in
                    guard let model = report.remoteModelRoles[role] else { return nil }
                    return "\(role)=\(model)"
                }
                .joined(separator: ", ")
        }
        if !report.remoteModels.isEmpty {
            return report.remoteModels.joined(separator: ", ")
        }
        return "a remote model"
    }

    var requiredModelSummary: String {
        if !report.requiredModels.isEmpty {
            return report.requiredModels.joined(separator: ", ")
        }
        let configured = [report.llmModel, report.embedModel]
            .compactMap(Self.localOllamaModelName)
        return configured.isEmpty ? "configured local models" : configured.joined(separator: ", ")
    }

    private static func localOllamaModelName(_ model: String?) -> String? {
        guard let model, !model.isEmpty else { return nil }
        if model.hasPrefix("ollama/") {
            return String(model.dropFirst("ollama/".count))
        }
        if model.hasPrefix("ollama_chat/") {
            return String(model.dropFirst("ollama_chat/".count))
        }
        return nil
    }

    var nextStepState: StepState {
        if isRefreshing { return .working }
        if localModelStatusState != .ok || accessibilityState != .ok {
            return .attention
        }
        if allowlist.isEmpty || !captureRunning || memoryStats.observations == 0 {
            return .attention
        }
        return .ok
    }

    var nextStepSummary: String {
        if isRefreshing {
            return "Checking setup..."
        }
        if localModelStatusState != .ok {
            if report.cloudBlocked {
                return "Next: opt in to the remote model route or switch to local Ollama."
            }
            if hasRemoteModelRoute {
                return "Next: re-check setup to verify the configured remote model route."
            }
            if report.ollamaReachable == false {
                return "Next: launch Ollama, then re-check setup."
            }
            if !report.missingModels.isEmpty {
                if !report.autoPullAllowed {
                    return "Next: use a local Ollama host, then re-check setup."
                }
                return "Next: pull missing models: \(report.missingModels.joined(separator: ", "))"
            }
            return "Next: re-check the active model route."
        }
        if accessibilityState != .ok {
            return "Next: grant Accessibility so OpenBird can read active-window text."
        }
        if allowlist.isEmpty {
            return "Next: add at least one app to the capture allowlist."
        }
        if !captureRunning {
            return "Next: start capture."
        }
        if memoryStats.observations == 0 {
            return "Next: bring an allowed app to the front so memory starts filling."
        }
        return "Ready: capture is storing memory. Ask a question when you need it."
    }

    var canStartCaptureNow: Bool {
        accessibilityState == .ok
            && !allowlist.isEmpty
            && service.canLaunchOpenBirdCLI()
            && !captureRunning
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
        // Single source of truth shared with the Spotlight panel (AskPanelModel).
        ChatErrorPresenter.describe(error)
    }

    /// True when the CORE text-capture path is ready: local models + Accessibility.
    /// Encryption is a data-at-rest enhancement (the product runs plaintext-0600 +
    /// FileVault by design), and Screen Recording / Microphone are meetings-only —
    /// none of them gate "Ready", so the badge never contradicts an optional or
    /// informational checklist row.
    var isFullyConfigured: Bool {
        localModelStatusState == .ok && accessibilityState == .ok
    }

    // MARK: - Derived step states

    var encryptionState: StepState {
        if report.encryptionEnabled { return .ok }
        if report.encryptionStatus == "unknown" { return .unknown }
        return .attention   // plaintext-0600
    }

    var accessibilityState: StepState { accessibilityEffectivelyGranted ? .ok : .attention }
    var screenRecordingState: StepState { screenRecordingGranted ? .ok : .attention }
    var microphoneState: StepState { microphoneGranted ? .ok : .attention }

    var accessibilityEffectivelyGranted: Bool {
        accessibilityGranted || preflightAccessibilityPassed
    }

    private var preflightAccessibilityPassed: Bool {
        report.grant("accessibility") == "passed"
    }

    // MARK: - Actions

    /// Cheap permission refresh for app activation/menu-open paths. This avoids
    /// spawning CLI/preflight children every time the user returns from Settings.
    func refreshPermissionStates() {
        accessibilityGranted = service.accessibilityGranted()
        screenRecordingGranted = service.screenRecordingGranted()
        microphoneGranted = service.microphoneGranted()
    }

    func refresh() async {
        isRefreshing = true
        clearProvisioningState()
        defer { isRefreshing = false }
        capturePaused = service.isCapturePaused()
        captureRunning = service.isCaptureRunning()
        helpers = service.helperStatuses()
        allowlist = service.allowlist()
        refreshPermissionStates()
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
        guard canPullMissingModels else {
            lastActionMessage = "Automatic model download is available only for local Ollama."
            return
        }
        provisioningError = nil
        var failed = false
        for model in missing {
            provisioningModel = model
            provisioningProgress = nil
            workingMessage = "Downloading \(model)… (this can take a few minutes)"
            let host = report.ollamaHost
            let outcome = await service.pullModel(model, host: host) { progress in
                Task { @MainActor in
                    guard self.provisioningModel == progress.model else { return }
                    self.provisioningProgress = progress
                    self.workingMessage = Self.pullWorkingMessage(progress)
                }
            }
            lastActionMessage = outcome.message
            if !outcome.ok {
                provisioningError = outcome.message
                failed = true
                break
            }
        }
        provisioningModel = nil
        provisioningProgress = nil
        workingMessage = nil
        if !failed {
            await refresh()
        }
    }

    private static func pullWorkingMessage(_ progress: ModelPullProgress) -> String {
        if let fraction = progress.fraction {
            let pct = Int((fraction * 100).rounded())
            return "Downloading \(progress.model)… \(pct)% · \(progress.status)"
        }
        return "Downloading \(progress.model)… \(progress.status)"
    }

    private func clearProvisioningState() {
        provisioningModel = nil
        provisioningProgress = nil
        provisioningError = nil
        workingMessage = nil
    }

    #if DEBUG
    func setProvisioningErrorForTesting(_ message: String) {
        provisioningError = message
    }
    #endif

    // These trigger native TCC prompts only when the relevant grant is not already
    // present. After granting, the user taps Re-check to refresh full setup state.
    func requestAccessibility() {
        refreshPermissionStates()
        if accessibilityEffectivelyGranted {
            lastActionMessage = "Accessibility is already granted."
            return
        }
        let outcome = service.requestAccessibility()
        refreshPermissionStates()
        switch outcome {
        case .alreadyGranted:
            lastActionMessage = "Accessibility is already granted."
        case .needsPrompt:
            lastActionMessage = "Approve OpenBird in the prompt (or System Settings), then Re-check."
        }
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
