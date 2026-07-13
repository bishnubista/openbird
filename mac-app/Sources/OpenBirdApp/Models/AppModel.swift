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

enum DeepBrainStatusState: Equatable {
    case unknown
    case loaded(DeepBrainStatus)
    case failed
}

enum DeepBrainPreviewState: Equatable {
    case unknown
    case loading
    case loaded(DeepBrainPreview)
    case failed
}

enum CaptureHealthState: Equatable {
    case unknown
    case loaded(CaptureHealthReport)
    case failed
}

enum ClaudeAssistantState: Equatable {
    case unknown
    case connected
    case disconnected
    case failed
}

enum CaptureRowTone: Equatable {
    case ok
    case attention
    case neutral
}

struct CaptureRowStatus: Equatable {
    let label: String
    let detail: String
    let tone: CaptureRowTone
}

enum ModelRouteProvisioningState: Equatable {
    case unknown
    case remoteRoute
    case ollamaUnavailable
    case ollamaTooOldForEmbeddingGemma
    case modelsMissing(canPull: Bool)
    case pulling
    case runtimeReady
    case error(String)
}

enum OnboardingPrimaryAction: Equatable {
    case checking
    case complete
    case start
    case blocked(String)

    var label: String {
        switch self {
        case .checking:
            "Checking setup..."
        case .complete:
            "Continue"
        case .start:
            "Start capturing"
        case .blocked:
            "Finish setup"
        }
    }

    var isDisabled: Bool {
        self == .checking
    }
}

struct OnboardingPresentationState: Equatable {
    var completed: Bool
    var presented: Bool

    mutating func presentIfNeeded() {
        presented = !completed
    }

    mutating func dismissWithoutCompleting() {
        presented = false
    }

    mutating func complete() {
        completed = true
        presented = false
    }
}

@MainActor
final class AppModel: ObservableObject {
    static let embeddingGemmaMinimumOllamaVersion = "0.11.10"

    /// The active navigation pane in the single app window. Runtime single source of
    /// truth that the sidebar, the menu bar, and the deep-link router all read/write —
    /// switching panes never opens a second window. Held on this long-lived model, so it
    /// persists for the app session (the window can close and reopen on the same pane);
    /// it intentionally resets to `.today` on a full relaunch.
    @Published var selection: AppDestination = .today
    /// Navigation seam invoked when a chat citation is clicked: it focuses the
    /// Today/day view on the citation's day and (best-effort) observation. Wired in
    /// `OpenBirdApp.init` to the shared `TodayModel` so AppModel never imports the
    /// day model directly; defaults to a no-op so unit tests and previews construct an
    /// `AppModel` without it. Privacy: only an integer day offset and an opaque
    /// observation id flow through here — never captured text/window/URL.
    var citationNavigator: ((_ dayOffset: Int, _ observationId: String?) -> Void)?
    @Published private(set) var report = PreflightReport()
    @Published private(set) var capturePaused = false
    @Published private(set) var captureRunning = false
    /// True iff this app is registered AND eligible to launch at login. Mirrors the
    /// live `SMAppService` status (never stored locally) so System-Settings changes
    /// are reflected; see `refreshLaunchAtLoginState()`.
    @Published private(set) var launchAtLogin = false
    /// True when capture last died because the on-disk index needs rebuilding under
    /// the current embedding model (capture daemon exit code `captureReindexExitCode`).
    /// Drives the one-click "Reindex" affordance instead of a dead-end error.
    @Published private(set) var captureNeedsReindex = false
    /// True while a user-triggered `reindex` is running (disables the button).
    @Published private(set) var isReindexing = false
    @Published private(set) var helpers: [HelperStatus] = []
    @Published private(set) var allowlist: [String] = []
    /// Deep-capture (OCR) opt-in list (Phase C2) — always a subset of
    /// `allowlist` by construction (the service filters on every write).
    @Published private(set) var ocrApps: [String] = []
    /// Explicit terminal/editor grants. Python validates the exact eligible set
    /// and forces SQLCipher whenever this list is non-empty.
    @Published private(set) var detailedCaptureApps: [String] = []
    // TCC grants are checked from the APP's own process (that is where macOS
    // records them — a nested helper's request is attributed to the app bundle).
    @Published private(set) var accessibilityGranted = false
    @Published private(set) var screenRecordingGranted = false
    @Published private(set) var microphoneGranted = false
    @Published private(set) var memoryStats = MemoryStats.empty
    @Published private(set) var memoryStatsState = MemoryStatsState.unknown
    @Published private(set) var captureHealthState = CaptureHealthState.unknown
    @Published private(set) var deepBrainStatusState = DeepBrainStatusState.unknown
    @Published private(set) var deepBrainPreviewState = DeepBrainPreviewState.unknown
    @Published private(set) var claudeAssistantState = ClaudeAssistantState.unknown
    @Published private(set) var claudeAssistantBusy = false
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

    /// Capture daemon exit code meaning "index built under a different embedding
    /// model — run `openbird reindex`". Mirrors `CAPTURE_EXIT_REINDEX_REQUIRED` in
    /// `openbird/capture/cli.py`; keep the two in sync.
    static let captureReindexExitCode: Int32 = 5

    /// Capture daemon exit code meaning another `capture --loop` daemon already
    /// holds the single-instance lock. BENIGN — capture is still running via the
    /// other daemon, so this must never surface as an "unexpected" failure. Mirrors
    /// `CAPTURE_EXIT_ALREADY_RUNNING` in `openbird/capture/cli.py`.
    static let captureAlreadyRunningExitCode: Int32 = 7

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
        self.ocrApps = service.ocrApps()
        self.detailedCaptureApps = service.detailedCaptureApps()
        self.captureRunning = service.isCaptureRunning()
        self.accessibilityGranted = service.accessibilityGranted()
        self.screenRecordingGranted = service.screenRecordingGranted()
        self.microphoneGranted = service.microphoneGranted()
        self.launchAtLogin = service.launchAtLoginState() == .enabled

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

    /// Default Ask prompt chips for the empty state — one source of truth for the compact
    /// Spotlight panel and the in-window/expanded Ask. Generic, grounded-question shaped
    /// (not demo ticket names), so they work against the user's real captured memory.
    /// Only shown when memory has content (`askUnavailableReason == nil`); when there's no
    /// data the views show that reason instead, so we never offer a prompt that can't answer.
    let askSuggestions = [
        "Summarize my day",
        "What did I work on yesterday?",
        "What should I follow up on?",
    ]

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
            if report.ollamaVersionOK == false {
                return "Ollama is too old for embeddinggemma. Update Ollama to \(Self.embeddingGemmaMinimumOllamaVersion) or newer, then re-check."
            }
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

    var deepBrainStatusTitle: String {
        switch deepBrainStatusState {
        case .unknown:
            return "Checking Deep Brain status"
        case .failed:
            return "Deep Brain status unavailable"
        case .loaded(let status):
            return status.routeLabel
        }
    }

    var deepBrainStatusSummary: String {
        switch deepBrainStatusState {
        case .unknown:
            return "Re-check setup to read the local Deep Brain consent status."
        case .failed:
            return "Could not read Deep Brain status from the local CLI."
        case .loaded(let status):
            let exclusions = Self.deepBrainExclusionSummary(status.exclusions)
            let check = "Status check is local; no memory packet or provider was used."
            let route: String
            if status.cloudGatesEnabled {
                route = "Cloud reasoning gates are enabled for Deep Brain when you ask."
            } else if status.askAvailable {
                route = "Deep Brain ask can use the local model route without cloud."
            } else if !status.deepBrainEnabled {
                route = "Deep Brain ask is off until you enable the feature gate."
            } else {
                route = "Deep Brain ask is blocked by the current consent/model route."
            }
            return "\(route) \(check) \(exclusions)"
        }
    }

    var deepBrainStatusBadge: String {
        switch deepBrainStatusState {
        case .unknown:
            return "Check"
        case .failed:
            return "Unavailable"
        case .loaded(let status):
            if status.cloudGatesEnabled { return "Cloud gates" }
            if status.askAvailable { return "Local ask" }
            if !status.deepBrainEnabled { return "Off" }
            return "Blocked"
        }
    }

    var deepBrainStatusNeedsAttention: Bool {
        switch deepBrainStatusState {
        case .unknown, .failed:
            return true
        case .loaded(let status):
            return status.deepBrainEnabled && !status.askAvailable
        }
    }

    var deepBrainPreviewSummary: String {
        switch deepBrainPreviewState {
        case .unknown:
            return "Preview today's local Deep Brain packet before any cloud reasoning. The preview is user-triggered and does not use a provider."
        case .loading:
            return "Building today's local packet preview from distilled day memory. No provider or cloud send is used."
        case .failed:
            return "Could not build the local packet preview. No provider or cloud send was used."
        case .loaded(let preview):
            let exclusions = preview.exclusions
            let groups = preview.sourcesTotal == 1 ? "source group" : "source groups"
            let eligible = exclusions.keptObservations == 1 ? "eligible observation" : "eligible observations"
            let excluded = exclusions.excludedObservations == 1 ? "excluded observation" : "excluded observations"
            let cloud = preview.cloudReady ? "Cloud gates are ready if you ask." : "Cloud gates are not fully enabled."
            return "Local snapshot for \(preview.localDate): \(exclusions.keptObservations) \(eligible), \(exclusions.excludedObservations) \(excluded), \(preview.sourcesTotal) available \(groups). No provider or cloud send was used. \(cloud)"
        }
    }

    var deepBrainPreviewBadge: String {
        switch deepBrainPreviewState {
        case .unknown:
            return "Preview"
        case .loading:
            return "Loading"
        case .failed:
            return "Unavailable"
        case .loaded:
            return "Local preview"
        }
    }

    func loadDeepBrainPreview() {
        guard deepBrainPreviewState != .loading else { return }
        deepBrainPreviewState = .loading
        let service = self.service
        Task {
            if let preview = await service.deepBrainPreview(dayOffset: 0) {
                deepBrainPreviewState = .loaded(preview)
                lastActionMessage = "Built local Deep Brain packet preview."
            } else {
                deepBrainPreviewState = .failed
                lastActionMessage = "Could not build local Deep Brain packet preview."
            }
        }
    }

    private static func deepBrainExclusionSummary(_ exclusions: DeepBrainStatus.Exclusions) -> String {
        var pieces: [String] = []
        if !exclusions.excludedAppsConfigured.isEmpty {
            pieces.append("apps: \(exclusions.excludedAppsConfigured.joined(separator: ", "))")
        }
        if !exclusions.excludedSourcesConfigured.isEmpty {
            pieces.append("sources: \(exclusions.excludedSourcesConfigured.joined(separator: ", "))")
        }
        if exclusions.excludedObservationIdsConfigured > 0 {
            let n = exclusions.excludedObservationIdsConfigured
            pieces.append("\(n) observation id\(n == 1 ? "" : "s")")
        }
        guard !pieces.isEmpty else {
            return "No Deep Brain exclusions are configured."
        }
        return "Configured exclusions: \(pieces.joined(separator: "; "))."
    }

    static let dataPruneCommand = "openbird data prune --older-than 90d"
    static let dataPurgeAllCommand = "openbird data purge --all"
    static let dataExportCommand = "openbird data export --output ~/openbird-memory-export.jsonl"
    static let deepBrainAskCommand = "OPENBIRD_DEEP_BRAIN_ENABLED=1 openbird deep-brain ask --day 0 --stdin"
    static let productivityCoachCommand = "OPENBIRD_DEEP_BRAIN_ENABLED=1 openbird productivity-coach --day 0 --stdin"

    var dataDeletionSummary: String {
        "Terminal deletion commands ask for confirmation by default. `openbird data prune --older-than 90d` cascade-deletes old observations plus orphaned blobs/chunks/FTS/vector rows; `openbird data vacuum` reclaims disk space after deletion. Total wipe is available as `openbird data purge --all`."
    }

    var dataExportSummary: String {
        "Copy a Terminal export command with a suggested local-only path for decrypted JSONL memory, including captured text in plaintext. Edit the path before running if you prefer elsewhere; iCloud, Dropbox, or other synced folders may upload the file outside OpenBird's control, and later purge/prune will not delete exported copies. The command asks for confirmation by default."
    }

    func copyDataPruneCommand() {
        service.copyToPasteboard(Self.dataPruneCommand)
        lastActionMessage = "Copied confirmation-gated prune command."
    }

    func copyDataExportCommand() {
        service.copyToPasteboard(Self.dataExportCommand)
        lastActionMessage = "Copied confirmation-gated export command."
    }

    var deepBrainAskCommandSummary: String {
        "Copy a one-shot Terminal command for Deep Brain ask. Preview the packet first; the command enables Deep Brain for that ask only, does not persist consent, and reads your question from stdin. In Terminal, run it, type your question, then press Ctrl-D. Remote LLM routes still require separate OPENBIRD_ALLOW_CLOUD=1."
    }

    func copyDeepBrainAskCommand() {
        service.copyToPasteboard(Self.deepBrainAskCommand)
        lastActionMessage = "Copied one-shot Deep Brain ask command."
    }

    var productivityCoachCommandSummary: String {
        "Copy a one-shot Terminal command for cited productivity coaching. First review local productivity facts with `openbird productivity` (local-only, no model). The command enables coaching for that Terminal run only, does not persist consent, and reads your question from stdin. In Terminal, run it, type your question, then press Ctrl-D. Remote LLM routes still require separate OPENBIRD_ALLOW_CLOUD=1."
    }

    func copyProductivityCoachCommand() {
        service.copyToPasteboard(Self.productivityCoachCommand)
        lastActionMessage = "Copied one-shot productivity coach command."
    }

    var modelRouteFooterLabel: String {
        if report.error != nil { return "model route unknown" }
        if report.cloudBlocked { return "remote blocked" }
        if hasRemoteModelRoute { return "remote model" }
        if localModelStatusState == .unknown { return "model route unknown" }
        // "on-device" (handoff sidebar footer copy) — truthful only on the local route;
        // the remote/blocked/unknown branches above keep their honest labels.
        return "on-device"
    }

    var meetingTranscriptionState: StepState {
        guard let readiness = report.meetingTranscription else { return .unknown }
        if readiness.backendAvailable
            || readiness.parakeetMLXAvailable
            || readiness.fasterWhisperAvailable {
            return .ok
        }
        return .attention
    }

    var meetingTranscriptionSummary: String {
        guard let readiness = report.meetingTranscription else {
            return "Re-check setup to detect parakeet-mlx or faster-whisper."
        }
        if readiness.parakeetMLXAvailable {
            return "\(readiness.recommendedBackend) ready · Apple Silicon recommended"
        }
        if readiness.fasterWhisperAvailable {
            return "\(readiness.fallbackBackend) ready · portable fallback"
        }
        return "No meeting transcription backend installed. \(readiness.recommendedBackend) is recommended on Apple Silicon; \(readiness.fallbackBackend) is the portable fallback. Install one in the openbird CLI environment."
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
        if report.ollamaReachable == true && report.ollamaVersionOK == false {
            return .ollamaTooOldForEmbeddingGemma
        }
        if !report.missingModels.isEmpty {
            return .modelsMissing(canPull: canPullMissingModels)
        }
        return .unknown
    }

    var modelRouteActionLabel: String? {
        switch modelRouteProvisioningState {
        case .ollamaUnavailable:
            return "Get Ollama"
        case .ollamaTooOldForEmbeddingGemma:
            return "Update Ollama"
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
        case .ollamaUnavailable, .ollamaTooOldForEmbeddingGemma:
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
            if report.ollamaReachable == true && report.ollamaVersionOK == false {
                return "Next: update Ollama to \(Self.embeddingGemmaMinimumOllamaVersion) or newer for embeddinggemma, then re-check setup."
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

    /// UserDefaults key for the first-run flag, owned by `AppShellView`'s
    /// `@AppStorage` (which defaults to `UserDefaults.standard`). Centralized here
    /// so the auto-resume gate reads the exact same key.
    static let onboardingCompletedKey = "openbird.onboarding.completed"
    static let onboardingRepairV1DoneKey = "openbird.onboarding.repair.v1.done"

    /// First-run onboarding asks for the fully useful stack (models + Accessibility +
    /// allowlist) before claiming success, so the first captured memory can be searched
    /// and answered. The Settings/MenuBar start path remains a lower-bar escape hatch
    /// for users who intentionally want text capture running before models are ready.
    var onboardingPrimaryAction: OnboardingPrimaryAction {
        if isRefreshing || lastRefresh == nil {
            return .checking
        }
        if captureRunning {
            return .complete
        }
        guard localModelStatusState == .ok else {
            return .blocked(nextStepSummary)
        }
        guard accessibilityState == .ok else {
            return .blocked(nextStepSummary)
        }
        guard !allowlist.isEmpty else {
            return .blocked(nextStepSummary)
        }
        guard service.canLaunchOpenBirdCLI() else {
            return .blocked("Could not find the openbird CLI. Re-check setup or reinstall OpenBird.")
        }
        return .start
    }

    static func repairIncompleteOnboardingCompletionIfNeeded(
        completed: Bool,
        repairDone: Bool,
        allowlistIsEmpty: Bool,
        captureRunning: Bool
    ) -> (completed: Bool, repairDone: Bool, repaired: Bool) {
        guard !repairDone else { return (completed, repairDone, false) }
        let shouldRepair = completed && allowlistIsEmpty && !captureRunning
        return (shouldRepair ? false : completed, true, shouldRepair)
    }

    @discardableResult
    func repairIncompleteOnboardingCompletionIfNeeded() -> Bool {
        let defaults = UserDefaults.standard
        let outcome = Self.repairIncompleteOnboardingCompletionIfNeeded(
            completed: defaults.bool(forKey: Self.onboardingCompletedKey),
            repairDone: defaults.bool(forKey: Self.onboardingRepairV1DoneKey),
            allowlistIsEmpty: allowlist.isEmpty,
            captureRunning: captureRunning
        )
        defaults.set(outcome.completed, forKey: Self.onboardingCompletedKey)
        defaults.set(outcome.repairDone, forKey: Self.onboardingRepairV1DoneKey)
        return outcome.repaired
    }

    /// Whether capture should be (re)started automatically at launch. Builds on
    /// `canStartCaptureNow` (Accessibility granted, non-empty allowlist, CLI
    /// available, not already running) and additionally requires that the user has
    /// finished onboarding and has NOT explicitly paused capture. Screen Recording
    /// is intentionally NOT required: text capture is Accessibility-based.
    var shouldAutoResumeCapture: Bool {
        canStartCaptureNow
            && !capturePaused
            && UserDefaults.standard.bool(forKey: Self.onboardingCompletedKey)
    }

    private var didAttemptAutoResume = false

    /// Resume capture if the user is already configured and hasn't paused. The app
    /// kills its daemon on quit (see `willTerminateNotification` above) and
    /// otherwise never restarts capture without a UI action, so without this a
    /// quit→relaunch leaves capture silently off.
    ///
    /// Retry-until-ready, then once: the one-shot flag is consumed ONLY after a
    /// successful start, so a launch where state isn't ready yet (e.g. Accessibility
    /// granted a moment later, or a re-invoked `.task`) can still start later. After
    /// the first successful start the flag — and `canStartCaptureNow`'s
    /// `!captureRunning` term — both prevent a second spawn (closing the race before
    /// the async `startCapture()` flips `captureRunning`); the daemon `flock` is the
    /// final backstop. The `start` seam keeps this testable without launching real
    /// service/process work. Returns whether it attempted a start.
    @discardableResult
    func autoResumeCaptureIfNeeded(start: () -> Void) -> Bool {
        guard !didAttemptAutoResume else { return false }
        guard shouldAutoResumeCapture else { return false }
        didAttemptAutoResume = true
        start()
        return true
    }

    @discardableResult
    func autoResumeCaptureIfNeeded() -> Bool {
        autoResumeCaptureIfNeeded(start: { self.startCapture() })
    }

    /// Re-read the live "launch at login" status into `launchAtLogin`. Cheap; call
    /// it wherever the user might have changed it out-of-band (launch, refresh, after
    /// a toggle, and on app re-activation) so the UI never shows a stale value.
    func refreshLaunchAtLoginState() {
        launchAtLogin = service.launchAtLoginState() == .enabled
    }

    /// Toggle whether OpenBird launches at login. Together with auto-resume this lets
    /// capture survive a reboot/logout. Never flips `launchAtLogin` optimistically —
    /// it always resyncs from the real `SMAppService` status and maps that status to
    /// an honest user message (success only on `.enabled`; `.requiresApproval` opens
    /// System Settings; anything else is reported as a failure).
    func setLaunchAtLogin(_ enabled: Bool) {
        do {
            try service.setLaunchAtLogin(enabled)
        } catch {
            refreshLaunchAtLoginState()
            lastActionMessage = "Could not update Launch at Login: \(error.localizedDescription)"
            return
        }
        // Read the resulting status ONCE and derive both the published flag and the
        // message from it, so they can never disagree (a second read could observe a
        // different status and e.g. show "enabled" while the toggle reads off).
        let state = service.launchAtLoginState()
        launchAtLogin = state == .enabled
        if enabled {
            switch state {
            case .enabled:
                lastActionMessage = "OpenBird will launch at login."
            case .requiresApproval:
                lastActionMessage = "Approve OpenBird in System Settings › General › Login Items."
                service.openLoginItemsSettings()
            case .disabled, .unavailable:
                lastActionMessage = "Could not enable Launch at Login. Open System Settings › General › Login Items."
            }
        } else {
            switch state {
            case .disabled:
                lastActionMessage = "OpenBird will not launch at login."
            case .enabled, .requiresApproval, .unavailable:
                lastActionMessage = "Could not disable Launch at Login. Check System Settings › General › Login Items."
            }
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
        // Single source of truth shared with the Spotlight panel (AskPanelModel).
        ChatErrorPresenter.describe(error)
    }

    /// True when the CORE text-memory path is ready enough to claim "Ready":
    /// route + Accessibility are OK, an allowlisted app is being captured, and the
    /// memory store has at least one observation. If stats cannot be read, this
    /// conservatively stays false. Encryption is a data-at-rest enhancement; optional
    /// Screen Recording / Microphone capabilities do not gate this core-memory label.
    var isFullyConfigured: Bool {
        nextStepState == .ok
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
        ocrApps = service.ocrApps()
        detailedCaptureApps = service.detailedCaptureApps()
        refreshLaunchAtLoginState()
        refreshPermissionStates()
        report = await service.preflightReport()
        await refreshMemoryStats()
        await refreshCaptureHealth()
        await refreshDeepBrainStatus()
        await refreshClaudeAssistantStatus()
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

    func refreshCaptureHealth() async {
        capturePaused = service.isCapturePaused()
        captureRunning = service.isCaptureRunning()
        if let health = await service.captureHealth() {
            captureHealthState = .loaded(health)
        } else {
            captureHealthState = .failed
        }
    }

    func refreshDeepBrainStatus() async {
        if let status = await service.deepBrainStatus() {
            deepBrainStatusState = .loaded(status)
        } else {
            deepBrainStatusState = .failed
        }
    }

    func refreshClaudeAssistantStatus() async {
        guard let status = await service.claudeAssistantStatus() else {
            claudeAssistantState = .failed
            return
        }
        claudeAssistantState = status.configured ? .connected : .disconnected
    }

    var claudeAssistantSummary: String {
        switch claudeAssistantState {
        case .unknown:
            return "Re-check setup to read Claude Desktop connection status."
        case .connected:
            return "Read-only access is configured. Excerpts leave only when Claude asks OpenBird."
        case .disconnected:
            return "Not connected. OpenBird will add one local, read-only Claude tool server."
        case .failed:
            return "Could not read Claude Desktop settings. Existing settings were not changed."
        }
    }

    func connectClaudeAssistant() {
        guard !claudeAssistantBusy else { return }
        claudeAssistantBusy = true
        lastActionMessage = "Connecting Claude Desktop..."
        Task {
            let connected = await service.connectClaudeAssistant()
            claudeAssistantBusy = false
            if connected {
                claudeAssistantState = .connected
                lastActionMessage = "Claude Desktop connected. Restart Claude to load OpenBird."
            } else {
                claudeAssistantState = .failed
                lastActionMessage = "Could not connect Claude Desktop. Existing settings were not changed."
            }
        }
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

    func setOnboardingStateForTesting(lastRefresh: Date?, captureRunning: Bool? = nil) {
        self.lastRefresh = lastRefresh
        self.isRefreshing = false
        if let captureRunning {
            self.captureRunning = captureRunning
        }
    }

    func setReadinessStateForTesting(
        allowlist: [String]? = nil,
        captureRunning: Bool? = nil,
        memoryStats: MemoryStats? = nil
    ) {
        self.isRefreshing = false
        if let allowlist {
            self.allowlist = allowlist
        }
        if let captureRunning {
            self.captureRunning = captureRunning
        }
        if let memoryStats {
            self.memoryStats = memoryStats
            self.memoryStatsState = .loaded(memoryStats)
        }
    }

    func setDeepBrainStatusForTesting(_ state: DeepBrainStatusState) {
        self.deepBrainStatusState = state
    }

    func setDeepBrainPreviewForTesting(_ state: DeepBrainPreviewState) {
        self.deepBrainPreviewState = state
    }

    func setCaptureHealthStateForTesting(_ state: CaptureHealthState) {
        self.captureHealthState = state
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
        Task { _ = await startCaptureAfterRefreshingStats() }
    }

    @discardableResult
    func startCaptureForOnboarding() async -> Bool {
        guard !allowlist.isEmpty else {
            lastActionMessage = "Add at least one app to the capture allowlist first."
            return false
        }
        guard onboardingPrimaryAction == .start else {
            lastActionMessage = nextStepSummary
            return false
        }
        lastActionMessage = "Starting capture..."
        return await startCaptureAfterRefreshingStats()
    }

    /// Start capture after recording the current observation count for health comparison.
    @discardableResult
    private func startCaptureAfterRefreshingStats() async -> Bool {
        await refreshMemoryStats()
        captureHealthCheckTask?.cancel()
        captureStopRequested = false
        let observationsBeforeStart = memoryStats.observations
        if service.startCapture(onExit: { [weak self] code in
            Task { @MainActor in
                await self?.handleCaptureExit(code: code)
            }
        }) {
            captureRunning = true
            capturePaused = false
            captureNeedsReindex = false
            UserDefaults.standard.set(true, forKey: Self.onboardingCompletedKey)
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
            return true
        } else {
            lastActionMessage = "Could not start capture (CLI not found)."
            return false
        }
    }

    /// React to the capture daemon terminating. Extracted from the `onExit` closure
    /// so the exit-code → state mapping (notably the reindex-required code) is unit
    /// testable without spawning a real process.
    func handleCaptureExit(code: Int32) async {
        captureRunning = service.isCaptureRunning()
        captureHealthCheckTask?.cancel()
        captureHealthCheckTask = nil
        await refreshMemoryStats()
        if captureStopRequested {
            captureStopRequested = false
        } else if code == Self.captureReindexExitCode {
            // Recoverable: the index was built under a different embedding model.
            // Offer a one-click reindex instead of a dead-end error.
            captureNeedsReindex = true
            lastActionMessage = "Capture needs a reindex (your embedding model changed). Click Reindex, then start capture."
        } else if code == Self.captureAlreadyRunningExitCode {
            // A daemon we optimistically spawned lost the single-instance race;
            // another daemon already held the lock. Benign — NOT an unexpected
            // failure. Trust the `isCaptureRunning()` re-check above rather than
            // forcing `true`: if the lock holder exited during the race window,
            // no daemon may remain and we must not get stuck claiming it runs.
            lastActionMessage = captureRunning
                ? "Capture is already running."
                : "Capture stopped."
        } else {
            lastActionMessage = code == 0
                ? "Capture stopped."
                : "Capture stopped unexpectedly (exit \(code)). Re-check setup."
        }
    }

    func stopCapture() {
        captureStopRequested = true
        captureHealthCheckTask?.cancel()
        captureHealthCheckTask = nil
        service.stopCapture()
        captureRunning = false
        captureNeedsReindex = false
        lastActionMessage = "Capture stopped."
        Task { await refreshMemoryStats() }
    }

    /// Rebuild the vector index under the current embedding model, then resume
    /// capture. Invoked from the "Reindex" affordance shown after capture exits
    /// with `captureReindexExitCode`. Re-embeds all stored memory, so it can take
    /// a few minutes; `isReindexing` gates the button so it cannot run twice.
    func reindexNow() {
        guard !isReindexing else { return }
        isReindexing = true
        lastActionMessage = "Reindexing memory… (this can take a few minutes)"
        Task {
            let ok = await service.reindex()
            isReindexing = false
            if ok {
                captureNeedsReindex = false
                await refreshMemoryStats()
                // Resume capture once; if it still exits with the reindex code the
                // onExit handler re-sets the flag — it never auto-reindexes again.
                startCapture()
            } else {
                lastActionMessage = "Reindex failed. Run `openbird reindex` in Terminal, then try again."
            }
        }
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
        applyPolicyChangeToRunningCapture()
        Task { await refreshCaptureHealth() }
    }

    func removeFromAllowlist(_ bundleID: String) {
        service.setAllowlist(allowlist.filter { $0 != bundleID })
        allowlist = service.allowlist()
        // The service pruned the OCR opt-in list to the new allowlist
        // (structural subset); mirror the persisted truth.
        ocrApps = service.ocrApps()
        detailedCaptureApps = service.detailedCaptureApps()
        lastActionMessage = "Removed \(bundleID) from capture allowlist."
        applyPolicyChangeToRunningCapture()
        Task { await refreshCaptureHealth() }
    }

    /// Toggle the deep-capture (OCR) opt-in for one allowlisted app. A running
    /// app-launched daemon is cycled so the new policy actually applies (the
    /// daemon reads OPENBIRD_CAPTURE_OCR_APPS at spawn time only — the same
    /// restart precedent as an allowlist edit).
    func setOcrCapture(_ bundleID: String, enabled: Bool) {
        var updated = ocrApps.filter { $0 != bundleID }
        if enabled {
            updated.append(bundleID)
        }
        service.setOcrApps(updated)
        ocrApps = service.ocrApps()
        lastActionMessage = enabled
            ? "Enabled deep capture (OCR) for \(bundleID)."
            : "Disabled deep capture (OCR) for \(bundleID)."
        applyPolicyChangeToRunningCapture()
        Task { await refreshCaptureHealth() }
    }

    /// Grant or revoke detailed local capture for one eligible terminal/editor.
    /// Enabling is refused unless the live preflight verified SQLCipher; Python
    /// independently forces strict encryption before opening the capture store.
    func setDetailedCapture(_ bundleID: String, enabled: Bool) {
        if enabled && encryptionState != .ok {
            lastActionMessage = "Detailed capture requires verified encrypted memory."
            return
        }
        var updated = detailedCaptureApps.filter { $0 != bundleID }
        if enabled {
            updated.append(bundleID)
        }
        service.setDetailedCaptureApps(updated)
        detailedCaptureApps = service.detailedCaptureApps()
        lastActionMessage = enabled
            ? "Enabled detailed local capture for \(bundleID)."
            : "Disabled detailed local capture for \(bundleID)."
        applyPolicyChangeToRunningCapture()
        Task { await refreshCaptureHealth() }
    }

    /// A running daemon reads policy at spawn time only, so an allowlist edit
    /// must cycle the app-launched daemon (bounded stop -> respawn) or, for an
    /// external daemon we must not touch, say so honestly instead of letting
    /// the old policy keep capturing (a removed app would otherwise still be
    /// recorded until the next manual restart).
    /// Serializes policy-change restarts: two quick allowlist edits must not
    /// overlap (both would wait on the same daemon exit and both would spawn).
    /// A second edit arriving mid-restart is COALESCED — the queued run picks
    /// up the latest saved policy, so no edit is lost.
    private var policyRestartInFlight = false
    private var policyRestartQueued = false

    private func applyPolicyChangeToRunningCapture() {
        guard captureRunning else { return }  // policy applies on next start
        if policyRestartInFlight {
            policyRestartQueued = true
            return
        }
        policyRestartInFlight = true
        Task {
            let outcome = await service.restartCaptureForPolicyChange(
                onExit: { [weak self] code in
                    Task { @MainActor in
                        await self?.handleCaptureExit(code: code)
                    }
                }
            )
            switch outcome {
            case .restarted:
                lastActionMessage += " Capture restarted with the updated policy."
            case .externalDaemon:
                lastActionMessage += " Restart your capture daemon to apply it."
            case .failed:
                lastActionMessage += " Restarting capture failed — stop and start capture to apply."
            case .notRunning:
                break
            }
            captureRunning = service.isCaptureRunning()
            policyRestartInFlight = false
            if policyRestartQueued {
                // An edit landed mid-restart: run once more with the latest
                // saved policy (never in parallel, never lost).
                policyRestartQueued = false
                applyPolicyChangeToRunningCapture()
            }
        }
    }

    func runningAppSuggestions() -> [String] {
        let current = Set(allowlist)
        return service.runningAppBundleIDs().filter { !current.contains($0) }
    }

    func captureHealthApp(for bundleID: String) -> CaptureHealthApp? {
        guard case let .loaded(report) = captureHealthState else { return nil }
        return report.apps.first { $0.bundleID == bundleID }
    }

    var detailedCaptureEligibleApps: [String] {
        guard case let .loaded(report) = captureHealthState else {
            return detailedCaptureApps
        }
        return report.apps.compactMap { row in
            row.detailedCapture == nil ? nil : row.bundleID
        }
    }

    var effectiveCaptureAllowedCount: Int {
        guard case let .loaded(report) = captureHealthState else { return allowlist.count }
        return report.apps.filter { $0.policy.capture }.count
    }

    var captureAllowedSummary: String {
        let loaded: Bool
        let count: Int
        if case let .loaded(report) = captureHealthState {
            loaded = true
            count = report.apps.filter { $0.policy.capture }.count
        } else {
            loaded = false
            count = allowlist.count
        }
        let noun = count == 1 ? "app" : "apps"
        return loaded ? "\(count) \(noun) effectively allowed" : "\(count) \(noun) allowed"
    }

    func captureRowStatus(for bundleID: String) -> CaptureRowStatus {
        guard accessibilityEffectivelyGranted else {
            return CaptureRowStatus(
                label: "Needs permission",
                detail: "Accessibility is required before OpenBird can read active-window text.",
                tone: .attention
            )
        }
        if capturePaused {
            return CaptureRowStatus(
                label: "Paused",
                detail: "Capture is paused; no apps are currently read.",
                tone: .attention
            )
        }
        guard let health = captureHealthApp(for: bundleID) else {
            return CaptureRowStatus(
                label: "Checking",
                detail: "Re-check setup to verify effective capture state.",
                tone: .neutral
            )
        }
        if !health.policy.capture {
            return CaptureRowStatus(
                label: blockedStatusLabel(reason: health.policy.reason),
                detail: blockedStatusDetail(
                    reason: health.policy.reason,
                    detailedCapture: health.detailedCapture),
                tone: .attention
            )
        }
        if !captureRunning {
            return CaptureRowStatus(
                label: "Ready",
                detail: captureHealthDetail(health),
                tone: .neutral
            )
        }
        if health.recentObservations > 0 && health.quality == "low_signal" {
            return CaptureRowStatus(
                label: "Low signal",
                detail: captureHealthDetail(health),
                tone: .attention
            )
        }
        if health.recentObservations > 0 {
            return CaptureRowStatus(
                label: "Capturing",
                detail: captureHealthDetail(health),
                tone: .ok
            )
        }
        return CaptureRowStatus(
            label: "No recent captures",
            detail: captureHealthDetail(health),
            tone: .attention
        )
    }

    private func captureHealthDetail(_ health: CaptureHealthApp) -> String {
        let quality = Self.captureQualityLabel(health.quality)
        let counts = "\(health.recentObservations) recent · \(health.totalObservations) total"
        if let ts = health.lastCapturedTS {
            let date = Date(timeIntervalSince1970: ts)
            return "\(quality) signal · \(counts) · last \(date.formatted(date: .omitted, time: .shortened))"
        }
        return "\(quality) signal · \(counts)"
    }

    static func captureQualityLabel(_ quality: String) -> String {
        switch quality {
        case "good": return "Good"
        case "partial": return "Partial"
        case "low_signal": return "Low"
        case "blocked": return "Blocked"
        case "no_recent": return "No recent"
        default: return "Unknown"
        }
    }

    private func blockedStatusLabel(reason: String) -> String {
        switch reason {
        case "blocklisted", "dangerous_app", "self_capture":
            return "Blocked by safety"
        case "not_allowlisted":
            return "Not allowed"
        default:
            return "Blocked"
        }
    }

    private func blockedStatusDetail(
        reason: String, detailedCapture: String? = nil
    ) -> String {
        switch reason {
        case "blocklisted":
            if detailedCapture == "available" {
                return "Terminal/editor capture is off. Enable Detailed local capture below."
            }
            return "Safety blocklist overrides this allowlist entry."
        case "dangerous_app":
            return "Password managers and vault-like apps are never captured."
        case "self_capture":
            return "OpenBird does not capture its own UI."
        case "not_allowlisted":
            return "This app is not effectively allowed by the capture policy."
        default:
            return "Capture policy rejected this app (\(reason))."
        }
    }

    var promptDirectoryPath: String { service.promptDirectoryPath }

    func openPromptsFolder() {
        service.openPromptsFolder()
        lastActionMessage = "Opened prompt customization folder."
    }

    func editPromptPersona(_ key: PromptPersonaKey) {
        lastActionMessage = "Opening \(key.label) prompt editor..."
        Task {
            let outcome = await service.editPromptPersona(key)
            switch outcome {
            case .launched:
                lastActionMessage = "Opened \(key.label) prompt editor."
            case .cliMissing:
                lastActionMessage = "Could not find the openbird CLI. Run `openbird prompts edit \(key.rawValue)` from Terminal."
            case .failed(let code):
                lastActionMessage = "Prompt editor exited with status \(code). Run `openbird prompts edit \(key.rawValue)` from Terminal."
            }
        }
    }

    func openDataFolder() { service.openDataFolder() }
    func openBundleFolder() { service.openBundleFolder() }

    // MARK: - Citation navigation

    /// Open the Today/day view on the day a chat citation came from and focus its
    /// source observation. Switches the pane to `.today`, then hands the computed day
    /// offset + observation id to the wired `citationNavigator` (the shared
    /// `TodayModel`). When `observationId` is absent — older citations may lack it —
    /// the day still opens (the documented minimum behavior). Privacy: only the day
    /// offset and the opaque observation id leave this method; the snippet/app/window
    /// are never read here.
    func navigateToCitation(_ citation: ChatCitation) {
        let offset = Self.dayOffset(forTimestamp: citation.ts)
        selection = .today
        citationNavigator?(offset, citation.observationId)
    }

    /// Day offset (0=today, 1=yesterday, …) of the local calendar day containing
    /// `ts`. Mirrors `TodayModel`'s day keying and the Python `_day_bounds` (local
    /// calendar days) so a citation maps to the SAME day the timeline/briefing show.
    /// A future timestamp clamps to 0 (today) since the Today view has no future days.
    nonisolated static func dayOffset(
        forTimestamp ts: Double,
        now: Date = Date(),
        calendar: Calendar = .current
    ) -> Int {
        let citationDay = calendar.startOfDay(for: Date(timeIntervalSince1970: ts))
        let today = calendar.startOfDay(for: now)
        let days = calendar.dateComponents([.day], from: citationDay, to: today).day ?? 0
        return max(0, days)
    }
}
