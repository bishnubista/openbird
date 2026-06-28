import AppKit
import ApplicationServices
import AVFoundation
import CoreGraphics
import Foundation
import ServiceManagement

/// UI-facing state of the "Launch OpenBird at login" setting, decoupled from
/// `SMAppService.Status` so `AppModel`/tests don't depend on ServiceManagement.
enum LaunchAtLoginState: Equatable {
    /// Registered and eligible to launch at login.
    case enabled
    /// Registered but the user must approve it in System Settings › Login Items.
    case requiresApproval
    /// Not registered (the off state).
    case disabled
    /// Cannot be determined / not available (e.g. unsigned dev build, not found).
    case unavailable

    init(_ status: SMAppService.Status) {
        switch status {
        case .enabled: self = .enabled
        case .requiresApproval: self = .requiresApproval
        case .notRegistered: self = .disabled
        case .notFound: self = .unavailable
        @unknown default: self = .unavailable
        }
    }
}

/// Optional meeting speech-to-text backend readiness decoded from preflight.
struct MeetingTranscriptionReadiness: Equatable {
    let parakeetMLXAvailable: Bool
    let fasterWhisperAvailable: Bool
    let backendAvailable: Bool
    let recommendedBackend: String
    let recommendedExtra: String
    let fallbackBackend: String
    let fallbackExtra: String
}

/// A decoded, UI-friendly slice of `openbird preflight --json`.
struct PreflightReport: Equatable {
    var ollamaReachable: Bool?           // nil = unknown / not probed
    var ollamaVersionOK: Bool?           // nil = not required / unreadable / older CLI
    var ollamaHost: String?
    var requiredModels: [String] = []
    var missingModels: [String] = []
    var llmModel: String?
    var embedModel: String?
    var remoteModelRoles: [String: String] = [:]
    var remoteModels: [String] = []
    var usesLocalOllama: Bool = true
    var autoPullAllowed: Bool = false
    var cloudBlocked: Bool = false
    var encryptionStatus: String = "unknown"
    var encryptionEnabled: Bool = false
    /// capability -> "passed" | "failed" | "unknown"
    var grants: [String: String] = [:]
    var helperPresent: Bool = false
    var runtimeOK: Bool = false
    var releaseOK: Bool = false
    var meetingTranscription: MeetingTranscriptionReadiness?
    var error: String?

    func grant(_ capability: String) -> String { grants[capability] ?? "unknown" }
}

/// Progress from Ollama model provisioning. `fraction` is nil when Ollama streams
/// a status update without byte counters.
struct ModelPullProgress: Equatable {
    let model: String
    let status: String
    let completed: Int64?
    let total: Int64?

    var fraction: Double? {
        guard let completed, let total, total > 0 else { return nil }
        return min(1.0, max(0.0, Double(completed) / Double(total)))
    }
}

/// macOS privacy panes the setup flow can deep-link into.
enum PrivacyPane: String {
    case accessibility = "Privacy_Accessibility"
    case screenRecording = "Privacy_ScreenCapture"
    case microphone = "Privacy_Microphone"

    var url: URL? {
        URL(string: "x-apple.systempreferences:com.apple.preference.security?\(rawValue)")
    }
}

enum PromptPersonaKey: String, CaseIterable, Identifiable {
    case rag
    case routine
    case meeting
    case signal

    var id: String { rawValue }

    var label: String {
        switch self {
        case .rag: return "Ask"
        case .routine: return "Routines"
        case .meeting: return "Meetings"
        case .signal: return "Signals"
        }
    }
}

enum AccessibilityRequestOutcome: Equatable {
    case alreadyGranted
    case needsPrompt
}

enum PromptEditOutcome: Equatable {
    case launched
    case cliMissing
    case failed(Int)
}

struct ProcessResult: Sendable {
    let exitCode: Int
    let stdout: String
    let stderr: String
}

typealias PromptEditRunner = @Sendable (String, [String], [String: String]) -> ProcessResult

/// `@unchecked Sendable`: the only mutable instance state is `captureProcess`,
/// which is touched solely on the main actor (AppModel actions and the main-queue
/// willTerminate handler). The static `run`/`runAsync` helpers operate on locals
/// guarded by their own lock. This lets the main-queue termination handler capture
/// the service without a strict-concurrency violation.
/// One cited source behind a chat answer (decoded from `chat --json`).
///
/// `observationId` / `chunkId` carry the source's stable identity so a clicked
/// citation can navigate the user to the originating occurrence. They are
/// OPTIONAL because not every emitted citation is guaranteed to carry them (older
/// CLI builds, or any future un-grounded/synthetic citation): navigation falls
/// back to the citation `ts` (the day) when `observationId` is absent.
struct ChatCitation: Codable, Identifiable, Equatable {
    let index: Int
    let observationId: String?
    let chunkId: String?
    let app: String?
    let window: String?
    let ts: Double
    let snippet: String
    var id: Int { index }

    enum CodingKeys: String, CodingKey {
        case index
        case observationId = "observation_id"
        case chunkId = "chunk_id"
        case app, window, ts, snippet
    }

    /// Memberwise init with `observationId`/`chunkId` defaulted to nil so the common
    /// callers (tests, previews) stay terse while decode still populates the ids.
    init(
        index: Int,
        observationId: String? = nil,
        chunkId: String? = nil,
        app: String?,
        window: String?,
        ts: Double,
        snippet: String
    ) {
        self.index = index
        self.observationId = observationId
        self.chunkId = chunkId
        self.app = app
        self.window = window
        self.ts = ts
        self.snippet = snippet
    }
}

/// One aggregate citation behind a deterministic derived answer.
///
/// Unlike `ChatCitation`, this does not identify one navigable occurrence. The
/// Python CLI emits `derived_from` observation ids for auditability, but the app
/// intentionally does not decode or store them; the UI only needs the derived
/// source's opaque id, label, snippet, and total backing-source count.
struct DerivedChatCitation: Codable, Identifiable, Equatable {
    let index: Int
    let sourceId: String
    let type: String
    let label: String
    let snippet: String
    let derivedFromTotal: Int
    var id: String { sourceId }

    enum CodingKeys: String, CodingKey {
        case index
        case sourceId = "source_id"
        case type, label, snippet
        case derivedFromTotal = "derived_from_total"
    }

    init(
        index: Int,
        sourceId: String,
        type: String = "day_memory",
        label: String,
        snippet: String,
        derivedFromTotal: Int
    ) {
        self.index = index
        self.sourceId = sourceId
        self.type = type
        self.label = label
        self.snippet = snippet
        self.derivedFromTotal = derivedFromTotal
    }
}

/// A source the chat UI can display. Occurrence sources are clickable and
/// navigable; derived sources are aggregate facts and render as non-clickable.
enum ChatSource: Identifiable, Equatable {
    case occurrence(ChatCitation)
    case derived(DerivedChatCitation)

    var id: String {
        switch self {
        case .occurrence(let citation): return "occurrence-\(citation.index)"
        case .derived(let citation): return "derived-\(citation.sourceId)"
        }
    }
}

/// A grounded chat answer plus its citations (decoded from `openbird chat --json`).
struct ChatResult: Codable, Equatable {
    let answer: String
    let grounded: Bool
    let grounding: String?
    let citations: [ChatCitation]
    let derivedCitations: [DerivedChatCitation]
    let reasoningRoute: String?

    enum CodingKeys: String, CodingKey {
        case answer, grounded, grounding, citations
        case derivedCitations = "derived_citations"
        case reasoningRoute = "reasoning_route"
    }

    init(
        answer: String,
        grounded: Bool,
        citations: [ChatCitation],
        grounding: String? = nil,
        derivedCitations: [DerivedChatCitation] = [],
        reasoningRoute: String? = nil
    ) {
        self.answer = answer
        self.grounded = grounded
        self.grounding = grounding
        self.citations = citations
        self.derivedCitations = derivedCitations
        self.reasoningRoute = reasoningRoute
    }

    /// Decode tolerantly around new source fields: older CLIs omit
    /// `derived_citations`, and a malformed optional derived array should not drop
    /// an otherwise valid answer.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        answer = try c.decode(String.self, forKey: .answer)
        grounded = try c.decode(Bool.self, forKey: .grounded)
        grounding = try c.decodeIfPresent(String.self, forKey: .grounding)
        citations = (try? c.decodeIfPresent([ChatCitation].self, forKey: .citations)) ?? []
        derivedCitations = (try? c.decodeIfPresent([DerivedChatCitation].self, forKey: .derivedCitations)) ?? []
        reasoningRoute = try c.decodeIfPresent(String.self, forKey: .reasoningRoute)
    }

    var displaySources: [ChatSource] {
        citations.map(ChatSource.occurrence) + derivedCitations.map(ChatSource.derived)
    }

    var sourceCount: Int { displaySources.count }

    var hasDisplaySources: Bool { sourceCount > 0 }

    var routeLabel: String? {
        switch reasoningRoute {
        case "local_deterministic":
            return "Local only"
        case "cloud_reasoning_active":
            return "Cloud reasoning active"
        case "local_model":
            return "Local model"
        default:
            return nil
        }
    }
}

enum ChatError: Error { case cliMissing, failed(String), decode }

/// One source in a briefing's "source trail" — an occurrence the prose was
/// grounded in (decoded from `openbird briefing --json`'s `sources` array). It
/// mirrors `ChatCitation`'s navigable fields (`observationId` / `app` / `window`
/// / `ts` / `snippet`) so a clicked briefing source reuses the SAME citation
/// navigation chat citations use — focusing that observation in its day. Privacy:
/// `snippet` is already redaction-handled CLI-side (same helpers as chat
/// citations); the app never sees a raw blob.
struct BriefingSource: Codable, Identifiable, Equatable {
    let observationId: String
    let app: String?
    let window: String?
    let ts: Double
    let snippet: String
    var id: String { observationId }

    enum CodingKeys: String, CodingKey {
        case observationId = "observation_id"
        case app, window, ts, snippet
    }

    /// Adapt to a `ChatCitation` so the existing citation navigation/rendering
    /// (`AppModel.navigateToCitation`, `SourcesRail` styling) applies unchanged.
    /// `index` is 1-based for display only; there is no chunk id for a briefing
    /// source (it points at the occurrence, not a retrieved chunk).
    func asCitation(index: Int) -> ChatCitation {
        ChatCitation(
            index: index, observationId: observationId, chunkId: nil,
            app: app, window: window, ts: ts, snippet: snippet
        )
    }
}

/// A day's prose briefing plus its source trail (decoded from `openbird briefing
/// --json`). `sourcesTotal` is the full count of grounding groups; when it
/// exceeds `sources.count` the trail was capped CLI-side and the UI says "N of M"
/// rather than silently truncating.
struct DayBriefing: Codable, Equatable {
    let text: String
    let reasoningRoute: String?
    let sources: [BriefingSource]
    let sourcesTotal: Int

    enum CodingKeys: String, CodingKey {
        case text
        case reasoningRoute = "reasoning_route"
        case sources
        case sourcesTotal = "sources_total"
    }

    var routeLabel: String? {
        switch reasoningRoute {
        case "local_deterministic":
            return "Local only"
        case "cloud_reasoning_active":
            return "Cloud reasoning active"
        case "local_model":
            return "Local model"
        default:
            return nil
        }
    }

    /// Decode tolerantly: `sources`/`sources_total` are absent on older CLI
    /// builds (and the experimental `--signals` path), and `reasoning_route` is
    /// absent on older CLIs that may still have used the configured model. Missing
    /// or unknown routes intentionally resolve to no label, never "Local only."
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        text = try c.decode(String.self, forKey: .text)
        reasoningRoute = try c.decodeIfPresent(String.self, forKey: .reasoningRoute)
        sources = try c.decodeIfPresent([BriefingSource].self, forKey: .sources) ?? []
        sourcesTotal = try c.decodeIfPresent(Int.self, forKey: .sourcesTotal) ?? sources.count
    }

    init(
        text: String,
        reasoningRoute: String? = nil,
        sources: [BriefingSource] = [],
        sourcesTotal: Int? = nil
    ) {
        self.text = text
        self.reasoningRoute = reasoningRoute
        self.sources = sources
        self.sourcesTotal = sourcesTotal ?? sources.count
    }
}

/// Local-only productivity facts decoded from `openbird productivity --json`.
/// This intentionally models only descriptive facts for the Today surface; the
/// source-id-bearing citation fields and coach packet are left undecoded.
struct DayProductivity: Codable, Equatable {
    let route: String
    let egress: String
    let localDate: String
    let sourceScope: String
    let productivity: ProductivityBody

    enum CodingKeys: String, CodingKey {
        case route, egress, productivity
        case localDate = "local_date"
        case sourceScope = "source_scope"
    }

    var facts: ProductivityFacts { productivity.facts }

    var routeLabel: String? {
        route == "productivity.local_facts" && egress == "none" ? "Local facts" : nil
    }

    var hasActivityFacts: Bool {
        facts.activeMinutes > 0
            || facts.contextSwitchCount > 0
            || facts.topCategory != nil
            || facts.topHour != nil
            || facts.longestFocusBlock != nil
    }
}

struct ProductivityBody: Codable, Equatable {
    let facts: ProductivityFacts
}

struct ProductivityFacts: Codable, Equatable {
    let activeMinutes: Double
    let contextSwitchCount: Int
    let contextSwitchesPerActiveHour: Double
    let topCategory: ProductivityTopCategory?
    let topHour: ProductivityTopHour?
    let longestFocusBlock: ProductivityFocusBlock?

    enum CodingKeys: String, CodingKey {
        case activeMinutes = "active_minutes"
        case contextSwitchCount = "context_switch_count"
        case contextSwitchesPerActiveHour = "context_switches_per_active_hour"
        case topCategory = "top_category"
        case topHour = "top_hour"
        case longestFocusBlock = "longest_focus_block"
    }
}

struct ProductivityTopCategory: Codable, Equatable {
    let category: String
    let minutes: Double
    let sourceCount: Int

    enum CodingKeys: String, CodingKey {
        case category, minutes
        case sourceCount = "source_count"
    }
}

struct ProductivityTopHour: Codable, Equatable {
    let hour: String
    let minutes: Double
    let sourceCount: Int

    enum CodingKeys: String, CodingKey {
        case hour, minutes
        case sourceCount = "source_count"
    }
}

struct ProductivityFocusBlock: Codable, Equatable {
    let category: String?
    let seconds: Double
    let sessionCount: Int

    enum CodingKeys: String, CodingKey {
        case category, seconds
        case sessionCount = "session_count"
    }
}

/// Local memory DB counters decoded from `openbird data stats`.
struct MemoryStats: Codable, Equatable {
    let observations: Int
    let blobs: Int
    let chunks: Int
    let vectors: Int

    static let empty = MemoryStats(observations: 0, blobs: 0, chunks: 0, vectors: 0)
}

/// One capture session in a day (decoded from `openbird timeline --json`).
struct TimelineSession: Codable, Identifiable, Equatable {
    let sessionId: String?
    let app: String?
    let start: Double
    let end: Double
    let count: Int
    /// Representative window title for the session (e.g. "rag.py — openbird"),
    /// surfaced by the CLI from real captured `window` metadata. Optional so a
    /// session with no captured window title — or a stale CLI that predates the
    /// field — decodes cleanly.
    let window: String?

    var id: String { "\(sessionId ?? "nil")|\(app ?? "nil")|\(start)" }

    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case app, start, end, count, window
    }
}

/// A day's capture timeline + stat-chip numbers (decoded from `openbird timeline --json`).
struct DayTimeline: Codable, Equatable {
    let dayOffset: Int
    let start: Double
    let end: Double
    let totalObservations: Int
    let distinctApps: Int
    let activeSeconds: Double
    let sessions: [TimelineSession]

    enum CodingKeys: String, CodingKey {
        case dayOffset = "day_offset"
        case start, end
        case totalObservations = "total_observations"
        case distinctApps = "distinct_apps"
        case activeSeconds = "active_seconds"
        case sessions
    }
}


final class OpenBirdService: @unchecked Sendable {
    private let fileManager = FileManager.default
    private let defaults = UserDefaults.standard
    private let allowlistKey = "openbird.captureAllowlist"
    private let accessibilityProbe: @Sendable () -> Bool
    private let accessibilityPrompter: @Sendable () -> Void
    private let privacyPaneOpener: @Sendable (PrivacyPane) -> Void
    private let promptFolderOpener: @Sendable (URL) -> Void
    private let promptEditRunner: PromptEditRunner
    private let openBirdCLIResolver: (@Sendable () -> String?)?
    /// Status-path probe for a long-lived external `openbird capture --loop` daemon.
    /// Defaults to the real `pgrep`-based body; injectable so `isCaptureRunning()` (and
    /// thus `AppModel.init`) can be made hermetic in tests regardless of host process
    /// state. NOTE: the safety-critical pre-spawn guard in `startCapture()` deliberately
    /// does NOT route through this seam — see `externalLoopDaemonRunning()`.
    private let externalLoopDaemonProbe: @Sendable () -> Bool
    /// Status-path probe for a transient external `capture-helper`. Defaults to the real
    /// `pgrep -x capture-helper`; injectable so a stray helper on the host cannot flip
    /// `isCaptureRunning()` under test.
    private let captureHelperRunningProbe: @Sendable () -> Bool
    /// "Launch at login" backend, injectable so the toggle logic is testable without
    /// touching the real `SMAppService` (which only behaves correctly for a signed,
    /// installed app). Defaults forward to `SMAppService.mainApp`.
    private let loginItemStateProbe: @Sendable () -> LaunchAtLoginState
    private let loginItemRegister: @Sendable () throws -> Void
    private let loginItemUnregister: @Sendable () throws -> Void
    private let loginItemsSettingsOpener: @Sendable () -> Void

    /// The capture daemon launched by the app (if any), so it can be stopped.
    private var captureProcess: Process?

    /// Write end of the "death pipe" handed to the launched capture daemon as its
    /// stdin. We hold it open for this app's lifetime and never write to it after
    /// the one-shot handshake token. If the app dies for ANY reason (graceful quit,
    /// crash, SIGKILL) the OS closes this fd, the daemon's stdin reaches EOF, and an
    /// app-supervised daemon self-exits instead of orphaning. Retaining it here also
    /// keeps ARC from closing it early. See `startCapture` and the Python
    /// `OPENBIRD_SUPERVISOR_TOKEN` contract in `openbird/capture/daemon.py`.
    private var captureSupervisorPipe: FileHandle?

    init(
        accessibilityProbe: @escaping @Sendable () -> Bool = { AXIsProcessTrusted() },
        accessibilityPrompter: @escaping @Sendable () -> Void = {
            let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
            _ = AXIsProcessTrustedWithOptions([key: true] as CFDictionary)
        },
        privacyPaneOpener: @escaping @Sendable (PrivacyPane) -> Void = { pane in
            guard let url = pane.url else { return }
            NSWorkspace.shared.open(url)
        },
        promptFolderOpener: @escaping @Sendable (URL) -> Void = { url in
            NSWorkspace.shared.open(url)
        },
        promptEditRunner: PromptEditRunner? = nil,
        openBirdCLIResolver: (@Sendable () -> String?)? = nil,
        externalLoopDaemonProbe: @escaping @Sendable () -> Bool = {
            OpenBirdService.realExternalLoopDaemonRunning()
        },
        captureHelperRunningProbe: @escaping @Sendable () -> Bool = {
            OpenBirdService.run("/usr/bin/pgrep", arguments: ["-x", "capture-helper"]).exitCode == 0
        },
        loginItemStateProbe: @escaping @Sendable () -> LaunchAtLoginState = {
            LaunchAtLoginState(SMAppService.mainApp.status)
        },
        loginItemRegister: @escaping @Sendable () throws -> Void = {
            try SMAppService.mainApp.register()
        },
        loginItemUnregister: @escaping @Sendable () throws -> Void = {
            try SMAppService.mainApp.unregister()
        },
        loginItemsSettingsOpener: @escaping @Sendable () -> Void = {
            SMAppService.openSystemSettingsLoginItems()
        }
    ) {
        self.accessibilityProbe = accessibilityProbe
        self.accessibilityPrompter = accessibilityPrompter
        self.privacyPaneOpener = privacyPaneOpener
        self.promptFolderOpener = promptFolderOpener
        self.promptEditRunner = promptEditRunner ?? { path, arguments, environment in
            OpenBirdService.run(path, arguments: arguments, timeout: 30, environment: environment)
        }
        self.openBirdCLIResolver = openBirdCLIResolver
        self.externalLoopDaemonProbe = externalLoopDaemonProbe
        self.captureHelperRunningProbe = captureHelperRunningProbe
        self.loginItemStateProbe = loginItemStateProbe
        self.loginItemRegister = loginItemRegister
        self.loginItemUnregister = loginItemUnregister
        self.loginItemsSettingsOpener = loginItemsSettingsOpener
    }

    private var dataDirectory: URL {
        Self.dataDirectoryURL()
    }

    /// The OpenBird data directory (`~/.openbird` or `$OPENBIRD_DATA_DIR`). Static
    /// so the launch-time DB-key bootstrap can resolve paths before any instance
    /// exists.
    static func dataDirectoryURL(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL {
        if let override = environment["OPENBIRD_DATA_DIR"],
           !override.isEmpty {
            return URL(fileURLWithPath: NSString(string: override).expandingTildeInPath)
        }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".openbird")
    }

    /// The effective prompt persona override directory, mirroring Python
    /// `Settings.prompts_dir` behavior. `OPENBIRD_PROMPTS_DIR` is intentionally
    /// treated as a literal path string: Python does not expand `~` for this
    /// explicit override, so the Settings display must not either.
    static func promptDirectoryPath(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String {
        if let override = environment["OPENBIRD_PROMPTS_DIR"], !override.isEmpty {
            return override
        }
        return dataDirectoryURL(environment: environment)
            .appendingPathComponent("prompts", isDirectory: true)
            .path
    }

    static func promptDirectoryURL(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL {
        fileURL(forEffectivePath: promptDirectoryPath(environment: environment))
    }

    private static func fileURL(forEffectivePath path: String) -> URL {
        if path.hasPrefix("/") {
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
            .appendingPathComponent(path, isDirectory: true)
    }

    /// The resolved DB file, mirroring Python `Settings` precedence (`config.py`):
    /// `OPENBIRD_DB_PATH` wins if set; otherwise `<data dir>/openbird.db`. The
    /// fail-closed key check MUST inspect this exact file — checking only the
    /// data-dir default could mint a fresh key while a custom-path encrypted DB
    /// elsewhere gets stranded.
    static func databaseURL() -> URL {
        if let override = ProcessInfo.processInfo.environment["OPENBIRD_DB_PATH"],
           !override.isEmpty {
            return URL(fileURLWithPath: NSString(string: override).expandingTildeInPath)
        }
        return dataDirectoryURL().appendingPathComponent("openbird.db")
    }

    // MARK: - DB encryption key (app-owned; injected into CLI children)
    //
    // The signed app resolves the DB key from its own Keychain item (stable
    // Developer-ID DR -> prompt reads "OpenBird", "Always Allow" persists) and
    // overlays it as OPENBIRD_DB_KEY into every CLI child, so the Python layer
    // never touches the Keychain itself. See docs/design/keychain-app-attribution.md.

    /// Resolved once at launch; nil when no key could be safely provided.
    private static var injectedDBKey: String?

    /// Resolve the app-owned DB key and make it available to all CLI children.
    /// Idempotent; call once at launch (applicationDidFinishLaunching) BEFORE any
    /// child process is spawned.
    static func bootstrapDBKey() {
        let dbPath = databaseURL().path
        let (key, _) = KeychainKeyProvider.resolveKey(dbPath: dbPath)  // outcome logged by provider
        guard let key else { return }
        injectedDBKey = key
        // Belt-and-suspenders: also export into our own environment so a child
        // that inherits env without an explicit overlay still receives the key.
        setenv("OPENBIRD_DB_KEY", key, 1)
    }

    /// Base environment for a CLI child with OPENBIRD_DB_KEY explicitly overlaid
    /// (Codex finding #3 — never rely on inherited env / setenv timing alone).
    static func childEnvironment(
        base: [String: String] = ProcessInfo.processInfo.environment
    ) -> [String: String] {
        var env = base
        // The notarized app bundles Python inside the signed app tree. Prevent
        // import-time .pyc writes from mutating the Developer ID seal at runtime.
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        if let key = injectedDBKey {
            env["OPENBIRD_DB_KEY"] = key
        } else {
            // If the signed app could not resolve the key, never let the bundled
            // Python interpreter try Keychain itself. That fallback is what makes
            // macOS prompt for "python3.13" instead of "OpenBird". Strict mode
            // also keeps this from silently degrading to plaintext.
            env["OPENBIRD_DISABLE_KEYRING"] = "1"
            // Refuse silent plaintext by default — BUT respect an explicit operator
            // opt-in. A normal launch never sets OPENBIRD_DISABLE_KEYRING in its own
            // environment, so this guard still fires for a real user whose Keychain
            // failed. A developer running against a plaintext dev DB (e.g. the
            // headless self-test, which launches with OPENBIRD_DISABLE_KEYRING=1)
            // has consciously opted into plaintext, so don't override that choice.
            // Recognize the SAME truthy set the Python keyring parser uses
            // (storage/crypto.py: {1,true,yes,on}), so `=true` works too — not just `=1`.
            let disableRaw = base["OPENBIRD_DISABLE_KEYRING"]?
                .trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? ""
            let keyringExplicitlyDisabled = ["1", "true", "yes", "on"].contains(disableRaw)
            if !keyringExplicitlyDisabled {
                env["OPENBIRD_REQUIRE_ENCRYPTION"] = "1"
            }
        }
        return env
    }

    private var pauseFile: URL {
        dataDirectory.appendingPathComponent("capture.paused")
    }

    // MARK: - Pause / capture lifecycle

    func isCapturePaused() -> Bool {
        fileManager.fileExists(atPath: pauseFile.path)
    }

    @discardableResult
    func setCapturePaused(_ paused: Bool) throws -> Bool {
        try fileManager.createDirectory(
            at: dataDirectory,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
        if paused {
            try Data().write(to: pauseFile, options: .atomic)
            try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: pauseFile.path)
        } else if fileManager.fileExists(atPath: pauseFile.path) {
            try fileManager.removeItem(at: pauseFile)
        }
        return isCapturePaused()
    }

    /// Whether a capture daemon is running — whether this app launched it, an
    /// external `capture-helper` is mid-spawn, or an external
    /// `openbird capture --loop` daemon is supervising.
    func isCaptureRunning() -> Bool {
        if let proc = captureProcess {
            if proc.isRunning { return true }
            captureProcess = nil
            // The launched daemon exited; drop our stale death-pipe write end.
            try? captureSupervisorPipe?.close()
            captureSupervisorPipe = nil
        }
        // An externally running helper counts as "capturing" — but the helper is
        // only alive intermittently between the daemon's re-spawn cadence, so a
        // miss here does NOT mean capture is stopped. Hence the loop check below.
        // Routed through the injectable probe so tests can run hermetically.
        if captureHelperRunningProbe() {
            return true
        }
        // The long-lived parent is `openbird capture --loop` (the CLI/wrapper),
        // which stays alive across helper re-spawns, so it is the reliable signal
        // for an externally-started daemon. `pgrep -x openbird` is too broad (it
        // matches any subcommand, e.g. a transient `openbird doctor`), and
        // `pgrep -f capture` is far too broad (it hits Chrome `video_capture`,
        // `cameracaptured`, …). So we over-match on the argv with a narrow regex,
        // then make the precise, testable decision in
        // `externalCaptureRunning(pgrepOutput:ownPID:)`.
        //
        // Routed through the injectable probe (default == real body) so the
        // status/init path is hermetic under test.
        return externalLoopDaemonProbe()
    }

    /// Is a long-lived external `openbird capture --loop` daemon running (one this
    /// app did not launch)? This is the NARROW, authoritative signal for "a
    /// supervising daemon already exists" — used by `isCaptureRunning()` (the
    /// status/init path, via the injectable `externalLoopDaemonProbe`) and by the
    /// pre-spawn duplicate guard in `startCapture()`. The transient `capture-helper`
    /// is deliberately NOT consulted here: it lives only between the daemon's
    /// re-spawn cadence, and a lone helper (e.g. from a bounded `--once` pass) does
    /// not imply a supervising loop daemon.
    ///
    /// This instance method honors the injected probe, so the status surface stays
    /// hermetic in tests. The safety-critical pre-spawn guard in `startCapture()`
    /// must NOT be defeatable by a test stub, so it calls
    /// `realExternalLoopDaemonRunning()` directly instead of this method.
    func externalLoopDaemonRunning() -> Bool {
        externalLoopDaemonProbe()
    }

    /// The real `pgrep`-based detection of a long-lived external
    /// `openbird capture --loop` daemon. This is the default backing for
    /// `externalLoopDaemonProbe` AND the direct, non-overridable call used by the
    /// `startCapture()` duplicate-spawn guard. Static (captures no `self`) so it can
    /// back the `@Sendable` default closure.
    ///
    /// `-f` matches against the full argv (the subcommand/flags live in argv, not
    /// in `comm`). `-l` with `-f` is documented by `man pgrep` to print "the
    /// process ID and the full argument list" — that argv is what the pure filter
    /// inspects to reject unrelated processes (e.g. an editor that merely has the
    /// string open, or the daemon's transient `--once` passes).
    ///
    /// Privacy-safe: operates only on process argv (binary + subcommand flags) — no
    /// window titles, URLs, or captured content are read or logged.
    static func realExternalLoopDaemonRunning() -> Bool {
        let result = Self.run(
            "/usr/bin/pgrep",
            arguments: ["-fl", "openbird(-cli)? capture( |$)"]
        )
        guard result.exitCode == 0 else { return false }
        return Self.externalCaptureRunning(
            pgrepOutput: result.stdout,
            ownPID: ProcessInfo.processInfo.processIdentifier
        )
    }

    /// Pure decision: given `pgrep -fl` output, is there a live external
    /// `openbird capture --loop` daemon? Factored out so the matching rules are
    /// unit-testable without spawning processes.
    ///
    /// Each `pgrep -fl` line is `"<pid> <full argv>"`. A line counts as a running
    /// external capture daemon when ALL hold:
    ///   1. ADJACENCY: some token T whose path basename is exactly the CLI
    ///      (`openbird` or `openbird-cli`) is immediately followed by the token
    ///      `capture`. This is the real typer invocation shape — the subcommand
    ///      always sits directly after the program. The basename rule rejects mere
    ///      substrings like `video_capture`, and the adjacency requirement rejects
    ///      a lone `openbird` token that happens to share a line with an unrelated
    ///      `capture`/`--loop`. Checked across ALL tokens (not just argv[0]) because
    ///      the real daemon launches as `…/python3 …/openbird capture --loop`.
    ///   2. the `--loop` flag appears as a token AFTER that `capture` (a bounded
    ///      `--once` pass is not a long-running daemon).
    ///   3. it is not this process (`ownPID`, so the app never reports itself) and
    ///      no token's basename is `pgrep` (so this very query never self-matches).
    ///
    /// Privacy-safe: operates only on process argv (binary + subcommand flags) —
    /// no window titles, URLs, or captured content are read or logged.
    ///
    /// Known limitation (accepted): `pgrep -fl` flattens argv into a single
    /// space-joined string, so true argv boundaries are lost. A contrived process
    /// whose SINGLE argument is literally the string `…/openbird capture --loop`
    /// (e.g. an editor opening a file with that exact name incl. spaces) would still
    /// match. This is inherent to any pgrep/ps text source; the adjacency + basename
    /// rules make it require a precisely-crafted path, and the only consequence is a
    /// status badge reading "capturing" — never a data/privacy effect.
    static func externalCaptureRunning(pgrepOutput: String, ownPID: Int32) -> Bool {
        for line in pgrepOutput.split(whereSeparator: \.isNewline) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { continue }

            // Split the leading PID off the argv.
            let parts = trimmed.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: true)
            guard parts.count == 2, let pid = Int32(parts[0]) else { continue }
            if pid == ownPID { continue }

            let tokens = String(parts[1]).split(separator: " ").map(String.init)

            // Rule 3 (pgrep self): reject only when a token's basename is `pgrep`,
            // never on a whole-line substring (a path could legitimately contain it).
            if tokens.contains(where: { ($0 as NSString).lastPathComponent == "pgrep" }) {
                continue
            }

            // Rules 1 & 2: find `<…/openbird|…/openbird-cli> capture` as ADJACENT
            // tokens, then require `--loop` somewhere after that `capture`.
            for index in tokens.indices.dropLast() {
                let exe = (tokens[index] as NSString).lastPathComponent
                guard exe == "openbird" || exe == "openbird-cli" else { continue }
                guard tokens[index + 1] == "capture" else { continue }
                if tokens[(index + 2)...].contains("--loop") {
                    return true
                }
            }
        }
        return false
    }

    /// Launch `openbird capture --loop` via the bundled wrapper, injecting the
    /// saved allowlist as OPENBIRD_ALLOWLIST so the daemon captures only the apps
    /// the user opted into. Returns false if the CLI cannot be resolved.
    @discardableResult
    func startCapture(onExit: (@Sendable (Int32) -> Void)? = nil) -> Bool {
        // If a launched daemon is still running, leave it (and its death pipe)
        // untouched. If one exists but has already exited, drop the now-stale
        // death-pipe write fd so it is not retained across this re-spawn / early
        // return (e.g. CLI unresolved or an external daemon already running).
        if let proc = captureProcess {
            if proc.isRunning { return true }
            captureProcess = nil
            try? captureSupervisorPipe?.close()
            captureSupervisorPipe = nil
        }
        guard let cli = resolveOpenBirdCLI() else {
            return false
        }
        // Resuming clears any pause gate so capture actually records. Done BEFORE
        // the adopt-external guard below: if we adopt an already-running daemon
        // while a stale `capture.paused` sidecar exists, the daemon would keep
        // honoring the pause while the UI reports capture resumed. If clearing the
        // gate fails (or the sidecar survives), fail the launch rather than report
        // success while capture stays silently paused.
        do {
            if try setCapturePaused(false) {
                return false  // sidecar still present -> daemon would stay paused
            }
        } catch {
            return false
        }
        // Advisory pre-spawn guard: if a long-lived external `capture --loop`
        // daemon is ALREADY running — an orphan from a prior app instance that
        // died non-gracefully, or one the user started by hand — do not spawn a
        // second one (two daemons double-capture). Only the supervising loop
        // counts here, NOT a transient `capture-helper` (a lone helper, e.g. from
        // a bounded `--once` pass, does not imply a running daemon). Best-effort
        // only: a TOCTOU window remains before `run()`, so the Python daemon's
        // flock is the real authority (a losing spawn exits code 7, handled
        // benignly upstream). We do NOT adopt/track the external PID — out of scope.
        // Calls the REAL detection directly (not the injectable status probe): a
        // test stub that fakes "no daemon" for the status surface must never be able
        // to silently disable this double-spawn safeguard.
        if Self.realExternalLoopDaemonRunning() {
            return true
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: cli)
        process.arguments = ["capture", "--loop"]
        var env = Self.childEnvironment()
        let allow = allowlist()
        if !allow.isEmpty {
            env["OPENBIRD_ALLOWLIST"] = allow.joined(separator: ",")
        }

        // App-supervised self-exit ("death pipe"). Hand the daemon a pipe as its
        // stdin and stamp a per-launch random token into the environment. We write
        // that exact token through the pipe right after launch; the daemon arms its
        // self-exit watcher ONLY after reading the matching token (a leaked env var
        // alone cannot arm it — see openbird/capture/daemon.py SUPERVISOR_TOKEN_ENV).
        // While this app lives we keep the write end open and never write again; when
        // the app dies (quit/crash/SIGKILL) the OS closes it, the daemon's stdin hits
        // EOF, and the daemon self-exits instead of orphaning under launchd.
        let deathPipe = Pipe()
        let supervisorWrite = deathPipe.fileHandleForWriting
        // Mark the write end close-on-exec so the spawned child inherits ONLY the
        // read end (its stdin). Otherwise a duplicate writer surviving exec would
        // hold the pipe open forever and the EOF would never arrive. If this fails
        // the self-exit contract is broken, so refuse to launch an unsupervised
        // daemon rather than orphan one under launchd.
        guard fcntl(supervisorWrite.fileDescriptor, F_SETFD, FD_CLOEXEC) != -1 else {
            try? supervisorWrite.close()
            return false
        }
        let token = UUID().uuidString
        env["OPENBIRD_SUPERVISOR_TOKEN"] = token
        process.standardInput = deathPipe

        process.environment = env
        // Discard helper stdout/stderr from the app: captured content must never
        // flow into the app's logs. The daemon persists to the local DB itself.
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        process.terminationHandler = { proc in
            onExit?(proc.terminationStatus)
        }
        do {
            try process.run()
            // One-shot handshake: prove to the daemon that THIS pipe carries the
            // app-supervisor token, then keep the write end open (held by the
            // retained FileHandle below) and never write again. The daemon arms its
            // self-exit watcher ONLY after reading this token, so a failed write
            // would leave the daemon running unsupervised (orphan-prone). Tear the
            // just-spawned daemon back down and report failure instead.
            do {
                try supervisorWrite.write(contentsOf: Data((token + "\n").utf8))
            } catch {
                // Detach the exit handler before terminating: this launch never
                // became `captureProcess`, so the UI must not process an onExit
                // for a daemon it never marked running.
                process.terminationHandler = nil
                process.terminate()
                try? supervisorWrite.close()
                return false
            }
            captureProcess = process
            captureSupervisorPipe = supervisorWrite
            return true
        } catch {
            try? supervisorWrite.close()
            return false
        }
    }

    /// Stop the app-launched capture daemon and any stray helper processes.
    func stopCapture() {
        terminateLaunchedCapture()
        _ = stopHelperProcesses()
    }

    /// Terminate ONLY the capture daemon this app launched (no pkill), so quitting
    /// the app never orphans a long-running `openbird capture --loop` child. Kept
    /// distinct from `stopCapture()` so app-quit cleanup does not also kill a
    /// capture daemon the user started independently of the app.
    func terminateLaunchedCapture() {
        if let proc = captureProcess, proc.isRunning {
            proc.terminate()
        }
        captureProcess = nil
        // Close our death-pipe write end. We already terminated the child above,
        // so this is a redundant EOF backstop; closing it also releases the fd.
        try? captureSupervisorPipe?.close()
        captureSupervisorPipe = nil
    }

    /// Rebuild the on-disk vector index under the current embedding model via
    /// `openbird reindex --yes`. Needed after the default embed model changes:
    /// an index built under the old model trips `EmbeddingCohortMismatch` and the
    /// capture daemon exits `CAPTURE_EXIT_REINDEX_REQUIRED`. Re-embeds every stored
    /// observation, so it is long-running (hence the generous timeout) and callers
    /// run it off the main actor. Returns true on a clean (exit 0) reindex.
    func reindex(timeout: TimeInterval = 600) async -> Bool {
        guard let cli = resolveOpenBirdCLI() else { return false }
        let result = await runAsync(cli, arguments: ["reindex", "--yes"], timeout: timeout)
        return result.exitCode == 0
    }

    // MARK: - Helpers

    func helperStatuses() -> [HelperStatus] {
        [
            helperStatus(id: "capture", label: "Capture helper", executable: "capture-helper"),
            helperStatus(id: "audio", label: "Audio helper", executable: "audio-helper")
        ]
    }

    func stopHelperProcesses() -> Bool {
        let processNames = ["capture-helper", "audio-helper", "CaptureHelper", "AudioHelper"]
        return processNames
            .map { Self.run("/usr/bin/pkill", arguments: ["-x", $0]).exitCode == 0 }
            .contains(true)
    }

    func canLaunchOpenBirdCLI() -> Bool {
        resolveOpenBirdCLI() != nil
    }

    // MARK: - Launch at login (SMAppService)

    /// Current "launch at login" state, read live from the system (never cached) so
    /// it reflects changes the user makes in System Settings › Login Items.
    func launchAtLoginState() -> LaunchAtLoginState {
        loginItemStateProbe()
    }

    /// Register or unregister this app as a macOS login item. Throws the underlying
    /// `SMAppService` error so the caller can resync state and surface a message.
    func setLaunchAtLogin(_ enabled: Bool) throws {
        if enabled {
            try loginItemRegister()
        } else {
            try loginItemUnregister()
        }
    }

    /// Open System Settings › General › Login Items so the user can approve the app
    /// when registration lands in the `.requiresApproval` state.
    func openLoginItemsSettings() {
        loginItemsSettingsOpener()
    }

    // MARK: - Folders & panes

    func openDataFolder() {
        try? fileManager.createDirectory(
            at: dataDirectory,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
        NSWorkspace.shared.open(dataDirectory)
    }

    var promptDirectoryPath: String {
        Self.promptDirectoryPath()
    }

    func openPromptsFolder() {
        let directory = Self.promptDirectoryURL()
        try? fileManager.createDirectory(
            at: directory,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
        promptFolderOpener(directory)
    }

    func editPromptPersona(_ key: PromptPersonaKey) async -> PromptEditOutcome {
        guard let cli = resolveOpenBirdCLI() else { return .cliMissing }
        var environment = Self.childEnvironment()
        // A GUI-launched Settings button cannot drive terminal editors like vim.
        // Force Launch Services to open the scaffolded .txt persona file.
        environment["EDITOR"] = "/usr/bin/open"
        let result = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                continuation.resume(returning: self.promptEditRunner(
                    cli,
                    ["prompts", "edit", key.rawValue],
                    environment
                ))
            }
        }
        return result.exitCode == 0 ? .launched : .failed(result.exitCode)
    }

    func openBundleFolder() {
        NSWorkspace.shared.activateFileViewerSelecting([Bundle.main.bundleURL])
    }

    /// Open the given macOS privacy pane in System Settings so the user can grant
    /// the permission. macOS does not allow an app to grant TCC itself; deep-link
    /// + re-check is the closest to "no manual work" the platform permits.
    func openPrivacyPane(_ pane: PrivacyPane) {
        privacyPaneOpener(pane)
    }

    // MARK: - TCC checked/requested from the APP process
    //
    // macOS attributes a nested helper's TCC request to its containing app bundle
    // (the "responsible process"), so grants land on OpenBird.app, not on the
    // flat helper binary. We therefore check AND request these permissions from
    // the app's own process — that is where the grant actually lives, and the
    // capture daemon (launched as a descendant of the app) inherits it at runtime.

    func accessibilityGranted() -> Bool {
        accessibilityProbe()
    }

    func screenRecordingGranted() -> Bool {
        CGPreflightScreenCaptureAccess()
    }

    func microphoneGranted() -> Bool {
        AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
    }

    static func decideAccessibilityRequest(isTrusted: Bool) -> AccessibilityRequestOutcome {
        isTrusted ? .alreadyGranted : .needsPrompt
    }

    /// Trigger the Accessibility prompt for the APP only when it is not already
    /// trusted. Re-prompting an already granted app creates a confusing loop when
    /// System Settings is already correct but the setup UI is stale.
    func requestAccessibility() -> AccessibilityRequestOutcome {
        let outcome = Self.decideAccessibilityRequest(isTrusted: accessibilityGranted())
        guard outcome == .needsPrompt else { return outcome }
        accessibilityPrompter()
        openPrivacyPane(.accessibility)
        return outcome
    }

    /// Trigger the Screen-Recording prompt for the APP, then open the pane.
    func requestScreenRecording() {
        _ = CGRequestScreenCaptureAccess()
        openPrivacyPane(.screenRecording)
    }

    /// Trigger the Microphone prompt for the APP, then open the pane.
    func requestMicrophone() {
        AVCaptureDevice.requestAccess(for: .audio) { _ in }
        openPrivacyPane(.microphone)
    }

    // MARK: - Allowlist (persisted in UserDefaults; injected into capture)

    func allowlist() -> [String] {
        defaults.stringArray(forKey: allowlistKey) ?? []
    }

    func setAllowlist(_ bundleIDs: [String]) {
        // De-dupe, trim, drop empties; keep stable order.
        var seen = Set<String>()
        let cleaned = bundleIDs
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && seen.insert($0).inserted }
        defaults.set(cleaned, forKey: allowlistKey)
    }

    /// Bundle IDs of other apps currently running with a regular UI, useful as
    /// allowlist suggestions.
    func runningAppBundleIDs() -> [String] {
        NSWorkspace.shared.runningApplications
            .filter { $0.activationPolicy == .regular && $0.bundleIdentifier != nil }
            .compactMap { $0.bundleIdentifier }
            .filter { $0 != Bundle.main.bundleIdentifier }
            .sorted()
    }

    // MARK: - Ollama

    func ollamaPath() -> String? {
        let candidates = ["/opt/homebrew/bin/ollama", "/usr/local/bin/ollama", "/usr/bin/ollama"]
        return candidates.first { fileManager.isExecutableFile(atPath: $0) }
    }

    /// Pull a model via the local Ollama API, falling back to the CLI when
    /// available. This can take minutes (multi-GB), so it uses a generous timeout
    /// and reports concise progress/outcome strings.
    func pullModel(
        _ model: String,
        host: String?,
        progress: (@Sendable (ModelPullProgress) -> Void)? = nil
    ) async -> (ok: Bool, message: String) {
        if let host, let url = Self.ollamaPullURL(host: host) {
            do {
                let message = try await pullModelViaAPI(model, url: url, progress: progress)
                return (true, message)
            } catch {
                if ollamaPath() == nil {
                    return (false, "Could not pull \(model): \(Self.describePullError(error))")
                }
            }
        }

        guard let ollama = ollamaPath() else {
            return (false, "Ollama CLI not found. Install Ollama from ollama.com.")
        }
        var environment = Self.childEnvironment()
        if let host = host?.trimmingCharacters(in: .whitespacesAndNewlines),
           !host.isEmpty {
            environment["OLLAMA_HOST"] = host
        }
        let result = await runAsync(
            ollama,
            arguments: ["pull", model],
            timeout: 1800,
            environment: environment
        )
        if result.exitCode == 0 {
            return (true, "Pulled \(model).")
        }
        let detail = result.stderr.isEmpty ? "exit \(result.exitCode)" : result.stderr
        return (false, "Could not pull \(model): \(detail)")
    }

    static func ollamaPullURL(host: String) -> URL? {
        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let normalized = trimmed.hasSuffix("/") ? trimmed : "\(trimmed)/"
        guard let base = URL(string: normalized) else { return nil }
        return URL(string: "api/pull", relativeTo: base)?.absoluteURL
    }

    static func parsePullProgressLine(_ line: String, model: String) throws -> ModelPullProgress? {
        guard let data = line.data(using: .utf8),
              let payload = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        if let error = payload["error"] as? String, !error.isEmpty {
            throw OllamaPullError.api(error)
        }
        guard let status = payload["status"] as? String, !status.isEmpty else {
            return nil
        }
        return ModelPullProgress(
            model: model,
            status: status,
            completed: Self.int64(payload["completed"]),
            total: Self.int64(payload["total"])
        )
    }

    private func pullModelViaAPI(
        _ model: String,
        url: URL,
        progress: (@Sendable (ModelPullProgress) -> Void)?
    ) async throws -> String {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 1800
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["model": model])

        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw OllamaPullError.http(http.statusCode)
        }

        for try await line in bytes.lines {
            guard let update = try Self.parsePullProgressLine(line, model: model) else { continue }
            progress?(update)
        }
        return "Pulled \(model)."
    }

    private static func int64(_ value: Any?) -> Int64? {
        if let value = value as? Int64 { return value }
        if let value = value as? Int { return Int64(value) }
        if let value = value as? NSNumber { return value.int64Value }
        return nil
    }

    private static func describePullError(_ error: Error) -> String {
        if let error = error as? OllamaPullError {
            return error.description
        }
        return error.localizedDescription
    }

    // MARK: - Preflight

    func preflightReport() async -> PreflightReport {
        guard let cli = resolveOpenBirdCLI() else {
            return PreflightReport(error: "openbird CLI not found in app bundle or PATH.")
        }
        let result = await runAsync(cli, arguments: ["preflight", "--json"], timeout: 30)
        guard result.exitCode == 0 || result.exitCode == 1 else {
            return PreflightReport(error: result.stderr.isEmpty
                ? "openbird preflight exited with \(result.exitCode)."
                : result.stderr)
        }
        return Self.parsePreflight(result.stdout)
    }

    func memoryStats() async -> MemoryStats? {
        guard let cli = resolveOpenBirdCLI() else { return nil }
        let result = await runAsync(cli, arguments: ["data", "stats"], timeout: 10)
        guard result.exitCode == 0,
              let decoded = Self.parseMemoryStats(result.stdout) else {
            return nil
        }
        return decoded
    }

    /// Decode the JSON emitted by `openbird data stats` into UI counters.
    static func parseMemoryStats(_ output: String) -> MemoryStats? {
        guard let data = output.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(MemoryStats.self, from: data)
    }

    /// Load a day's capture timeline (sessions + stat numbers). Pure local read;
    /// `openbird timeline` opens the store without an LLM/cloud gate, so this is fast.
    func dayTimeline(dayOffset: Int) async -> DayTimeline? {
        guard let cli = resolveOpenBirdCLI() else { return nil }
        let result = await runAsync(
            cli, arguments: ["timeline", "--day", String(dayOffset), "--json"], timeout: 20
        )
        guard result.exitCode == 0 else { return nil }
        return Self.parseDayTimeline(result.stdout)
    }

    static func parseDayTimeline(_ output: String) -> DayTimeline? {
        guard let data = output.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(DayTimeline.self, from: data)
    }

    /// Generate a day's grounded briefing + source trail (`openbird briefing
    /// --json`). The default CLI route is local deterministic day memory; callers
    /// cache per day.
    func dailyBriefing(dayOffset: Int, timeout: TimeInterval = 60) async -> DayBriefing? {
        guard let cli = resolveOpenBirdCLI() else { return nil }
        let result = await runAsync(
            cli, arguments: ["briefing", "--day", String(dayOffset), "--json"], timeout: timeout
        )
        guard result.exitCode == 0 else { return nil }
        return Self.parseBriefing(result.stdout)
    }

    static func parseBriefing(_ output: String) -> DayBriefing? {
        guard let data = output.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(DayBriefing.self, from: data)
    }

    /// Load deterministic, local-only productivity facts for a day. This may build
    /// day memory on first use, so it gets the same order-of-magnitude timeout as
    /// briefing rather than the cheap timeline read.
    func dailyProductivity(dayOffset: Int, timeout: TimeInterval = 60) async -> DayProductivity? {
        guard let cli = resolveOpenBirdCLI() else { return nil }
        let result = await runAsync(
            cli, arguments: ["productivity", "--day", String(dayOffset), "--json"], timeout: timeout
        )
        guard result.exitCode == 0 else { return nil }
        return Self.parseProductivity(result.stdout)
    }

    static func parseProductivity(_ output: String) -> DayProductivity? {
        guard let data = output.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(DayProductivity.self, from: data)
    }

    // MARK: - Internals

    private func helperStatus(id: String, label: String, executable: String) -> HelperStatus {
        let url = Bundle.main.bundleURL
            .appendingPathComponent("Contents/MacOS")
            .appendingPathComponent(executable)
        return HelperStatus(
            id: id,
            label: label,
            isBundled: fileManager.isExecutableFile(atPath: url.path),
            path: url.path
        )
    }

    private func resolveOpenBirdCLI() -> String? {
        if let openBirdCLIResolver {
            return openBirdCLIResolver()
        }
        let bundled = Bundle.main.bundleURL
            .appendingPathComponent("Contents/MacOS/openbird-cli")
        let candidates = [bundled.path, "/opt/homebrew/bin/openbird", "/usr/local/bin/openbird"]
        return candidates.first { fileManager.isExecutableFile(atPath: $0) }
    }

    /// Ask a grounded question over captured memory via `openbird chat --json`.
    /// Runs the bundled CLI (inheriting the app's environment, incl. any
    /// OPENBIRD_DATA_DIR) and decodes the structured answer + citations. The LLM
    /// call can take a while, so the timeout is generous. Synchronous — callers
    /// run it off the main actor.
    ///
    /// When `dayOffset` is non-nil the CLI receives `--day N` (0=today,
    /// 1=yesterday, ...), which HARD-SCOPES retrieval and every citation to that
    /// calendar day. The Today view's "Ask about this day" passes the viewed day;
    /// the generic/global Ask passes nil so retrieval stays unscoped.
    func askChat(
        _ question: String, dayOffset: Int? = nil, timeout: TimeInterval = 90
    ) throws -> ChatResult {
        guard let cli = resolveOpenBirdCLI() else { throw ChatError.cliMissing }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: cli)
        // The question goes via STDIN, never argv — chat text must not be visible
        // to local process inspection (consistent with the capture pipeline).
        // `--day N` (day scope) is config, not content, so it's safe in argv.
        process.arguments = Self.chatArguments(dayOffset: dayOffset)
        process.environment = Self.childEnvironment()

        let stdinPipe = Pipe()
        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardInput = stdinPipe
        process.standardOutput = outPipe
        process.standardError = errPipe

        // Launch FIRST. Starting the pipe-drain readers before run() would leak
        // them (blocked on readDataToEndOfFile waiting for an EOF that never comes)
        // if run() throws.
        do { try process.run() } catch { throw ChatError.failed("Could not launch chat.") }

        // Drain stdout/stderr on background queues so a full pipe buffer cannot
        // deadlock against our wait loop. Started only after a successful launch.
        var outData = Data()
        var errData = Data()
        let lock = NSLock()
        let group = DispatchGroup()
        for pipe in [outPipe, errPipe] {
            group.enter()
            let isOut = pipe === outPipe
            DispatchQueue.global(qos: .utility).async {
                let d = pipe.fileHandleForReading.readDataToEndOfFile()
                lock.lock()
                if isOut {
                    outData = d
                } else {
                    errData = d
                }
                lock.unlock()
                group.leave()
            }
        }

        // Feed the question, then close stdin so the CLI sees EOF.
        if let qData = (question + "\n").data(using: .utf8) {
            stdinPipe.fileHandleForWriting.write(qData)
        }
        try? stdinPipe.fileHandleForWriting.close()

        // Hard timeout: wait, then SIGTERM, grace, then SIGKILL.
        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.05) }
        if process.isRunning {
            process.terminate()  // SIGTERM
            let grace = Date().addingTimeInterval(2)
            while process.isRunning && Date() < grace { Thread.sleep(forTimeInterval: 0.05) }
            if process.isRunning { kill(process.processIdentifier, SIGKILL) }
            process.waitUntilExit()
            group.wait()
            throw ChatError.failed("Chat timed out while waiting for the local model.")
        }

        process.waitUntilExit()
        group.wait()
        lock.lock()
        let data = outData
        let stderr = String(data: errData, encoding: .utf8) ?? ""
        lock.unlock()
        guard process.terminationStatus == 0 else {
            throw ChatError.failed(Self.chatFailureSummary(
                exitCode: process.terminationStatus,
                stderr: stderr
            ))
        }
        guard let decoded = try? JSONDecoder().decode(ChatResult.self, from: data) else {
            throw ChatError.decode
        }
        return decoded
    }

    /// Build the `openbird chat` argv. The question is fed via stdin (`--stdin`),
    /// never argv; `--day N` is appended ONLY when a day scope is requested, so the
    /// unscoped path is byte-for-byte unchanged. Pure + static so it's unit-testable
    /// without spawning a process.
    static func chatArguments(dayOffset: Int?) -> [String] {
        var args = ["chat", "--json", "--stdin"]
        if let dayOffset {
            args += ["--day", String(dayOffset)]
        }
        return args
    }

    static func chatFailureSummary(exitCode: Int32, stderr: String) -> String {
        let lower = stderr.lowercased()
        if lower.contains("openbird_allow_cloud") || lower.contains("cloud model configured") {
            return "Chat blocked because a cloud model is configured without opt-in."
        }
        if lower.contains("model") && (lower.contains("not found") || lower.contains("missing")) {
            return "Chat failed because a required local model is missing."
        }
        if lower.contains("connection refused") || lower.contains("ollama") {
            return "Chat failed because the local Ollama model request did not complete."
        }
        return "Chat failed (exit \(exitCode)). Run openbird doctor for details."
    }

    static func parsePreflight(_ output: String) -> PreflightReport {
        guard let data = output.data(using: .utf8),
              let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return PreflightReport(error: "Could not parse preflight JSON.")
        }
        var report = PreflightReport()
        report.runtimeOK = payload["runtime_ok"] as? Bool ?? false
        report.releaseOK = payload["release_gate_ok"] as? Bool ?? false

        if let ollama = payload["ollama"] as? [String: Any] {
            report.ollamaReachable = ollama["reachable"] as? Bool   // nil if "unknown"/"n/a"
            report.ollamaHost = ollama["host"] as? String
            report.requiredModels = ollama["required_models"] as? [String] ?? []
            report.missingModels = ollama["missing_models"] as? [String] ?? []
            report.autoPullAllowed = ollama["auto_pull_allowed"] as? Bool ?? false
            report.ollamaVersionOK = ollama["version_ok"] as? Bool
        }
        if let cloud = payload["cloud"] as? [String: Any] {
            report.llmModel = cloud["llm_model"] as? String
            report.embedModel = cloud["embed_model"] as? String
            if let remoteModelRoles = cloud["remote_models"] as? [String: String] {
                report.remoteModelRoles = remoteModelRoles
                report.remoteModels = remoteModelRoles
                    .keys
                    .sorted()
                    .compactMap { remoteModelRoles[$0] }
            } else if let remoteModels = cloud["remote_models"] as? [String] {
                report.remoteModels = remoteModels
            }
            report.usesLocalOllama = cloud["uses_local_ollama"] as? Bool ?? true
            report.cloudBlocked = cloud["blocked"] as? Bool ?? false
        }
        if let enc = payload["encryption"] as? [String: Any] {
            report.encryptionStatus = enc["status"] as? String ?? "unknown"
            report.encryptionEnabled = enc["enabled"] as? Bool ?? false
        }
        if let macos = payload["macos"] as? [String: Any] {
            report.helperPresent = macos["helper_present"] as? Bool ?? false
            for cap in ["accessibility", "screen_recording", "microphone", "system_audio"] {
                report.grants[cap] = macos[cap] as? String ?? "unknown"
            }
        }
        if let meetings = payload["meetings"] as? [String: Any],
           let transcription = meetings["transcription"] as? [String: Any] {
            report.meetingTranscription = Self.parseMeetingTranscription(transcription)
        }
        return report
    }

    private static func parseMeetingTranscription(
        _ payload: [String: Any]
    ) -> MeetingTranscriptionReadiness? {
        guard
            let parakeetMLXAvailable = payload["parakeet_mlx_available"] as? Bool,
            let fasterWhisperAvailable = payload["faster_whisper_available"] as? Bool,
            let backendAvailable = payload["backend_available"] as? Bool,
            let recommendedBackend = payload["recommended_backend"] as? String,
            let recommendedExtra = payload["recommended_extra"] as? String,
            let fallbackBackend = payload["fallback_backend"] as? String,
            let fallbackExtra = payload["fallback_extra"] as? String
        else {
            return nil
        }
        return MeetingTranscriptionReadiness(
            parakeetMLXAvailable: parakeetMLXAvailable,
            fasterWhisperAvailable: fasterWhisperAvailable,
            backendAvailable: backendAvailable,
            recommendedBackend: recommendedBackend,
            recommendedExtra: recommendedExtra,
            fallbackBackend: fallbackBackend,
            fallbackExtra: fallbackExtra
        )
    }

    private static func run(
        _ path: String,
        arguments: [String],
        timeout: TimeInterval = 4,
        environment: [String: String]? = nil
    ) -> ProcessResult {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = arguments
        // Overlay OPENBIRD_DB_KEY so preflight/data-stats children read the
        // app-owned key and never raise their own Keychain prompt.
        process.environment = environment ?? childEnvironment()

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr

        // Read pipes on background queues so a child that fills the 64 KB pipe
        // buffer cannot deadlock against our wait loop.
        var outData = Data()
        var errData = Data()
        let ioGroup = DispatchGroup()
        let lock = NSLock()
        for (pipe, append) in [(stdout, { (d: Data) in outData.append(d) }),
                               (stderr, { (d: Data) in errData.append(d) })] {
            ioGroup.enter()
            DispatchQueue.global(qos: .utility).async {
                let d = pipe.fileHandleForReading.readDataToEndOfFile()
                lock.lock(); append(d); lock.unlock()
                ioGroup.leave()
            }
        }

        do {
            try process.run()
        } catch {
            return ProcessResult(exitCode: 127, stdout: "", stderr: error.localizedDescription)
        }

        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        if process.isRunning {
            process.terminate()
        }
        process.waitUntilExit()
        ioGroup.wait()

        let out = String(data: outData, encoding: .utf8) ?? ""
        let err = String(data: errData, encoding: .utf8) ?? ""
        return ProcessResult(
            exitCode: Int(process.terminationStatus),
            stdout: out.trimmingCharacters(in: .whitespacesAndNewlines),
            stderr: err.trimmingCharacters(in: .whitespacesAndNewlines)
        )
    }

    private func runAsync(
        _ path: String,
        arguments: [String],
        timeout: TimeInterval,
        environment: [String: String]? = nil
    ) async -> ProcessResult {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                continuation.resume(returning: Self.run(
                    path,
                    arguments: arguments,
                    timeout: timeout,
                    environment: environment
                ))
            }
        }
    }
}

private enum OllamaPullError: Error, CustomStringConvertible {
    case api(String)
    case http(Int)

    var description: String {
        switch self {
        case .api(let message):
            return message
        case .http(let status):
            return "Ollama API returned HTTP \(status)"
        }
    }
}
