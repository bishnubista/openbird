import AppKit
import ApplicationServices
import AVFoundation
import CoreGraphics
import Foundation

/// A decoded, UI-friendly slice of `openbird preflight --json`.
struct PreflightReport: Equatable {
    var ollamaReachable: Bool?           // nil = unknown / not probed
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

enum AccessibilityRequestOutcome: Equatable {
    case alreadyGranted
    case needsPrompt
}

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

/// A grounded chat answer plus its citations (decoded from `openbird chat --json`).
struct ChatResult: Codable, Equatable {
    let answer: String
    let grounded: Bool
    let citations: [ChatCitation]
}

enum ChatError: Error { case cliMissing, failed(String), decode }

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
    private let openBirdCLIResolver: (@Sendable () -> String?)?

    /// The capture daemon launched by the app (if any), so it can be stopped.
    private var captureProcess: Process?

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
        openBirdCLIResolver: (@Sendable () -> String?)? = nil
    ) {
        self.accessibilityProbe = accessibilityProbe
        self.accessibilityPrompter = accessibilityPrompter
        self.privacyPaneOpener = privacyPaneOpener
        self.openBirdCLIResolver = openBirdCLIResolver
    }

    private var dataDirectory: URL {
        Self.dataDirectoryURL()
    }

    /// The OpenBird data directory (`~/.openbird` or `$OPENBIRD_DATA_DIR`). Static
    /// so the launch-time DB-key bootstrap can resolve paths before any instance
    /// exists.
    static func dataDirectoryURL() -> URL {
        if let override = ProcessInfo.processInfo.environment["OPENBIRD_DATA_DIR"],
           !override.isEmpty {
            return URL(fileURLWithPath: NSString(string: override).expandingTildeInPath)
        }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".openbird")
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
            env["OPENBIRD_REQUIRE_ENCRYPTION"] = "1"
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
        }
        // An externally running helper counts as "capturing" — but the helper is
        // only alive intermittently between the daemon's re-spawn cadence, so a
        // miss here does NOT mean capture is stopped. Hence the loop check below.
        if Self.run("/usr/bin/pgrep", arguments: ["-x", "capture-helper"]).exitCode == 0 {
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
        return externalLoopDaemonRunning()
    }

    /// Is a long-lived external `openbird capture --loop` daemon running (one this
    /// app did not launch)? This is the NARROW, authoritative signal for "a
    /// supervising daemon already exists" — used both by `isCaptureRunning()` and
    /// by the pre-spawn duplicate guard in `startCapture()`. The transient
    /// `capture-helper` is deliberately NOT consulted here: it lives only between
    /// the daemon's re-spawn cadence, and a lone helper (e.g. from a bounded
    /// `--once` pass) does not imply a supervising loop daemon.
    ///
    /// `-f` matches against the full argv (the subcommand/flags live in argv, not
    /// in `comm`). `-l` with `-f` is documented by `man pgrep` to print "the
    /// process ID and the full argument list" — that argv is what the pure filter
    /// inspects to reject unrelated processes (e.g. an editor that merely has the
    /// string open, or the daemon's transient `--once` passes).
    func externalLoopDaemonRunning() -> Bool {
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
        guard captureProcess?.isRunning != true, let cli = resolveOpenBirdCLI() else {
            return captureProcess?.isRunning == true
        }
        // Resuming clears any pause gate so capture actually records. Done BEFORE
        // the adopt-external guard below: if we adopt an already-running daemon
        // while a stale `capture.paused` sidecar exists, the daemon would keep
        // honoring the pause while the UI reports capture resumed.
        _ = try? setCapturePaused(false)
        // Advisory pre-spawn guard: if a long-lived external `capture --loop`
        // daemon is ALREADY running — an orphan from a prior app instance that
        // died non-gracefully, or one the user started by hand — do not spawn a
        // second one (two daemons double-capture). Only the supervising loop
        // counts here, NOT a transient `capture-helper` (a lone helper, e.g. from
        // a bounded `--once` pass, does not imply a running daemon). Best-effort
        // only: a TOCTOU window remains before `run()`, so the Python daemon's
        // flock is the real authority (a losing spawn exits code 7, handled
        // benignly upstream). We do NOT adopt/track the external PID — out of scope.
        if externalLoopDaemonRunning() {
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
            captureProcess = process
            return true
        } catch {
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

    // MARK: - Folders & panes

    func openDataFolder() {
        try? fileManager.createDirectory(
            at: dataDirectory,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
        NSWorkspace.shared.open(dataDirectory)
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

    /// Generate a day's grounded prose briefing (`openbird briefing --json`). This
    /// makes one local LLM call, so the timeout is generous; an empty day returns a
    /// deterministic line cheaply. Callers should cache per day.
    func dailyBriefing(dayOffset: Int, timeout: TimeInterval = 120) async -> String? {
        guard let cli = resolveOpenBirdCLI() else { return nil }
        let result = await runAsync(
            cli, arguments: ["briefing", "--day", String(dayOffset), "--json"], timeout: timeout
        )
        guard result.exitCode == 0 else { return nil }
        return Self.parseBriefing(result.stdout)
    }

    static func parseBriefing(_ output: String) -> String? {
        struct Briefing: Decodable { let text: String }
        guard let data = output.data(using: .utf8) else { return nil }
        return (try? JSONDecoder().decode(Briefing.self, from: data))?.text
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
        return report
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

private struct ProcessResult {
    let exitCode: Int
    let stdout: String
    let stderr: String
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
