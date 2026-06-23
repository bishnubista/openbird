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
/// One cited source behind a chat answer (decoded from `chat --json`). Extra
/// fields in the JSON (observation_id, chunk_id) are intentionally ignored.
struct ChatCitation: Codable, Identifiable, Equatable {
    let index: Int
    let app: String?
    let window: String?
    let ts: Double
    let snippet: String
    var id: Int { index }
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

    /// Whether a capture daemon launched by this app is still running.
    func isCaptureRunning() -> Bool {
        if let proc = captureProcess {
            if proc.isRunning { return true }
            captureProcess = nil
        }
        // Also treat an externally running helper as "capturing" for status.
        return Self.run("/usr/bin/pgrep", arguments: ["-x", "capture-helper"]).exitCode == 0
    }

    /// Launch `openbird capture --loop` via the bundled wrapper, injecting the
    /// saved allowlist as OPENBIRD_ALLOWLIST so the daemon captures only the apps
    /// the user opted into. Returns false if the CLI cannot be resolved.
    @discardableResult
    func startCapture(onExit: (@Sendable (Int32) -> Void)? = nil) -> Bool {
        guard captureProcess?.isRunning != true, let cli = resolveOpenBirdCLI() else {
            return captureProcess?.isRunning == true
        }
        // Resuming also clears any pause gate so capture actually records.
        _ = try? setCapturePaused(false)

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

    /// Pull a model via the Ollama CLI. This can take minutes (multi-GB), so it
    /// runs with a generous timeout and reports a concise outcome string.
    func pullModel(_ model: String) async -> (ok: Bool, message: String) {
        guard let ollama = ollamaPath() else {
            return (false, "Ollama CLI not found. Install Ollama from ollama.com.")
        }
        let result = await runAsync(ollama, arguments: ["pull", model], timeout: 1800)
        if result.exitCode == 0 {
            return (true, "Pulled \(model).")
        }
        let detail = result.stderr.isEmpty ? "exit \(result.exitCode)" : result.stderr
        return (false, "Could not pull \(model): \(detail)")
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
    func askChat(_ question: String, timeout: TimeInterval = 90) throws -> ChatResult {
        guard let cli = resolveOpenBirdCLI() else { throw ChatError.cliMissing }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: cli)
        // The question goes via STDIN, never argv — chat text must not be visible
        // to local process inspection (consistent with the capture pipeline).
        process.arguments = ["chat", "--json", "--stdin"]
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
        _ path: String, arguments: [String], timeout: TimeInterval = 4
    ) -> ProcessResult {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = arguments
        // Overlay OPENBIRD_DB_KEY so preflight/data-stats children read the
        // app-owned key and never raise their own Keychain prompt.
        process.environment = childEnvironment()

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
        _ path: String, arguments: [String], timeout: TimeInterval
    ) async -> ProcessResult {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                continuation.resume(returning: Self.run(path, arguments: arguments, timeout: timeout))
            }
        }
    }
}

private struct ProcessResult {
    let exitCode: Int
    let stdout: String
    let stderr: String
}
