// OpenBird capture-helper — frontmost active-window text via the Accessibility API.
//
// Emits one JSON object per capture on stdout, matching the schema the Python
// capture daemon parses (`openbird/capture/daemon.py`):
//
//     {"app": "<bundle id>", "window": "<title>", "url": null,
//      "text": "<AX text>", "ts": <epoch seconds>, "incognito": false}
//
// Privacy by prevention:
//   * Captured text is written ONLY to stdout (the JSON line). It is NEVER placed
//     in stderr, argv, env, or any log line. stderr carries only non-content
//     diagnostics (status words + counts).
//   * AX traversal is bounded by hard depth / node / time limits so a pathological
//     accessibility tree cannot hang or exhaust memory.
//
// TCC: requires Accessibility (and, for some apps, the target app to expose a
// usable AX tree). `AXIsProcessTrustedWithOptions` prompts the user once.

import ApplicationServices
import AppKit
import CaptureHelperCore
import Foundation

// MARK: - Tunable limits (bound traversal so a bad AX tree can't hang us)

private enum Limits {
    static let maxDepth = 40            // max AX subtree depth to descend
    static let maxNodes = 5_000         // max AX nodes to visit per capture
    static let maxTextBytes = 1_000_000 // cap aggregated text (daemon also caps)
    static let deadlineSeconds = 2.0    // wall-clock budget for the AX walk
    // Phase C2: COMBINED per-capture wall budget. AX walk + optional OCR
    // fallback together may never exceed this, so one capture can never occupy
    // the walk queue for ~5s and starve triggers while heartbeats keep
    // liveness looking healthy. OCR gets min(ocrMaxSeconds, remaining).
    static let combinedDeadlineSeconds = 4.0
    static let ocrMaxSeconds = 2.5
}

// MARK: - Non-content diagnostics (stderr only)

/// Write a NON-CONTENT diagnostic line to stderr. Never pass captured text here.
/// Internal (not private): StreamEngine.swift shares it for the same contract.
func diag(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

// MARK: - Trust / TCC

/// Pure accept/reject decision for a stdout-target `stat`/`fstat` result, split
/// out from any I/O so it can be unit-tested directly (given stat fields ->
/// accept/reject) without a real pipe or Accessibility access.
///
/// A stdout target qualifies as a **private pipe** only when ALL hold:
///   * type is a FIFO (anonymous `subprocess.PIPE` / named `mkfifo`) or a socket
///     (a `socketpair` IPC channel) — never a TTY or a redirected regular file,
///     which could persist captured content to scrollback or a log;
///   * it is owned by the current effective user (`st_uid == euid`) — a pipe
///     owned by another user must never receive captured screen content;
///   * its permission bits are safe **for its kind**, decided by `nlink`:
///     - A **nameless** endpoint (`nlink == 0`: an anonymous `pipe(2)` from
///       `subprocess.PIPE`, or a `socketpair`) has NO filesystem path, so no other
///       process can `open()` it by name regardless of its mode bits. Darwin
///       reports such pipes as 0660 (and socketpairs as 0666); those group/other
///       bits are inert because the only handles in existence are the two the
///       kernel handed the daemon and this helper. We require type + owner only.
///     - A **named** endpoint (`nlink > 0`: a `mkfifo` FIFO, or a bound socket)
///       IS openable by path, so its bits matter: we require NO group or other
///       permission bits at all (the same 0600 rigor as the audio `--out`
///       named-FIFO policy), rejecting a same-user 0660 FIFO that group members
///       could otherwise read.
///
/// Using `nlink` (not mode alone) is what lets us safely tolerate the legitimate
/// daemon's 0660 anonymous pipe WITHOUT also accepting a group/world-accessible
/// *named* FIFO — `fstat` mode by itself cannot tell the two apart.
func stdoutPipeStatIsPrivate(mode: mode_t, uid: uid_t, euid: uid_t, nlink: nlink_t) -> Bool {
    let type = mode & S_IFMT
    guard type == S_IFIFO || type == S_IFSOCK else { return false }
    guard uid == euid else { return false }
    if nlink == 0 {
        // Nameless kernel endpoint: unreachable by path; bits are inert.
        return true
    }
    // Named endpoint (path-openable): require owner-only bits (no group/other).
    return (mode & (mode_t(S_IRWXG) | mode_t(S_IRWXO))) == 0
}

/// Whether stdout is a private pipe/socket (opened by the Python daemon), not a
/// TTY or a redirected regular file, owned by us, with no world access. Captured
/// AX text is only ever written here, so we refuse to run unless this holds
/// (fail-closed privacy boundary).
///
/// Uses `fstat` on the already-open fd (not `stat` on a path) so the bits we
/// validate belong to the exact file the helper will write to — closing the
/// TOCTOU window a path-based check would leave open.
func stdoutIsPrivatePipe() -> Bool {
    var st = stat()
    guard fstat(FileHandle.standardOutput.fileDescriptor, &st) == 0 else { return false }
    return stdoutPipeStatIsPrivate(
        mode: st.st_mode, uid: st.st_uid, euid: geteuid(), nlink: st.st_nlink)
}

/// Prompt for (and report) Accessibility trust. Returns whether we are trusted.
func ensureAccessibilityTrust(prompt: Bool) -> Bool {
    let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
    let options = [key: prompt] as CFDictionary
    return AXIsProcessTrustedWithOptions(options)
}

private struct GrantReport: Encodable {
    let accessibility: String
    // Phase C2: the CAPTURE helper owns ScreenCaptureKit now, so it reports
    // the Screen Recording grant (preflight.py routes this capability here;
    // microphone/system-audio stay on the audio helper). Preflight-only —
    // CGPreflightScreenCaptureAccess never prompts.
    let screen_recording: String
}

private func emitGrantReport() {
    let report = GrantReport(
        accessibility: ensureAccessibilityTrust(prompt: false) ? "passed" : "failed",
        screen_recording: CGPreflightScreenCaptureAccess() ? "passed" : "failed"
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.withoutEscapingSlashes]
    guard let data = try? encoder.encode(report) else {
        diag("capture: preflight_encode_failed")
        exit(2)
    }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
}

// MARK: - AX helpers

/// A small budget tracker shared across a single capture traversal.
private final class TraversalBudget {
    var nodesVisited = 0
    var bytesCollected = 0
    var sensitiveSkipped = 0   // nodes skipped by the secure/role policy
    var axTimedOut = false     // a batched AX read hit kAXErrorCannotComplete
    let deadline: Date

    init(deadlineSeconds: Double) {
        self.deadline = Date().addingTimeInterval(deadlineSeconds)
    }

    var expired: Bool {
        axTimedOut
            || nodesVisited >= Limits.maxNodes
            || bytesCollected >= Limits.maxTextBytes
            || Date() >= deadline
    }
}

// MARK: - Sensitive-node policy (privacy by prevention)

/// AX subroles whose content is sensitive and must NEVER be aggregated. A secure
/// text field (password box) is the canonical case; AX exposes it as a text field
/// with the `AXSecureTextField` subrole. We skip the node entirely — neither its
/// value/title nor its subtree is read.
private let sensitiveSubroles: Set<String> = [
    "AXSecureTextField",
]

/// AX roles we never extract text from (containers that may hold credential UI or
/// carry no useful textual content but can be large).
private let skipValueRoles: Set<String> = [
    "AXSecureTextField",
]

/// Whether this role/subrole pair marks a node as sensitive (skip wholesale).
/// Takes the already-fetched strings so the SENSITIVITY DECISION NEVER REQUIRES
/// READING VALUE/TITLE: the two-stage batched fetch in `collectText` reads
/// role/subrole first and only requests content attributes for nodes this
/// function clears (the secure-field invariant, enforced by fetch ORDER).
private func isSensitiveMeta(role: String?, subrole: String?) -> Bool {
    if let subrole, sensitiveSubroles.contains(subrole) { return true }
    if let role, skipValueRoles.contains(role) { return true }
    return false
}

/// Read a string attribute from an AX element, or nil.
private func axString(_ element: AXUIElement, _ attribute: String) -> String? {
    var value: CFTypeRef?
    let err = AXUIElementCopyAttributeValue(element, attribute as CFString, &value)
    guard err == .success else { return nil }
    if let s = value as? String { return s }
    return nil
}

/// Batched attribute read: one IPC round-trip for N attributes (vs N trips).
/// Returns positional values (error placeholders surface as non-castable
/// entries -> nil at the use site), or nil on wholesale failure. Flags
/// `budget.axTimedOut` on kAXErrorCannotComplete — with the 1s messaging
/// timeout set, that means the target app is not responding; the caller
/// accepts the partial text collected so far instead of stalling the capture.
private func axBatch(
    _ element: AXUIElement, _ attributes: [String], budget: TraversalBudget
) -> [AnyObject]? {
    var values: CFArray?
    let err = AXUIElementCopyMultipleAttributeValues(
        element, attributes as CFArray, AXCopyMultipleAttributeOptions(), &values)
    if err == .cannotComplete {
        budget.axTimedOut = true
        return nil
    }
    guard err == .success, let arr = values as? [AnyObject], arr.count == attributes.count
    else { return nil }
    return arr
}

/// Live pause-state check. The pause sidecar is intentionally read on every
/// checkpoint; the helper never caches it for a whole capture.
func capturePaused(_ pauseFile: String?) -> Bool {
    guard let pauseFile else { return false }
    if pauseFile.isEmpty {
        diag("capture: pause_state_unknown")
        return true
    }
    let fm = FileManager.default
    if fm.fileExists(atPath: pauseFile) {
        return true
    }

    let parent = (pauseFile as NSString).deletingLastPathComponent
    var isDir = ObjCBool(false)
    if parent.isEmpty
        || !fm.fileExists(atPath: parent, isDirectory: &isDir)
        || !isDir.boolValue
        || !fm.isReadableFile(atPath: parent) {
        diag("capture: pause_state_unknown")
        return true
    }
    return false
}

func skipIfPaused(_ pauseFile: String?) -> Bool {
    if capturePaused(pauseFile) {
        diag("capture: skipped_paused")
        return true
    }
    return false
}

/// Depth/node/time-bounded AX text aggregation.
///
/// Collects `AXValue`/`AXTitle`/`AXDescription` strings from the element subtree,
/// stopping when any budget limit is hit. Returns the joined text.
private func collectText(
    from element: AXUIElement,
    depth: Int,
    budget: TraversalBudget,
    into parts: inout [String],
    pauseFile: String?
) {
    if skipIfPaused(pauseFile) || depth > Limits.maxDepth || budget.expired { return }
    budget.nodesVisited += 1

    // STAGE 1 — role/subrole ONLY, batched (1 IPC round-trip). The sensitivity
    // decision happens before any content attribute is requested, preserving
    // the secure-field invariant: a password field's value/title is never even
    // asked for, let alone read.
    if skipIfPaused(pauseFile) { return }
    guard let meta = axBatch(
        element,
        [kAXRoleAttribute as String, kAXSubroleAttribute as String],
        budget: budget)
    else {
        // FAIL CLOSED: if role/subrole cannot be read at all, we cannot prove
        // the node is non-sensitive, so its value/title are never requested
        // and its subtree is skipped. (Per-attribute absence inside a
        // successful batch is different — placeholders yield nil strings and
        // the node is judged on what IS present.)
        return
    }
    if budget.expired { return }
    if isSensitiveMeta(role: meta[0] as? String, subrole: meta[1] as? String) {
        budget.sensitiveSkipped += 1
        return
    }

    // STAGE 2 — content + children for proven non-sensitive nodes, batched
    // (value/title/children in one round-trip; was 5 single-attribute trips).
    if skipIfPaused(pauseFile) { return }
    let content = axBatch(
        element,
        [
            kAXValueAttribute as String,
            kAXTitleAttribute as String,
            kAXChildrenAttribute as String,
        ],
        budget: budget)
    if budget.expired { return }

    for candidate in [content?[0] as? String, content?[1] as? String] {
        if let s = candidate, !s.isEmpty {
            parts.append(s)
            budget.bytesCollected += s.utf8.count
            if budget.expired { return }
        }
    }

    if skipIfPaused(pauseFile) { return }
    for child in (content?[2] as? [AXUIElement]) ?? [] {
        if skipIfPaused(pauseFile) || budget.expired { return }
        collectText(
            from: child,
            depth: depth + 1,
            budget: budget,
            into: &parts,
            pauseFile: pauseFile
        )
    }
}

// MARK: - JSON emission

// NOTE: `CaptureEvent` now lives in CaptureHelperCore (CaptureEvent.swift) so
// the raw wire shape — one-shot: no type/trigger/ocr keys; stream AX:
// type+trigger, no ocr; OCR frame: ocr:true — is pinned by `swift test`.

/// Emit a capture event as a single JSON line on STDOUT (the only content sink).
func emit(_ event: CaptureEvent) {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.withoutEscapingSlashes]
    guard let data = try? encoder.encode(event) else {
        diag("capture: encode_failed")
        return
    }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
}

// MARK: - Capture

/// Match a bundle id against one allow/block entry, mirroring the Python
/// `redact._bundle_matches`: exact id (default), `glob:<pattern>`, or `re:<regex>`.
private func bundleMatches(_ app: String, _ entry: String) -> Bool {
    let appL = app.lowercased()
    if entry.hasPrefix("glob:") {
        let pat = String(entry.dropFirst(5)).trimmingCharacters(in: .whitespaces).lowercased()
        return globMatch(pat, appL)
    }
    if entry.hasPrefix("re:") {
        let pat = String(entry.dropFirst(3)).trimmingCharacters(in: .whitespaces)
        guard let re = try? NSRegularExpression(pattern: "^(?:\(pat))$", options: [.caseInsensitive])
        else { return false }
        let range = NSRange(app.startIndex..., in: app)
        return re.firstMatch(in: app, options: [], range: range) != nil
    }
    return appL == entry.lowercased()
}

/// Minimal `*`/`?` glob match (case-insensitive inputs already lowercased).
private func globMatch(_ pattern: String, _ text: String) -> Bool {
    let escaped = pattern.map { ch -> String in
        switch ch {
        case "*": return ".*"
        case "?": return "."
        default: return NSRegularExpression.escapedPattern(for: String(ch))
        }
    }.joined()
    guard let re = try? NSRegularExpression(pattern: "^\(escaped)$") else { return false }
    return re.firstMatch(in: text, options: [], range: NSRange(text.startIndex..., in: text)) != nil
}

private func anyMatch(_ bundleId: String, _ entries: Set<String>) -> Bool {
    for e in entries where bundleMatches(bundleId, e) { return true }
    return false
}

// OpenBird's own bundle-id root (app + bundled helpers). MUST mirror the
// Python `redact._SELF_BUNDLE_ROOT` — a two-copy parity test
// (tests/unit/test_capture_self_exclusion.py) parses THIS literal and fails
// if the two drift. Matching is EXACT or dotted-child, NEVER substring: the
// literal "openbird" appears in legitimate Chrome/terminal rows
// (github.com/…/openbird), so a substring match would delete real dev signal.
private let selfBundleRoot = "ai.openbird.openbird"

/// True iff `bundleId` is OpenBird's own process (the app or a bundled
/// helper): exact root or dotted child, case-insensitively. Mirrors
/// `redact._is_self_capture`. Runs BEFORE the allowlist gate / any AX element
/// creation / SCK, so OpenBird's own UI is never read — with OCR in play the
/// old Python-only backstop would otherwise permit a SCREENSHOT of our own UI.
func isSelfCapture(_ bundleId: String?) -> Bool {
    guard let id = bundleId?.lowercased() else { return false }
    return id == selfBundleRoot || id.hasPrefix(selfBundleRoot + ".")
}

// MARK: - OCR fallback runtime (Phase C2, stream mode only)

/// Everything one OCR fallback decision needs, constructed ONCE by the
/// StreamEngine when `--ocr-apps` is non-empty. One-shot mode never builds
/// one (`ocr: nil` keeps that path byte-identical): the per-app throttle
/// needs process-lifetime state, and a one-shot helper is a fresh process
/// every spawn — helper-local throttle state cannot exist there (the same
/// stream-only precedent as MicMonitor/AFK).
///
/// Threading: `gate`/`lastTccState` are mutated on the walk queue ONLY
/// (captures run one-at-a-time there). The mic bit is a LOCK-BACKED LIVE READ
/// (`micHotNow`), not a dispatch-time snapshot: a call can start during the
/// multi-second AX walk, and the OCR decision must see the mic state AT
/// DECISION TIME — a stale snapshot could run SCK/Vision GPU work into a live
/// call (the exact contention the mic gate exists to prevent). `emitSystem`
/// is the locked StreamEmitter (thread-safe).
final class OcrRuntime {
    /// The opt-in set (`--ocr-apps`), matched via the same `bundleMatches`
    /// grammar as the allowlist. Opted-in ⊆ allowlisted holds by construction:
    /// this branch is only reachable after `contentAllowed` returned true.
    let apps: Set<String>
    /// Pure gate/throttle state machine (CaptureHelperCore). Walk-queue-confined.
    var gate: OcrGate
    /// Cancellable async-in-sync bridge over the HAL (CaptureHelperCore).
    let bridge: OcrBridge
    /// `CGPreflightScreenCaptureAccess` (injected closure). NEVER prompts.
    private let tccPreflight: () -> Bool
    /// Locked stream emitter for `ocr_available`/`ocr_unavailable` system events.
    private let emitSystem: (String) -> Void
    /// Lock-backed live mic state (see threading note above). Written on the
    /// main thread on every MicMonitor flip; read on the walk queue at OCR
    /// decision time.
    private let micLock = NSLock()
    private var _micHot = false
    var micHot: Bool {
        get {
            micLock.lock()
            defer { micLock.unlock() }
            return _micHot
        }
        set {
            micLock.lock()
            defer { micLock.unlock() }
            _micHot = newValue
        }
    }
    /// Last TCC state we told the daemon about (walk-queue-confined after the
    /// startup emission; used to re-emit only on an observed flip).
    private var lastTccState: Bool?

    init(
        apps: Set<String>,
        minInterval: Double,
        hal: OcrHAL,
        tccPreflight: @escaping () -> Bool,
        emitSystem: @escaping (String) -> Void
    ) {
        self.apps = apps
        self.gate = OcrGate(minInterval: minInterval)
        self.bridge = OcrBridge(hal: hal)
        self.tccPreflight = tccPreflight
        self.emitSystem = emitSystem
    }

    /// Preflight the Screen Recording grant, emitting `ocr_available` /
    /// `ocr_unavailable` on the FIRST reading (startup) and on every observed
    /// flip afterwards (grant revoked/restored mid-run) — metadata only.
    func tccGranted() -> Bool {
        let granted = tccPreflight()
        if granted != lastTccState {
            lastTccState = granted
            emitSystem(granted ? "ocr_available" : "ocr_unavailable")
        }
        return granted
    }
}

// SINGLE SOURCE OF TRUTH: bundle-id substrings whose content is never
// captured even if (mis)allowlisted (password managers / vaults). This baked
// constant MUST mirror `redact._DANGEROUS_BUNDLE_SUBSTRINGS` (Python) and the
// committed canonical list `dangerous_apps.json`. The parity unit test in
// tests/unit/test_capture.py parses THIS literal, the JSON, and the Python
// tuple and fails the build if any of the three drift — so editing the list
// means editing all three together.
//
// Why a baked constant and NOT a runtime `Bundle.module` JSON read: the shipped
// helper is a bare executable copied into `OpenBird.app` (see
// script/build_and_run.sh) WITHOUT SwiftPM's generated resource bundle.
// `Bundle.module` `fatalError`s when that bundle is absent, which would crash
// the helper before any fallback could run — defeating the fail-closed intent
// and breaking capture for every allowlisted app. Baking the list in keeps the
// backstop always complete and dependency-free at runtime; the JSON exists as
// the canonical drift-detection source for the parity test (and for any future
// build-time codegen), not as a runtime input.
private let dangerousBundleSubstrings: [String] = [
    "1password", "onepassword", "lastpass", "bitwarden", "dashlane",
    "keepass", "keychain", "keychainaccess", "keeper", "nordpass",
    "enpass", "protonpass",
].map { $0.lowercased() }

// Window-title markers that signal a private/incognito window.
private let incognitoTitleMarkers = [
    "incognito", "private browsing", "inprivate", "private window",
]

private func isDangerousBundle(_ bundleId: String?) -> Bool {
    guard let id = bundleId?.lowercased() else { return false }
    return dangerousBundleSubstrings.contains { id.contains($0) }
}

private func isIncognitoTitle(_ title: String?) -> Bool {
    guard let t = title?.lowercased() else { return false }
    return incognitoTitleMarkers.contains { t.contains($0) }
}

/// Decide whether an app's CONTENT may be read, per the **allowlist-ONLY** policy
/// (mirrors `redact.decide`): nothing is captured unless its bundle id matches the
/// allowlist; the blocklist can only subtract. An EMPTY allowlist captures nothing
/// (opt-in first run). Runs BEFORE any AX traversal so disallowed app text is never
/// read or transmitted — enforcement at the boundary, not after IPC.
private func contentAllowed(bundleId: String?, allow: Set<String>, block: Set<String>) -> Bool {
    guard let id = bundleId else { return false }   // unknown app -> deny
    if anyMatch(id, block) { return false }         // blocklist subtracts
    if allow.isEmpty { return false }               // allowlist-only: empty = capture nothing
    return anyMatch(id, allow)
}

// MARK: - Browser URL via Apple Events (opt-in)

/// Map a browser bundle id to its AppleScript application name, for the
/// Chromium family ONLY. Chromium exposes a per-window `mode` property, which is
/// what lets us reliably skip private windows (see :func:`activeTabURL`). Safari
/// is deliberately excluded for now: it has NO scriptable private-browsing flag,
/// so we cannot prove a window is non-private and must not risk capturing a
/// private URL. (Safari support is a follow-up that needs the Window-menu check.)
private func browserScriptTarget(_ bundleId: String?) -> String? {
    switch bundleId {
    case "com.google.Chrome", "com.google.Chrome.canary": return "Google Chrome"
    case "com.microsoft.edgemac": return "Microsoft Edge"
    case "com.brave.Browser": return "Brave Browser"
    case "company.thebrowser.Browser": return "Arc"
    case "com.vivaldi.Vivaldi": return "Vivaldi"
    default: return nil
    }
}

/// The active tab's privacy mode + committed URL, read together in one Apple
/// Event so the SAME reliable `mode` signal can gate both URL and text capture.
private struct BrowserTab {
    /// True ONLY when the window's `mode` is definitively `"incognito"`. A browser
    /// whose `mode` can't be read (older Arc) is NOT marked private here — so text
    /// capture is never wrongly suppressed; the URL is still withheld (see `url`).
    let confirmedPrivate: Bool
    /// The committed URL, present ONLY when `mode == "normal"` (fail-closed).
    let url: String?
}

/// Probe a Chromium browser's front window for its privacy mode and active-tab
/// URL via Apple Events.
///
/// The script returns a sentinel: `"PRIVATE"` (incognito), `"URL:<url>"` (a
/// confirmed-normal window), or `""` (mode unreadable / no window / denied). This
/// lets the caller (a) SKIP TEXT for a confirmed-private window — closing the gap
/// where a private Chromium window whose title lacks an "incognito" marker would
/// otherwise leak its text — and (b) capture the URL only when normal. Never logs
/// the URL; the Python side scrubs it before storage.
private func browserTabInfo(appName: String) -> BrowserTab {
    let source = """
    tell application "\(appName)"
        if (count of windows) is 0 then return ""
        set theWindow to front window
        set windowMode to ""
        try
            set windowMode to (mode of theWindow) as text
        end try
        if windowMode is "incognito" then return "PRIVATE"
        if windowMode is not "normal" then return ""
        return "URL:" & (URL of active tab of theWindow)
    end tell
    """
    var errorInfo: NSDictionary?
    guard let script = NSAppleScript(source: source) else {
        return BrowserTab(confirmedPrivate: false, url: nil)
    }
    let result = script.executeAndReturnError(&errorInfo)
    if errorInfo != nil {
        // Denied automation (-1743), not scriptable, or no window: no URL, and we
        // do NOT claim privacy (don't suppress text on a transient script error).
        diag("capture: url_unavailable")
        return BrowserTab(confirmedPrivate: false, url: nil)
    }
    let s = result.stringValue ?? ""
    if s == "PRIVATE" { return BrowserTab(confirmedPrivate: true, url: nil) }
    if s.hasPrefix("URL:") {
        let u = String(s.dropFirst(4))
        return BrowserTab(confirmedPrivate: false, url: u.isEmpty ? nil : u)
    }
    return BrowserTab(confirmedPrivate: false, url: nil)
}

/// Capture the frontmost app's active-window text once and emit it.
///
/// Stream mode passes `trigger` (stamped onto the event), `emitter` (the
/// locked POSIX-write sink), and optionally `ocr` (the Phase C2 fallback
/// runtime); one-shot mode uses the defaults — `ocr: nil` in particular keeps
/// that path byte-identical with older daemons.
func captureFrontmost(
    allow: Set<String>, block: Set<String>, pauseFile: String?, captureUrls: Bool,
    trigger: String? = nil, emitter: ((CaptureEvent) -> Void)? = nil,
    ocr: OcrRuntime? = nil
) {
    let send = emitter ?? emit
    // Start of the COMBINED AX+OCR wall budget (monotonic).
    let captureStart = DispatchTime.now()
    if skipIfPaused(pauseFile) { return }

    guard let frontApp = NSWorkspace.shared.frontmostApplication else {
        diag("capture: no_frontmost_app")
        return
    }
    let bundleId = frontApp.bundleIdentifier
    let pid = frontApp.processIdentifier
    if skipIfPaused(pauseFile) { return }

    // Self-capture gate FIRST (before the allowlist gate, any AX element
    // creation, and SCK): OpenBird must never read — or, with OCR in play,
    // screenshot — its own UI, even if (mis)allowlisted. Mirrors the Python
    // `redact._is_self_capture` backstop, but enforced AT THE SOURCE so the
    // text never crosses IPC. Emits the metadata-only frame like the other
    // early returns so the daemon still sees the focus change.
    if isSelfCapture(bundleId) {
        diag("capture: skipped_self_capture")
        if skipIfPaused(pauseFile) { return }
        send(CaptureEvent(
            app: bundleId, window: nil, url: nil, text: "",
            ts: Date().timeIntervalSince1970, incognito: false, trigger: trigger))
        return
    }

    // Allowlist-first gate BEFORE reading any AX text. A disallowed app emits
    // metadata only (empty text) so the daemon sees the focus change but no
    // captured content ever crosses the IPC boundary.
    if !contentAllowed(bundleId: bundleId, allow: allow, block: block) {
        diag("capture: skipped_not_allowlisted")
        if skipIfPaused(pauseFile) { return }
        send(CaptureEvent(
            app: bundleId, window: nil, url: nil, text: "",
            ts: Date().timeIntervalSince1970, incognito: false, trigger: trigger))
        return
    }

    // Dangerous-app backstop runs BEFORE any AX access (no AX element is even
    // created for a password manager): mirrors `redact.decide`'s dangerous gate
    // and ensures a vault's title/text is never read, even if (mis)allowlisted.
    if isDangerousBundle(bundleId) {
        diag("capture: skipped_dangerous_app")
        if skipIfPaused(pauseFile) { return }
        send(CaptureEvent(app: bundleId, window: nil, url: nil, text: "",
                          ts: Date().timeIntervalSince1970, incognito: false,
                          trigger: trigger))
        return
    }

    if skipIfPaused(pauseFile) { return }
    let appElement = AXUIElementCreateApplication(pid)
    // Bound every AX message to this app at 1s (kAXErrorCannotComplete on
    // breach -> axBatch flags the budget and the walk accepts partial text).
    AXUIElementSetMessagingTimeout(appElement, 1.0)
    if skipIfPaused(pauseFile) { return }

    // Resolve the focused window (fall back to the first window).
    var focused: CFTypeRef?
    var windowElement: AXUIElement?
    if AXUIElementCopyAttributeValue(
        appElement, kAXFocusedWindowAttribute as CFString, &focused) == .success,
        let w = focused, CFGetTypeID(w) == AXUIElementGetTypeID() {
        windowElement = (w as! AXUIElement)
    } else {
        var windowsValue: CFTypeRef?
        if AXUIElementCopyAttributeValue(
            appElement, kAXWindowsAttribute as CFString, &windowsValue) == .success,
            let windows = windowsValue as? [AXUIElement], let first = windows.first {
            windowElement = first
        }
    }

    guard let window = windowElement else {
        diag("capture: no_window pid=\(pid)")
        return
    }

    if skipIfPaused(pauseFile) { return }
    let windowTitle = axString(window, kAXTitleAttribute as String)
    if skipIfPaused(pauseFile) { return }

    // Incognito/private windows: emit metadata only, with incognito=true and no
    // (potentially sensitive) window title, before any text traversal.
    if isIncognitoTitle(windowTitle) {
        diag("capture: skipped_incognito")
        if skipIfPaused(pauseFile) { return }
        send(CaptureEvent(app: bundleId, window: nil, url: nil, text: "",
                          ts: Date().timeIntervalSince1970, incognito: true,
                          trigger: trigger))
        return
    }

    // Browser URL probe runs BEFORE text traversal (opt-in only). The Apple Events
    // `mode` is a far more reliable private-window signal than the title heuristic
    // above, so a window it confirms private skips TEXT too — closing the leak
    // where a private Chromium window whose title lacks an "incognito" marker would
    // otherwise have its text captured. A non-private window yields the URL to emit.
    var capturedURL: String? = nil
    if captureUrls, let appName = browserScriptTarget(bundleId) {
        if skipIfPaused(pauseFile) { return }
        let tab = browserTabInfo(appName: appName)
        if tab.confirmedPrivate {
            diag("capture: skipped_incognito_mode")
            if skipIfPaused(pauseFile) { return }
            send(CaptureEvent(app: bundleId, window: nil, url: nil, text: "",
                              ts: Date().timeIntervalSince1970, incognito: true,
                              trigger: trigger))
            return
        }
        capturedURL = tab.url
    }

    let budget = TraversalBudget(deadlineSeconds: Limits.deadlineSeconds)
    var parts: [String] = []
    if skipIfPaused(pauseFile) { return }
    collectText(from: window, depth: 0, budget: budget, into: &parts, pauseFile: pauseFile)
    if skipIfPaused(pauseFile) { return }

    var text = parts.joined(separator: "\n")
    if text.utf8.count > Limits.maxTextBytes {
        let prefix = text.utf8.prefix(Limits.maxTextBytes)
        text = String(decoding: Array(prefix), as: UTF8.self)
    }

    // OCR fallback (Phase C2, stream-mode only — `ocr` is nil elsewhere).
    // This branch sits INSIDE the same gated tail as the AX emit: every
    // earlier privacy return (self-capture, not-allowlisted, dangerous,
    // incognito title, confirmed-private mode, paused) already exited, so the
    // gate ordering is inherited, not re-proven here.
    var ocrUsed = false
    if let runtime = ocr,
       text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
       let id = bundleId {
        let now = Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000_000.0
        let decision = runtime.gate.decide(
            bundleId: id,
            axTextEmpty: true,
            optedIn: anyMatch(id, runtime.apps),
            tccGranted: runtime.tccGranted(),
            micHot: runtime.micHot,
            now: now)
        switch decision {
        case .skip(let reason):
            diag("ocr: skipped_\(reason)")
        case .run:
            // Live pause re-check immediately before the HAL call: a pause
            // that landed during the AX walk must also stop the screenshot.
            if skipIfPaused(pauseFile) { return }
            // COMBINED budget: AX + OCR <= combinedDeadlineSeconds total.
            // The deadline races the CANCELLABLE pre-image (SCK) phase only;
            // once pixels exist the bridge waits out the synchronous Vision
            // pass (bounded, sub-second typical) so the image scope closes
            // before this capture returns — worst case documented in
            // OcrBridge.swift.
            let elapsed = Double(
                DispatchTime.now().uptimeNanoseconds - captureStart.uptimeNanoseconds
            ) / 1_000_000_000.0
            let ocrBudget = min(
                Limits.ocrMaxSeconds, Limits.combinedDeadlineSeconds - elapsed)
            let started = DispatchTime.now()
            switch runtime.bridge.recognize(pid: pid, axTitle: windowTitle, timeout: ocrBudget) {
            case .text(let recognized):
                text = recognized
                if text.utf8.count > Limits.maxTextBytes {
                    let prefix = text.utf8.prefix(Limits.maxTextBytes)
                    text = String(decoding: Array(prefix), as: UTF8.self)
                }
                ocrUsed = true
                let ms = Int(
                    Double(DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds)
                        / 1_000_000.0)
                // Non-content diag: byte count + duration only.
                diag("ocr: ok bytes=\(text.utf8.count) ms=\(ms)")
            case .skipped(let reason):
                diag("ocr: skipped_\(reason)")
            case .timeout:
                // The acquire (pre-image) phase timed out: the task was
                // cancelled and a late-acquired image is dropped UNREAD by
                // the bridge's closed generation box. Once pixels exist the
                // bridge never times out — it waits for the synchronous
                // Vision pass so the image cannot outlive this capture (see
                // OcrBridge.swift). The throttle slot was consumed at
                // decision time, so this cannot retry-storm.
                diag("ocr: skipped_ocr_timeout")
            }
        }
    }

    // Emit only NON-content metadata to stderr. `capturedURL` was probed above
    // (opt-in only); the diag carries a presence bit (0/1), never the URL string.
    diag("capture: ok nodes=\(budget.nodesVisited) bytes=\(text.utf8.count) "
        + "secure_skipped=\(budget.sensitiveSkipped) url=\(capturedURL != nil ? 1 : 0)"
        + (budget.axTimedOut ? " partial=ax_timeout" : "")
        + (ocrUsed ? " ocr=1" : ""))

    let event = CaptureEvent(
        app: bundleId,
        window: windowTitle,
        url: capturedURL,
        text: text,
        ts: Date().timeIntervalSince1970,
        incognito: false,
        trigger: trigger,
        ocr: ocrUsed ? true : nil
    )
    if skipIfPaused(pauseFile) { return }
    send(event)
}

// MARK: - Entry point

/// Parse a repeatable/comma-separated list flag (e.g. `--allow a,b --allow c`).
func listArg(_ args: [String], _ name: String) -> Set<String> {
    var out: Set<String> = []
    var i = 0
    while i < args.count {
        if args[i] == name, i + 1 < args.count {
            for piece in args[i + 1].split(separator: ",") {
                let s = piece.trimmingCharacters(in: .whitespaces)
                if !s.isEmpty { out.insert(s) }
            }
            i += 1
        }
        i += 1
    }
    return out
}

/// Parse a single value flag (e.g. `--pause-file /path/to/capture.paused`).
func valueArg(_ args: [String], _ name: String) -> String? {
    var i = 0
    while i < args.count {
        if args[i] == name, i + 1 < args.count, !args[i + 1].hasPrefix("--") {
            return args[i + 1]
        }
        if args[i] == name { return "" }
        i += 1
    }
    return nil
}

private func run() {
    let args = CommandLine.arguments
    if args.contains("--preflight-grants") {
        emitGrantReport()
        return
    }

    // Trigger the Accessibility authorization prompt for THIS helper binary, so
    // macOS registers it in System Settings > Privacy > Accessibility (a binary
    // never appears there until it has requested the grant once). The system
    // prompt's "Open System Settings" button lands the user on the right pane
    // with this helper already listed — the prerequisite for granting capture.
    if args.contains("--request-accessibility") {
        _ = ensureAccessibilityTrust(prompt: true)
        return
    }

    let noPrompt = args.contains("--no-prompt")
    // Allowlist-first content policy, enforced before any AX text is read.
    let allow = listArg(args, "--allow")
    let block = listArg(args, "--block")
    let pauseFile = valueArg(args, "--pause-file")
    // Opt-in: capture the active browser tab's URL via Apple Events (off unless the
    // daemon passes --capture-urls, which it does only when the user enables it).
    let captureUrls = args.contains("--capture-urls")

    // Persistent event-driven mode. Timing knobs arrive pre-clamped from the
    // Python config; SchedulerConfig defensively re-clamps. Stream mode never
    // returns (RunLoop.main.run() / exit paths inside).
    if args.contains("--stream") {
        func doubleArg(_ name: String) -> Double? {
            valueArg(args, name).flatMap(Double.init)
        }
        let config = SchedulerConfig(
            minGap: doubleArg("--min-gap") ?? 1.0,
            idleTick: doubleArg("--idle-tick") ?? 5.0,
            forceCeiling: doubleArg("--ceiling") ?? 60.0,
            afkThreshold: doubleArg("--afk-threshold") ?? 150.0
        )
        // OCR fallback opt-in (Phase C2): STREAM MODE ONLY — the per-app
        // throttle needs process-lifetime state a fresh-per-spawn one-shot
        // helper cannot hold. The daemon passes --ocr-apps only when the user
        // opted apps in; one-shot spawns never receive (and would ignore) it.
        StreamEngine(
            allow: allow, block: block, pauseFile: pauseFile,
            captureUrls: captureUrls, config: config,
            ocrApps: listArg(args, "--ocr-apps"),
            ocrMinInterval: doubleArg("--ocr-min-interval") ?? 30.0
        ).run(noPrompt: noPrompt)
    }

    if skipIfPaused(pauseFile) { return }

    // Privacy boundary: captured AX text must only flow to the daemon's private
    // pipe. Refuse if stdout is a terminal or a redirected file, so content can
    // never land in scrollback or a log file.
    if !stdoutIsPrivatePipe() {
        diag("capture: stdout is not a private pipe (launch via the daemon); refusing (fail-closed)")
        exit(3)
    }

    let trusted = ensureAccessibilityTrust(prompt: !noPrompt)
    if !trusted {
        // Honest failure: without Accessibility we cannot read AX text. Report a
        // non-content diagnostic and exit non-zero so the daemon can surface it.
        diag("capture: accessibility_not_trusted")
        exit(2)
    }

    captureFrontmost(allow: allow, block: block, pauseFile: pauseFile, captureUrls: captureUrls)
}

run()
