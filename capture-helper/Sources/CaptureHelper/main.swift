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
import Foundation

// MARK: - Tunable limits (bound traversal so a bad AX tree can't hang us)

private enum Limits {
    static let maxDepth = 40            // max AX subtree depth to descend
    static let maxNodes = 5_000         // max AX nodes to visit per capture
    static let maxTextBytes = 1_000_000 // cap aggregated text (daemon also caps)
    static let deadlineSeconds = 2.0    // wall-clock budget per capture
}

// MARK: - Non-content diagnostics (stderr only)

/// Write a NON-CONTENT diagnostic line to stderr. Never pass captured text here.
private func diag(_ message: String) {
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
private func stdoutIsPrivatePipe() -> Bool {
    var st = stat()
    guard fstat(FileHandle.standardOutput.fileDescriptor, &st) == 0 else { return false }
    return stdoutPipeStatIsPrivate(
        mode: st.st_mode, uid: st.st_uid, euid: geteuid(), nlink: st.st_nlink)
}

/// Prompt for (and report) Accessibility trust. Returns whether we are trusted.
private func ensureAccessibilityTrust(prompt: Bool) -> Bool {
    let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
    let options = [key: prompt] as CFDictionary
    return AXIsProcessTrustedWithOptions(options)
}

private struct GrantReport: Encodable {
    let accessibility: String
}

private func emitGrantReport() {
    let report = GrantReport(
        accessibility: ensureAccessibilityTrust(prompt: false) ? "passed" : "failed"
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
    let deadline: Date

    init(deadlineSeconds: Double) {
        self.deadline = Date().addingTimeInterval(deadlineSeconds)
    }

    var expired: Bool {
        nodesVisited >= Limits.maxNodes
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

/// Whether this element's role/subrole marks it as sensitive (skip wholesale).
private func isSensitive(_ element: AXUIElement) -> Bool {
    if let subrole = axString(element, kAXSubroleAttribute as String),
       sensitiveSubroles.contains(subrole) {
        return true
    }
    if let role = axString(element, kAXRoleAttribute as String),
       skipValueRoles.contains(role) {
        return true
    }
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

/// Read the children of an AX element, or an empty array.
private func axChildren(_ element: AXUIElement) -> [AXUIElement] {
    var value: CFTypeRef?
    let err = AXUIElementCopyAttributeValue(
        element, kAXChildrenAttribute as CFString, &value)
    guard err == .success, let children = value as? [AXUIElement] else { return [] }
    return children
}

/// Live pause-state check. The pause sidecar is intentionally read on every
/// checkpoint; the helper never caches it for a whole capture.
private func capturePaused(_ pauseFile: String?) -> Bool {
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

private func skipIfPaused(_ pauseFile: String?) -> Bool {
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

    // Role/subrole policy BEFORE reading any value: never aggregate secure text
    // fields (passwords), and never descend into them.
    if skipIfPaused(pauseFile) { return }
    if isSensitive(element) {
        budget.sensitiveSkipped += 1
        return
    }

    for attr in [kAXValueAttribute as String, kAXTitleAttribute as String] {
        if skipIfPaused(pauseFile) { return }
        if let s = axString(element, attr), !s.isEmpty {
            parts.append(s)
            budget.bytesCollected += s.utf8.count
            if budget.expired { return }
        }
    }

    if skipIfPaused(pauseFile) { return }
    for child in axChildren(element) {
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

/// One capture record. Encoded to a single JSON line on stdout.
private struct CaptureEvent: Encodable {
    let app: String?
    let window: String?
    let url: String?
    let text: String
    let ts: Double
    let incognito: Bool
}

/// Emit a capture event as a single JSON line on STDOUT (the only content sink).
private func emit(_ event: CaptureEvent) {
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

/// The active tab's committed URL via Apple Events, or nil.
///
/// FAIL-CLOSED on privacy: the script returns a URL ONLY when the window's `mode`
/// is definitively `"normal"`. A private/incognito window (`mode == "incognito"`)
/// OR a browser where `mode` can't be read (unsupported / older Arc) both yield
/// the empty string → no URL. Also nil for: no front window, a non-Chromium app,
/// or absent Automation consent (AppleScript error -1743). The URL is untrusted:
/// the Python side runs `redact.scrub_url` (drops query/fragment + tokens) before
/// storage, and the helper never logs the URL itself — only a presence bit.
private func activeTabURL(bundleId: String?) -> String? {
    guard let appName = browserScriptTarget(bundleId) else { return nil }
    let source = """
    tell application "\(appName)"
        if (count of windows) is 0 then return ""
        set theWindow to front window
        set windowMode to ""
        try
            set windowMode to (mode of theWindow) as text
        end try
        if windowMode is not "normal" then return ""
        return URL of active tab of theWindow
    end tell
    """
    var errorInfo: NSDictionary?
    guard let script = NSAppleScript(source: source) else { return nil }
    let result = script.executeAndReturnError(&errorInfo)
    if errorInfo != nil {
        // Denied automation (-1743), app not scriptable, or no window: skip the
        // URL silently. Reason code only — never the URL or the error detail.
        diag("capture: url_unavailable")
        return nil
    }
    guard let s = result.stringValue, !s.isEmpty else { return nil }
    return s
}

/// Capture the frontmost app's active-window text once and emit it.
private func captureFrontmost(
    allow: Set<String>, block: Set<String>, pauseFile: String?, captureUrls: Bool
) {
    if skipIfPaused(pauseFile) { return }

    guard let frontApp = NSWorkspace.shared.frontmostApplication else {
        diag("capture: no_frontmost_app")
        return
    }
    let bundleId = frontApp.bundleIdentifier
    let pid = frontApp.processIdentifier
    if skipIfPaused(pauseFile) { return }

    // Allowlist-first gate BEFORE reading any AX text. A disallowed app emits
    // metadata only (empty text) so the daemon sees the focus change but no
    // captured content ever crosses the IPC boundary.
    if !contentAllowed(bundleId: bundleId, allow: allow, block: block) {
        diag("capture: skipped_not_allowlisted")
        if skipIfPaused(pauseFile) { return }
        emit(CaptureEvent(
            app: bundleId, window: nil, url: nil, text: "",
            ts: Date().timeIntervalSince1970, incognito: false))
        return
    }

    // Dangerous-app backstop runs BEFORE any AX access (no AX element is even
    // created for a password manager): mirrors `redact.decide`'s dangerous gate
    // and ensures a vault's title/text is never read, even if (mis)allowlisted.
    if isDangerousBundle(bundleId) {
        diag("capture: skipped_dangerous_app")
        if skipIfPaused(pauseFile) { return }
        emit(CaptureEvent(app: bundleId, window: nil, url: nil, text: "",
                          ts: Date().timeIntervalSince1970, incognito: false))
        return
    }

    if skipIfPaused(pauseFile) { return }
    let appElement = AXUIElementCreateApplication(pid)
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
        emit(CaptureEvent(app: bundleId, window: nil, url: nil, text: "",
                          ts: Date().timeIntervalSince1970, incognito: true))
        return
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

    // Active-tab URL via Apple Events, only when opted in (--capture-urls). Skips
    // incognito/private and denied-automation; nil for non-browsers. Python scrubs
    // it; the diag carries a presence bit (0/1), never the URL string.
    var url: String? = nil
    if captureUrls, !skipIfPaused(pauseFile) {
        url = activeTabURL(bundleId: bundleId)
    }

    // Emit only NON-content metadata to stderr (node/byte counts + redactions).
    diag("capture: ok nodes=\(budget.nodesVisited) bytes=\(text.utf8.count) "
        + "secure_skipped=\(budget.sensitiveSkipped) url=\(url != nil ? 1 : 0)")

    let event = CaptureEvent(
        app: bundleId,
        window: windowTitle,
        url: url,
        text: text,
        ts: Date().timeIntervalSince1970,
        incognito: false
    )
    if skipIfPaused(pauseFile) { return }
    emit(event)
}

// MARK: - Entry point

/// Parse a repeatable/comma-separated list flag (e.g. `--allow a,b --allow c`).
private func listArg(_ args: [String], _ name: String) -> Set<String> {
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
private func valueArg(_ args: [String], _ name: String) -> String? {
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
