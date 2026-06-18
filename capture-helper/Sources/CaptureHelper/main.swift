// OpenBird capture-helper — frontmost active-window text via the Accessibility API.
//
// Emits one JSON object per capture on stdout, matching the schema the Python
// capture daemon parses (`openbird/capture/daemon.py`):
//
//     {"app": "<bundle id>", "window": "<title>", "url": "<url|null>",
//      "text": "<AX text>", "ts": <epoch seconds>, "incognito": false}
//
// Privacy by prevention (PLAN.md "Whole-data-path privacy" / subprocess hygiene):
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

/// Whether stdout is a private pipe/socket (opened by the Python daemon), not a
/// TTY or a redirected regular file. Captured AX text is only ever written here,
/// so we refuse to run unless this holds (fail-closed privacy boundary).
private func stdoutIsPrivatePipe() -> Bool {
    var st = stat()
    guard fstat(FileHandle.standardOutput.fileDescriptor, &st) == 0 else { return false }
    let mode = st.st_mode & S_IFMT
    return mode == S_IFIFO || mode == S_IFSOCK
}

/// Prompt for (and report) Accessibility trust. Returns whether we are trusted.
private func ensureAccessibilityTrust(prompt: Bool) -> Bool {
    let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
    let options = [key: prompt] as CFDictionary
    return AXIsProcessTrustedWithOptions(options)
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

/// Depth/node/time-bounded AX text aggregation.
///
/// Collects `AXValue`/`AXTitle`/`AXDescription` strings from the element subtree,
/// stopping when any budget limit is hit. Returns the joined text.
private func collectText(
    from element: AXUIElement,
    depth: Int,
    budget: TraversalBudget,
    into parts: inout [String]
) {
    if depth > Limits.maxDepth || budget.expired { return }
    budget.nodesVisited += 1

    // Role/subrole policy BEFORE reading any value: never aggregate secure text
    // fields (passwords), and never descend into them.
    if isSensitive(element) {
        budget.sensitiveSkipped += 1
        return
    }

    for attr in [kAXValueAttribute as String, kAXTitleAttribute as String] {
        if let s = axString(element, attr), !s.isEmpty {
            parts.append(s)
            budget.bytesCollected += s.utf8.count
            if budget.expired { return }
        }
    }

    for child in axChildren(element) {
        if budget.expired { return }
        collectText(from: child, depth: depth + 1, budget: budget, into: &parts)
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

// Hardcoded backstop FALLBACK: bundle-id substrings whose content is never
// captured even if (mis)allowlisted. This MUST mirror
// `redact._DANGEROUS_BUNDLE_SUBSTRINGS` (Python) and the committed
// `dangerous_apps.json` resource — the parity unit test in
// tests/unit/test_capture.py fails if these three drift. Edit ALL THREE
// together. This baked constant is the FALLBACK; the effective list is this
// fallback UNION the JSON resource, so the backstop can only ever grow, never
// shrink to empty/partial (fail-closed). See `dangerousBundleSubstrings`.
private let dangerousBundleSubstringsFallback: [String] = [
    "1password", "onepassword", "lastpass", "bitwarden", "dashlane",
    "keepass", "keychain", "keychainaccess", "keeper", "nordpass",
    "enpass", "protonpass",
]

/// The effective dangerous-bundle substrings: the baked fallback UNION the
/// committed `dangerous_apps.json` resource (loaded via `Bundle.module`). On a
/// missing, unreadable, malformed, or empty JSON resource we fall back to the
/// full baked list — the backstop is fail-closed and never empty or partial.
private let dangerousBundleSubstrings: [String] = {
    var set = Set(dangerousBundleSubstringsFallback.map { $0.lowercased() })
    if let url = Bundle.module.url(forResource: "dangerous_apps", withExtension: "json"),
       let data = try? Data(contentsOf: url),
       let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
       let list = obj["dangerous_bundle_substrings"] as? [String] {
        for entry in list { set.insert(entry.lowercased()) }
    } else {
        diag("capture: dangerous_list_fallback")  // non-content diagnostic only
    }
    return Array(set)
}()

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

/// Capture the frontmost app's active-window text once and emit it.
private func captureFrontmost(allow: Set<String>, block: Set<String>) {
    guard let frontApp = NSWorkspace.shared.frontmostApplication else {
        diag("capture: no_frontmost_app")
        return
    }
    let bundleId = frontApp.bundleIdentifier
    let pid = frontApp.processIdentifier

    // Allowlist-first gate BEFORE reading any AX text. A disallowed app emits
    // metadata only (empty text) so the daemon sees the focus change but no
    // captured content ever crosses the IPC boundary.
    if !contentAllowed(bundleId: bundleId, allow: allow, block: block) {
        diag("capture: skipped_not_allowlisted")
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
        emit(CaptureEvent(app: bundleId, window: nil, url: nil, text: "",
                          ts: Date().timeIntervalSince1970, incognito: false))
        return
    }

    let appElement = AXUIElementCreateApplication(pid)

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

    let windowTitle = axString(window, kAXTitleAttribute as String)

    // Incognito/private windows: emit metadata only, with incognito=true and no
    // (potentially sensitive) window title, before any text traversal.
    if isIncognitoTitle(windowTitle) {
        diag("capture: skipped_incognito")
        emit(CaptureEvent(app: bundleId, window: nil, url: nil, text: "",
                          ts: Date().timeIntervalSince1970, incognito: true))
        return
    }

    let budget = TraversalBudget(deadlineSeconds: Limits.deadlineSeconds)
    var parts: [String] = []
    collectText(from: window, depth: 0, budget: budget, into: &parts)

    var text = parts.joined(separator: "\n")
    if text.utf8.count > Limits.maxTextBytes {
        let prefix = text.utf8.prefix(Limits.maxTextBytes)
        text = String(decoding: Array(prefix), as: UTF8.self)
    }

    // Emit only NON-content metadata to stderr (node/byte counts + redactions).
    diag("capture: ok nodes=\(budget.nodesVisited) bytes=\(text.utf8.count) "
        + "secure_skipped=\(budget.sensitiveSkipped)")

    let event = CaptureEvent(
        app: bundleId,
        window: windowTitle,
        url: nil, // URL extraction is per-app (Safari/Chrome AX) — out of MVP scope here.
        text: text,
        ts: Date().timeIntervalSince1970,
        incognito: false
    )
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

private func run() {
    let args = CommandLine.arguments
    let noPrompt = args.contains("--no-prompt")
    // Allowlist-first content policy, enforced before any AX text is read.
    let allow = listArg(args, "--allow")
    let block = listArg(args, "--block")

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

    captureFrontmost(allow: allow, block: block)
}

run()
