import AppKit

/// Resolve a human-readable app name from a capture bundle id (e.g.
/// `com.apple.finder` → "Finder", `com.microsoft.VSCode` → "Visual Studio Code").
/// Uses LaunchServices for the real localized name of any installed app, and falls
/// back to a capitalized last path-component when the app can't be resolved.
enum AppDisplay {
    static func name(_ bundleID: String?) -> String {
        guard let bundleID, !bundleID.isEmpty else { return "Unknown" }
        if let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleID) {
            let display = FileManager.default.displayName(atPath: url.path)
            let trimmed = display.hasSuffix(".app") ? String(display.dropLast(4)) : display
            if !trimmed.isEmpty { return trimmed }
        }
        return fallbackName(bundleID)
    }

    /// Deterministic, machine-independent fallback: the last dotted component,
    /// capitalized (`com.foo.barApp` → "BarApp"). Used when LaunchServices has no
    /// match (e.g. the app isn't installed on this Mac).
    static func fallbackName(_ bundleID: String) -> String {
        let last = bundleID.split(separator: ".").last.map(String.init) ?? bundleID
        guard let first = last.first else { return bundleID }
        return first.uppercased() + last.dropFirst()
    }
}
