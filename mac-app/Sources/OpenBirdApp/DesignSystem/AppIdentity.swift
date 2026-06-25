import AppKit
import SwiftUI

/// Resolves a bundle id to a human display name + the real macOS app icon, memoized.
///
/// The Settings "Capture allowlist" stores raw bundle ids (e.g. `com.apple.Safari`), but
/// renders them as the app's actual Finder name + icon ("Safari" + the Safari icon). All
/// APIs used here predate the macOS 13 floor, so no availability guards are needed.
///
/// Lookups hit Launch Services + disk, so they are cached per bundle id: calling this from
/// a SwiftUI `body` is safe because each id resolves at most once per session. `@MainActor`
/// keeps the shared cache race-free (NSWorkspace is main-actor-friendly anyway).
@MainActor
enum AppIdentity {
    struct Identity {
        let name: String
        let icon: NSImage?
        /// False when the app isn't installed (icon is nil; name derived from the id).
        let isInstalled: Bool
    }

    private static var cache: [String: Identity] = [:]

    static func forBundleID(_ bundleID: String) -> Identity {
        if let hit = cache[bundleID] { return hit }
        let ws = NSWorkspace.shared
        let identity: Identity
        if let url = ws.urlForApplication(withBundleIdentifier: bundleID) {
            // Finder-localized name (strips ".app"); the real multi-rep app icon.
            let name = FileManager.default.displayName(atPath: url.path)
            let icon = ws.icon(forFile: url.path)
            identity = Identity(name: name, icon: icon, isInstalled: true)
        } else {
            // Not installed: derive a readable name from the last bundle-id segment so the
            // row still reads cleanly ("com.tinyspeck.slackmacgap" → "Slackmacgap").
            let tail = bundleID.split(separator: ".").last.map(String.init) ?? bundleID
            identity = Identity(name: tail.capitalized, icon: nil, isInstalled: false)
        }
        cache[bundleID] = identity
        return identity
    }
}
