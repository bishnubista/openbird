import AppKit

/// A borderless, non-activating panel that can still become key so its hosted
/// SwiftUI `TextField` receives keystrokes. A plain borderless `NSPanel` returns
/// `false` for `canBecomeKey`, which would leave the Spotlight input unfocusable —
/// hence this subclass (per the panel-focus design requirement).
final class AskPanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
}
