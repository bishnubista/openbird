import AppKit

/// The expanded Ask shell: a borderless glass window that — unlike the compact
/// Spotlight `AskPanel` — activates and persists (no click-away dismiss). A borderless
/// `NSWindow` does not become key/main by default, so these overrides are required for
/// the text field and Esc handling to work (Codex review).
final class AskExpandedWindow: NSWindow {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
}
