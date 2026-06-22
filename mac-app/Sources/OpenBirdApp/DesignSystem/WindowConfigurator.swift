import AppKit
import SwiftUI

/// Makes the hosting window draggable by any empty background region — the
/// macOS-13-safe way to keep a `.hiddenTitleBar` window movable without
/// `WindowDragGesture` (macOS 15+, above our deployment floor). Configured from
/// `viewDidMoveToWindow` so it applies once the view is actually attached to a window
/// (no `view.window == nil` attachment race) and re-applies if it re-attaches
/// (Codex review). Drop into a view's `.background(WindowConfigurator())`.
struct WindowConfigurator: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView { ConfiguringView() }
    func updateNSView(_ nsView: NSView, context: Context) {}

    private final class ConfiguringView: NSView {
        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            window?.isMovableByWindowBackground = true
        }
    }
}
