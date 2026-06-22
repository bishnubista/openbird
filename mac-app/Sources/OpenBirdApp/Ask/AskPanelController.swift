import AppKit
import Carbon.HIToolbox
import SwiftUI

/// Owns the unified Ask experience and its ⌥Space global hotkey. There is ONE Ask
/// conversation (`askModel`), surfaced through two AppKit shells that are never visible
/// at once (Codex review — deterministic surface state):
///
/// - **compact**: the floating Spotlight `AskPanel` (`.nonactivatingPanel`, click-away
///   dismiss). The default; ⌥Space toggles it.
/// - **expanded**: a separate `AskExpandedWindow` (activating, persistent, no click-away
///   dismiss) showing the same conversation plus optional Timeline/Sources rails.
///
/// Both are AppKit-owned (no SwiftUI `Window` scene), so they open reliably even when no
/// scene is mounted — the macOS-13-safe way to open the rich surface from the detached
/// hotkey (Apple's `openWindow` is view-hierarchy-scoped). Created once in
/// `OpenBirdApp.init()` with the shared `askModel`/`timelineModel`.
@MainActor
final class AskPanelController: NSObject, ObservableObject {
    private let appModel: AppModel
    private let askModel: AskPanelModel
    private let timelineModel: TimelineModel

    private var compactPanel: AskPanel?
    private var expandedWindow: AskExpandedWindow?
    private var hotKey: GlobalHotKey?
    private var hotKeyInstalled = false
    /// Screen-space y of the compact panel's top edge, held constant so the card grows
    /// downward (Spotlight-style) as answers arrive.
    private var anchorTopY: CGFloat?

    private let compactWidth: CGFloat = 620
    private let expandedWidth: CGFloat = 960

    init(model: AppModel, service: OpenBirdService, askModel: AskPanelModel, timelineModel: TimelineModel) {
        self.appModel = model
        self.askModel = askModel
        self.timelineModel = timelineModel
        super.init()
    }

    // MARK: - Hotkey

    /// Register ⌥Space once. Idempotent.
    func installHotKeyIfNeeded() {
        guard !hotKeyInstalled else { return }
        // Mark installed only on SUCCESS, so a transient registration failure doesn't
        // permanently block later retries for this run (CodeRabbit).
        let candidate = GlobalHotKey(keyCode: UInt32(kVK_Space), modifiers: UInt32(optionKey)) { [weak self] in
            DispatchQueue.main.async { self?.toggle() }
        }
        guard let candidate else {
            NSLog("openbird: hotkey-register-failed (⌥Space already in use?) — will retry")
            return
        }
        hotKey = candidate
        hotKeyInstalled = true
    }

    /// ⌥Space deterministically yields the compact panel: hide it if it's already
    /// showing, otherwise show it (closing the expanded window first if open). It never
    /// reopens the 960px expanded surface — that is only reached via the expand button
    /// (CodeRabbit).
    func toggle() {
        if isCompactVisible { hide() } else { show() }
    }

    private var isCompactVisible: Bool { compactPanel?.isVisible ?? false }
    private var isExpandedVisible: Bool { expandedWindow?.isVisible ?? false }

    // MARK: - Surface state machine (compact / expanded / hidden — mutually exclusive)

    /// Show the compact Spotlight panel (closing the expanded window first). The single
    /// "Ask" entry point for the hotkey, the menu, the sidebar, and the deep-link.
    func show() {
        orderOutExpanded()
        let p = compactPanel ?? makeCompactPanel()
        compactPanel = p
        positionCompact(p, freshAnchor: true)
        // A .nonactivatingPanel becomes key on its own; do not activate the app, so the
        // user's foreground app stays active (true Spotlight-utility behavior).
        p.makeKeyAndOrderFront(nil)
        p.orderFrontRegardless()
        focusInput(of: p)
    }

    /// Hide all Ask surfaces.
    func hide() {
        compactPanel?.orderOut(nil)
        anchorTopY = nil
        orderOutExpanded()
    }

    /// Compact → expanded: swap the floating panel for the persistent window with rails.
    func expand() {
        compactPanel?.orderOut(nil)
        anchorTopY = nil
        let w = expandedWindow ?? makeExpandedWindow()
        expandedWindow = w
        centerExpanded(w)
        // A normal activating window must bring the app forward to take key/focus.
        NSApp.activate(ignoringOtherApps: true)
        w.makeKeyAndOrderFront(nil)
        focusInput(of: w)
    }

    /// Expanded → compact: return to the floating panel.
    func collapse() {
        orderOutExpanded()
        show()
    }

    private func orderOutExpanded() {
        expandedWindow?.orderOut(nil)
    }

    /// Route first-responder into the SwiftUI field on the next runloop turn, after the
    /// window is key (a borderless window won't auto-focus it otherwise).
    private func focusInput(of window: NSWindow) {
        DispatchQueue.main.async { [weak window] in
            guard let window else { return }
            window.makeFirstResponder(window.contentView)
        }
    }

    // MARK: - Compact panel

    private func makeCompactPanel() -> AskPanel {
        let root = AskPanelView(
            askModel: askModel,
            appModel: appModel,
            onEscape: { [weak self] in self?.hide() },
            onExpand: { [weak self] in self?.expand() },
            onSizeChange: { [weak self] size in self?.handleCompactSize(size) }
        )
        let hosting = NSHostingView(rootView: root)
        hosting.autoresizingMask = [.width, .height]

        let panel = AskPanel(
            contentRect: NSRect(x: 0, y: 0, width: compactWidth, height: 140),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        panel.contentView = hosting
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.level = .statusBar
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .transient]
        panel.isMovableByWindowBackground = false
        panel.hidesOnDeactivate = false
        panel.animationBehavior = .utilityWindow
        panel.delegate = self
        return panel
    }

    /// Resize the compact panel to fit SwiftUI content, keeping the top edge anchored.
    /// Applies ONLY to the compact panel; the expanded window sizes itself.
    private func handleCompactSize(_ size: CGSize) {
        guard let panel = compactPanel, panel.isVisible else { return }
        let newHeight = ceil(size.height)
        guard newHeight > 0, abs(panel.frame.height - newHeight) > 0.5 else { return }
        panel.setContentSize(NSSize(width: compactWidth, height: newHeight))
        positionCompact(panel, freshAnchor: false)
    }

    private func positionCompact(_ panel: AskPanel, freshAnchor: Bool) {
        let screen = (panel.screen ?? NSScreen.main)?.visibleFrame ?? .zero
        let height = panel.frame.height
        if freshAnchor || anchorTopY == nil {
            anchorTopY = screen.minY + screen.height * 0.80
        }
        let topY = anchorTopY ?? (screen.midY + height / 2)
        let originX = screen.midX - compactWidth / 2
        panel.setFrameOrigin(NSPoint(x: originX, y: topY - height))
    }

    // MARK: - Expanded window

    private func makeExpandedWindow() -> AskExpandedWindow {
        let root = ExpandedAskView(
            askModel: askModel,
            appModel: appModel,
            timelineModel: timelineModel,
            onCollapse: { [weak self] in self?.collapse() },
            onClose: { [weak self] in self?.hide() }
        )
        let hosting = NSHostingView(rootView: root)
        hosting.autoresizingMask = [.width, .height]

        let window = AskExpandedWindow(
            contentRect: NSRect(x: 0, y: 0, width: expandedWidth, height: 600),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        window.contentView = hosting
        window.title = "Ask"                     // borderless: not shown, but exposed via AX for E2E
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = false                 // the glass surface draws its own shadow
        window.level = .normal                   // a regular window, not a floating utility
        window.isMovableByWindowBackground = true
        window.collectionBehavior = [.fullScreenAuxiliary]
        window.animationBehavior = .documentWindow
        window.delegate = self
        return window
    }

    private func centerExpanded(_ window: AskExpandedWindow) {
        let screen = (window.screen ?? NSScreen.main)?.visibleFrame ?? .zero
        // Clamp BOTH dimensions and the origin to the visible frame so the rails never
        // run off-screen on a narrow display/space (CodeRabbit).
        let width = min(expandedWidth, screen.width - 40)
        let height = min(window.frame.height, screen.height - 40)
        window.setContentSize(NSSize(width: width, height: height))
        let originX = max(screen.minX, screen.midX - width / 2)
        let originY = max(screen.minY, screen.midY - height / 2)
        window.setFrameOrigin(NSPoint(x: originX, y: originY))
    }
}

extension AskPanelController: NSWindowDelegate {
    /// Only the COMPACT panel dismisses on click-away (Spotlight). The expanded window is
    /// persistent — resigning key (e.g. transiently during activation) must not hide it.
    func windowDidResignKey(_ notification: Notification) {
        guard let window = notification.object as? NSWindow else { return }
        if window === compactPanel {
            hide()
        }
    }
}
