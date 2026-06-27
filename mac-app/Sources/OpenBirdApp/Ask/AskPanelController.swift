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
/// `OpenBirdApp.init()` with the shared `askModel`.
@MainActor
final class AskPanelController: NSObject, ObservableObject {
    private let appModel: AppModel
    private let askModel: AskPanelModel

    private var compactPanel: AskPanel?
    private var expandedWindow: AskExpandedWindow?
    private var hotKey: GlobalHotKey?
    private var hotKeyInstalled = false
    /// Screen-space y of the compact panel's top edge, held constant so the card grows
    /// downward (Spotlight-style) as answers arrive.
    private var anchorTopY: CGFloat?

    private let compactWidth: CGFloat = 620
    // Single-column Ask (no side rails) — the expanded overlay no longer needs the wide
    // chat+rails footprint, so it's snug around the readable chat column.
    private let expandedWidth: CGFloat = 720

    /// Opens the SwiftUI main window (`Window(id: "main")`). Wired from the scene's
    /// `@Environment(\.openWindow)` because that environment value is only reachable
    /// inside a `View`. Used by citation click-through so navigating from an Ask
    /// overlay still works when the main window was closed (the MenuBarExtra keeps the
    /// app alive with no window). Falls back to fronting an existing window when unset.
    var openMainWindow: (() -> Void)?

    init(model: AppModel, service: OpenBirdService, askModel: AskPanelModel) {
        self.appModel = model
        self.askModel = askModel
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
    ///
    /// `dayScope` hard-scopes every ask in the panel to one calendar day
    /// (0=today, 1=yesterday, ...): the Today view's "Ask about this day" passes the
    /// viewed day; the generic entry points (hotkey, menu, deep-link) pass nil, which
    /// CLEARS any prior scope so the global Ask never inherits a stale day.
    func show(dayScope: Int? = nil) {
        askModel.dayScope = dayScope
        showPanel()
    }

    /// Order the compact panel front WITHOUT touching `askModel.dayScope`. Used by
    /// `show(dayScope:)` (after setting the scope) and by `collapse()` (which must
    /// PRESERVE the active scope when returning from the expanded surface).
    private func showPanel() {
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

    /// Hide all Ask surfaces. Ends any Today-scoped compact session, so the next
    /// Ask starts unscoped unless it is itself opened from Today.
    func hide() {
        compactPanel?.orderOut(nil)
        anchorTopY = nil
        orderOutExpanded()
        askModel.dayScope = nil
    }

    /// Compact → expanded: swap the floating panel for the persistent window with rails.
    /// The expanded window is a GENERIC global Ask surface, so entering it CLEARS any
    /// day scope — a scope set by Today's compact panel must not leak into it (and
    /// the in-window Ask pane shares the same model). Scope is intentionally confined
    /// to the compact Today-initiated session.
    func expand() {
        compactPanel?.orderOut(nil)
        anchorTopY = nil
        askModel.dayScope = nil
        let w = expandedWindow ?? makeExpandedWindow()
        expandedWindow = w
        centerExpanded(w)
        // A normal activating window must bring the app forward to take key/focus.
        NSApp.activate(ignoringOtherApps: true)
        w.makeKeyAndOrderFront(nil)
        focusInput(of: w)
    }

    /// Expanded → compact: return to the floating panel. The scope was already
    /// cleared by `expand()` (the expanded surface is generic), so this returns an
    /// unscoped compact panel — it must NOT route through `show()` (which would
    /// re-clear and re-anchor identically, but keeping it explicit avoids surprise).
    func collapse() {
        orderOutExpanded()
        showPanel()
    }

    private func orderOutExpanded() {
        expandedWindow?.orderOut(nil)
    }

    // MARK: - Citation click-through

    /// A chat citation was clicked in an Ask OVERLAY surface (the compact Spotlight
    /// panel or the expanded window). Dismiss the overlay, bring the main app window
    /// forward, and route to the source's day/observation in the Today pane. The
    /// in-window Ask *pane* doesn't use this — it navigates `AppModel` directly since
    /// it already lives inside the main window (no overlay to dismiss).
    func navigateToCitation(_ citation: ChatCitation) {
        hide()
        NSApp.activate(ignoringOtherApps: true)
        showMainWindow()
        appModel.navigateToCitation(citation)
    }

    /// Ensure the main window exists and is frontmost. Prefer the injected SwiftUI
    /// opener (it RECREATES the window if the user closed it — the MenuBarExtra keeps
    /// the app running with no window); fall back to fronting an existing window when
    /// the opener wasn't wired (e.g. unit context).
    private func showMainWindow() {
        if let openMainWindow {
            openMainWindow()
            return
        }
        for window in NSApp.windows where window.identifier?.rawValue == "main" {
            window.makeKeyAndOrderFront(nil)
        }
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
            onSizeChange: { [weak self] size in self?.handleCompactSize(size) },
            onSelectCitation: { [weak self] citation in self?.navigateToCitation(citation) }
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

    /// Guards SYNCHRONOUS re-entry of `handleCompactSize` — `setContentSize` re-lays-out
    /// the hosting view, which can flush preferences again within this call. Not the
    /// convergence mechanism (that's `compactResizeTarget`'s clamp+ceil+epsilon); just a
    /// belt-and-suspenders against a same-callstack recurse.
    private var isApplyingCompactSize = false

    /// Resize the compact panel to fit SwiftUI content, keeping the top edge anchored.
    /// Applies ONLY to the compact panel; the expanded window sizes itself.
    private func handleCompactSize(_ size: CGSize) {
        guard let panel = compactPanel, panel.isVisible, !isApplyingCompactSize else { return }
        // Bound the request by what the screen can actually display so the resize loop
        // always reaches a fixed point (#169).
        let maxHeight = (panel.screen ?? NSScreen.main)?.visibleFrame.height ?? size.height
        guard let target = Self.compactResizeTarget(
            reportedHeight: size.height,
            currentHeight: panel.frame.height,
            maxHeight: maxHeight
        ) else { return }
        isApplyingCompactSize = true
        defer { isApplyingCompactSize = false }
        panel.setContentSize(NSSize(width: compactWidth, height: target))
        positionCompact(panel, freshAnchor: false)
    }

    /// Pure resize decision (no AppKit state) so it is unit-testable and its convergence
    /// is auditable. Returns the panel height to apply, or `nil` when already settled.
    ///
    /// Clamping the request to `maxHeight` (the screen's displayable height) is what
    /// breaks the #169 layout loop: if the SwiftUI content reports a height larger than
    /// the window server will grant, `panel.frame.height` could never reach an unclamped
    /// target, so `abs(current - target) > 0.5` would stay true and the Core-Animation
    /// display cycle would resize the panel every frame forever. `ceil` + the 0.5 epsilon
    /// give a stable fixed point once the content height settles (a stable report yields
    /// an identical clamped target, so the second call returns nil).
    nonisolated static func compactResizeTarget(reportedHeight: CGFloat,
                                                currentHeight: CGFloat,
                                                maxHeight: CGFloat) -> CGFloat? {
        guard reportedHeight.isFinite, maxHeight > 0 else { return nil }
        let target = min(ceil(reportedHeight), floor(maxHeight))
        guard target > 0, abs(currentHeight - target) > 0.5 else { return nil }
        return target
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
            onCollapse: { [weak self] in self?.collapse() },
            onClose: { [weak self] in self?.hide() },
            onSelectCitation: { [weak self] citation in self?.navigateToCitation(citation) }
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
