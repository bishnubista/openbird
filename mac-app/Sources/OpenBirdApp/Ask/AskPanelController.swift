import AppKit
import Carbon.HIToolbox
import SwiftUI

/// Owns the floating Spotlight Ask panel and its ⌥Space global hotkey. Created once
/// in `OpenBirdApp.init()` with the shared `AppModel`/`OpenBirdService` (so there is
/// never a second model and no configure-after-launch race).
@MainActor
final class AskPanelController: NSObject, ObservableObject {
    private let appModel: AppModel
    private let askModel: AskPanelModel
    private var panel: AskPanel?
    private var hotKey: GlobalHotKey?
    private var hotKeyInstalled = false
    /// Screen-space y of the panel's top edge, held constant so the card grows
    /// downward (Spotlight-style) as answers arrive.
    private var anchorTopY: CGFloat?

    private let panelWidth: CGFloat = 620

    init(model: AppModel, service: OpenBirdService) {
        self.appModel = model
        self.askModel = AskPanelModel(service: service, appModel: model)
        super.init()
    }

    /// Register ⌥Space once. Idempotent, so a scene `.task` can call it regardless of
    /// lifecycle timing without risking a double registration.
    func installHotKeyIfNeeded() {
        guard !hotKeyInstalled else { return }
        hotKeyInstalled = true
        hotKey = GlobalHotKey(keyCode: UInt32(kVK_Space), modifiers: UInt32(optionKey)) { [weak self] in
            // Carbon fires on the main thread; hop explicitly to satisfy main-actor
            // isolation of `toggle()`.
            DispatchQueue.main.async { self?.toggle() }
        }
        if hotKey == nil {
            // Degrade gracefully: the menu item still opens the panel.
            NSLog("openbird: hotkey-register-failed (⌥Space already in use?)")
        }
    }

    func toggle() {
        if let panel, panel.isVisible { hide() } else { show() }
    }

    func show() {
        let p = panel ?? makePanel()
        panel = p
        positionForCurrentSize(p, freshAnchor: true)
        // Do NOT activate the app: a .nonactivatingPanel becomes key on its own, so
        // it accepts typing while the user's foreground app stays active and focus
        // returns there on dismiss (true Spotlight-utility behavior).
        p.makeKeyAndOrderFront(nil)
        p.orderFrontRegardless()   // ensure it surfaces even while our app is inactive
        // Route first-responder into the SwiftUI field on the next runloop turn,
        // after the panel is key (a borderless panel won't auto-focus it otherwise).
        DispatchQueue.main.async { [weak p] in
            guard let p else { return }
            p.makeFirstResponder(p.contentView)
        }
    }

    func hide() {
        panel?.orderOut(nil)
        anchorTopY = nil
    }

    // MARK: - Panel construction

    private func makePanel() -> AskPanel {
        let root = AskPanelView(
            askModel: askModel,
            appModel: appModel,
            onEscape: { [weak self] in self?.hide() },
            onSizeChange: { [weak self] size in self?.handleContentSize(size) }
        )
        let hosting = NSHostingView(rootView: root)
        hosting.autoresizingMask = [.width, .height]

        let panel = AskPanel(
            contentRect: NSRect(x: 0, y: 0, width: panelWidth, height: 140),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        panel.contentView = hosting
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false                 // the glass surface draws its own shadow
        panel.level = .statusBar                // reliably above other apps / full-screen
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .transient]
        panel.isMovableByWindowBackground = false
        panel.hidesOnDeactivate = false
        panel.animationBehavior = .utilityWindow
        panel.delegate = self
        return panel
    }

    /// Resize the panel to fit SwiftUI content, keeping the top edge anchored.
    private func handleContentSize(_ size: CGSize) {
        guard let panel, panel.isVisible else { return }
        let newHeight = ceil(size.height)
        guard newHeight > 0, abs(panel.frame.height - newHeight) > 0.5 else { return }
        panel.setContentSize(NSSize(width: panelWidth, height: newHeight))
        positionForCurrentSize(panel, freshAnchor: false)
    }

    private func positionForCurrentSize(_ panel: AskPanel, freshAnchor: Bool) {
        let screen = (panel.screen ?? NSScreen.main)?.visibleFrame ?? .zero
        let height = panel.frame.height
        if freshAnchor || anchorTopY == nil {
            // Top edge ~80% up the visible area (upper third) — Spotlight placement.
            anchorTopY = screen.minY + screen.height * 0.80
        }
        let topY = anchorTopY ?? (screen.midY + height / 2)
        let originX = screen.midX - panelWidth / 2
        panel.setFrameOrigin(NSPoint(x: originX, y: topY - height))
    }
}

extension AskPanelController: NSWindowDelegate {
    /// Click-away / app switch dismisses, like Spotlight.
    func windowDidResignKey(_ notification: Notification) {
        hide()
    }
}
