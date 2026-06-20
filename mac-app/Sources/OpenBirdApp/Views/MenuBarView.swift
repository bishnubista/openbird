import AppKit
import SwiftUI

/// The menu-bar dropdown, styled as a Liquid Glass panel (handoff §1). Rendered via
/// `.menuBarExtraStyle(.window)` so it's a real SwiftUI surface (not the system
/// menu), letting it use the shared glass material, custom rows with hover accent,
/// a live capturing indicator, and helper/encryption health dots.
struct MenuBarView: View {
    @ObservedObject var model: AppModel
    @Environment(\.openWindow) private var openWindow
    /// Summon the Spotlight Ask panel (also bound to the global ⌥Space hotkey).
    var openAskPanel: () -> Void

    @Environment(\.colorScheme) private var scheme
    /// The popover window hosting this view. A `.window`-style MenuBarExtra does not
    /// auto-dismiss on a button tap the way a system menu does, so command rows
    /// close it explicitly (matching native menu behavior).
    @State private var menuWindow: NSWindow?

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            MenuRow(icon: "macwindow", title: "Open OpenBird") {
                run { openWindow(id: "main"); NSApp.activate(ignoringOtherApps: true) }
            }
            // ⌥Space is shown as a trailing hint, not a `.keyboardShortcut`, so it
            // can't double-fire with the global Carbon hotkey.
            MenuRow(icon: "sparkle.magnifyingglass", title: "Ask OpenBird…", shortcut: "⌥Space") {
                run { openAskPanel() }
            }

            divider

            captureStatusRow

            if model.captureRunning {
                MenuRow(icon: "stop.fill", title: "Stop Capture") { run { model.stopCapture() } }
            } else {
                MenuRow(icon: "play.fill", title: "Start Capture",
                        disabled: model.allowlist.isEmpty) { run { model.startCapture() } }
            }
            MenuRow(icon: model.capturePaused ? "play.circle" : "pause.circle",
                    title: model.capturePaused ? "Resume Capture" : "Pause Capture") {
                run { model.toggleCapturePause() }
            }
            MenuRow(icon: "arrow.clockwise", title: "Re-check Setup") {
                run { Task { await model.refresh() } }
            }

            divider

            ForEach(model.helpers) { helper in
                HealthRow(ok: helper.isBundled,
                          label: "\(helper.label) \(helper.isBundled ? "OK" : "missing")")
            }
            HealthRow(state: model.encryptionState, label: encryptionLabel)

            divider

            MenuRow(icon: "folder", title: "Data Folder") { run { model.openDataFolder() } }
            MenuRow(icon: "power", title: "Quit OpenBird") { run { model.quit() } }
        }
        .padding(OB.Space.s)
        .frame(width: 288)
        .glassSurface(cornerRadius: OB.Radius.dropdown)
        .padding(OB.Space.sm)
        .background(WindowAccessor { menuWindow = $0 })
    }

    /// Dismiss the dropdown, then run the action on the next runloop turn so closing
    /// the popover never races a window/panel the action itself brings up.
    private func run(_ action: @escaping () -> Void) {
        menuWindow?.orderOut(nil)
        DispatchQueue.main.async(execute: action)
    }

    private var divider: some View {
        Divider().overlay(OB.separator(scheme)).padding(.vertical, OB.Space.xs)
    }

    private var captureStatusRow: some View {
        HStack(spacing: OB.Space.sm) {
            PulsingDot(active: model.captureRunning && !model.capturePaused)
            VStack(alignment: .leading, spacing: 1) {
                Text(captureStateText)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(OB.textPrimary(scheme))
                Text(model.memorySummary)
                    .font(.system(size: 11))
                    .foregroundStyle(OB.textSecondary(scheme))
                    .lineLimit(1)
            }
            Spacer()
        }
        .padding(.horizontal, OB.Space.sm)
        .padding(.vertical, OB.Space.s)
    }

    private var captureStateText: String {
        if model.capturePaused { return "Paused" }
        return model.captureRunning ? "Capturing" : "Idle"
    }

    private var encryptionLabel: String {
        switch model.encryptionState {
        case .ok: return "Encryption at rest on"
        case .attention: return "Encryption at rest off"
        default: return "Encryption status unknown"
        }
    }
}

/// A tappable dropdown row with a leading SF Symbol, optional trailing shortcut,
/// and the handoff's hover = accent-background / white-text treatment.
private struct MenuRow: View {
    let icon: String
    let title: String
    var shortcut: String? = nil
    var disabled: Bool = false
    let action: () -> Void

    @Environment(\.colorScheme) private var scheme
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: OB.Space.sm) {
                Image(systemName: icon)
                    .frame(width: 16)
                Text(title)
                if let shortcut {
                    Spacer()
                    Text(shortcut)
                        .font(.system(size: 11))
                        .opacity(hovering ? 0.9 : 0.5)
                } else {
                    Spacer()
                }
            }
            .font(.system(size: 13))
            .foregroundStyle(rowForeground)
            .padding(.horizontal, OB.Space.sm)
            .padding(.vertical, 5)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
            .background(
                RoundedRectangle(cornerRadius: OB.Radius.control)
                    .fill(hovering && !disabled ? OB.accent : .clear)
            )
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .onHover { hovering = $0 }
    }

    private var rowForeground: Color {
        if disabled { return OB.textTertiary(scheme) }
        return hovering ? .white : OB.textPrimary(scheme)
    }
}

/// A non-interactive status row with a colored health dot.
private struct HealthRow: View {
    let ok: Bool
    let label: String
    var state: StepState? = nil

    @Environment(\.colorScheme) private var scheme

    init(ok: Bool, label: String) {
        self.ok = ok
        self.label = label
    }

    init(state: StepState, label: String) {
        self.state = state
        self.ok = state == .ok
        self.label = label
    }

    var body: some View {
        HStack(spacing: OB.Space.sm) {
            Circle().fill(dotColor).frame(width: 7, height: 7)
            Text(label)
                .font(.system(size: 12))
                .foregroundStyle(OB.textSecondary(scheme))
            Spacer()
        }
        .padding(.horizontal, OB.Space.sm)
        .padding(.vertical, 3)
    }

    private var dotColor: Color {
        switch state {
        case .some(.attention): return .orange
        case .some(.unknown): return OB.textTertiary(scheme)
        default: return ok ? OB.ok(scheme) : .red
        }
    }
}

/// A capture indicator dot that pulses while actively capturing. The animation is
/// driven by `active` (not just `.onAppear`), so it starts/stops correctly when
/// capture state changes while the dropdown is open.
private struct PulsingDot: View {
    let active: Bool
    @State private var pulsing = false

    var body: some View {
        Circle()
            .fill(active ? OB.capturingDot : Color.secondary)
            .frame(width: 9, height: 9)
            .scaleEffect(pulsing ? 1.0 : 0.7)
            .opacity(pulsing ? 1.0 : 0.6)
            .onAppear { applyAnimation() }
            .onChange(of: active) { _ in applyAnimation() }
    }

    private func applyAnimation() {
        if active {
            withAnimation(.easeInOut(duration: 0.9).repeatForever(autoreverses: true)) {
                pulsing = true
            }
        } else {
            withAnimation(.default) { pulsing = false }
        }
    }
}

/// Captures the hosting `NSWindow` of a SwiftUI view so the dropdown can dismiss
/// itself programmatically (the `.window` MenuBarExtra has no system dismissal).
private struct WindowAccessor: NSViewRepresentable {
    let onResolve: (NSWindow?) -> Void

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async { onResolve(view.window) }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        DispatchQueue.main.async { onResolve(nsView.window) }
    }
}
