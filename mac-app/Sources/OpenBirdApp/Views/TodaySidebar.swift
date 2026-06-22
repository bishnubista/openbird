import AppKit
import SwiftUI

/// The 222px Today-window sidebar (handoff §3): the OpenBird wordmark, the section
/// nav, and a live capture-status footer. Only **Today** has a destination today,
/// so the other nav rows are rendered pixel-faithfully but are honestly inert —
/// `.disabled(true)` + an accessibility "not enabled" trait — rather than looking
/// clickable while doing nothing (macOS HIG / Codex review).
struct TodaySidebar: View {
    @ObservedObject var appModel: AppModel
    /// Show the Ask panel (the `.ask` destination — a controller action, not a window).
    var onAsk: () -> Void
    @Environment(\.colorScheme) private var scheme
    @Environment(\.openWindow) private var openWindow

    /// The sidebar mirrors the menu bar via the shared `AppDestination` set. It lives
    /// only in the Today window, so `.today` is always the active row.
    private func select(_ destination: AppDestination) {
        switch destination {
        case .ask: onAsk()
        case .today: break                       // already the active surface
        case .setup:
            openWindow(id: "main")
            NSApp.activate(ignoringOtherApps: true)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            wordmark
            nav
            Spacer(minLength: 0)
            footer
        }
        .frame(width: 222, alignment: .leading)
        .frame(maxHeight: .infinity, alignment: .top)
        .background(sidebarFill)
        .overlay(alignment: .trailing) {
            Rectangle().fill(OB.separator(scheme)).frame(width: 0.5)
        }
    }

    // MARK: Wordmark

    private var wordmark: some View {
        HStack(spacing: OB.Space.sm) {
            BirdLogo().fill(OB.accent).frame(width: 22, height: 22)
            Text("OpenBird")
                .font(.system(size: 15, weight: .bold))
                .foregroundStyle(OB.textPrimary(scheme))
        }
        // Leave room for the window's traffic lights, which overlap the top-left.
        .padding(.top, 34)
        .padding(.horizontal, OB.Space.l)
        .padding(.bottom, OB.Space.l)
    }

    // MARK: Nav

    private var nav: some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach(AppDestination.allCases) { dest in
                NavRow(
                    title: dest.title,
                    systemImage: dest.systemImage,
                    isActive: dest == .today,
                    action: { select(dest) }
                )
            }
        }
        .padding(.horizontal, OB.Space.m)
    }

    // MARK: Footer

    private var footer: some View {
        HStack(alignment: .top, spacing: OB.Space.s) {
            FooterPulseDot(active: appModel.captureRunning && !appModel.capturePaused)
                .padding(.top, 3)
            Text(footerText)
                .font(.system(size: 11))
                .foregroundStyle(OB.textSecondary(scheme))
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(OB.Space.l)
        .overlay(alignment: .top) {
            Rectangle().fill(OB.separator(scheme)).frame(height: 0.5)
        }
    }

    /// Real footer status: capture state, allowlist size, and the actual at-rest
    /// encryption state. The encryption word is derived from `encryptionState`, not
    /// hardcoded — claiming "encrypted" when it isn't would be a false privacy claim.
    private var footerText: String {
        let state = appModel.capturePaused
            ? "Paused"
            : (appModel.captureRunning ? "Capturing" : "Idle")
        let n = appModel.allowlist.count
        let apps = "\(n) app\(n == 1 ? "" : "s") allowed"
        let encryption: String
        switch appModel.encryptionState {
        case .ok: encryption = "encrypted"
        case .attention: encryption = "encryption off"
        default: encryption = "encryption unknown"
        }
        return "\(state) · \(apps) · on-device · \(encryption)"
    }

    private var sidebarFill: Color {
        scheme == .dark ? Color.white.opacity(0.03) : Color.black.opacity(0.02)
    }
}

/// One sidebar nav row. A row with a `windowID` is a real button that opens/focuses
/// that window (hover = subtle wash, per the handoff); the active row ("Today") is a
/// non-interactive accent highlight; rows with neither are inert placeholders.
private struct NavRow: View {
    let title: String
    let systemImage: String
    let isActive: Bool
    let action: () -> Void

    @Environment(\.colorScheme) private var scheme
    @State private var hovered = false

    var body: some View {
        let row = HStack(spacing: OB.Space.sm) {
            Image(systemName: systemImage).frame(width: 18)
            Text(title).font(.system(size: 13.5, weight: isActive ? .semibold : .regular))
            Spacer()
        }
        .foregroundStyle(foreground)
        .padding(.horizontal, OB.Space.m)
        .padding(.vertical, OB.Space.sm)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: OB.Radius.control).fill(background))
        .contentShape(Rectangle())

        if isActive {
            row   // the surface you're already in — highlighted, non-interactive
        } else {
            Button(action: action) { row }
                .buttonStyle(.plain)
                .onHover { hovered = $0 }
        }
    }

    private var foreground: Color {
        if isActive { return .white }
        return hovered ? OB.textPrimary(scheme) : OB.textSecondary(scheme)
    }

    private var background: Color {
        if isActive { return OB.accent }
        if hovered { return scheme == .dark ? Color.white.opacity(0.09) : Color.black.opacity(0.06) }
        return .clear
    }
}

/// A small status dot that pulses while capture is live (handoff `obPulse`).
private struct FooterPulseDot: View {
    let active: Bool
    @State private var pulsing = false

    var body: some View {
        Circle()
            .fill(active ? OB.capturingDot : Color.secondary)
            .frame(width: 7, height: 7)
            .scaleEffect(pulsing ? 1.0 : 0.7)
            .opacity(pulsing ? 1.0 : 0.6)
            .onAppear { apply() }
            .onChange(of: active) { _ in apply() }
    }

    private func apply() {
        if active {
            withAnimation(.easeInOut(duration: 0.9).repeatForever(autoreverses: true)) {
                pulsing = true
            }
        } else {
            withAnimation(.default) { pulsing = false }
        }
    }
}
