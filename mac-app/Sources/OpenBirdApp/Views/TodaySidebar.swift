import AppKit
import SwiftUI

/// The 222px Today-window sidebar (handoff §3): the OpenBird wordmark, the section
/// nav, and a live capture-status footer. Only **Today** has a destination today,
/// so the other nav rows are rendered pixel-faithfully but are honestly inert —
/// `.disabled(true)` + an accessibility "not enabled" trait — rather than looking
/// clickable while doing nothing (macOS HIG / Codex review).
struct TodaySidebar: View {
    @ObservedObject var appModel: AppModel
    @Environment(\.colorScheme) private var scheme

    /// One nav entry. The sidebar is *launch navigation* — it lives only in the Today
    /// window, so "Today" is always the active row. A row with a `windowID` opens that
    /// window on tap; the rest are present-but-inert placeholders for sections that
    /// have no window yet (honestly non-interactive, not clickable-looking no-ops).
    private struct NavItem {
        let icon: String
        let label: String
        /// Highlighted as the surface you're already in (Today only).
        let isActiveHere: Bool
        /// Non-nil → tapping opens/focuses that window scene.
        let windowID: String?
    }

    private let items: [NavItem] = [
        .init(icon: "calendar", label: "Today", isActiveHere: true, windowID: nil),
        .init(icon: "list.bullet", label: "Timeline", isActiveHere: false, windowID: "timeline"),
        .init(icon: "waveform", label: "Meetings", isActiveHere: false, windowID: nil),
        .init(icon: "arrow.triangle.2.circlepath", label: "Routines", isActiveHere: false, windowID: nil),
        .init(icon: "magnifyingglass", label: "Search", isActiveHere: false, windowID: nil),
        .init(icon: "slider.horizontal.3", label: "Settings", isActiveHere: false, windowID: nil),
    ]

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
            ForEach(items.indices, id: \.self) { i in
                NavRow(
                    icon: items[i].icon,
                    label: items[i].label,
                    isActiveHere: items[i].isActiveHere,
                    windowID: items[i].windowID
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
    let icon: String
    let label: String
    let isActiveHere: Bool
    let windowID: String?

    @Environment(\.colorScheme) private var scheme
    @Environment(\.openWindow) private var openWindow
    @State private var hovered = false

    var body: some View {
        let row = HStack(spacing: OB.Space.sm) {
            Image(systemName: icon).frame(width: 18)
            Text(label).font(.system(size: 13.5, weight: isActiveHere ? .semibold : .regular))
            Spacer()
        }
        .foregroundStyle(foreground)
        .padding(.horizontal, OB.Space.m)
        .padding(.vertical, OB.Space.sm)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: OB.Radius.control).fill(background))
        .contentShape(Rectangle())

        if let windowID {
            Button {
                openWindow(id: windowID)
                NSApp.activate(ignoringOtherApps: true)
            } label: { row }
                .buttonStyle(.plain)
                .onHover { hovered = $0 }
        } else {
            // Inert placeholder: honestly non-interactive, not a clickable no-op.
            row.disabled(true)
        }
    }

    private var foreground: Color {
        if isActiveHere { return .white }
        if windowID != nil { return hovered ? OB.textPrimary(scheme) : OB.textSecondary(scheme) }
        return OB.textTertiary(scheme)
    }

    private var background: Color {
        if isActiveHere { return OB.accent }
        if windowID != nil, hovered {
            return scheme == .dark ? Color.white.opacity(0.09) : Color.black.opacity(0.06)
        }
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
