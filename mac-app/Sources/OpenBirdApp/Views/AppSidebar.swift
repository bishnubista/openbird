import SwiftUI

/// The 222px app shell sidebar (handoff §3): the OpenBird wordmark, the section nav,
/// and a live capture-status footer. Every row is now a real navigation destination —
/// selecting one swaps the single window's detail pane via `appModel.selection` rather
/// than opening a separate window (Ask, Today, and Setup all live in one window now).
struct AppSidebar: View {
    @ObservedObject var appModel: AppModel
    @Environment(\.colorScheme) private var scheme

    /// Switch the in-window detail pane. The sidebar mirrors the menu bar via the
    /// shared `AppDestination` set; the active row is whichever pane is selected.
    private func select(_ destination: AppDestination) {
        appModel.selection = destination
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
                    isActive: dest == appModel.selection,
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

    /// Real footer status: capture state, allowlist size, model route, and the
    /// actual at-rest encryption state. Privacy words are derived from state, not
    /// hardcoded, so the footer never claims local-only behavior for a remote route.
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
        return "\(state) · \(apps) · \(appModel.modelRouteFooterLabel) · \(encryption)"
    }

    private var sidebarFill: Color {
        scheme == .dark ? Color.white.opacity(0.03) : Color.black.opacity(0.02)
    }
}

/// One sidebar nav row. The active row (the selected pane) is a non-interactive accent
/// highlight; every other row is a real button that switches the detail pane on click
/// (hover = subtle wash, per the handoff).
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
