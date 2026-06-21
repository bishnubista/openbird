import SwiftUI

/// The 222px Today-window sidebar (handoff §3): the OpenBird wordmark, the section
/// nav, and a live capture-status footer. Only **Today** has a destination today,
/// so the other nav rows are rendered pixel-faithfully but are honestly inert —
/// `.disabled(true)` + an accessibility "not enabled" trait — rather than looking
/// clickable while doing nothing (macOS HIG / Codex review).
struct TodaySidebar: View {
    @ObservedObject var appModel: AppModel
    @Environment(\.colorScheme) private var scheme

    /// One nav entry. `enabled == false` rows are present-but-inert placeholders for
    /// sections that have no window yet.
    private struct NavItem {
        let icon: String
        let label: String
        let enabled: Bool
    }

    private let items: [NavItem] = [
        .init(icon: "calendar", label: "Today", enabled: true),
        .init(icon: "list.bullet", label: "Timeline", enabled: false),
        .init(icon: "waveform", label: "Meetings", enabled: false),
        .init(icon: "arrow.triangle.2.circlepath", label: "Routines", enabled: false),
        .init(icon: "magnifyingglass", label: "Search", enabled: false),
        .init(icon: "slider.horizontal.3", label: "Settings", enabled: false),
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
                navRow(items[i])
            }
        }
        .padding(.horizontal, OB.Space.m)
    }

    @ViewBuilder
    private func navRow(_ item: NavItem) -> some View {
        let isActive = item.enabled && item.label == "Today"
        HStack(spacing: OB.Space.sm) {
            Image(systemName: item.icon)
                .frame(width: 18)
            Text(item.label)
                .font(.system(size: 13.5, weight: isActive ? .semibold : .regular))
            Spacer()
        }
        .foregroundStyle(navForeground(item, isActive: isActive))
        .padding(.horizontal, OB.Space.m)
        .padding(.vertical, OB.Space.sm)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: OB.Radius.control)
                .fill(isActive ? OB.accent : .clear)
        )
        // Inert rows are plain (non-Button) text with no hover/selection/action —
        // honestly non-interactive, not a clickable-looking no-op. Today is the
        // only live destination. `.disabled` is a belt-and-suspenders signal.
        .disabled(!item.enabled)
    }

    private func navForeground(_ item: NavItem, isActive: Bool) -> Color {
        if isActive { return .white }
        return item.enabled ? OB.textPrimary(scheme) : OB.textTertiary(scheme)
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
