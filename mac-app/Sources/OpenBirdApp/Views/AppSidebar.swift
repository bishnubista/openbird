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
        // Two lines (handoff sidebar footer): a bold status word with the pulsing dot on
        // top, then the dimmed detail line below — not one run-on caption.
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: OB.Space.sm) {
                FooterPulseDot(active: appModel.captureRunning && !appModel.capturePaused)
                Text(footerStatus)
                    .font(.system(size: 12.5, weight: .semibold))
                    .foregroundStyle(OB.textPrimary(scheme))
            }
            Text(footerDetail)
                .font(.system(size: 11))
                .foregroundStyle(OB.textTertiary(scheme))
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, OB.Space.ml)
        .padding(.vertical, OB.Space.m)
        .overlay(alignment: .top) {
            Rectangle().fill(OB.separator(scheme)).frame(height: 0.5)
        }
    }

    /// The bold status word: the live capture state (handoff "Capturing").
    private var footerStatus: String {
        appModel.capturePaused
            ? "Paused"
            : (appModel.captureRunning ? "Capturing" : "Idle")
    }

    /// Real footer detail line: allowlist size, model route, and the actual at-rest
    /// encryption state. Words are derived from state, not hardcoded, so the footer never
    /// claims local-only behavior for a remote route.
    private var footerDetail: String {
        let encryption: String
        switch appModel.encryptionState {
        case .ok: encryption = "encrypted"
        case .attention: encryption = "encryption off"
        default: encryption = "encryption unknown"
        }
        return "\(appModel.captureAllowedSummary) · \(appModel.modelRouteFooterLabel) · \(encryption)"
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
            // Red while capturing (handoff #ff453a); a neutral dot otherwise.
            .fill(active ? OB.capturingDot : Color.secondary)
            .frame(width: 7, height: 7)
            // handoff `obPulse`: opacity 1→.35, scale 1→.85 over 1.7s, ease-in-out.
            .scaleEffect(pulsing ? 0.85 : 1.0)
            .opacity(pulsing ? 0.35 : 1.0)
            .onAppear { apply() }
            .onChange(of: active) { _ in apply() }
    }

    private func apply() {
        if active {
            withAnimation(.easeInOut(duration: 0.85).repeatForever(autoreverses: true)) {
                pulsing = true
            }
        } else {
            withAnimation(.default) { pulsing = false }
        }
    }
}
