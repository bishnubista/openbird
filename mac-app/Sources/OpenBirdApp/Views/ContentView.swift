import SwiftUI

struct ContentView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: OB.Space.l) {
                HeaderView(model: model)
                GlassCard { SetupView(model: model) }
                GlassCard { ChatView(model: model) }
                GlassCard { HelperListView(helpers: model.helpers) }
                GlassCard { TrustControlsView(model: model) }
            }
            .padding(OB.Space.xl)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(minWidth: 560, minHeight: 560)
        .background(WindowBackdrop())
    }
}

/// A Liquid Glass section card: padded content on the shared glass material. Gives
/// the window the same material language as the Spotlight panel.
private struct GlassCard<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        content
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(OB.Space.l)
            .glassSurface(cornerRadius: OB.Radius.card)
    }
}

/// Opaque-window backdrop: a soft base plus the handoff's blurred color "orbs" so
/// the glass cards have something colorful to refract (a window has no desktop
/// behind it, unlike the floating panel — so here the orbs are intentional, not
/// fake). Kept low-opacity for legibility in both appearances.
private struct WindowBackdrop: View {
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        ZStack {
            (scheme == .dark ? Color(hex: 0x1B1B20) : Color(hex: 0xF2F2F5))
            orb(Color(hex: 0x785AFF), x: 0.12, y: 0.08, size: 460)   // purple
            orb(Color(hex: 0x2F7FF2), x: 0.92, y: 0.30, size: 520)   // blue
            orb(Color(hex: 0xFF78AA), x: 0.70, y: 0.95, size: 420)   // pink
        }
        .ignoresSafeArea()
    }

    private func orb(_ color: Color, x: CGFloat, y: CGFloat, size: CGFloat) -> some View {
        GeometryReader { geo in
            RadialGradient(
                gradient: Gradient(colors: [color.opacity(scheme == .dark ? 0.30 : 0.18), .clear]),
                center: .center, startRadius: 0, endRadius: size / 2
            )
            .frame(width: size, height: size)
            .position(x: geo.size.width * x, y: geo.size.height * y)
            .blur(radius: 20)
        }
    }
}

private struct HeaderView: View {
    @ObservedObject var model: AppModel
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        VStack(alignment: .leading, spacing: OB.Space.sm) {
            HStack(spacing: OB.Space.m) {
                BirdLogo()
                    .fill(OB.accent)
                    .frame(width: 34, height: 34)
                VStack(alignment: .leading, spacing: 2) {
                    Text("OpenBird")
                        .font(.system(size: 21, weight: .bold))
                        .foregroundStyle(OB.textPrimary(scheme))
                    Text(model.isFullyConfigured ? "Ready to capture" : "Finish setup below")
                        .font(.subheadline)
                        .foregroundStyle(model.isFullyConfigured ? OB.ok(scheme) : OB.textSecondary(scheme))
                }
                Spacer()
            }
            if let lastRefresh = model.lastRefresh {
                Text("Last checked \(lastRefresh.formatted(date: .omitted, time: .shortened))")
                    .font(.caption)
                    .foregroundStyle(OB.textTertiary(scheme))
            }
            Label(model.nextStepSummary, systemImage: nextStepIcon)
                .font(.callout)
                .foregroundStyle(nextStepColor)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, OB.Space.xs)
    }

    private var nextStepIcon: String {
        switch model.nextStepState {
        case .ok: return "checkmark.circle.fill"
        case .attention: return "arrow.right.circle.fill"
        case .working: return "arrow.clockwise.circle.fill"
        case .unknown: return "questionmark.circle"
        }
    }

    private var nextStepColor: Color {
        switch model.nextStepState {
        case .ok: return OB.ok(scheme)
        case .attention: return .orange
        case .working: return .blue
        case .unknown: return .secondary
        }
    }
}

private struct HelperListView: View {
    let helpers: [HelperStatus]

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Packaged Helpers")
                .font(.headline)
            ForEach(helpers) { helper in
                HStack {
                    Image(systemName: helper.isBundled ? "checkmark.circle.fill" : "xmark.circle")
                        .foregroundStyle(helper.isBundled ? .green : .red)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(helper.label)
                        Text(helper.isBundled ? helper.path : "Missing from app bundle")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                    Spacer()
                }
            }
        }
    }
}

private struct TrustControlsView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Trust Controls")
                .font(.headline)
            Toggle("Pause capture", isOn: Binding(
                get: { model.capturePaused },
                set: { _ in model.toggleCapturePause() }
            ))
            HStack {
                Button {
                    model.stopHelpers()
                } label: {
                    Label("Stop Helpers", systemImage: "stop.fill")
                }
                Button {
                    model.openBundleFolder()
                } label: {
                    Label("App Bundle", systemImage: "app")
                }
            }
        }
    }
}
