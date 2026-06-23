import SwiftUI

/// The Setup detail pane — guided setup, packaged-helper status, and trust controls,
/// rendered in the app shell's detail column for the `.setup` destination. This view is
/// now ONLY the Setup content: the window-level concerns it used to own (min-frame,
/// `GlassBackdrop`, the background drag region, the onboarding `.sheet`, and the gated
/// `openbird://` deep-link router) moved to the always-mounted `AppShellView` so they
/// survive whichever pane is selected (Codex review — `onOpenURL` here would unmount
/// when the user is on Today or Ask).
struct ContentView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: OB.Space.l) {
                HeaderView(model: model)
                GlassCard { SetupView(model: model) }
                GlassCard { HelperListView(helpers: model.helpers) }
                GlassCard { TrustControlsView(model: model) }
            }
            .padding(OB.Space.xl)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
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
