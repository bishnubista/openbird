import AppKit
import SwiftUI

struct ContentView: View {
    @ObservedObject var model: AppModel
    /// Summon the compact Spotlight Ask panel (for the `openbird://ask` deep-link).
    var onAsk: () -> Void = {}
    /// Open the expanded Ask window (for the `openbird://ask-expanded` E2E deep-link).
    var onAskExpanded: () -> Void = {}
    /// One-time first-run flag; the onboarding sheet shows until the user taps
    /// "Start capturing" (or dismisses), then never auto-presents again.
    @AppStorage("openbird.onboarding.completed") private var onboardingCompleted = false
    @Environment(\.openWindow) private var openWindow

    /// Drives the onboarding `.sheet` from the inverse of the completed flag; the
    /// sheet (and its "Start capturing" button) sets it to dismiss.
    private var onboardingBinding: Binding<Bool> {
        Binding(get: { !onboardingCompleted }, set: { onboardingCompleted = !$0 })
    }

    var body: some View {
        VStack(spacing: 0) {
            // Clear strip beneath the overlaid traffic lights (the window now uses a
            // hidden titlebar). Doubles as a drag region, and — because the ScrollView
            // starts below it and clips to its bounds — scrolled content never slides
            // up under the traffic lights.
            Color.clear.frame(height: 28)
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
        .frame(minWidth: 560, minHeight: 560)
        .background(GlassBackdrop())
        .background(WindowConfigurator())   // draggable by background (macOS-13-safe)
        .sheet(isPresented: onboardingBinding) {
            OnboardingSheet(model: model, isPresented: onboardingBinding)
        }
        // Deep-link router: `openbird://today|main|ask` opens a surface without the
        // menu bar. Attached here because the main Window scene auto-opens at launch,
        // so this handler always exists to receive the URL.
        .onOpenURL { url in route(url) }
    }

    private func route(_ url: URL) {
        // TEST-ONLY affordance for the E2E screenshot harness — NOT a public product
        // surface. A routable openbird:// scheme would let any local app or web page
        // force OpenBird to foreground private activity data (Today) or steal focus
        // (Ask). So the router is inert unless the app was launched explicitly with
        // `--enable-e2e-deeplinks` (the harness passes it via `open --args`).
        guard ProcessInfo.processInfo.arguments.contains("--enable-e2e-deeplinks") else { return }
        guard url.scheme == "openbird" else { return }
        switch url.host {
        case "today": openWindow(id: "today")
        case "main": openWindow(id: "main")
        case "ask": onAsk()
        case "ask-expanded": onAskExpanded()
        default: return
        }
        NSApp.activate(ignoringOtherApps: true)
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
