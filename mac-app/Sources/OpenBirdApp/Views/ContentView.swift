import SwiftUI

struct ContentView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                HeaderView(model: model)
                SetupView(model: model)
                Divider()
                ChatView(model: model)
                Divider()
                HelperListView(helpers: model.helpers)
                TrustControlsView(model: model)
            }
            .padding(24)
        }
        .frame(minWidth: 560, minHeight: 560)
    }
}

private struct HeaderView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Image(systemName: "bird")
                    .font(.largeTitle)
                    .foregroundStyle(.tint)
                VStack(alignment: .leading, spacing: 2) {
                    Text("OpenBird")
                        .font(.title)
                    Text(model.isFullyConfigured ? "Ready to capture" : "Finish setup below")
                        .font(.subheadline)
                        .foregroundStyle(model.isFullyConfigured ? .green : .secondary)
                }
                Spacer()
            }
            if let lastRefresh = model.lastRefresh {
                Text("Last checked \(lastRefresh.formatted(date: .omitted, time: .shortened))")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            Label(model.nextStepSummary, systemImage: nextStepIcon)
                .font(.callout)
                .foregroundStyle(nextStepColor)
                .fixedSize(horizontal: false, vertical: true)
        }
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
        case .ok: return .green
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
