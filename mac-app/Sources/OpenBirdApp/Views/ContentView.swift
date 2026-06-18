import SwiftUI

struct ContentView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        NavigationSplitView {
            List {
                Label("Status", systemImage: "bird")
                Label("Helpers", systemImage: "wrench.and.screwdriver")
                Label("Privacy", systemImage: "lock.shield")
            }
            .listStyle(.sidebar)
            .navigationTitle("OpenBird")
        } detail: {
            VStack(alignment: .leading, spacing: 24) {
                HeaderView(model: model)
                HelperListView(helpers: model.helpers)
                PrivacyControlsView(model: model)
                Spacer()
            }
            .padding(24)
        }
    }
}

private struct HeaderView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("OpenBird")
                        .font(.title)
                    Text(model.preflight.status)
                        .foregroundStyle(model.preflight.status == "Runtime Ready" ? .green : .secondary)
                }
                Spacer()
                Button {
                    Task { await model.refresh() }
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .disabled(model.isRefreshing)
            }

            Text(model.preflight.detail)
                .foregroundStyle(.secondary)

            if let lastRefresh = model.lastRefresh {
                Text("Last checked \(lastRefresh.formatted(date: .omitted, time: .shortened))")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
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

private struct PrivacyControlsView: View {
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
                    model.openDataFolder()
                } label: {
                    Label("Data Folder", systemImage: "folder")
                }
                Button {
                    model.openBundleFolder()
                } label: {
                    Label("App Bundle", systemImage: "app")
                }
            }
            if !model.lastActionMessage.isEmpty {
                Text(model.lastActionMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}
