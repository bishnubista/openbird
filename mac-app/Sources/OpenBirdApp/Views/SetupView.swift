import SwiftUI

/// The guided setup checklist: every step needed to make OpenBird capture work
/// on this Mac, each with a one-click action. macOS forbids an app from granting
/// its own TCC permissions, so permission steps deep-link into System Settings
/// and then re-check via the bundled helper probe.
struct SetupView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Guided Setup")
                    .font(.headline)
                Spacer()
                if model.isFullyConfigured {
                    Label("Ready", systemImage: "checkmark.seal.fill")
                        .foregroundStyle(.green)
                }
                Button {
                    Task { await model.refresh() }
                } label: {
                    Label("Re-check", systemImage: "arrow.clockwise")
                }
                .disabled(model.isRefreshing)
            }

            if let working = model.workingMessage {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(working).font(.callout).foregroundStyle(.secondary)
                }
            }

            VStack(spacing: 0) {
                StepRow(
                    state: model.ollamaState,
                    title: "Ollama running",
                    detail: ollamaDetail,
                    actionLabel: model.ollamaState == .ok ? nil : "Get Ollama",
                    action: { NSWorkspace.shared.open(URL(string: "https://ollama.com")!) }
                )
                Divider()
                StepRow(
                    state: model.modelsState,
                    title: "Local models",
                    detail: modelsDetail,
                    actionLabel: model.report.missingModels.isEmpty ? nil : "Pull models",
                    action: { Task { await model.pullMissingModels() } }
                )
                Divider()
                StepRow(
                    state: model.encryptionState,
                    title: "Encrypted memory",
                    detail: encryptionDetail,
                    actionLabel: nil,
                    action: {}
                )
                Divider()
                StepRow(
                    state: model.accessibilityState,
                    title: "Accessibility permission",
                    detail: "Lets the capture helper read active-window text. Click Grant, then approve OpenBird in the prompt.",
                    actionLabel: model.accessibilityState == .ok ? nil : "Grant",
                    action: { model.requestAccessibility() }
                )
                Divider()
                StepRow(
                    state: model.screenRecordingState,
                    title: "Screen Recording permission",
                    detail: "Optional · needed only for meeting/system-audio capture.",
                    actionLabel: model.screenRecordingState == .ok ? nil : "Grant",
                    optional: true,
                    action: { model.requestScreenRecording() }
                )
                Divider()
                StepRow(
                    state: model.microphoneState,
                    title: "Microphone permission",
                    detail: "Optional · records your side of a meeting as a separate track.",
                    actionLabel: model.microphoneState == .ok ? nil : "Grant",
                    optional: true,
                    action: { model.requestMicrophone() }
                )
            }
            .padding(.vertical, 4)
            .background(RoundedRectangle(cornerRadius: 10).fill(Color(NSColor.controlBackgroundColor)))

            AllowlistEditor(model: model)
            CaptureControls(model: model)

            if !model.lastActionMessage.isEmpty {
                Text(model.lastActionMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var ollamaDetail: String {
        switch model.ollamaState {
        case .ok: return "Reachable at the local Ollama endpoint."
        case .attention: return "Not reachable. Install and launch Ollama, then re-check."
        default: return "Status unknown — re-check after launching Ollama."
        }
    }

    private var modelsDetail: String {
        if model.report.missingModels.isEmpty && model.modelsState == .ok {
            return "All required models present."
        }
        if !model.report.missingModels.isEmpty {
            return "Missing: \(model.report.missingModels.joined(separator: ", "))"
        }
        return "Start Ollama to verify required models."
    }

    private var encryptionDetail: String {
        switch model.encryptionState {
        case .ok: return "At-rest SQLCipher encryption is active."
        case .attention:
            return "Running plaintext (0600 file). Install the encryption extra "
                + "(bundled with the Homebrew build) to enable SQLCipher."
        default: return "Encryption status could not be verified."
        }
    }
}

/// One checklist row: status glyph, title/detail, optional action button.
private struct StepRow: View {
    let state: StepState
    let title: String
    let detail: String
    let actionLabel: String?
    var optional: Bool = false
    let action: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: glyph)
                .foregroundStyle(color)
                .font(.title3)
                .frame(width: 22)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if let label = actionLabel {
                Button(label, action: action)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
    }

    private var glyph: String {
        switch state {
        case .ok: return "checkmark.circle.fill"
        // Optional steps that aren't done are not problems — show a neutral glyph
        // so they never read as an error next to a "Ready" badge.
        case .attention: return optional ? "circle" : "exclamationmark.triangle.fill"
        case .working: return "clock.fill"
        case .unknown: return optional ? "circle" : "questionmark.circle"
        }
    }

    private var color: Color {
        switch state {
        case .ok: return .green
        case .attention: return optional ? .secondary : .orange
        case .working: return .blue
        case .unknown: return .secondary
        }
    }
}

/// Edit the per-app capture allowlist. Capture records nothing until at least one
/// app is allowed (allowlist-first privacy), so this is part of setup.
private struct AllowlistEditor: View {
    @ObservedObject var model: AppModel
    @State private var newBundleID = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Capture allowlist")
                .font(.headline)
            Text("OpenBird captures text only from apps you list here. Nothing is recorded otherwise.")
                .font(.caption)
                .foregroundStyle(.secondary)

            if model.allowlist.isEmpty {
                Text("No apps allowed yet — capture will record nothing.")
                    .font(.caption)
                    .foregroundStyle(.orange)
            } else {
                ForEach(model.allowlist, id: \.self) { bundleID in
                    HStack {
                        Image(systemName: "app.dashed")
                        Text(bundleID).font(.callout)
                        Spacer()
                        Button(role: .destructive) {
                            model.removeFromAllowlist(bundleID)
                        } label: {
                            Image(systemName: "minus.circle")
                        }
                        .buttonStyle(.borderless)
                    }
                }
            }

            HStack {
                TextField("Bundle id (e.g. com.tinyspeck.slackmacgap)", text: $newBundleID)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(addEntry)
                Button("Add", action: addEntry)
                    .disabled(newBundleID.trimmingCharacters(in: .whitespaces).isEmpty)
            }

            let suggestions = model.runningAppSuggestions()
            if !suggestions.isEmpty {
                Text("Running apps")
                    .font(.caption).foregroundStyle(.secondary)
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack {
                        ForEach(suggestions.prefix(12), id: \.self) { id in
                            Button {
                                model.addToAllowlist(id)
                            } label: {
                                Text(id).font(.caption)
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                }
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color(NSColor.controlBackgroundColor)))
    }

    private func addEntry() {
        model.addToAllowlist(newBundleID)
        newBundleID = ""
    }
}

/// Start/stop the capture daemon directly from the app.
private struct CaptureControls: View {
    @ObservedObject var model: AppModel

    var body: some View {
        HStack(spacing: 12) {
            if model.captureRunning {
                Button {
                    model.stopCapture()
                } label: {
                    Label("Stop Capture", systemImage: "stop.circle.fill")
                }
                .tint(.red)
                Label("Capturing", systemImage: "dot.radiowaves.left.and.right")
                    .foregroundStyle(.green)
            } else {
                Button {
                    model.startCapture()
                } label: {
                    Label("Start Capture", systemImage: "play.circle.fill")
                }
                .disabled(model.allowlist.isEmpty)
            }
            Spacer()
            Button {
                model.openDataFolder()
            } label: {
                Label("Data Folder", systemImage: "folder")
            }
        }
    }
}
