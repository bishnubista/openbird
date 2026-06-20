import AppKit
import SwiftUI

struct MenuBarView: View {
    @ObservedObject var model: AppModel
    @Environment(\.openWindow) private var openWindow
    /// Summon the Spotlight Ask panel (also bound to the global ⌥Space hotkey).
    var openAskPanel: () -> Void

    var body: some View {
        Button("Open OpenBird") {
            openWindow(id: "main")
            NSApp.activate(ignoringOtherApps: true)
        }

        // The ⌥Space hint is in the title (not a `.keyboardShortcut`) so it cannot
        // double-fire alongside the global Carbon hotkey when the app is active.
        Button("Ask OpenBird…  ⌥Space") {
            openAskPanel()
        }

        Divider()

        if model.captureRunning {
            Button("Stop Capture") { model.stopCapture() }
        } else {
            Button("Start Capture") { model.startCapture() }
                .disabled(model.allowlist.isEmpty)
        }

        Button(model.capturePaused ? "Resume Capture" : "Pause Capture") {
            model.toggleCapturePause()
        }

        Button("Re-check Setup") {
            Task { await model.refresh() }
        }

        Divider()

        Text(statusLine)
        Text(model.memorySummary)
        ForEach(model.helpers) { helper in
            Text(helper.isBundled ? "\(helper.label): OK" : "\(helper.label): Missing")
        }

        Divider()

        Button("Data Folder") { model.openDataFolder() }
        Button("Quit") { model.quit() }
    }

    private var statusLine: String {
        if model.isFullyConfigured {
            return model.captureRunning ? "Capture running" : "Ready"
        }
        return "Setup incomplete"
    }
}
