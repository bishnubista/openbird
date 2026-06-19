import AppKit
import SwiftUI

struct MenuBarView: View {
    @ObservedObject var model: AppModel
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Button("Open OpenBird") {
            openWindow(id: "main")
            NSApp.activate(ignoringOtherApps: true)
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
        ForEach(model.helpers) { helper in
            Text(helper.isBundled ? "\(helper.label): OK" : "\(helper.label): Missing")
        }

        Divider()

        Button("Data Folder") { model.openDataFolder() }
        Button("Quit") { model.quit() }
    }

    private var statusLine: String {
        if model.isFullyConfigured {
            return model.captureRunning ? "Capturing" : "Ready"
        }
        return "Setup incomplete"
    }
}
