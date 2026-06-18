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

        Button(model.capturePaused ? "Resume Capture" : "Pause Capture") {
            model.toggleCapturePause()
        }

        Button("Refresh Status") {
            Task { await model.refresh() }
        }

        Divider()

        Text(shortStatus)
        ForEach(model.helpers) { helper in
            Text(helper.isBundled ? "\(helper.label): OK" : "\(helper.label): Missing")
        }

        Divider()

        Button("Stop Helpers") {
            model.stopHelpers()
        }

        Button("Data Folder") {
            model.openDataFolder()
        }

        Divider()

        Button("Quit") {
            NSApplication.shared.terminate(nil)
        }
    }

    private var shortStatus: String {
        if model.preflight.status.count <= 30 {
            return model.preflight.status
        }
        return String(model.preflight.status.prefix(27)) + "..."
    }
}
