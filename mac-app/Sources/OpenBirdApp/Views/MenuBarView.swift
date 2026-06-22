import AppKit
import SwiftUI

/// Native menu-bar menu. Keep this as plain menu content rather than a custom
/// `.window` popover so the status item behaves like older Homebrew builds:
/// visible icon, click opens a standard menu, click outside dismisses.
struct MenuBarView: View {
    @ObservedObject var model: AppModel
    @Environment(\.openWindow) private var openWindow
    /// Summon the Spotlight Ask panel (also bound to the global Option-Space hotkey).
    var openAskPanel: () -> Void

    var body: some View {
        Group {
            Button("Open OpenBird") {
                openMainWindow()
            }

            Button("Ask OpenBird...") {
                openAskPanel()
            }

            Divider()

            Text(statusLine)
            Text(model.memorySummary)

            if model.captureRunning {
                Button(model.capturePaused ? "Resume Capture" : "Pause Capture") {
                    model.toggleCapturePause()
                }
            } else {
                Button("Start Capture") {
                    model.startCapture()
                }
                .disabled(model.allowlist.isEmpty)
            }

            Button("Today's Activity") {
                openWindow(id: "today")
                NSApp.activate(ignoringOtherApps: true)
            }

            Divider()

            ForEach(model.helpers) { helper in
                Text(helper.isBundled ? "\(helper.label): OK" : "\(helper.label): Missing")
            }
            Text("Encryption at rest: \(encryptionStatus)")

            Divider()

            Button("Re-check Setup") {
                Task { await model.refresh() }
            }
            Button("Data Folder") {
                model.openDataFolder()
            }
            Button("About OpenBird") {
                showAboutPanel()
            }
            Button("Settings...") {
                openMainWindow()
            }
            .keyboardShortcut(",", modifiers: .command)

            Divider()

            Button("Quit OpenBird") {
                model.quit()
            }
            .keyboardShortcut("q", modifiers: .command)
        }
        .onAppear {
            model.refreshPermissionStates()
        }
    }

    private func openMainWindow() {
        openWindow(id: "main")
        NSApp.activate(ignoringOtherApps: true)
    }

    private func showAboutPanel() {
        NSApp.activate(ignoringOtherApps: true)
        NSApp.orderFrontStandardAboutPanel(options: [:])
    }

    private var statusLine: String {
        if model.capturePaused { return "Paused" }
        if model.captureRunning { return "Capturing" }
        return model.isFullyConfigured ? "Ready" : "Setup incomplete"
    }

    private var encryptionStatus: String {
        switch model.encryptionState {
        case .ok: return "on"
        case .attention: return "off"
        default: return "unknown"
        }
    }
}
