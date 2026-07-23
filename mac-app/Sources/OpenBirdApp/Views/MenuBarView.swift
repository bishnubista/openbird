import AppKit
import SwiftUI

/// Native menu-bar menu. Keep this as plain `.menu` content rather than a custom
/// `.window` popover so the status item behaves like older Homebrew builds:
/// visible icon, click opens a standard menu, click outside dismisses.
///
/// The Liquid Glass handoff specifies a 288px glass dropdown (rounded, accent-hover
/// rows). That is **intentionally deferred** — native macOS menus can't be skinned to
/// glass, and the only way to render custom chrome is to drop `MenuBarExtra` for a
/// hand-owned `NSStatusItem` + `NSPanel`. That reopens menu-bar status-item/icon
/// stability that was just fixed (#51 added a glass window-style menu; #57 reverted it
/// to the native menu), and its correctness depends on runtime behavior (full menu bar,
/// multi-display, off-screen item, click-away/Escape/VoiceOver) that needs hands-on GUI
/// verification. See `docs/design/menu-dropdown-deferral.md`. This file keeps the menu
/// content aligned to the handoff where the native menu allows (item order, the ⌘P
/// Pause shortcut); the glass chrome is a separate, soak-tested change.
struct MenuBarView: View {
    @ObservedObject var model: AppModel
    @Environment(\.openWindow) private var openWindow
    /// Summon the Spotlight Ask panel (also bound to the global Option-Space hotkey).
    var openAskPanel: () -> Void

    var body: some View {
        Group {
            // Primary destinations — mirrored with the Today sidebar via AppDestination.
            Button(AppDestination.ask.title) {
                openAskPanel()
            }

            Button(AppDestination.today.title) {
                showMainWindow(.today)
            }

            Button(AppDestination.setup.title) {
                showMainWindow(.setup)
            }
            .keyboardShortcut(",", modifiers: .command)

            Divider()

            Text("Status: \(statusLine)")
            Text("Memory: \(model.memorySummary)")

            if model.captureRunning {
                Button(model.capturePaused ? "Resume Capture" : "Pause Capture") {
                    model.toggleCapturePause()
                }
                .keyboardShortcut("p", modifiers: .command)   // handoff: Pause Capture ⌘P
            } else {
                Button("Start Capture") {
                    model.startCapture()
                }
                .disabled(model.allowlist.isEmpty)
            }

            switch model.meetingState {
            case .idle:
                Button("Start Meeting Recording") {
                    showMainWindow(.today)
                    model.requestStartMeeting()
                }
                .disabled(!model.meetingCanStart)
            case .consent:
                Button("Continue Meeting Setup…") { showMainWindow(.today) }
            case .preparing:
                Button("Cancel Meeting Preparation") { model.cancelMeetingAction() }
            case .recording:
                Button("Stop Meeting Recording") { model.stopMeetingRecording() }
            case .finalizing:
                Text("Meeting: finalizing…")
            }
            if model.meetingState.isBusy {
                Button("Force Stop Meeting Audio") { model.forceStopMeetingAudio() }
            }

            Divider()

            ForEach(model.helpers) { helper in
                Text(helper.isBundled ? "\(helper.label): present" : "\(helper.label): missing")
            }
            Text("Encryption: \(encryptionStatus)")

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

    /// Select a pane and bring the single app window forward. For a `Window` (singleton)
    /// scene, `openWindow(id:"main")` focuses the existing window — or RE-CREATES it if
    /// the user previously closed it (which `NSApp.activate` alone cannot do, Codex
    /// review) — and never spawns a duplicate. Setting `selection` first means the
    /// window comes up already showing the requested pane.
    private func showMainWindow(_ destination: AppDestination) {
        model.selection = destination
        openWindow(id: "main")
        NSApp.activate(ignoringOtherApps: true)
    }

    private func showAboutPanel() {
        NSApp.activate(ignoringOtherApps: true)
        NSApp.orderFrontStandardAboutPanel(options: [:])
    }

    private var statusLine: String {
        if model.meetingState.isRecording { return "Recording meeting" }
        if model.capturePaused { return "Paused" }
        if model.isFullyConfigured { return "Ready" }
        if model.captureRunning { return "Capturing" }
        return "Setup incomplete"
    }

    private var encryptionStatus: String {
        switch model.encryptionState {
        case .ok: return "on"
        case .attention: return "off"
        default: return "unknown"
        }
    }
}
