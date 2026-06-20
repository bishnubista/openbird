import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Resolve the app-owned DB key and export OPENBIRD_DB_KEY before any CLI
        // child is spawned (the first spawn is the main Window's refresh .task,
        // which runs after launch). This makes the Keychain prompt read "OpenBird"
        // and stops the Python layer from prompting. See KeychainKeyProvider.
        OpenBirdService.bootstrapDBKey()
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
}

@main
struct OpenBirdApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var model = AppModel(service: OpenBirdService())

    var body: some Scene {
        Window("OpenBird", id: "main") {
            ContentView(model: model)
                .frame(minWidth: 560, minHeight: 420)
                .task {
                    await model.refresh()
                }
        }

        MenuBarExtra("OpenBird", systemImage: model.menuBarSymbol) {
            MenuBarView(model: model)
        }
    }
}
