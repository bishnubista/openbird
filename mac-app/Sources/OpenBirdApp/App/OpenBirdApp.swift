import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
}

@main
struct OpenBirdApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var model = AppModel(service: OpenBirdService())

    var body: some Scene {
        WindowGroup("OpenBird", id: "main") {
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
