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
    @StateObject private var model: AppModel
    @StateObject private var askPanel: AskPanelController
    @StateObject private var todayModel: TodayModel
    @StateObject private var timelineModel: TimelineModel
    /// Per-window chat thread for the Timeline window (Direction C). Built once here
    /// sharing the one service; the thread persists across window reopen and is reset
    /// on day change inside the view.
    @StateObject private var timelineChat: AskPanelModel
    /// Per-window chat thread for the Sources window (Direction B). Window-local — no
    /// history shared with the Spotlight panel or the Timeline window.
    @StateObject private var sourcesChat: AskPanelModel

    /// Construct the service, model, Ask-panel controller, and Today model TOGETHER,
    /// sharing the one `AppModel`/`OpenBirdService`, so there is never a second model
    /// instance and no configure-after-launch ordering race (see the design doc).
    init() {
        let service = OpenBirdService()
        let model = AppModel(service: service)
        _model = StateObject(wrappedValue: model)
        _askPanel = StateObject(wrappedValue: AskPanelController(model: model, service: service))
        _todayModel = StateObject(wrappedValue: TodayModel(service: service))
        _timelineModel = StateObject(wrappedValue: TimelineModel(service: service))
        _timelineChat = StateObject(wrappedValue: AskPanelModel(service: service, appModel: model))
        _sourcesChat = StateObject(wrappedValue: AskPanelModel(service: service, appModel: model))
    }

    var body: some Scene {
        Window("OpenBird", id: "main") {
            ContentView(model: model, onAsk: { askPanel.show() })
                .frame(minWidth: 560, minHeight: 420)
                .task {
                    askPanel.installHotKeyIfNeeded()   // idempotent ⌥Space registration
                    await model.refresh()
                }
                .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
                    model.refreshPermissionStates()
                }
        }

        Window("Today", id: "today") {
            TodayView(model: todayModel, appModel: model, onAsk: { askPanel.show() })
        }

        Window("Timeline", id: "timeline") {
            TimelineAskView(model: timelineModel, chat: timelineChat)
        }
        .windowStyle(.hiddenTitleBar)

        Window("Sources", id: "sources") {
            AskSourcesView(chat: sourcesChat)
        }
        .windowStyle(.hiddenTitleBar)

        MenuBarExtra("OpenBird", systemImage: model.menuBarSymbol) {
            MenuBarView(model: model, openAskPanel: { askPanel.show() })
        }
    }
}
