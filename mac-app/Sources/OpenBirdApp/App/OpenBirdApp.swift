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
    /// The ONE Ask conversation, shared by the compact Spotlight panel and the expanded
    /// Ask window (both owned by `askPanel`). Lifted here so the thread is continuous
    /// across both surfaces.
    @StateObject private var askModel: AskPanelModel
    @StateObject private var askPanel: AskPanelController
    @StateObject private var todayModel: TodayModel
    @StateObject private var timelineModel: TimelineModel

    /// Construct the service, model, Ask-panel controller, and Today model TOGETHER,
    /// sharing the one `AppModel`/`OpenBirdService`, so there is never a second model
    /// instance and no configure-after-launch ordering race (see the design doc).
    init() {
        let service = OpenBirdService()
        let model = AppModel(service: service)
        let askModel = AskPanelModel(service: service, appModel: model)
        let timelineModel = TimelineModel(service: service)
        let todayModel = TodayModel(service: service)
        // Wire citation click-through: a clicked source switches to the Today pane
        // (done in AppModel) and focuses the citation's day/observation here. Strong
        // captures are fine — these models live for the app's lifetime.
        model.citationNavigator = { dayOffset, observationId in
            Task { await todayModel.focus(dayOffset: dayOffset, observationId: observationId) }
        }
        _model = StateObject(wrappedValue: model)
        _askModel = StateObject(wrappedValue: askModel)
        _todayModel = StateObject(wrappedValue: todayModel)
        _timelineModel = StateObject(wrappedValue: timelineModel)
        _askPanel = StateObject(wrappedValue: AskPanelController(
            model: model, service: service, askModel: askModel, timelineModel: timelineModel))
    }

    var body: some Scene {
        // ONE window for the whole app: the shell renders Ask / Today / Setup as
        // selectable detail panes (no second `Window` scene). Today's surface is now a
        // pane, reached by setting `model.selection`, never by opening another window.
        Window("OpenBird", id: "main") {
            MainWindowRoot(
                model: model,
                todayModel: todayModel,
                askModel: askModel,
                timelineModel: timelineModel,
                askPanel: askPanel
            )
        }
        .windowStyle(.hiddenTitleBar)

        MenuBarExtra("OpenBird", systemImage: model.menuBarSymbol) {
            MenuBarView(model: model, openAskPanel: { askPanel.show() })
        }
    }
}

/// The main window's root view. Exists so it can read `@Environment(\.openWindow)`
/// (only reachable from inside a `View`) and hand the controller a way to RE-OPEN the
/// main window — needed when a citation is clicked from an Ask overlay after the user
/// closed the main window (the MenuBarExtra keeps the app alive with no window).
private struct MainWindowRoot: View {
    @ObservedObject var model: AppModel
    @ObservedObject var todayModel: TodayModel
    @ObservedObject var askModel: AskPanelModel
    @ObservedObject var timelineModel: TimelineModel
    let askPanel: AskPanelController

    @Environment(\.openWindow) private var openWindow

    var body: some View {
        AppShellView(
            model: model,
            todayModel: todayModel,
            askModel: askModel,
            timelineModel: timelineModel,
            onAsk: { day in askPanel.show(dayScope: day) },
            onAskExpanded: { askPanel.expand() }
        )
        .task {
            askPanel.installHotKeyIfNeeded()   // idempotent ⌥Space registration
            askPanel.openMainWindow = { openWindow(id: "main") }
            await model.refresh()
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            model.refreshPermissionStates()
        }
    }
}
