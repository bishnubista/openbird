import AppKit
import os
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    private static let log = Logger(subsystem: "ai.openbird.OpenBird", category: "selftest")

    /// Detects the #169 main-thread layout feedback loop (CPU pin) and logs a privacy-safe
    /// reason code. Held for the process lifetime; nil when disabled (release builds without
    /// OPENBIRD_LAYOUT_WATCHDOG). Not installed in headless self-test mode.
    private var layoutWatchdog: LayoutLoopWatchdog?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // DEVELOP/SELF-TEST MODE: when OPENBIRD_SELFTEST_ASK=<query> is set, run that
        // ask headlessly through the SAME OpenBirdService.askChat seam the Ask panel
        // uses, emit a deterministic outcome (stdout + os_log, counts/booleans only —
        // never the query/answer/citation text), and exit. No window, no global
        // hotkey, no keystroke automation — so an automated verifier can validate the
        // real app's ask pipeline with NO Accessibility/Automation permission prompt
        // (the human bottleneck for self-testing). By default it resolves the
        // app-owned DB key so beta rehearsal exercises the encrypted real store.
        // Set OPENBIRD_DISABLE_KEYRING=1 only for prompt-free plaintext-dev DB runs.
        if let query = ProcessInfo.processInfo.environment["OPENBIRD_SELFTEST_ASK"],
           !query.isEmpty {
            // Stay fully headless: no dock icon / window, so the SwiftUI scene's
            // live work (refresh, auto-resume capture) never runs. The window's
            // `.task` ALSO guards on this env var as belt-and-suspenders, since the
            // ask runs off-thread and the scene could otherwise mount before exit.
            NSApp.setActivationPolicy(.prohibited)
            if Self.shouldBootstrapDBKeyForSelfTest() {
                Self.emitSelfTestSignal("SELFTEST db_key.bootstrap start")
                let ok = OpenBirdService.bootstrapDBKey(timeout: Self.selfTestDBKeyTimeout())
                Self.emitSelfTestSignal("SELFTEST db_key.bootstrap done ok=\(ok ? 1 : 0)")
                if !ok {
                    Self.emitSelfTestOutcomeAndExit(
                        "SELFTEST ask.outcome error=1 kind=db_key_unavailable",
                        code: 2
                    )
                    return
                }
            }
            Self.runSelfTestAndExit(query: query)
            return  // runSelfTestAndExit exits the process when the ask completes
        }
        // Resolve the app-owned DB key and export OPENBIRD_DB_KEY before any CLI
        // child is spawned (the first spawn is the main Window's refresh .task,
        // which runs after launch). This makes the Keychain prompt read "OpenBird"
        // and stops the Python layer from prompting. See KeychainKeyProvider.
        OpenBirdService.bootstrapDBKey()
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        startLayoutWatchdogIfEnabled()
    }

    /// Install the #169 layout-loop watchdog when enabled (DEBUG, or
    /// OPENBIRD_LAYOUT_WATCHDOG set). Reached only past the self-test early-return above,
    /// so the headless verifier never starts it.
    private func startLayoutWatchdogIfEnabled() {
        #if DEBUG
        let isDebug = true
        #else
        let isDebug = false
        #endif
        let environment = ProcessInfo.processInfo.environment
        guard LayoutLoopWatchdog.isEnabled(
            environment: environment, isDebug: isDebug
        ) else { return }
        // Build from the same environment so OPENBIRD_LAYOUT_WATCHDOG_SECONDS is honored.
        let watchdog = LayoutLoopWatchdog(environment: environment)
        layoutWatchdog = watchdog
        watchdog.start()
    }

    static func shouldBootstrapDBKeyForSelfTest(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        !OpenBirdService.isKeyringExplicitlyDisabled(environment: environment)
    }

    static func selfTestDBKeyTimeout(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> TimeInterval {
        let fallback: TimeInterval = 15
        guard let raw = environment["OPENBIRD_SELFTEST_DB_KEY_TIMEOUT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty,
              let value = TimeInterval(raw),
              value > 0
        else {
            return fallback
        }
        return value
    }

    static func selfTestErrorSignal(_ error: Error) -> String {
        switch error {
        case ChatError.cliMissing:
            return "cli_missing"
        case ChatError.failed:
            return "chat_failed"
        case ChatError.decode:
            return "decode"
        default:
            return "unknown"
        }
    }

    private static func emitSelfTestOutcomeAndExit(_ line: String, code: Int32) {
        if code == 0 {
            log.info("\(line, privacy: .public)")
        } else {
            log.error("\(line, privacy: .public)")
        }
        FileHandle.standardOutput.write(Data((line + "\n").utf8))
        exit(code)
    }

    private static func emitSelfTestSignal(_ line: String) {
        log.info("\(line, privacy: .public)")
        FileHandle.standardOutput.write(Data((line + "\n").utf8))
    }

    /// Run one ask through the real service off the main thread, print a parseable
    /// outcome line to stdout, and exit (0=grounded, 1=ungrounded, 2=error). Stays
    /// headless: no dock icon / window / Accessibility automation. The default
    /// self-test resolves the app-owned DB key for encrypted real-store evidence;
    /// OPENBIRD_DISABLE_KEYRING=1 remains the explicit plaintext-dev path.
    private static func runSelfTestAndExit(query: String) {
        let service = OpenBirdService()
        DispatchQueue.global(qos: .userInitiated).async {
            let line: String
            let code: Int32
            do {
                let r = try service.askChat(query, dayOffset: nil)
                let grounded = r.grounded && r.hasDisplaySources
                line = "SELFTEST ask.outcome grounded=\(grounded ? 1 : 0) citations=\(r.citations.count) derived=\(r.derivedCitations.count) sources=\(r.sourceCount)"
                code = grounded ? 0 : 1
            } catch {
                line = "SELFTEST ask.outcome error=1 kind=\(Self.selfTestErrorSignal(error))"
                code = 2
            }
            Self.emitSelfTestOutcomeAndExit(line, code: code)
        }
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

    /// Construct the service, model, Ask-panel controller, and Today model TOGETHER,
    /// sharing the one `AppModel`/`OpenBirdService`, so there is never a second model
    /// instance and no configure-after-launch ordering race (see the design doc).
    init() {
        let service = OpenBirdService()
        let model = AppModel(service: service)
        let askModel = AskPanelModel(service: service, appModel: model)
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
        _askPanel = StateObject(wrappedValue: AskPanelController(
            model: model, service: service, askModel: askModel))
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
    let askPanel: AskPanelController

    @Environment(\.openWindow) private var openWindow

    var body: some View {
        AppShellView(
            model: model,
            todayModel: todayModel,
            askModel: askModel,
            onAsk: { day in askPanel.show(dayScope: day) },
            onAskExpanded: { askPanel.expand() }
        )
        .task {
            // Self-test mode runs the ask headlessly and exits; never start the
            // live UI work here (capture auto-resume / hotkey / refresh), which
            // could spawn capture or raise a permission prompt before exit.
            guard ProcessInfo.processInfo.environment["OPENBIRD_SELFTEST_ASK"] == nil
            else { return }
            askPanel.installHotKeyIfNeeded()   // idempotent ⌥Space registration
            askPanel.openMainWindow = { openWindow(id: "main") }
            model.repairIncompleteOnboardingCompletionIfNeeded()
            await model.refresh()
            // Resume capture if the user is already configured and didn't pause it.
            // After refresh() so allowlist / pause / running state is current.
            // Idempotent, so re-running this .task cannot double-spawn.
            model.autoResumeCaptureIfNeeded()
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            model.refreshPermissionStates()
            // The user may have changed Login Items in System Settings while away.
            model.refreshLaunchAtLoginState()
            // Refresh the day view on refocus while it's open, so activity captured in the
            // background shows up without manually leaving and returning to the pane.
            if model.selection == .today {
                Task { await todayModel.load() }
            }
        }
    }
}
