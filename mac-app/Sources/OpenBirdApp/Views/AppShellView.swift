import AppKit
import SwiftUI

/// The single app window's root shell: the persistent nav sidebar plus a detail pane
/// switched by `model.selection` (Ask / Today / Setup all live in ONE window now). This
/// view is ALWAYS mounted for the window's lifetime, so it owns everything that must
/// outlive any individual pane: the window chrome (min-size, glass backdrop, background
/// drag region), the first-run onboarding sheet, and the gated `openbird://` deep-link
/// router. (Codex review: hanging `onOpenURL`/`.sheet` off the Setup pane would unmount
/// them whenever the user is on Today or Ask.)
struct AppShellView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var todayModel: TodayModel
    @ObservedObject var askModel: AskPanelModel
    /// Summon the compact Spotlight Ask panel hard-scoped to a day offset (used by the
    /// Today pane's "Ask about this day"; 0=today, 1=yesterday, ...).
    var onAsk: (Int) -> Void = { _ in }
    /// Open the expanded Ask overlay window (the `openbird://ask-expanded` E2E deep-link).
    var onAskExpanded: () -> Void = {}

    /// One-time first-run flag; only a real capture start/completion flips it. Plain
    /// dismissal can hide the sheet for this session without claiming setup is done.
    @AppStorage("openbird.onboarding.completed") private var onboardingCompleted = false
    @State private var onboardingPresented = false
    @State private var didPrepareOnboarding = false

    var body: some View {
        HStack(spacing: 0) {
            AppSidebar(appModel: model)
            detailPane
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        // One coherent floor that fits every pane: sidebar (222) + the widest pane's
        // usable content (Settings cards / Today). The Ask pane centers its single chat
        // column, so it stays legible at any width above this floor.
        .frame(minWidth: 920, minHeight: 560)
        .background(GlassBackdrop())
        .background(WindowConfigurator())   // draggable by background (macOS-13-safe)
        .sheet(isPresented: $onboardingPresented) {
            OnboardingSheet(
                model: model,
                onComplete: completeOnboarding,
                onOpenSetup: openSetupFromOnboarding
            )
        }
        .onAppear(perform: prepareOnboardingPresentation)
        .onOpenURL { url in route(url) }
    }

    @ViewBuilder
    private var detailPane: some View {
        switch model.selection {
        case .ask:
            AskPaneView(askModel: askModel, appModel: model)
        case .today:
            TodayView(model: todayModel, appModel: model, onAsk: onAsk)
        case .setup:
            SettingsView(model: model)
        }
    }

    private func route(_ url: URL) {
        // TEST-ONLY affordance for the E2E screenshot harness — NOT a public product
        // surface. A routable openbird:// scheme would let any local app or web page
        // force OpenBird to foreground private activity data (Today) or steal focus
        // (Ask). So the router is inert unless the app was launched explicitly with
        // `--enable-e2e-deeplinks` (the harness passes it via `open --args`).
        guard ProcessInfo.processInfo.arguments.contains("--enable-e2e-deeplinks") else { return }
        guard url.scheme == "openbird" else { return }
        switch url.host {
        case "today": model.selection = .today
        case "main", "setup": model.selection = .setup
        case "ask": model.selection = .ask          // in-window pane (was the compact overlay)
        case "ask-expanded": onAskExpanded()         // the borderless expanded overlay (E2E)
        default: return
        }
        NSApp.activate(ignoringOtherApps: true)
    }

    private func prepareOnboardingPresentation() {
        guard !didPrepareOnboarding else { return }
        didPrepareOnboarding = true
        model.repairIncompleteOnboardingCompletionIfNeeded()
        onboardingCompleted = UserDefaults.standard.bool(forKey: AppModel.onboardingCompletedKey)
        var state = OnboardingPresentationState(
            completed: onboardingCompleted,
            presented: onboardingPresented
        )
        state.presentIfNeeded()
        onboardingPresented = state.presented
    }

    private func completeOnboarding() {
        var state = OnboardingPresentationState(
            completed: onboardingCompleted,
            presented: onboardingPresented
        )
        state.complete()
        onboardingCompleted = state.completed
        onboardingPresented = state.presented
    }

    private func openSetupFromOnboarding() {
        var state = OnboardingPresentationState(
            completed: onboardingCompleted,
            presented: onboardingPresented
        )
        state.dismissWithoutCompleting()
        onboardingCompleted = state.completed
        onboardingPresented = state.presented
        model.selection = .setup
    }
}
