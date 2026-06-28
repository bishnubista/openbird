import SwiftUI

/// First-run onboarding sheet (handoff §4 / screenshot 06). This is a thin host for
/// the pixel-faithful `SetupSheetView`, adding the glass backdrop, state refresh, and
/// primary action that completes onboarding only after capture is already running or
/// successfully starts.
struct OnboardingSheet: View {
    @ObservedObject var model: AppModel
    @State private var actionInFlight = false
    var onComplete: () -> Void
    var onOpenSetup: () -> Void

    var body: some View {
        SetupSheetView(
            model: model,
            primaryLabel: model.onboardingPrimaryAction.label,
            primaryDisabled: actionInFlight || model.onboardingPrimaryAction.isDisabled,
            onPrimary: start
        )
            .padding(OB.Space.xl)
            .background(GlassBackdrop())
            .task { await model.refresh() }
    }

    private func start() {
        guard !actionInFlight else { return }
        actionInFlight = true
        // The initial `.task` refresh may still be in flight on a very fast click, so
        // re-check before deciding whether this is a true completion or a setup handoff.
        Task { @MainActor in
            defer { actionInFlight = false }
            if model.lastRefresh == nil { await model.refresh() }
            switch model.onboardingPrimaryAction {
            case .checking:
                return
            case .complete:
                onComplete()
            case .start:
                if await model.startCaptureForOnboarding() {
                    onComplete()
                }
            case .blocked(let message):
                model.lastActionMessage = message
                onOpenSetup()
            }
        }
    }
}
