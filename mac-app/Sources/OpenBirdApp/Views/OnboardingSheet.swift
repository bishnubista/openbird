import SwiftUI

/// First-run onboarding sheet (handoff §4 / screenshot 06). This is now a thin host for
/// the shared `SetupSheetView`: the same pixel-faithful sheet the navigable Setup tab
/// renders, so onboarding and Settings can never drift. This wrapper only adds the
/// first-run concerns — the glass backdrop, a state refresh, and the "Start capturing"
/// action that ALSO dismisses the sheet.
struct OnboardingSheet: View {
    @ObservedObject var model: AppModel
    @Binding var isPresented: Bool

    var body: some View {
        SetupSheetView(model: model, onPrimary: start)
            .padding(OB.Space.xl)
            .background(GlassBackdrop())
            .task { await model.refresh() }
    }

    private func start() {
        // Only start the capture daemon when there's something to capture — an empty
        // allowlist would launch a no-op daemon (capture is allowlist-first). The initial
        // `.task` refresh may still be in flight on a very fast click, so make sure the
        // allowlist reflects real service state before deciding. Either way we dismiss to
        // the Settings tab, which carries the full allowlist editor.
        Task { @MainActor in
            if model.lastRefresh == nil { await model.refresh() }
            if !model.allowlist.isEmpty {
                model.startCapture()
            }
            isPresented = false
        }
    }
}
