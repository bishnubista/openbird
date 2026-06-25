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
        // allowlist would launch a no-op daemon (capture is allowlist-first). Either way
        // we dismiss to the main window, which carries the full Setup tab + allowlist
        // editor, so the setup path is never hidden behind a one-time sheet.
        if !model.allowlist.isEmpty {
            model.startCapture()
        }
        isPresented = false
    }
}
