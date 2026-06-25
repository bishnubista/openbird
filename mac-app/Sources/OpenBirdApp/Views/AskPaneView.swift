import SwiftUI

/// The in-window Ask pane — the sidebar's `.ask` destination. A thin PANE wrapper around
/// the shared `AskContentView` with NO window chrome (no collapse/close buttons, no
/// cast-shadow padding, no drag region: the app shell owns the window). It renders the
/// SAME conversation (`askModel`) as the ⌥Space `ExpandedAskView` overlay, so a thread
/// started in one is continuous in the other. Sits on the shell's `GlassBackdrop`.
struct AskPaneView: View {
    @ObservedObject var askModel: AskPanelModel
    @ObservedObject var appModel: AppModel

    var body: some View {
        AskContentView(
            askModel: askModel,
            appModel: appModel,
            // In-window pane: no overlay to dismiss, so navigate AppModel directly
            // (it switches to the Today pane and focuses the source's day).
            onSelectCitation: { appModel.navigateToCitation($0) }
        )
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        // This is the GENERIC in-window Ask surface (sidebar `.ask` / `openbird://ask`),
        // reached without going through the controller. It shares the one `askModel`, so
        // a day scope set by Today's compact panel would otherwise leak here — clear it
        // on appear so the in-window Ask is always unscoped.
        .onAppear { askModel.dayScope = nil }
    }
}
