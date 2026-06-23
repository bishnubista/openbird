import SwiftUI

/// The in-window Ask pane — the sidebar's `.ask` destination. A thin PANE wrapper around
/// the shared `AskContentView` with NO window chrome (no collapse/close buttons, no
/// cast-shadow padding, no drag region: the app shell owns the window). It renders the
/// SAME conversation (`askModel`) as the ⌥Space `ExpandedAskView` overlay, so a thread
/// started in one is continuous in the other. Sits on the shell's `GlassBackdrop`.
struct AskPaneView: View {
    @ObservedObject var askModel: AskPanelModel
    @ObservedObject var appModel: AppModel
    @ObservedObject var timelineModel: TimelineModel

    var body: some View {
        AskContentView(
            askModel: askModel,
            appModel: appModel,
            timelineModel: timelineModel
        )
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
