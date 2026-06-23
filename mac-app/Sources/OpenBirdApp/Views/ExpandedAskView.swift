import SwiftUI

/// The expanded Ask surface hosted in the borderless `AskExpandedWindow` (the ⌥Space
/// overlay's "expand" state). A thin WINDOW wrapper around the shared `AskContentView`:
/// it adds only the window-only chrome — the collapse/close header buttons (via the
/// closures), the cast-shadow padding, the background drag region, and Escape-to-close.
/// The in-window sidebar pane uses the same content via `AskPaneView` instead.
struct ExpandedAskView: View {
    @ObservedObject var askModel: AskPanelModel
    @ObservedObject var appModel: AppModel
    @ObservedObject var timelineModel: TimelineModel
    var onCollapse: () -> Void
    var onClose: () -> Void

    var body: some View {
        AskContentView(
            askModel: askModel,
            appModel: appModel,
            timelineModel: timelineModel,
            onCollapse: onCollapse,
            onClose: onClose
        )
        .glassSurface(cornerRadius: OB.Radius.window)
        .padding(24)                       // room for the cast glass shadow
        .background(WindowConfigurator())  // draggable by background
        .onExitCommand(perform: onClose)
    }
}
