import SwiftUI

/// The expanded Ask surface hosted in the borderless `AskExpandedWindow` (the ⌥Space
/// overlay's "expand" state). A thin WINDOW wrapper around the shared `AskContentView`:
/// it adds only the window-only chrome — the collapse/close header buttons (via the
/// closures), the cast-shadow padding, the background drag region, and Escape-to-close.
/// The in-window sidebar pane uses the same content via `AskPaneView` instead.
struct ExpandedAskView: View {
    @ObservedObject var askModel: AskPanelModel
    @ObservedObject var appModel: AppModel
    var onCollapse: () -> Void
    var onClose: () -> Void
    /// Invoked when a citation is clicked — the controller dismisses this window and
    /// routes to the source's day in Today. Defaults to a no-op for previews.
    var onSelectCitation: (ChatCitation) -> Void = { _ in }

    var body: some View {
        AskContentView(
            askModel: askModel,
            appModel: appModel,
            onCollapse: onCollapse,
            onClose: onClose,
            onSelectCitation: onSelectCitation
        )
        .glassSurface(cornerRadius: OB.Radius.window)
        .padding(24)                       // room for the cast glass shadow
        .background(WindowConfigurator())  // draggable by background
        .onExitCommand(perform: onClose)
    }
}
