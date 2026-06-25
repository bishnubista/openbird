import Foundation

/// The app's primary navigation destinations — a single source of truth so the menu
/// bar and the Today sidebar mirror each other and can't drift. Carries display data
/// only (title + icon + order); each renderer supplies the behavior (show the Ask
/// panel vs. open a window) and the active-state highlight.
enum AppDestination: String, CaseIterable, Identifiable {
    case ask, today, setup

    var id: String { rawValue }

    var title: String {
        switch self {
        case .ask: return "Ask OpenBird"
        case .today: return "Today"
        // The `.setup` case keeps its raw value (deep-links/tests stay valid) but presents
        // as "Settings" — the durable permissions/capture/privacy pane (design rename).
        case .setup: return "Settings"
        }
    }

    /// SF Symbol name (all available on the macOS 13 floor).
    var systemImage: String {
        switch self {
        case .ask: return "magnifyingglass"
        case .today: return "calendar"
        case .setup: return "slider.horizontal.3"
        }
    }
}
