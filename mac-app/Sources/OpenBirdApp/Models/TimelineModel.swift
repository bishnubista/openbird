import Foundation

/// Backs the standalone Timeline window (handoff Direction C — "Timeline-grounded").
/// Loads ONLY the capture timeline (fast SQL) — deliberately NOT the briefing, which
/// is a slow on-demand LLM call the Today view owns. Uses the same generation-guarded
/// supersede behavior as `TodayModel`, so switching days or a refresh-on-activation
/// can't let a stale in-flight response overwrite the current day.
@MainActor
final class TimelineModel: ObservableObject {
    @Published private(set) var timeline: DayTimeline?
    @Published private(set) var loading = false
    @Published private(set) var error: String?
    /// App display names resolved once per distinct app when the timeline loads, so the
    /// rail never hits LaunchServices per row (a NULL-session day can have many rows).
    @Published private(set) var appNames: [String: String] = [:]
    /// 0 = today, 1 = yesterday, … Opens on today (most recent); the prototype seeds
    /// "Yesterday" as demo content, but a live timeline defaults to the current day.
    @Published var dayOffset = 0

    /// Base date the currently-loaded data was fetched against. `dayTitle`/`dayHeading`
    /// derive from this rather than `Date()` so an always-open window past midnight
    /// keeps labels aligned with the loaded sessions until the next load (CodeRabbit).
    private var dayReference = Date()

    private let service: OpenBirdService
    /// Bumped on every `load()`; an in-flight fetch only commits if it is still the
    /// current generation when it resumes (drops superseded day-switch/refresh races).
    private var loadGeneration = 0

    init(service: OpenBirdService) {
        self.service = service
    }

    /// Title for the current day ("Today" / "Yesterday" / a weekday name).
    var dayTitle: String {
        switch dayOffset {
        case 0: return "Today"
        case 1: return "Yesterday"
        default:
            guard let day = Calendar.current.date(byAdding: .day, value: -dayOffset, to: dayReference)
            else { return "\(dayOffset) days ago" }
            return day.formatted(.dateTime.weekday(.wide))
        }
    }

    /// Uppercase full-date rail header ("FRIDAY, JUNE 19"), matching the handoff.
    var dayHeading: String {
        let date = Calendar.current.date(byAdding: .day, value: -dayOffset, to: dayReference) ?? dayReference
        return date.formatted(.dateTime.weekday(.wide).month(.wide).day()).uppercased()
    }

    /// Subtitle digest: session + app counts ("6 sessions across 6 apps").
    var sessionSummary: String {
        guard let timeline else { return "" }
        let s = timeline.sessions.count
        let a = timeline.distinctApps
        return "\(s) session\(s == 1 ? "" : "s") across \(a) app\(a == 1 ? "" : "s")"
    }

    func load() async {
        dayReference = Date()   // anchor day labels to this fetch
        loadGeneration += 1
        let generation = loadGeneration
        let day = dayOffset
        loading = true
        defer { if generation == loadGeneration { loading = false } }
        let result = await service.dayTimeline(dayOffset: day)
        guard generation == loadGeneration else { return }   // superseded → drop
        timeline = result
        error = result == nil ? "Could not load the timeline." : nil
        var names: [String: String] = [:]
        for app in Set((result?.sessions ?? []).compactMap(\.app)) {
            names[app] = AppDisplay.name(app)
        }
        appNames = names
    }

    /// Display name for a session's app, from the prebuilt map (falls back to the
    /// resolver for anything not in it).
    func displayName(_ app: String?) -> String {
        guard let app, !app.isEmpty else { return "Unknown" }
        return appNames[app] ?? AppDisplay.name(app)
    }

    /// Switch the displayed day and reload. Caller is responsible for resetting any
    /// associated chat thread (the day-scoped recap changes underneath it).
    func setDay(_ offset: Int) async {
        guard offset >= 0, offset != dayOffset else { return }
        dayOffset = offset
        // Drop the previous day's rail/recap immediately so the loading state shows
        // during the switch — otherwise the new heading renders over stale sessions
        // until the CLI returns (Codex review). NOT done in `refresh()`, where the
        // same-day data should stay visible until its replacement arrives.
        timeline = nil
        appNames = [:]
        error = nil
        await load()
    }

    /// Re-fetch the current day in place (e.g. on app re-activation). Keeps the
    /// existing rail visible until the fresh result replaces it — no loading flash.
    func refresh() async {
        await load()
    }
}
