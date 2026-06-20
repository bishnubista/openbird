import Foundation

/// Backs the Today/day view: loads the capture timeline (fast SQL) and the prose
/// briefing (one on-demand LLM call, cached per day so reopening Today is free).
@MainActor
final class TodayModel: ObservableObject {
    @Published private(set) var timeline: DayTimeline?
    @Published private(set) var briefing: String?
    @Published private(set) var loadingTimeline = false
    @Published private(set) var loadingBriefing = false
    @Published private(set) var timelineError: String?
    /// Display names resolved once per distinct app when the timeline loads, so the
    /// view never hits LaunchServices per row (a NULL-session day can have hundreds
    /// of single-observation sessions).
    @Published private(set) var appNames: [String: String] = [:]
    /// 0 = today, 1 = yesterday, … Defaults to today, matching the "Today's
    /// Activity" menu entry that opens this view.
    @Published var dayOffset = 0

    private let service: OpenBirdService
    /// Per-day briefing cache (this app session), keyed by ABSOLUTE local day (not
    /// the relative offset) so a session left open across midnight doesn't serve a
    /// stale briefing for what is now a different calendar day.
    private var briefingCache: [String: String] = [:]
    /// Bumped on every `load()`; an in-flight fetch only commits its result if this
    /// is still the current generation when it resumes — so switching days (or a
    /// second refresh) mid-load can't let a stale response overwrite the new day.
    private var loadGeneration = 0

    init(service: OpenBirdService) {
        self.service = service
    }

    /// Stable absolute-day key for the given offset (local calendar day).
    private func dayKey(_ offset: Int) -> String {
        let date = Calendar.current.date(byAdding: .day, value: -offset, to: Date()) ?? Date()
        let c = Calendar.current.dateComponents([.year, .month, .day], from: date)
        return "\(c.year ?? 0)-\(c.month ?? 0)-\(c.day ?? 0)"
    }

    /// Title for the current day ("Today" / "Yesterday" / a weekday name).
    var dayTitle: String {
        switch dayOffset {
        case 0: return "Today"
        case 1: return "Yesterday"
        default:
            guard let day = Calendar.current.date(byAdding: .day, value: -dayOffset, to: Date())
            else { return "\(dayOffset) days ago" }
            return day.formatted(.dateTime.weekday(.wide))
        }
    }

    /// Subtitle: full date + session/app counts (handoff "Friday, June 19 · 6 sessions across 6 apps").
    var daySubtitle: String {
        let date = Calendar.current.date(byAdding: .day, value: -dayOffset, to: Date()) ?? Date()
        let dateText = date.formatted(.dateTime.weekday(.wide).month(.wide).day())
        guard let timeline else { return dateText }
        let s = timeline.sessions.count
        let a = timeline.distinctApps
        return "\(dateText) · \(s) session\(s == 1 ? "" : "s") across \(a) app\(a == 1 ? "" : "s")"
    }

    func load() async {
        loadGeneration += 1
        let generation = loadGeneration
        let day = dayOffset
        await loadTimeline(day: day, generation: generation)
        await loadBriefing(day: day, generation: generation)
    }

    private func loadTimeline(day: Int, generation: Int) async {
        loadingTimeline = true
        defer { if generation == loadGeneration { loadingTimeline = false } }
        let result = await service.dayTimeline(dayOffset: day)
        guard generation == loadGeneration else { return }   // superseded → drop
        timeline = result
        timelineError = result == nil ? "Could not load the timeline." : nil
        // Resolve app display names ONCE per distinct app (off the render path).
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

    private func loadBriefing(day: Int, generation: Int) async {
        // load() awaits loadTimeline first, so this can start already superseded —
        // bail before any state mutation in that case.
        guard generation == loadGeneration else { return }
        let key = dayKey(day)
        if let cached = briefingCache[key] {
            briefing = cached   // still current (no await since the guard above)
            return
        }
        loadingBriefing = true
        defer { if generation == loadGeneration { loadingBriefing = false } }
        briefing = nil
        if let text = await service.dailyBriefing(dayOffset: day) {
            briefingCache[key] = text   // cache regardless (keyed by absolute day)
            if generation == loadGeneration { briefing = text }
        }
    }

    /// Re-fetch both, bypassing the briefing cache for the current day.
    func refresh() async {
        briefingCache[dayKey(dayOffset)] = nil
        await load()
    }

    /// Switch the displayed day and reload.
    func setDay(_ offset: Int) async {
        guard offset >= 0, offset != dayOffset else { return }
        dayOffset = offset
        await load()
    }
}
