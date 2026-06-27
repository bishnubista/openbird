import Foundation

/// Backs the Today/day view: loads the capture timeline (fast SQL) and the
/// briefing (local deterministic by default, cached per day so reopening Today is free).
@MainActor
final class TodayModel: ObservableObject {
    @Published private(set) var timeline: DayTimeline?
    @Published private(set) var briefing: String?
    /// The current briefing's source trail (the occurrences the prose was grounded
    /// in) and the full grounding-group count. Empty when there is no briefing or
    /// the day had no activity. `briefingSourcesTotal > briefingSources.count` means
    /// the CLI capped the trail, so the view shows "N of M" rather than silently
    /// truncating. Privacy: ids + already-redacted snippets only — no raw blobs.
    @Published private(set) var briefingSources: [BriefingSource] = []
    @Published private(set) var briefingSourcesTotal = 0
    /// When the currently-shown briefing was generated (real fetch time, cached per
    /// day so reopening Today keeps the original time). Drives the "generated H:MM"
    /// label on the briefing card.
    @Published private(set) var briefingGeneratedAt: Date?
    /// Truthful route label from the CLI's `reasoning_route`. Missing/unknown means
    /// no label, never a local-only affirmation.
    @Published private(set) var briefingRouteLabel: String?
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
    /// Source observation a clicked chat citation asked us to focus, set by
    /// `focus(dayOffset:observationId:)`. The day view renders *sessions* (not
    /// individual observations), so there is no per-observation row to scroll to yet;
    /// this is carried so the field is ready when observation-level rows land and so a
    /// view can highlight the owning session when a match becomes available.
    /// Privacy: an opaque id only — never captured text/window/URL.
    @Published private(set) var focusedObservationId: String?

    private let service: OpenBirdService
    /// Per-day briefing cache (this app session), keyed by ABSOLUTE local day (not
    /// the relative offset) so a session left open across midnight doesn't serve a
    /// stale briefing for what is now a different calendar day.
    private var briefingCache: [String: DayBriefing] = [:]
    /// Generation time per cached briefing (keyed by the same absolute-day key), so
    /// a cache hit reuses the original "generated H:MM" instead of stamping now.
    private var briefingTimeCache: [String: Date] = [:]
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
            apply(cached)   // still current (no await since the guard above)
            briefingGeneratedAt = briefingTimeCache[key]
            return
        }
        loadingBriefing = true
        defer { if generation == loadGeneration { loadingBriefing = false } }
        briefing = nil
        briefingSources = []
        briefingSourcesTotal = 0
        briefingGeneratedAt = nil
        briefingRouteLabel = nil
        if let result = await service.dailyBriefing(dayOffset: day) {
            // A superseded generation must NOT write the cache. With the reload-on-open and
            // refocus paths, two same-day loads can overlap; if an OLDER load finishes after
            // a newer one, writing here would overwrite fresh data with stale, and the next
            // cache-hit would then apply it. Drop the superseded result entirely.
            guard generation == loadGeneration else { return }
            let now = Date()
            briefingCache[key] = result
            briefingTimeCache[key] = now
            apply(result)
            briefingGeneratedAt = now
        }
    }

    /// Publish a fetched/cached briefing into the view-facing fields.
    private func apply(_ briefing: DayBriefing) {
        self.briefing = briefing.text
        briefingRouteLabel = briefing.routeLabel
        briefingSources = briefing.sources
        briefingSourcesTotal = briefing.sourcesTotal
    }

    /// Re-fetch both, bypassing the briefing cache for the current day.
    func refresh() async {
        let key = dayKey(dayOffset)
        briefingCache[key] = nil
        briefingTimeCache[key] = nil
        await load()
    }

    /// Switch the displayed day and reload.
    func setDay(_ offset: Int) async {
        guard offset >= 0, offset != dayOffset else { return }
        dayOffset = offset
        await load()
    }

    /// Focus the day view on a clicked chat citation: switch to its day (reloading
    /// the timeline/briefing if it changed) and record the source observation to
    /// focus. Opening the correct day is the guaranteed behavior; the observation id
    /// is best-effort context for highlighting once the day's sessions are loaded.
    /// Invoked via `AppModel.citationNavigator`. Privacy: `observationId` is an opaque
    /// id; no captured content is read here.
    func focus(dayOffset offset: Int, observationId: String?) async {
        focusedObservationId = observationId
        let target = max(0, offset)
        if target == dayOffset {
            if timeline == nil { await load() }   // first open of an already-current day
        } else {
            await setDay(target)
        }
    }
}
