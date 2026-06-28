import Foundation

/// The grounded indicator state for the Ask sources rail.
enum GroundedState: Equatable { case hidden, thinking, grounded, ungrounded }

/// The sources-rail + grounded-indicator state derived from the conversation. Kept
/// EXACT (Codex review — no "latest completed answer" loosening): while busy →
/// `.thinking` plus the last successful answer's sources as prior context; idle with
/// the last turn having a result → `.grounded`/`.ungrounded` plus that result's
/// sources; idle with the last turn errored or no turns → `.hidden` + empty (so a
/// failed follow-up never leaves a stale "grounded" rail next to the error).
struct SourcesDisplay: Equatable {
    var indicator: GroundedState
    var sources: [ChatSource]

    static func make(thread: [AskPanelModel.Turn], busy: Bool) -> SourcesDisplay {
        if busy {
            let prior = thread.last { $0.result != nil }?.result?.displaySources ?? []
            return SourcesDisplay(indicator: .thinking, sources: prior)
        }
        if let result = thread.last?.result {
            return SourcesDisplay(indicator: result.grounded ? .grounded : .ungrounded,
                                  sources: result.displaySources)
        }
        return SourcesDisplay(indicator: .hidden, sources: [])
    }
}
