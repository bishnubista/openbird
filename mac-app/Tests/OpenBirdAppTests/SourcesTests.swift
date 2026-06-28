import XCTest
@testable import OpenBirdApp

final class SourcesRailTests: XCTestCase {
    private func citation(index: Int, app: String?, window: String?, snippet: String = "x") -> ChatCitation {
        ChatCitation(index: index, app: app, window: window, ts: 0, snippet: snippet)
    }

    func testCardTitlePrefersWindowThenApp() {
        XCTAssertEqual(SourcesRail.cardTitle(citation(index: 1, app: "VS Code", window: "rag.py — openbird")),
                       "rag.py — openbird")
        XCTAssertEqual(SourcesRail.cardTitle(citation(index: 1, app: "Zoom", window: "")), "Zoom")
        XCTAssertEqual(SourcesRail.cardTitle(citation(index: 1, app: "Zoom", window: nil)), "Zoom")
        XCTAssertEqual(SourcesRail.cardTitle(citation(index: 1, app: nil, window: nil)), "Source")
        // Whitespace-only metadata falls through instead of rendering blank.
        XCTAssertEqual(SourcesRail.cardTitle(citation(index: 1, app: "Zoom", window: "   ")), "Zoom")
        XCTAssertEqual(SourcesRail.cardTitle(citation(index: 1, app: "  ", window: "  ")), "Source")
    }

    func testDerivedCardTitleAndCountUseAggregateMetadata() {
        let source = DerivedChatCitation(
            index: 1,
            sourceId: "D1",
            label: "  Daily productivity facts  ",
            snippet: "12 active minutes",
            derivedFromTotal: 14
        )

        XCTAssertEqual(SourcesRail.derivedCardTitle(source), "Daily productivity facts")
        XCTAssertEqual(SourcesRail.derivedCountText(source), "14 source observations")
    }

    func testDerivedCardTitleFallsBack() {
        let source = DerivedChatCitation(
            index: 1,
            sourceId: "D1",
            label: "   ",
            snippet: "",
            derivedFromTotal: 1
        )

        XCTAssertEqual(SourcesRail.derivedCardTitle(source), "Derived source")
        XCTAssertEqual(SourcesRail.derivedCountText(source), "1 source observation")
    }
}

final class SourcesDisplayTests: XCTestCase {
    private func citation(_ index: Int) -> ChatCitation {
        ChatCitation(index: index, app: "VS Code", window: "f.py", ts: 0, snippet: "s")
    }

    private func derived(_ id: String) -> DerivedChatCitation {
        DerivedChatCitation(
            index: 1,
            sourceId: id,
            label: "Daily productivity facts",
            snippet: "12 active minutes",
            derivedFromTotal: 2
        )
    }

    private func turn(
        _ question: String,
        grounded: Bool? = nil,
        citations: [ChatCitation] = [],
        derivedCitations: [DerivedChatCitation] = [],
        error: String? = nil
    ) -> AskPanelModel.Turn {
        let result = grounded.map {
            ChatResult(
                answer: "a",
                grounded: $0,
                citations: citations,
                derivedCitations: derivedCitations
            )
        }
        return AskPanelModel.Turn(question: question, result: result, error: error)
    }

    func testDisplayHiddenWhenNoAnswerYet() {
        XCTAssertEqual(SourcesDisplay.make(thread: [], busy: false).indicator, .hidden)
        XCTAssertTrue(SourcesDisplay.make(thread: [], busy: false).sources.isEmpty)
    }

    func testDisplayThinkingWhileBusyKeepsPriorSources() {
        let thread = [
            turn("q1", grounded: true, citations: [citation(1)]),
            turn("q2"),   // pending
        ]
        let d = SourcesDisplay.make(thread: thread, busy: true)
        XCTAssertEqual(d.indicator, .thinking)
        XCTAssertEqual(d.sources.map(\.id), ["occurrence-1"])   // last successful answer's sources
    }

    func testDisplayThinkingWhileBusyKeepsPriorDerivedSources() {
        let thread = [
            turn("q1", grounded: true, derivedCitations: [derived("D1")]),
            turn("q2"),   // pending
        ]
        let d = SourcesDisplay.make(thread: thread, busy: true)
        XCTAssertEqual(d.indicator, .thinking)
        XCTAssertEqual(d.sources.map(\.id), ["derived-D1"])
    }

    func testDisplayNeutralAndEmptyWhenLastTurnErrored() {
        // The Codex-flagged bug: after a grounded Q1, a failed Q2 must NOT keep "grounded"
        // + Q1's rail next to the error.
        let thread = [
            turn("q1", grounded: true, citations: [citation(1)]),
            turn("q2", error: "failed"),
        ]
        let d = SourcesDisplay.make(thread: thread, busy: false)
        XCTAssertEqual(d.indicator, .hidden)
        XCTAssertTrue(d.sources.isEmpty)
    }

    func testDisplayReflectsLastSuccessfulAnswer() {
        let thread = [
            turn("q1", grounded: true, citations: [citation(1)]),
            turn("q2", grounded: false, citations: [citation(2), citation(3)]),
        ]
        let d = SourcesDisplay.make(thread: thread, busy: false)
        XCTAssertEqual(d.indicator, .ungrounded)
        XCTAssertEqual(d.sources.map(\.id), ["occurrence-2", "occurrence-3"])
    }

    func testDisplayReflectsDerivedOnlyAnswer() {
        let thread = [
            turn("q1", grounded: true, derivedCitations: [derived("D1")]),
        ]
        let d = SourcesDisplay.make(thread: thread, busy: false)
        XCTAssertEqual(d.indicator, .grounded)
        XCTAssertEqual(d.sources.map(\.id), ["derived-D1"])
    }
}
