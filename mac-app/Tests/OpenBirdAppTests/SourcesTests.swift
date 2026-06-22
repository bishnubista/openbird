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
    }
}

final class AskSourcesViewTests: XCTestCase {
    private func citation(_ index: Int) -> ChatCitation {
        ChatCitation(index: index, app: "VS Code", window: "f.py", ts: 0, snippet: "s")
    }

    private func turn(_ question: String, grounded: Bool? = nil, citations: [ChatCitation] = [], error: String? = nil) -> AskPanelModel.Turn {
        let result = grounded.map { ChatResult(answer: "a", grounded: $0, citations: citations) }
        return AskPanelModel.Turn(question: question, result: result, error: error)
    }

    func testDisplayHiddenWhenNoAnswerYet() {
        XCTAssertEqual(AskSourcesView.makeDisplay(thread: [], busy: false).indicator, .hidden)
        XCTAssertTrue(AskSourcesView.makeDisplay(thread: [], busy: false).citations.isEmpty)
    }

    func testDisplayThinkingWhileBusyKeepsPriorSources() {
        let thread = [
            turn("q1", grounded: true, citations: [citation(1)]),
            turn("q2"),   // pending
        ]
        let d = AskSourcesView.makeDisplay(thread: thread, busy: true)
        XCTAssertEqual(d.indicator, .thinking)
        XCTAssertEqual(d.citations.map(\.index), [1])   // last successful answer's sources
    }

    func testDisplayNeutralAndEmptyWhenLastTurnErrored() {
        // The Codex-flagged bug: after a grounded Q1, a failed Q2 must NOT keep "grounded"
        // + Q1's rail next to the error.
        let thread = [
            turn("q1", grounded: true, citations: [citation(1)]),
            turn("q2", error: "failed"),
        ]
        let d = AskSourcesView.makeDisplay(thread: thread, busy: false)
        XCTAssertEqual(d.indicator, .hidden)
        XCTAssertTrue(d.citations.isEmpty)
    }

    func testDisplayReflectsLastSuccessfulAnswer() {
        let thread = [
            turn("q1", grounded: true, citations: [citation(1)]),
            turn("q2", grounded: false, citations: [citation(2), citation(3)]),
        ]
        let d = AskSourcesView.makeDisplay(thread: thread, busy: false)
        XCTAssertEqual(d.indicator, .ungrounded)
        XCTAssertEqual(d.citations.map(\.index), [2, 3])
    }
}
