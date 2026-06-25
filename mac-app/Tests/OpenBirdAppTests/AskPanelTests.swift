import XCTest
@testable import OpenBirdApp

final class SourceIdentityTests: XCTestCase {
    func testKnownAppsMapToHandoffGlyphs() {
        XCTAssertEqual(SourceIdentity.forApp("Visual Studio Code").glyph, "{}")
        XCTAssertEqual(SourceIdentity.forApp("zoom.us").glyph, "Z")
        XCTAssertEqual(SourceIdentity.forApp("Linear").glyph, "L")
        XCTAssertEqual(SourceIdentity.forApp("Notion").glyph, "N")
    }

    func testUnknownAppFallsBackToInitial() {
        XCTAssertEqual(SourceIdentity.forApp("Figma").glyph, "F")
    }

    func testNilAppFallsBackToNeutralDot() {
        XCTAssertEqual(SourceIdentity.forApp(nil).glyph, "•")
    }
}

final class CitationFormattingTests: XCTestCase {
    func testSourceLabelJoinsAppAndWindow() {
        XCTAssertEqual(
            CitationFormatting.sourceLabel(app: "Code", window: "main.swift"),
            "Code / main.swift"
        )
    }

    func testSourceLabelDropsEmptyParts() {
        XCTAssertEqual(CitationFormatting.sourceLabel(app: "Code", window: ""), "Code")
        XCTAssertEqual(CitationFormatting.sourceLabel(app: nil, window: nil), "unknown")
    }

    func testTimeLabelsAreNonEmpty() {
        // Exact text is locale/timezone-dependent; assert only that it renders.
        XCTAssertFalse(CitationFormatting.timeLabel(1_700_000_000).isEmpty)
        XCTAssertFalse(CitationFormatting.shortTime(1_700_000_000).isEmpty)
    }
}

@MainActor
final class AskPanelModelTests: XCTestCase {
    private func makeAppModel() -> AppModel {
        // Models ready + memoryStatsState .unknown keeps askUnavailableReason nil
        // so these tests exercise panel concurrency/result flow, not setup gating.
        var report = PreflightReport()
        report.ollamaReachable = true
        report.runtimeOK = true
        return AppModel(service: OpenBirdService(), initialReport: report)
    }

    func testSingleAskAppendsOneTurnAndFillsResult() async {
        let result = ChatResult(
            answer: "You worked on the citation pipeline.",
            grounded: true,
            citations: [ChatCitation(index: 1, app: "Code", window: "rag.py", ts: 1_700_000_000, snippet: "...")]
        )
        let model = AskPanelModel(appModel: makeAppModel()) { _, _ in result }

        guard let turn = model.beginAsk("what did I do?") else {
            return XCTFail("beginAsk should return a turn for a valid question")
        }
        XCTAssertEqual(model.thread.count, 1)
        XCTAssertTrue(model.busy)

        await model.complete(turn)
        XCTAssertEqual(model.thread.count, 1, "no duplicate append")
        XCTAssertEqual(model.thread.first?.result, result)
        XCTAssertNil(model.thread.first?.error)
        XCTAssertFalse(model.busy)
    }

    func testEmptyQuestionIsNoOp() {
        let model = AskPanelModel(appModel: makeAppModel()) { _, _ in
            ChatResult(answer: "x", grounded: false, citations: [])
        }
        XCTAssertNil(model.beginAsk("   \n  "))
        XCTAssertTrue(model.thread.isEmpty)
        XCTAssertFalse(model.busy)
    }

    func testErrorPopulatesTurnErrorNotResult() async {
        let model = AskPanelModel(appModel: makeAppModel()) { _, _ in
            throw ChatError.failed("local model offline")
        }
        guard let turn = model.beginAsk("hello") else {
            return XCTFail("beginAsk should return a turn")
        }
        await model.complete(turn)
        XCTAssertEqual(model.thread.first?.error, "local model offline")
        XCTAssertNil(model.thread.first?.result)
        XCTAssertFalse(model.busy)
    }

    func testClearDuringAskDropsStaleResultAndResetsBusy() async {
        let model = AskPanelModel(appModel: makeAppModel()) { _, _ in
            ChatResult(answer: "stale", grounded: true, citations: [])
        }
        guard let turn = model.beginAsk("slow question") else {
            return XCTFail("beginAsk should return a turn")
        }
        // Simulate the user clearing the panel while the CLI call is still in flight.
        model.clear()
        XCTAssertFalse(model.busy, "clear() must reset busy immediately")
        XCTAssertTrue(model.thread.isEmpty)

        await model.complete(turn)   // stale completion arrives after clear
        XCTAssertTrue(model.thread.isEmpty, "stale result must not resurrect a turn")
        XCTAssertFalse(model.busy)
    }

    func testBusyBlocksConcurrentBegin() {
        let model = AskPanelModel(appModel: makeAppModel()) { _, _ in
            ChatResult(answer: "x", grounded: false, citations: [])
        }
        XCTAssertNotNil(model.beginAsk("first"))   // sets busy = true
        XCTAssertNil(model.beginAsk("second"), "a second ask is blocked while busy")
        XCTAssertEqual(model.thread.count, 1)
    }

    // MARK: - Day scope forwarding

    /// Thread-safe box so the `@Sendable` ask closure (run on a detached task) can
    /// record the scope it received for the test to assert.
    private final class ScopeBox: @unchecked Sendable {
        private let lock = NSLock()
        private var value: Int??
        func set(_ v: Int?) { lock.lock(); value = v; lock.unlock() }
        var recorded: Int?? { lock.lock(); defer { lock.unlock() }; return value }
    }

    func testDayScopeIsForwardedToAsk() async {
        let box = ScopeBox()
        let model = AskPanelModel(appModel: makeAppModel()) { _, scope in
            box.set(scope)
            return ChatResult(answer: "ok", grounded: false, citations: [])
        }
        model.dayScope = 2
        guard let turn = model.beginAsk("what happened?") else {
            return XCTFail("beginAsk should return a turn")
        }
        XCTAssertEqual(model.thread.first?.dayScope, 2, "turn snapshots the scope")
        await model.complete(turn)
        XCTAssertEqual(box.recorded, .some(2), "scope 2 forwarded to the ask")
    }

    func testNilDayScopeForwardsNil() async {
        let box = ScopeBox()
        let model = AskPanelModel(appModel: makeAppModel()) { _, scope in
            box.set(scope)
            return ChatResult(answer: "ok", grounded: false, citations: [])
        }
        // dayScope defaults to nil (the generic/global Ask).
        guard let turn = model.beginAsk("anything?") else {
            return XCTFail("beginAsk should return a turn")
        }
        XCTAssertNil(model.thread.first?.dayScope)
        await model.complete(turn)
        XCTAssertEqual(box.recorded, .some(nil), "unscoped ask forwards nil")
    }

    func testTurnSnapshotsScopeSoLaterChangeDoesNotRetargetInFlightAsk() async {
        let box = ScopeBox()
        let model = AskPanelModel(appModel: makeAppModel()) { _, scope in
            box.set(scope)
            return ChatResult(answer: "ok", grounded: false, citations: [])
        }
        model.dayScope = 1
        guard let turn = model.beginAsk("q") else {
            return XCTFail("beginAsk should return a turn")
        }
        model.dayScope = 5   // user navigates to another day mid-flight
        await model.complete(turn)
        XCTAssertEqual(box.recorded, .some(1), "in-flight ask keeps its captured scope")
    }
}

final class OpenBirdServiceChatArgumentsTests: XCTestCase {
    func testUnscopedArgumentsOmitDay() {
        // Byte-for-byte unchanged from the pre-feature argv (question via stdin).
        XCTAssertEqual(
            OpenBirdService.chatArguments(dayOffset: nil),
            ["chat", "--json", "--stdin"]
        )
    }

    func testScopedArgumentsAppendDay() {
        XCTAssertEqual(
            OpenBirdService.chatArguments(dayOffset: 0),
            ["chat", "--json", "--stdin", "--day", "0"]
        )
        XCTAssertEqual(
            OpenBirdService.chatArguments(dayOffset: 3),
            ["chat", "--json", "--stdin", "--day", "3"]
        )
    }
}
