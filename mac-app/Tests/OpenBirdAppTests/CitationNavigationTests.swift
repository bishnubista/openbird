import XCTest
@testable import OpenBirdApp

/// Decoding contract: `chat --json` citations carry `observation_id` / `chunk_id`
/// (Python `AnswerResult.to_public_dict`), and the Swift `ChatCitation` must decode
/// them — while still tolerating their absence (older CLI builds / un-grounded
/// citations). This pins the cross-component shape so click-through has the ids it
/// needs to navigate to the source.
final class ChatCitationDecodingTests: XCTestCase {
    private func decodeResult(_ json: String) throws -> ChatResult {
        let data = Data(json.utf8)
        return try JSONDecoder().decode(ChatResult.self, from: data)
    }

    func testDecodesObservationAndChunkIdsFromRealisticChatJson() throws {
        // Mirrors `openbird chat --json` (cli.py -> AnswerResult.to_public_dict).
        let json = """
        {
          "answer": "Storage uses SQLite.",
          "grounded": true,
          "citations": [
            {
              "index": 1,
              "observation_id": "obs-42",
              "chunk_id": "chunk-7",
              "app": "Notes",
              "window": "Plan",
              "ts": 1700000000.0,
              "snippet": "store in sqlite"
            }
          ]
        }
        """
        let result = try decodeResult(json)
        XCTAssertEqual(result.citations.count, 1)
        let citation = try XCTUnwrap(result.citations.first)
        XCTAssertEqual(citation.observationId, "obs-42")
        XCTAssertEqual(citation.chunkId, "chunk-7")
        XCTAssertEqual(citation.app, "Notes")
        XCTAssertEqual(citation.ts, 1_700_000_000.0)
    }

    func testToleratesMissingObservationAndChunkIds() throws {
        // A citation WITHOUT the ids (older CLI / un-grounded) must still decode; the
        // ids are simply nil so navigation falls back to the day from `ts`.
        let json = """
        {
          "answer": "ok",
          "grounded": false,
          "citations": [
            {"index": 1, "app": "Code", "window": "rag.py", "ts": 1700000000.0, "snippet": "..."}
          ]
        }
        """
        let result = try decodeResult(json)
        let citation = try XCTUnwrap(result.citations.first)
        XCTAssertNil(citation.observationId)
        XCTAssertNil(citation.chunkId)
        XCTAssertEqual(citation.ts, 1_700_000_000.0)
    }

    func testDecodesNullIdsAsNil() throws {
        let json = """
        {
          "answer": "ok",
          "grounded": true,
          "citations": [
            {"index": 1, "observation_id": null, "chunk_id": null, "app": "Code",
             "window": "rag.py", "ts": 1700000000.0, "snippet": "..."}
          ]
        }
        """
        let result = try decodeResult(json)
        let citation = try XCTUnwrap(result.citations.first)
        XCTAssertNil(citation.observationId)
        XCTAssertNil(citation.chunkId)
    }
}

/// `AppModel.dayOffset(forTimestamp:)` must map a citation timestamp to the SAME
/// local calendar day the Today view / Python `_day_bounds` use, so a clicked
/// citation lands on the day that actually contains its source.
final class CitationDayOffsetTests: XCTestCase {
    private var calendar: Calendar {
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "America/New_York")!
        return c
    }

    /// A timestamp later the same calendar day as `now` is offset 0 (today).
    func testSameDayIsOffsetZero() {
        let cal = calendar
        let now = cal.date(from: DateComponents(year: 2026, month: 6, day: 24, hour: 18))!
        let earlierToday = cal.date(from: DateComponents(year: 2026, month: 6, day: 24, hour: 9))!
        let offset = AppModel.dayOffset(
            forTimestamp: earlierToday.timeIntervalSince1970, now: now, calendar: cal
        )
        XCTAssertEqual(offset, 0)
    }

    func testPreviousDayIsOffsetOne() {
        let cal = calendar
        let now = cal.date(from: DateComponents(year: 2026, month: 6, day: 24, hour: 2))!
        let yesterday = cal.date(from: DateComponents(year: 2026, month: 6, day: 23, hour: 23))!
        let offset = AppModel.dayOffset(
            forTimestamp: yesterday.timeIntervalSince1970, now: now, calendar: cal
        )
        XCTAssertEqual(offset, 1)
    }

    func testSeveralDaysAgo() {
        let cal = calendar
        let now = cal.date(from: DateComponents(year: 2026, month: 6, day: 24, hour: 12))!
        let past = cal.date(from: DateComponents(year: 2026, month: 6, day: 19, hour: 8))!
        let offset = AppModel.dayOffset(
            forTimestamp: past.timeIntervalSince1970, now: now, calendar: cal
        )
        XCTAssertEqual(offset, 5)
    }

    /// A future timestamp clamps to today (the Today view has no future days).
    func testFutureTimestampClampsToToday() {
        let cal = calendar
        let now = cal.date(from: DateComponents(year: 2026, month: 6, day: 24, hour: 12))!
        let future = cal.date(from: DateComponents(year: 2026, month: 6, day: 26, hour: 8))!
        let offset = AppModel.dayOffset(
            forTimestamp: future.timeIntervalSince1970, now: now, calendar: cal
        )
        XCTAssertEqual(offset, 0)
    }
}

/// Clicking a citation must (a) switch the active pane to Today and (b) hand the
/// computed day offset + observation id to the navigator seam. We assert the action
/// directly (no UI), since the views just forward to `AppModel.navigateToCitation`.
@MainActor
final class CitationNavigationActionTests: XCTestCase {
    private func makeModel() -> AppModel {
        AppModel(service: OpenBirdService(openBirdCLIResolver: { "/tmp/openbird" }))
    }

    func testNavigateSwitchesToTodayAndForwardsDayAndObservation() {
        let model = makeModel()
        model.selection = .ask   // start somewhere else to prove the switch

        var captured: (dayOffset: Int, observationId: String?)?
        model.citationNavigator = { day, obs in captured = (day, obs) }

        // ts = ~now so the expected offset is 0 regardless of the run date.
        let citation = ChatCitation(
            index: 1,
            observationId: "obs-99",
            chunkId: "chunk-1",
            app: "Code",
            window: "rag.py",
            ts: Date().timeIntervalSince1970,
            snippet: "..."
        )
        model.navigateToCitation(citation)

        XCTAssertEqual(model.selection, .today)
        XCTAssertEqual(captured?.dayOffset, 0)
        XCTAssertEqual(captured?.observationId, "obs-99")
    }

    /// A citation lacking an observation id still navigates (day-only) — the
    /// documented minimum behavior — passing nil through to the navigator.
    func testNavigateWithoutObservationIdStillOpensTheDay() {
        let model = makeModel()
        var captured: (dayOffset: Int, observationId: String?)?
        model.citationNavigator = { day, obs in captured = (day, obs) }

        let citation = ChatCitation(
            index: 1,
            app: "Code",
            window: "rag.py",
            ts: Date().timeIntervalSince1970,
            snippet: "..."
        )
        model.navigateToCitation(citation)

        XCTAssertEqual(model.selection, .today)
        XCTAssertEqual(captured?.dayOffset, 0)
        XCTAssertNil(captured?.observationId)
    }
}

/// A clicked briefing source reuses the SAME citation navigation as chat sources:
/// `BriefingSource.asCitation(...)` -> `AppModel.navigateToCitation` switches to
/// Today and forwards the day offset + observation id. Because a briefing source is
/// in the currently-shown day, this focuses the observation in that same day.
@MainActor
final class BriefingSourceNavigationTests: XCTestCase {
    func testBriefingSourceClickForwardsToCitationNavigator() {
        let model = AppModel(service: OpenBirdService(openBirdCLIResolver: { "/tmp/openbird" }))
        model.selection = .ask   // prove the switch to Today

        var captured: (dayOffset: Int, observationId: String?)?
        model.citationNavigator = { day, obs in captured = (day, obs) }

        let source = BriefingSource(
            observationId: "obs-src", app: "Code", window: "rag.py",
            ts: Date().timeIntervalSince1970, snippet: "edited rag.py"
        )
        // The TodayView source row's action: map to a citation, then navigate.
        model.navigateToCitation(source.asCitation(index: 1))

        XCTAssertEqual(model.selection, .today)
        XCTAssertEqual(captured?.dayOffset, 0)        // a "today" source -> today
        XCTAssertEqual(captured?.observationId, "obs-src")
    }
}

/// `TodayModel` publishes the briefing's source trail (text + sources + total) from
/// a decoded `DayBriefing`, so the view can render and link the trail.
@MainActor
final class TodayModelBriefingSourcesTests: XCTestCase {
    func testNoSourcesByDefault() {
        let model = TodayModel(service: OpenBirdService(openBirdCLIResolver: { nil }))
        XCTAssertTrue(model.briefingSources.isEmpty)
        XCTAssertEqual(model.briefingSourcesTotal, 0)
    }
}

/// `TodayModel.focus` records the observation to focus and aligns the day offset, so
/// a citation click lands the day view on the source's day.
@MainActor
final class TodayModelFocusTests: XCTestCase {
    func testFocusRecordsObservationAndSetsDay() async {
        // CLI resolver returns nil -> dayTimeline yields nil, so load() is a cheap
        // no-op here; we assert the day offset + focused id, not a real timeline fetch.
        let model = TodayModel(service: OpenBirdService(openBirdCLIResolver: { nil }))
        XCTAssertEqual(model.dayOffset, 0)

        await model.focus(dayOffset: 3, observationId: "obs-7")

        XCTAssertEqual(model.dayOffset, 3)
        XCTAssertEqual(model.focusedObservationId, "obs-7")
    }

    func testFocusClampsNegativeOffsetToZero() async {
        let model = TodayModel(service: OpenBirdService(openBirdCLIResolver: { nil }))
        await model.focus(dayOffset: -5, observationId: nil)
        XCTAssertEqual(model.dayOffset, 0)
        XCTAssertNil(model.focusedObservationId)
    }
}
