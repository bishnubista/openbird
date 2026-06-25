import XCTest
@testable import OpenBirdApp

final class TodayTimelineDecodingTests: XCTestCase {
    func testParseDayTimelineDecodesCliJson() {
        let json = """
        {
          "day_offset": 1,
          "start": 1.0,
          "end": 2.0,
          "total_observations": 3,
          "distinct_apps": 2,
          "active_seconds": 120.5,
          "sessions": [
            {"session_id": "s1", "app": "com.google.Chrome", "start": 1.0, "end": 1.5, "count": 2},
            {"session_id": null, "app": "com.apple.finder", "start": 1.6, "end": 1.6, "count": 1}
          ]
        }
        """
        let timeline = OpenBirdService.parseDayTimeline(json)
        XCTAssertEqual(timeline?.dayOffset, 1)
        XCTAssertEqual(timeline?.totalObservations, 3)
        XCTAssertEqual(timeline?.distinctApps, 2)
        XCTAssertEqual(timeline?.activeSeconds, 120.5)
        XCTAssertEqual(timeline?.sessions.count, 2)
        XCTAssertEqual(timeline?.sessions.first?.app, "com.google.Chrome")
        XCTAssertEqual(timeline?.sessions.first?.count, 2)
        XCTAssertNil(timeline?.sessions.last?.sessionId)
    }

    func testParseDayTimelineRejectsNonJson() {
        XCTAssertNil(OpenBirdService.parseDayTimeline("not json"))
    }

    func testParseBriefingExtractsText() {
        let briefing = OpenBirdService.parseBriefing("{\"text\": \"You worked on X.\"}")
        XCTAssertEqual(briefing?.text, "You worked on X.")
        // No sources key -> empty trail, total defaults to 0 (a stale-CLI briefing).
        XCTAssertEqual(briefing?.sources, [])
        XCTAssertEqual(briefing?.sourcesTotal, 0)
        XCTAssertNil(OpenBirdService.parseBriefing("nope"))
    }

    func testParseBriefingDecodesSourceTrail() {
        // Mirrors `openbird briefing --json` (cli.py -> select_briefing_sources).
        let json = """
        {
          "day_offset": 0,
          "start": 1.0,
          "end": 2.0,
          "text": "You edited rag.py and took notes.",
          "sources": [
            {"observation_id": "o2", "app": "Code", "window": "rag.py",
             "ts": 1700000060.0, "snippet": "edited rag.py"},
            {"observation_id": "o1", "app": "Notes", "window": null,
             "ts": 1700000000.0, "snippet": "took notes"}
          ],
          "sources_total": 5
        }
        """
        let briefing = OpenBirdService.parseBriefing(json)
        XCTAssertEqual(briefing?.text, "You edited rag.py and took notes.")
        XCTAssertEqual(briefing?.sources.count, 2)
        XCTAssertEqual(briefing?.sourcesTotal, 5)   // capped: 2 of 5 surfaced
        let first = try? XCTUnwrap(briefing?.sources.first)
        XCTAssertEqual(first?.observationId, "o2")
        XCTAssertEqual(first?.window, "rag.py")
        XCTAssertEqual(first?.ts, 1_700_000_060.0)
        XCTAssertNil(briefing?.sources.last?.window)   // null window decodes to nil
    }

    func testParseBriefingEmptyDayHasNoSources() {
        let json = """
        {"day_offset": 0, "start": 1.0, "end": 2.0,
         "text": "[yesterday] No activity recorded in the selected window.",
         "sources": [], "sources_total": 0}
        """
        let briefing = OpenBirdService.parseBriefing(json)
        XCTAssertEqual(briefing?.sources, [])
        XCTAssertEqual(briefing?.sourcesTotal, 0)
    }

    /// A briefing source maps to a `ChatCitation` carrying the navigable fields, so
    /// a clicked source reuses the chat citation navigation unchanged.
    func testBriefingSourceMapsToCitation() {
        let source = BriefingSource(
            observationId: "obs-7", app: "Code", window: "rag.py",
            ts: 1_700_000_000.0, snippet: "edited rag.py"
        )
        let citation = source.asCitation(index: 3)
        XCTAssertEqual(citation.index, 3)
        XCTAssertEqual(citation.observationId, "obs-7")
        XCTAssertNil(citation.chunkId)
        XCTAssertEqual(citation.app, "Code")
        XCTAssertEqual(citation.window, "rag.py")
        XCTAssertEqual(citation.ts, 1_700_000_000.0)
        XCTAssertEqual(citation.snippet, "edited rag.py")
    }
}

final class AppDisplayTests: XCTestCase {
    func testNilOrEmptyBundleIsUnknown() {
        XCTAssertEqual(AppDisplay.name(nil), "Unknown")
        XCTAssertEqual(AppDisplay.name(""), "Unknown")
    }

    func testFallbackCapitalizesLastComponent() {
        // Deterministic (machine-independent) path for an unresolvable bundle id.
        XCTAssertEqual(AppDisplay.fallbackName("com.example.someApp"), "SomeApp")
        XCTAssertEqual(AppDisplay.fallbackName("widget"), "Widget")
    }
}

final class TodayFormattingTests: XCTestCase {
    func testDurationLabel() {
        XCTAssertEqual(TodayView.durationLabel(30), "30s")
        XCTAssertEqual(TodayView.durationLabel(90), "1m")
        XCTAssertEqual(TodayView.durationLabel(3661), "1h 1m")
        XCTAssertEqual(TodayView.durationLabel(0), "0s")
    }
}

@MainActor
final class TodayModelTests: XCTestCase {
    func testDayTitleForTodayAndYesterday() {
        let model = TodayModel(service: OpenBirdService())
        model.dayOffset = 0
        XCTAssertEqual(model.dayTitle, "Today")
        model.dayOffset = 1
        XCTAssertEqual(model.dayTitle, "Yesterday")
    }
}
