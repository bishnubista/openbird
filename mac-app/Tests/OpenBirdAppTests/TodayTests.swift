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
        XCTAssertEqual(OpenBirdService.parseBriefing("{\"text\": \"You worked on X.\"}"), "You worked on X.")
        XCTAssertNil(OpenBirdService.parseBriefing("nope"))
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
