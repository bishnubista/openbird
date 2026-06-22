import XCTest
@testable import OpenBirdApp

final class TimelineFormattingTests: XCTestCase {
    func testDurationLabel() {
        XCTAssertEqual(TimelineFormatting.durationLabel(0), "0s")
        XCTAssertEqual(TimelineFormatting.durationLabel(30), "30s")
        XCTAssertEqual(TimelineFormatting.durationLabel(90), "1m")
        XCTAssertEqual(TimelineFormatting.durationLabel(3661), "1h 1m")
        // 2h 28m — the handoff's first session duration.
        XCTAssertEqual(TimelineFormatting.durationLabel(8880), "2h 28m")
    }

    func testGroupedSeparatesThousands() {
        XCTAssertEqual(TimelineFormatting.grouped(1284), "1,284")
        XCTAssertEqual(TimelineFormatting.grouped(0), "0")
    }

    func testTodayViewForwardsToSharedFormatter() {
        // The Today rail must format identically to the compact rail (one source of truth).
        XCTAssertEqual(TodayView.durationLabel(8880), TimelineFormatting.durationLabel(8880))
        XCTAssertEqual(TodayView.grouped(1284), TimelineFormatting.grouped(1284))
    }
}

@MainActor
final class TimelineModelTests: XCTestCase {
    func testDayTitleForTodayAndYesterday() {
        let model = TimelineModel(service: OpenBirdService())
        model.dayOffset = 0
        XCTAssertEqual(model.dayTitle, "Today")
        model.dayOffset = 1
        XCTAssertEqual(model.dayTitle, "Yesterday")
    }

    func testDayHeadingIsUppercaseFullDate() {
        let model = TimelineModel(service: OpenBirdService())
        model.dayOffset = 0
        let heading = model.dayHeading
        // Uppercased weekday/month/day, e.g. "MONDAY, JUNE 22".
        XCTAssertEqual(heading, heading.uppercased())
        XCTAssertTrue(heading.contains(","))
    }

    func testSessionSummaryEmptyBeforeLoad() {
        let model = TimelineModel(service: OpenBirdService())
        XCTAssertEqual(model.sessionSummary, "")
    }

    func testDefaultsToTodayOffset() {
        let model = TimelineModel(service: OpenBirdService())
        XCTAssertEqual(model.dayOffset, 0)
    }
}
