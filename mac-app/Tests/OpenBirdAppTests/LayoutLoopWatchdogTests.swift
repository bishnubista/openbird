import XCTest
@testable import OpenBirdApp

final class LayoutLoopWatchdogTests: XCTestCase {
    // MARK: - stallReport

    func testFreshHeartbeatDoesNotReport() {
        // Ticked 1s ago, threshold 5s → not stalled.
        XCTAssertNil(LayoutLoopWatchdog.stallReport(
            lastTick: 100, now: 101, threshold: 5, alreadyReported: false))
    }

    func testStaleHeartbeatReportsRoundedSeconds() {
        // Ticked 7.4s ago, threshold 5s, not yet reported → report 7.
        XCTAssertEqual(LayoutLoopWatchdog.stallReport(
            lastTick: 100, now: 107.4, threshold: 5, alreadyReported: false), 7)
    }

    func testStaleHeartbeatDedupesWithinOneEpisode() {
        // Already reported this stall → stay silent even though still stale.
        XCTAssertNil(LayoutLoopWatchdog.stallReport(
            lastTick: 100, now: 110, threshold: 5, alreadyReported: true))
    }

    func testExactThresholdReports() {
        // elapsed == threshold counts as stalled (>=).
        XCTAssertEqual(LayoutLoopWatchdog.stallReport(
            lastTick: 100, now: 105, threshold: 5, alreadyReported: false), 5)
    }

    // MARK: - stallCleared

    func testStallClearsWhenHeartbeatFresh() {
        XCTAssertTrue(LayoutLoopWatchdog.stallCleared(lastTick: 100, now: 101, threshold: 5))
    }

    func testStallNotClearedWhileStillStale() {
        XCTAssertFalse(LayoutLoopWatchdog.stallCleared(lastTick: 100, now: 108, threshold: 5))
    }

    // MARK: - isEnabled

    func testEnvExplicitlyEnables() {
        XCTAssertTrue(LayoutLoopWatchdog.isEnabled(
            environment: ["OPENBIRD_LAYOUT_WATCHDOG": "1"], isDebug: false))
    }

    func testEnvZeroDisablesEvenInDebug() {
        XCTAssertFalse(LayoutLoopWatchdog.isEnabled(
            environment: ["OPENBIRD_LAYOUT_WATCHDOG": "0"], isDebug: true))
    }

    func testEnvEmptyStringDisablesEvenInDebug() {
        XCTAssertFalse(LayoutLoopWatchdog.isEnabled(
            environment: ["OPENBIRD_LAYOUT_WATCHDOG": ""], isDebug: true))
    }

    func testAbsentEnvFollowsDebugFlag() {
        XCTAssertTrue(LayoutLoopWatchdog.isEnabled(environment: [:], isDebug: true))
        XCTAssertFalse(LayoutLoopWatchdog.isEnabled(environment: [:], isDebug: false))
    }
}
