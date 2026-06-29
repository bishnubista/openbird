import XCTest
@testable import OpenBirdApp

final class SettingsBannerDecisionTests: XCTestCase {
    func testActiveCaptureWithNoObservationsWaitsForMemory() {
        let decision = readyPrerequisites(
            captureRunning: true,
            capturePaused: false,
            observationCount: 0,
            nextStepSummary: "Next: bring an allowed app to the front so memory starts filling."
        )

        XCTAssertEqual(decision.tone, .amber)
        XCTAssertEqual(decision.title, "Capturing · waiting for memory")
        XCTAssertEqual(decision.subtitle, "Next: bring an allowed app to the front so memory starts filling.")
        XCTAssertNil(decision.buttonLabel)
        XCTAssertEqual(decision.actionKind, .none)
    }

    func testPausedCaptureWithNoObservationsDoesNotClaimActiveCapture() {
        let decision = readyPrerequisites(
            captureRunning: true,
            capturePaused: true,
            observationCount: 0,
            nextStepSummary: "Next: bring an allowed app to the front so memory starts filling."
        )

        XCTAssertEqual(decision.tone, .amber)
        XCTAssertEqual(decision.title, "Paused · waiting for memory")
        XCTAssertEqual(
            decision.subtitle,
            "Resume capture and bring an allowed app to the front so memory starts filling."
        )
        XCTAssertNil(decision.buttonLabel)
        XCTAssertEqual(decision.actionKind, .none)
    }

    func testCapturedMemoryShowsGreenMemoryReady() {
        let decision = readyPrerequisites(
            captureRunning: true,
            capturePaused: false,
            observationCount: 3
        )

        XCTAssertEqual(decision.tone, .green)
        XCTAssertEqual(decision.title, "Capturing · memory ready")
        XCTAssertEqual(decision.subtitle, "2 apps effectively allowed · encrypted · on-device")
        XCTAssertNil(decision.buttonLabel)
        XCTAssertEqual(decision.actionKind, .none)
    }

    func testRefreshingWithCapturedMemoryStaysGreen() {
        let decision = readyPrerequisites(
            captureRunning: true,
            capturePaused: false,
            observationCount: 3,
            memoryStatsState: .loaded(MemoryStats(observations: 3, blobs: 1, chunks: 1, vectors: 1)),
            isRefreshing: true
        )

        XCTAssertEqual(decision.tone, .green)
        XCTAssertEqual(decision.title, "Capturing · memory ready")
    }

    func testUnknownMemoryStatsChecksWithoutClaimingEmptyMemory() {
        let decision = readyPrerequisites(
            captureRunning: true,
            capturePaused: false,
            observationCount: 0,
            memoryStatsState: .unknown,
            isRefreshing: true,
            nextStepSummary: "Next: bring an allowed app to the front so memory starts filling."
        )

        XCTAssertEqual(decision.tone, .amber)
        XCTAssertEqual(decision.title, "Capturing · checking memory")
        XCTAssertEqual(decision.subtitle, "OpenBird is verifying whether captured memory is available.")
        XCTAssertFalse(decision.subtitle.contains("starts filling"))
    }

    func testFailedMemoryStatsDoesNotClaimEmptyMemory() {
        let decision = readyPrerequisites(
            captureRunning: true,
            capturePaused: false,
            observationCount: 0,
            memoryStatsState: .failed,
            nextStepSummary: "Next: bring an allowed app to the front so memory starts filling."
        )

        XCTAssertEqual(decision.tone, .amber)
        XCTAssertEqual(decision.title, "Memory status unavailable")
        XCTAssertEqual(decision.subtitle, "Re-check setup to verify captured memory.")
        XCTAssertFalse(decision.subtitle.contains("starts filling"))
    }

    private func readyPrerequisites(
        captureRunning: Bool,
        capturePaused: Bool,
        observationCount: Int,
        memoryStatsState: MemoryStatsState? = nil,
        isRefreshing: Bool = false,
        nextStepSummary: String = "Ready: capture is storing memory. Ask a question when you need it."
    ) -> SettingsBannerDecision {
        let statsState = memoryStatsState
            ?? .loaded(MemoryStats(observations: observationCount, blobs: 1, chunks: 1, vectors: 1))
        return SettingsBannerDecision.resolve(
            captureNeedsReindex: false,
            isReindexing: false,
            accessibilityState: .ok,
            localModelStatusState: .ok,
            localModelStatusSummary: "Models ready.",
            modelRouteActionLabel: nil,
            allowlistCount: 2,
            captureRunning: captureRunning,
            capturePaused: capturePaused,
            observationCount: observationCount,
            memoryStatsState: statsState,
            isRefreshing: isRefreshing,
            nextStepSummary: nextStepSummary,
            allGoodSubtitle: "2 apps effectively allowed · encrypted · on-device"
        )
    }
}
