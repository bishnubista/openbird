import Foundation
import XCTest
@testable import CaptureHelperCore

final class CaptureAttemptTests: XCTestCase {
    private func jsonObject(_ event: CaptureAttemptEvent) throws -> [String: Any] {
        let data = try JSONEncoder().encode(event)
        return try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    func testStartedAttemptHasOnlyContentFreeMetadata() throws {
        let event = CaptureAttemptEvent(
            status: .started,
            attemptId: "123e4567-e89b-12d3-a456-426614174000",
            helperEpoch: "123e4567-e89b-12d3-a456-426614174001",
            triggerSeq: 7,
            triggerTs: 1000.0,
            startedTs: 1000.1,
            trigger: .appActivated,
            adapterId: "generic_ax",
            extractorVersion: "generic_ax_v1")
        let obj = try jsonObject(event)
        XCTAssertEqual(
            Set(obj.keys),
            ["type", "status", "attempt_id", "helper_epoch", "trigger_seq",
             "trigger_ts", "started_ts", "trigger", "adapter_id",
             "extractor_version", "nodes_visited", "bytes_emitted",
             "elapsed_ms", "reason_codes", "coalesced_trigger_count"])
        XCTAssertEqual(obj["type"] as? String, "capture_attempt")
        XCTAssertNil(obj["outcome"])
        XCTAssertNil(obj["finished_ts"])
    }

    func testFinishedCoalescedAttemptPinsAggregateAndSuccessor() throws {
        let event = CaptureAttemptEvent(
            status: .finished,
            attemptId: "123e4567-e89b-12d3-a456-426614174010",
            helperEpoch: "123e4567-e89b-12d3-a456-426614174001",
            triggerSeq: 8,
            triggerTs: 1001.0,
            finishedTs: 1001.1,
            trigger: .titleChanged,
            outcome: .coalescedInflight,
            completeness: CaptureCompleteness.none,
            coalescedTriggerCount: 3,
            earliestCoalescedTs: 1000.8,
            successorAttemptId: "123e4567-e89b-12d3-a456-426614174011")
        let obj = try jsonObject(event)
        XCTAssertEqual(obj["outcome"] as? String, "coalesced_inflight")
        XCTAssertEqual(obj["coalesced_trigger_count"] as? Int, 3)
        XCTAssertEqual(
            obj["successor_attempt_id"] as? String,
            "123e4567-e89b-12d3-a456-426614174011")
        XCTAssertNil(obj["text"])
        XCTAssertNil(obj["window"])
        XCTAssertNil(obj["url"])
    }
}
