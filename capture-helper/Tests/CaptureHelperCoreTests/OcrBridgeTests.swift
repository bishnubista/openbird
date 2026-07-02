// OcrBridgeTests — fake-HAL tests for the cancellable async-in-sync bridge.
//
// The consensus invariants proven here:
//   * a hung HAL times out WITHIN the deadline (the walk queue is released on
//     time — walkInFlight cannot be held for the HAL's full duration);
//   * the timed-out task is actually cancelled;
//   * a LATE completion is owned by the closed generation box: it is never
//     delivered, and it cannot bleed into a subsequent recognize() call.

import Foundation
import XCTest
@testable import CaptureHelperCore

/// A controllable fake HAL: blocks until the test releases it, then returns
/// the next scripted outcome (one per call). Records whether cancellation was
/// observed.
private final class GateFakeHAL: OcrHAL, @unchecked Sendable {
    private let lock = NSLock()
    private var _release = false
    private var _sawCancellation = false
    private var _completions = 0
    private var outcomes: [OcrHALOutcome]

    init(outcomes: [OcrHALOutcome]) {
        self.outcomes = outcomes
    }

    var sawCancellation: Bool {
        lock.lock(); defer { lock.unlock() }
        return _sawCancellation
    }

    var completions: Int {
        lock.lock(); defer { lock.unlock() }
        return _completions
    }

    func release() {
        lock.lock(); _release = true; lock.unlock()
    }

    /// Synchronous helpers (NSLock is not directly usable in async contexts).
    private func noteCancellationIfCancelled(_ cancelled: Bool) {
        guard cancelled else { return }
        lock.lock(); _sawCancellation = true; lock.unlock()
    }

    private func released() -> Bool {
        lock.lock(); defer { lock.unlock() }
        return _release
    }

    private func complete() -> OcrHALOutcome {
        lock.lock(); defer { lock.unlock() }
        _completions += 1
        return outcomes.isEmpty ? .unavailable(reason: "ocr_error") : outcomes.removeFirst()
    }

    func recognizeFrontWindow(pid: Int32, axTitle: String?) async -> OcrHALOutcome {
        // Spin-with-sleep until released (observing cancellation like the real
        // SCK/Vision awaits do at their suspension points).
        while !released() {
            noteCancellationIfCancelled(Task.isCancelled)
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        noteCancellationIfCancelled(Task.isCancelled)
        return complete()
    }
}

/// Immediate fake for the happy/skip paths.
private struct ImmediateFakeHAL: OcrHAL {
    let outcome: OcrHALOutcome
    func recognizeFrontWindow(pid: Int32, axTitle: String?) async -> OcrHALOutcome {
        outcome
    }
}

final class OcrBridgeTests: XCTestCase {

    func testTextOutcomePassesThrough() {
        let bridge = OcrBridge(hal: ImmediateFakeHAL(outcome: .text("recognized")))
        XCTAssertEqual(
            bridge.recognize(pid: 1, axTitle: nil, timeout: 2.0),
            .text("recognized"))
    }

    func testUnavailableOutcomeBecomesSkippedReason() {
        let bridge = OcrBridge(hal: ImmediateFakeHAL(outcome: .unavailable(reason: "ocr_no_window")))
        XCTAssertEqual(
            bridge.recognize(pid: 1, axTitle: nil, timeout: 2.0),
            .skipped(reason: "ocr_no_window"))
    }

    func testNonPositiveTimeoutIsAnImmediateTimeout() {
        // The combined AX+OCR budget can be exhausted before OCR starts; the
        // bridge must not launch work it cannot wait for.
        let bridge = OcrBridge(hal: ImmediateFakeHAL(outcome: .text("never")))
        XCTAssertEqual(bridge.recognize(pid: 1, axTitle: nil, timeout: 0), .timeout)
    }

    func testHungHalTimesOutWithinDeadlineAndIsCancelled() {
        let hal = GateFakeHAL(outcomes: [.text("late text")])
        let bridge = OcrBridge(hal: hal)

        let start = DispatchTime.now()
        let outcome = bridge.recognize(pid: 1, axTitle: nil, timeout: 0.2)
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1e9

        // Deadline honored: the caller (walk queue) is released promptly —
        // this IS the walkInFlight-releases-within-deadline guarantee, since
        // dispatchCapture clears the flag as soon as captureFrontmost returns.
        XCTAssertEqual(outcome, .timeout)
        XCTAssertLessThan(elapsed, 1.0)

        // The retained task was cancelled; the fake observes it at its next
        // suspension point.
        let cancelDeadline = Date().addingTimeInterval(2.0)
        while !hal.sawCancellation && Date() < cancelDeadline {
            RunLoop.current.run(until: Date().addingTimeInterval(0.01))
        }
        XCTAssertTrue(hal.sawCancellation)
        hal.release()  // let the task finish (late) — nothing may be delivered
    }

    func testLateCompletionIsNeverDeliveredAndCannotPolluteNextCall() {
        // Call 1 will complete LATE with "stale text"; call 2 (post-release)
        // completes immediately with its own distinct outcome.
        let hal = GateFakeHAL(outcomes: [.text("stale text"), .text("second call")])
        let bridge = OcrBridge(hal: hal)

        // First call times out while the fake is still hung.
        XCTAssertEqual(bridge.recognize(pid: 1, axTitle: nil, timeout: 0.1), .timeout)

        // Release the stale task NOW so it completes "late" and wait for it.
        hal.release()
        let deadline = Date().addingTimeInterval(2.0)
        while hal.completions == 0 && Date() < deadline {
            RunLoop.current.run(until: Date().addingTimeInterval(0.01))
        }
        XCTAssertEqual(hal.completions, 1, "late task never completed")

        // A second call on the SAME bridge gets a fresh generation box and
        // ITS OWN result — the first call's stale text can never surface.
        XCTAssertEqual(
            bridge.recognize(pid: 1, axTitle: nil, timeout: 2.0),
            .text("second call"))
    }
}
