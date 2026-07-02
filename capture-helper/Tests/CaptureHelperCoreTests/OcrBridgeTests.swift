// OcrBridgeTests — fake-HAL tests for the two-phase cancellable bridge.
//
// The consensus + review invariants proven here:
//   * a hung ACQUIRE phase times out WITHIN the deadline (the walk queue is
//     released on time — walkInFlight cannot be held for the HAL's full
//     duration) and the task is actually cancelled;
//   * a LATE acquire completion is owned by the closed generation box: its
//     image is dropped UNREAD (never recognized), never delivered, and cannot
//     bleed into a subsequent recognize() call;
//   * SCOPE-BOUND PIXELS (the reviewed HIGH finding): once an image exists,
//     the bridge WAITS for the synchronous recognize phase — which
//     cancellation cannot interrupt — so the bridge never returns while the
//     HAL still holds pixels, even when recognition blows past the deadline.

import Foundation
import XCTest
@testable import CaptureHelperCore

/// Immediate fake for the happy/skip paths.
private struct ImmediateHAL: OcrHAL {
    let acquire: OcrAcquireOutcome
    let recognized: OcrHALOutcome

    func acquireImage(pid: Int32, axTitle: String?) async -> OcrAcquireOutcome {
        acquire
    }

    func recognize(image: OcrImageHandle) -> OcrHALOutcome {
        recognized
    }
}

/// A controllable acquire-phase fake: blocks until the test releases it, then
/// returns the scripted outcome. Records cancellation observations and
/// whether recognize() was ever reached.
private final class HungAcquireHAL: OcrHAL, @unchecked Sendable {
    private let lock = NSLock()
    private var _release = false
    private var _sawCancellation = false
    private var _acquireCompletions = 0
    private var _recognizeCalls = 0
    let acquireOutcome: OcrAcquireOutcome
    let recognizeOutcome: OcrHALOutcome

    init(acquireOutcome: OcrAcquireOutcome, recognizeOutcome: OcrHALOutcome) {
        self.acquireOutcome = acquireOutcome
        self.recognizeOutcome = recognizeOutcome
    }

    var sawCancellation: Bool {
        lock.lock(); defer { lock.unlock() }
        return _sawCancellation
    }

    var acquireCompletions: Int {
        lock.lock(); defer { lock.unlock() }
        return _acquireCompletions
    }

    var recognizeCalls: Int {
        lock.lock(); defer { lock.unlock() }
        return _recognizeCalls
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

    private func completeAcquire() -> OcrAcquireOutcome {
        lock.lock(); defer { lock.unlock() }
        _acquireCompletions += 1
        return acquireOutcome
    }

    func acquireImage(pid: Int32, axTitle: String?) async -> OcrAcquireOutcome {
        // Spin-with-sleep until released (observing cancellation like the
        // real SCK awaits do at their suspension points).
        while !released() {
            noteCancellationIfCancelled(Task.isCancelled)
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        noteCancellationIfCancelled(Task.isCancelled)
        return completeAcquire()
    }

    func recognize(image: OcrImageHandle) -> OcrHALOutcome {
        lock.lock(); _recognizeCalls += 1; lock.unlock()
        return recognizeOutcome
    }
}

/// The 'pixel retained' regression fake: acquire yields an image instantly;
/// recognize marks the pixels as live, then blocks NON-COOPERATIVELY
/// (Thread.sleep — like Vision's synchronous perform, it ignores task
/// cancellation) well past the bridge deadline before releasing them.
private final class SlowVisionHAL: OcrHAL, @unchecked Sendable {
    private let lock = NSLock()
    private var _pixelRetained = false
    let visionSeconds: TimeInterval

    init(visionSeconds: TimeInterval) {
        self.visionSeconds = visionSeconds
    }

    var pixelRetained: Bool {
        lock.lock(); defer { lock.unlock() }
        return _pixelRetained
    }

    func acquireImage(pid: Int32, axTitle: String?) async -> OcrAcquireOutcome {
        .image(OcrImageHandle("fake pixels"))
    }

    func recognize(image: OcrImageHandle) -> OcrHALOutcome {
        lock.lock(); _pixelRetained = true; lock.unlock()
        Thread.sleep(forTimeInterval: visionSeconds)  // non-cooperative, like Vision
        lock.lock(); _pixelRetained = false; lock.unlock()
        return .text("slow vision")
    }
}

final class OcrBridgeTests: XCTestCase {

    func testTextOutcomePassesThrough() {
        let bridge = OcrBridge(hal: ImmediateHAL(
            acquire: .image(OcrImageHandle("px")), recognized: .text("recognized")))
        XCTAssertEqual(
            bridge.recognize(pid: 1, axTitle: nil, timeout: 2.0),
            .text("recognized"))
    }

    func testAcquireUnavailableBecomesSkippedReason() {
        let bridge = OcrBridge(hal: ImmediateHAL(
            acquire: .unavailable(reason: "ocr_no_window"),
            recognized: .text("never reached")))
        XCTAssertEqual(
            bridge.recognize(pid: 1, axTitle: nil, timeout: 2.0),
            .skipped(reason: "ocr_no_window"))
    }

    func testRecognizeUnavailableBecomesSkippedReason() {
        let bridge = OcrBridge(hal: ImmediateHAL(
            acquire: .image(OcrImageHandle("px")),
            recognized: .unavailable(reason: "ocr_empty")))
        XCTAssertEqual(
            bridge.recognize(pid: 1, axTitle: nil, timeout: 2.0),
            .skipped(reason: "ocr_empty"))
    }

    func testNonPositiveTimeoutIsAnImmediateTimeout() {
        // The combined AX+OCR budget can be exhausted before OCR starts; the
        // bridge must not launch work it cannot wait for.
        let bridge = OcrBridge(hal: ImmediateHAL(
            acquire: .image(OcrImageHandle("px")), recognized: .text("never")))
        XCTAssertEqual(bridge.recognize(pid: 1, axTitle: nil, timeout: 0), .timeout)
    }

    // MARK: acquire phase — deadline-raced and cancellable

    func testHungAcquireTimesOutWithinDeadlineAndIsCancelled() {
        let hal = HungAcquireHAL(
            acquireOutcome: .unavailable(reason: "ocr_error"),
            recognizeOutcome: .text("never"))
        let bridge = OcrBridge(hal: hal)

        let start = DispatchTime.now()
        let outcome = bridge.recognize(pid: 1, axTitle: nil, timeout: 0.2)
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1e9

        // Deadline honored for the PRE-IMAGE phase: the caller (walk queue)
        // is released promptly — this IS the walkInFlight-releases-within-
        // deadline guarantee, since dispatchCapture clears the flag as soon
        // as captureFrontmost returns.
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

    func testLateAcquiredImageIsDroppedUnreadAndCannotPolluteNextCall() {
        // Call 1's acquire completes LATE with an image; the closed box must
        // drop it WITHOUT ever recognizing it (a late image never reaches the
        // Vision phase — pixel + content invariant in one).
        let hal = HungAcquireHAL(
            acquireOutcome: .image(OcrImageHandle("stale pixels")),
            recognizeOutcome: .text("stale text"))
        let bridge = OcrBridge(hal: hal)

        // First call times out while the fake acquire is still hung.
        XCTAssertEqual(bridge.recognize(pid: 1, axTitle: nil, timeout: 0.1), .timeout)
        XCTAssertEqual(hal.recognizeCalls, 0)

        // Release the stale acquire NOW so it completes "late" and wait.
        hal.release()
        let deadline = Date().addingTimeInterval(2.0)
        while hal.acquireCompletions == 0 && Date() < deadline {
            RunLoop.current.run(until: Date().addingTimeInterval(0.01))
        }
        XCTAssertEqual(hal.acquireCompletions, 1, "late acquire never completed")
        // The late image was dropped unread: recognize was NEVER invoked.
        XCTAssertEqual(hal.recognizeCalls, 0)

        // A second call on the SAME bridge gets a fresh generation box and
        // ITS OWN result — the stale first call can never surface through it.
        XCTAssertEqual(
            bridge.recognize(pid: 1, axTitle: nil, timeout: 2.0),
            .text("stale text"))
        XCTAssertEqual(hal.recognizeCalls, 1)  // recognized ITS OWN image, once
    }

    // MARK: recognize phase — scope-bound pixels (the reviewed HIGH finding)

    func testBridgeWaitsOutNonCooperativeVisionSoPixelsNeverOutliveReturn() {
        // Regression: recognize enters a 'pixel retained' phase and blocks
        // non-cooperatively far past the acquire deadline. The bridge must
        // NOT return while that flag is true — it waits for the image scope
        // to close and returns the real result, deadline notwithstanding.
        let hal = SlowVisionHAL(visionSeconds: 0.4)
        let bridge = OcrBridge(hal: hal)

        let start = DispatchTime.now()
        let outcome = bridge.recognize(pid: 1, axTitle: nil, timeout: 0.1)
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1e9

        XCTAssertEqual(outcome, .text("slow vision"))  // waited, no .timeout
        XCTAssertGreaterThanOrEqual(elapsed, 0.4)      // for the WHOLE pass
        XCTAssertFalse(hal.pixelRetained)              // pixels dead at return
    }
}
