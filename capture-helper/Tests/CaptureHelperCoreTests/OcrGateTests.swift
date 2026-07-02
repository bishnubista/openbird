// OcrGateTests — deterministic tests for the pure OCR fallback gate.
//
// Everything runs against an injected monotonic clock (plain Doubles): no AX,
// no ScreenCaptureKit, no Vision, no timers. The gate-order contract, the
// closed reason vocabulary, and the consume-at-decision-time throttle
// semantics are all pinned here.

import XCTest
@testable import CaptureHelperCore

final class OcrGateTests: XCTestCase {

    /// All-pass inputs; individual tests flip exactly one gate.
    private func decide(
        _ gate: inout OcrGate,
        bundleId: String = "com.example.app",
        axTextEmpty: Bool = true,
        optedIn: Bool = true,
        tccGranted: Bool = true,
        micHot: Bool = false,
        now: Double
    ) -> OcrDecision {
        gate.decide(
            bundleId: bundleId, axTextEmpty: axTextEmpty, optedIn: optedIn,
            tccGranted: tccGranted, micHot: micHot, now: now)
    }

    // MARK: gate order + distinct reason codes

    func testAllGatesPassYieldsRun() {
        var gate = OcrGate(minInterval: 30)
        XCTAssertEqual(decide(&gate, now: 100), .run)
    }

    func testAxTextPresentSkips() {
        var gate = OcrGate(minInterval: 30)
        XCTAssertEqual(
            decide(&gate, axTextEmpty: false, now: 100),
            .skip(reason: "ax_text_present"))
    }

    func testNotOptedInSkips() {
        var gate = OcrGate(minInterval: 30)
        XCTAssertEqual(
            decide(&gate, optedIn: false, now: 100),
            .skip(reason: "not_opted_in"))
    }

    func testTccDeniedSkips() {
        var gate = OcrGate(minInterval: 30)
        XCTAssertEqual(
            decide(&gate, tccGranted: false, now: 100),
            .skip(reason: "tcc_denied"))
    }

    func testMicHotSkips() {
        var gate = OcrGate(minInterval: 30)
        XCTAssertEqual(
            decide(&gate, micHot: true, now: 100),
            .skip(reason: "mic_hot"))
    }

    func testGateOrderEarlierGateWins() {
        // Every gate failing at once must report the FIRST reason in the
        // documented order (ax -> opt-in -> tcc -> mic -> throttle).
        var gate = OcrGate(minInterval: 30)
        XCTAssertEqual(
            decide(
                &gate, axTextEmpty: false, optedIn: false, tccGranted: false,
                micHot: true, now: 100),
            .skip(reason: "ax_text_present"))
        // Drop the ax gate: opt-in is next.
        XCTAssertEqual(
            decide(&gate, optedIn: false, tccGranted: false, micHot: true, now: 100),
            .skip(reason: "not_opted_in"))
        // Drop opt-in: TCC is next.
        XCTAssertEqual(
            decide(&gate, tccGranted: false, micHot: true, now: 100),
            .skip(reason: "tcc_denied"))
        // Drop TCC: mic is next (before the throttle).
        XCTAssertEqual(
            decide(&gate, micHot: true, now: 100),
            .skip(reason: "mic_hot"))
    }

    // MARK: throttle — consumed at decision time, per-app, boundary-exact

    func testRunConsumesThrottleSlotAtDecisionTime() {
        var gate = OcrGate(minInterval: 30)
        XCTAssertEqual(decide(&gate, now: 100), .run)
        // Immediately again: throttled — the slot was consumed by the DECISION,
        // regardless of whether the OCR attempt later succeeded or timed out.
        XCTAssertEqual(decide(&gate, now: 100.1), .skip(reason: "throttled"))
    }

    func testThrottleIntervalBoundary() {
        var gate = OcrGate(minInterval: 30)
        XCTAssertEqual(decide(&gate, now: 100), .run)
        // Just inside the interval: still throttled.
        XCTAssertEqual(decide(&gate, now: 129.999), .skip(reason: "throttled"))
        // Exactly at the interval: allowed again.
        XCTAssertEqual(decide(&gate, now: 130), .run)
    }

    func testThrottleIsPerAppIndependent() {
        var gate = OcrGate(minInterval: 30)
        XCTAssertEqual(decide(&gate, bundleId: "com.example.a", now: 100), .run)
        // A different app is NOT throttled by app A's slot.
        XCTAssertEqual(decide(&gate, bundleId: "com.example.b", now: 101), .run)
        // Each app's own slot holds independently.
        XCTAssertEqual(
            decide(&gate, bundleId: "com.example.a", now: 102),
            .skip(reason: "throttled"))
        XCTAssertEqual(
            decide(&gate, bundleId: "com.example.b", now: 102),
            .skip(reason: "throttled"))
    }

    func testSkipsDoNotConsumeThrottleSlot() {
        var gate = OcrGate(minInterval: 30)
        // A mic-hot skip must not start the interval…
        XCTAssertEqual(decide(&gate, micHot: true, now: 100), .skip(reason: "mic_hot"))
        // …so the very next mic-quiet capture may run immediately.
        XCTAssertEqual(decide(&gate, now: 101), .run)
    }

    func testTccRevokedMidRunSkipsWithoutPrompting() {
        var gate = OcrGate(minInterval: 30)
        XCTAssertEqual(decide(&gate, now: 100), .run)
        // Grant revoked between captures: the gate reports tcc_denied (the
        // caller only ever preflights — it never prompts).
        XCTAssertEqual(
            decide(&gate, tccGranted: false, now: 200),
            .skip(reason: "tcc_denied"))
        // Re-granted: runs again (interval long since elapsed).
        XCTAssertEqual(decide(&gate, now: 300), .run)
    }

    // MARK: defensive clamp (argv is operator-editable)

    func testMinIntervalIsDefensivelyClampedToFloor() {
        var gate = OcrGate(minInterval: 0)  // typo'd flag must not defeat the budget
        XCTAssertEqual(decide(&gate, now: 100), .run)
        XCTAssertEqual(decide(&gate, now: 105), .skip(reason: "throttled"))
        XCTAssertEqual(decide(&gate, now: 110), .run)  // floor is 10s
    }

    func testNonFiniteMinIntervalFallsBackToDefault() {
        var gate = OcrGate(minInterval: .nan)
        XCTAssertEqual(decide(&gate, now: 100), .run)
        XCTAssertEqual(decide(&gate, now: 129), .skip(reason: "throttled"))
        XCTAssertEqual(decide(&gate, now: 130), .run)  // default is 30s
    }
}
