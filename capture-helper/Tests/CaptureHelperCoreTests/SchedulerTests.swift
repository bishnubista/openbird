import XCTest
@testable import CaptureHelperCore

/// Deterministic tests for the capture-cadence state machine. Every rule that
/// reached review consensus has a test here: debounce coalescing, the >=1s
/// floor, the force-capture ceiling ("stuck forever" regression), AFK
/// suppression of BOTH idle ticks and the ceiling, trigger-while-AFK HID
/// re-sampling, and monotonic-only comparisons.
final class SchedulerTests: XCTestCase {
    private func makeScheduler(
        minGap: Double = 1.0,
        idleTick: Double = 5.0,
        ceiling: Double = 60.0,
        afk: Double = 150.0
    ) -> Scheduler {
        Scheduler(
            config: SchedulerConfig(
                minGap: minGap, idleTick: idleTick,
                forceCeiling: ceiling, afkThreshold: afk
            ))
    }

    private func captures(_ actions: [SchedulerAction]) -> [TriggerKind] {
        actions.compactMap {
            if case let .capture(kind) = $0 { return kind }
            return nil
        }
    }

    // MARK: debounce

    func testAppTriggerDebounces300ms() {
        var s = makeScheduler()
        XCTAssertEqual(s.trigger(.appActivated, now: 10.0), [])
        // Before the deadline: nothing fires (heartbeat only).
        XCTAssertEqual(captures(s.tick(now: 10.2, idleSeconds: 0.1)), [])
        // After the deadline: the debounced capture fires.
        XCTAssertEqual(captures(s.tick(now: 10.35, idleSeconds: 0.1)), [.appActivated])
    }

    func testTriggerStormCoalescesToOneCapture() {
        var s = makeScheduler()
        _ = s.trigger(.appActivated, now: 10.0)
        _ = s.trigger(.titleChanged, now: 10.1)
        _ = s.trigger(.focusChanged, now: 10.2)
        // One capture, at the EARLIEST armed deadline (10.3), latest kind.
        let fired = captures(s.tick(now: 10.31, idleSeconds: 0.1))
        XCTAssertEqual(fired, [.focusChanged])
        // Storm fully drained: next tick inside idleTick window is quiet.
        XCTAssertEqual(captures(s.tick(now: 10.5, idleSeconds: 0.1)), [])
    }

    // MARK: floor

    func testMinGapFloorDefersBackToBackCaptures() {
        var s = makeScheduler(minGap: 1.0)
        _ = s.trigger(.appActivated, now: 10.0)
        XCTAssertEqual(captures(s.tick(now: 10.31, idleSeconds: 0.1)), [.appActivated])
        // A second trigger 100ms later must NOT capture before the 1s floor.
        _ = s.trigger(.windowChanged, now: 10.4)
        XCTAssertEqual(captures(s.tick(now: 10.75, idleSeconds: 0.1)), [])
        // Once the floor clears (>= 11.31), it fires.
        XCTAssertEqual(captures(s.tick(now: 11.32, idleSeconds: 0.1)), [.windowChanged])
    }

    // MARK: ceiling — the "stuck at 31 frames forever" regression

    func testForceCeilingFiresWithoutAnyTriggers() {
        var s = makeScheduler(idleTick: 5.0, ceiling: 60.0)
        // First tick captures (nothing captured yet this run).
        XCTAssertEqual(captures(s.tick(now: 0.0, idleSeconds: 0.1)), [.idleTick])
        // Active user (idleSeconds low) but NO further triggers: the idle tick
        // keeps content flowing; suppress it by pretending ticks are sparse.
        // Jump far past the ceiling with no interim tick:
        XCTAssertEqual(captures(s.tick(now: 61.0, idleSeconds: 0.2)), [.forceCeiling])
    }

    func testIdleTickBackstopCadence() {
        var s = makeScheduler(idleTick: 5.0)
        XCTAssertEqual(captures(s.tick(now: 0.0, idleSeconds: 0.1)), [.idleTick])
        // Inside the tick window: heartbeat only.
        XCTAssertEqual(s.tick(now: 3.0, idleSeconds: 0.1), [.heartbeat])
        // Past it: capture.
        XCTAssertEqual(captures(s.tick(now: 5.5, idleSeconds: 0.1)), [.idleTick])
    }

    // MARK: AFK

    func testAfkEntryEmitsTransitionAndSuppressesCaptures() {
        var s = makeScheduler(afk: 150.0)
        _ = s.tick(now: 0.0, idleSeconds: 0.1)
        let actions = s.tick(now: 200.0, idleSeconds: 151.0)
        XCTAssertEqual(actions.first, .afkTransition(afk: true, idleSeconds: 151.0))
        XCTAssertTrue(s.isAfk)
        XCTAssertEqual(captures(actions), [])
    }

    func testAfkSuppressesIdleTickAndForceCeiling() {
        var s = makeScheduler(idleTick: 5.0, ceiling: 60.0, afk: 150.0)
        _ = s.tick(now: 0.0, idleSeconds: 0.1)
        _ = s.tick(now: 200.0, idleSeconds: 151.0)  // enter AFK
        // Far past BOTH the idle tick and the ceiling while AFK: no captures.
        for t in stride(from: 210.0, through: 600.0, by: 10.0) {
            let actions = s.tick(now: t, idleSeconds: 151.0 + t)
            XCTAssertEqual(captures(actions), [], "capture fired while AFK at t=\(t)")
            XCTAssertEqual(actions, [.heartbeat])
        }
    }

    func testTriggerWhileAfkDoesNotWakeCapture() {
        var s = makeScheduler(afk: 150.0)
        _ = s.tick(now: 0.0, idleSeconds: 0.1)
        _ = s.tick(now: 200.0, idleSeconds: 151.0)  // enter AFK
        // Background title change while the user is still away: suppressed.
        XCTAssertEqual(s.trigger(.titleChanged, now: 210.0), [])
        // Still AFK on the next tick (HID idle still high): heartbeat only,
        // and the suppressed trigger must NOT have armed a pending capture.
        XCTAssertEqual(s.tick(now: 215.0, idleSeconds: 166.0), [.heartbeat])
        XCTAssertTrue(s.isAfk)
    }

    func testAfkExitRequiresHidEvidenceThenCaptures() {
        var s = makeScheduler(afk: 150.0)
        _ = s.tick(now: 0.0, idleSeconds: 0.1)
        _ = s.tick(now: 200.0, idleSeconds: 151.0)  // enter AFK
        let actions = s.tick(now: 300.0, idleSeconds: 0.2)  // user typed
        XCTAssertEqual(actions.first, .afkTransition(afk: false, idleSeconds: 0.2))
        XCTAssertEqual(captures(actions), [.returnFromAfk])
        XCTAssertFalse(s.isAfk)
    }

    func testAfkEntryClearsPendingDebounce() {
        var s = makeScheduler(afk: 150.0)
        _ = s.tick(now: 0.0, idleSeconds: 0.1)
        _ = s.trigger(.appActivated, now: 100.0)  // armed, not yet fired
        _ = s.tick(now: 200.0, idleSeconds: 151.0)  // enter AFK
        // Return: the stale pre-AFK debounce must not fire alongside/instead
        // of return_from_afk.
        let actions = s.tick(now: 300.0, idleSeconds: 0.1)
        XCTAssertEqual(captures(actions), [.returnFromAfk])
    }

    // MARK: config hygiene

    func testConfigReclampsHostileValues() {
        let cfg = SchedulerConfig(minGap: 0.01, idleTick: 0.01, forceCeiling: 0.01, afkThreshold: 1.0)
        XCTAssertEqual(cfg.minGap, 1.0)
        XCTAssertGreaterThanOrEqual(cfg.forceCeiling, cfg.idleTick)
        XCTAssertGreaterThanOrEqual(cfg.afkThreshold, 30.0)
    }

    func testStartupTriggerFiresImmediately() {
        var s = makeScheduler()
        XCTAssertEqual(captures(s.trigger(.startup, now: 0.0)), [.startup])
    }
}
