import XCTest

@testable import CaptureHelperCore

/// Deterministic tests for the mic run-state monitor against a fake HAL.
/// Every consensus rule from the Phase C1 plan is pinned here: initial-hot
/// emission, aggregate-flip-only emission (OR across devices), the input-scope
/// contract (duplex output-only playback is NOT mic-hot), the per-device
/// global-scope fallback with its one-time diagnostic, and listener removal
/// BEFORE re-enumeration on device churn (the leak guard).
final class MicMonitorTests: XCTestCase {

    final class FakeMicHAL: MicHAL {
        struct Device {
            var hasInput: Bool
            /// nil = the input-scope query fails for this device (HAL quirk).
            var inputRunning: Bool?
            var globalRunning: Bool

            init(hasInput: Bool = true, inputRunning: Bool? = false, globalRunning: Bool = false) {
                self.hasInput = hasInput
                self.inputRunning = inputRunning
                self.globalRunning = globalRunning
            }
        }

        var devices: [MicDeviceID: Device] = [:]
        var order: [MicDeviceID] = []
        private(set) var deviceListHandler: (() -> Void)?
        private(set) var runHandlers: [MicDeviceID: () -> Void] = [:]
        /// Listener lifecycle journal: "add:<id>" / "remove:<id>" in call order.
        private(set) var journal: [String] = []

        func setDevices(_ list: [(MicDeviceID, Device)]) {
            devices = Dictionary(uniqueKeysWithValues: list)
            order = list.map(\.0)
        }

        func listDevices() -> [MicDeviceID] { order }
        func hasInputStreams(_ device: MicDeviceID) -> Bool {
            devices[device]?.hasInput ?? false
        }
        func isRunningInputScope(_ device: MicDeviceID) -> Bool? {
            devices[device]?.inputRunning
        }
        func isRunningGlobalScope(_ device: MicDeviceID) -> Bool {
            devices[device]?.globalRunning ?? false
        }
        func addDeviceListListener(_ handler: @escaping () -> Void) {
            deviceListHandler = handler
        }
        func addRunStateListener(_ device: MicDeviceID, handler: @escaping () -> Void) {
            journal.append("add:\(device)")
            runHandlers[device] = handler
        }
        func removeRunStateListener(_ device: MicDeviceID) {
            journal.append("remove:\(device)")
            runHandlers[device] = nil
        }

        // -- test drivers -----------------------------------------------------
        func fireRunState(_ device: MicDeviceID) { runHandlers[device]?() }
        func fireDeviceListChanged() { deviceListHandler?() }
        func setInputRunning(_ device: MicDeviceID, _ running: Bool?) {
            devices[device]?.inputRunning = running
        }
    }

    private func makeMonitor(
        _ hal: FakeMicHAL
    ) -> (MicMonitor, flips: () -> [Bool], diags: () -> [String]) {
        // Reference boxes so the closures observe appended values.
        final class Log {
            var flips: [Bool] = []
            var diags: [String] = []
        }
        let log = Log()
        let monitor = MicMonitor(
            hal: hal,
            onFlip: { log.flips.append($0) },
            diag: { log.diags.append($0) })
        return (monitor, { log.flips }, { log.diags })
    }

    // MARK: initial state

    func testInitialHotEmitsMicStarted() {
        let hal = FakeMicHAL()
        hal.setDevices([(1, .init(inputRunning: true))])
        let (monitor, flips, _) = makeMonitor(hal)
        monitor.start()
        XCTAssertEqual(flips(), [true])
        XCTAssertTrue(monitor.micHot)
    }

    func testInitialColdEmitsNothing() {
        let hal = FakeMicHAL()
        hal.setDevices([(1, .init(inputRunning: false))])
        let (monitor, flips, _) = makeMonitor(hal)
        monitor.start()
        XCTAssertEqual(flips(), [])
        XCTAssertFalse(monitor.micHot)
    }

    func testOutputOnlyDevicesAreNeverListened() {
        let hal = FakeMicHAL()
        hal.setDevices([
            (1, .init(hasInput: false, inputRunning: true, globalRunning: true)),
            (2, .init(hasInput: true, inputRunning: false)),
        ])
        let (monitor, flips, _) = makeMonitor(hal)
        monitor.start()
        XCTAssertEqual(flips(), [])
        XCTAssertEqual(hal.journal, ["add:2"])  // no listener on the speaker
    }

    // MARK: aggregate flips

    func testAggregateOrFlipSuppresssesDuplicateEdges() {
        let hal = FakeMicHAL()
        hal.setDevices([(1, .init()), (2, .init())])
        let (monitor, flips, _) = makeMonitor(hal)
        monitor.start()
        XCTAssertEqual(flips(), [])

        hal.setInputRunning(1, true)
        hal.fireRunState(1)
        XCTAssertEqual(flips(), [true])  // first hot device flips the aggregate

        hal.setInputRunning(2, true)
        hal.fireRunState(2)
        XCTAssertEqual(flips(), [true])  // second hot device: NO duplicate edge

        hal.setInputRunning(1, false)
        hal.fireRunState(1)
        XCTAssertEqual(flips(), [true])  // one still hot: aggregate unchanged

        hal.setInputRunning(2, false)
        hal.fireRunState(2)
        XCTAssertEqual(flips(), [true, false])  // last one off: mic_stopped
        XCTAssertFalse(monitor.micHot)
    }

    func testSpuriousNotificationWithoutStateChangeEmitsNothing() {
        let hal = FakeMicHAL()
        hal.setDevices([(1, .init(inputRunning: true))])
        let (monitor, flips, _) = makeMonitor(hal)
        monitor.start()
        XCTAssertEqual(flips(), [true])
        hal.fireRunState(1)  // notification, same state
        hal.fireRunState(1)
        XCTAssertEqual(flips(), [true])
    }

    // MARK: input-scope contract

    func testDuplexOutputOnlyPlaybackIsNotMicHot() {
        // AirPods playing music: global-scope "running somewhere" is TRUE but
        // the input scope is idle — the mic must NOT read as hot.
        let hal = FakeMicHAL()
        hal.setDevices([(7, .init(inputRunning: false, globalRunning: true))])
        let (monitor, flips, diags) = makeMonitor(hal)
        monitor.start()
        XCTAssertEqual(flips(), [])
        hal.fireRunState(7)  // playback start/stop notifications
        XCTAssertEqual(flips(), [])
        XCTAssertFalse(monitor.micHot)
        XCTAssertEqual(diags(), [])  // input scope worked: no fallback diag
    }

    func testInputScopeFailureFallsBackToGlobalWithOneTimeDiag() {
        let hal = FakeMicHAL()
        hal.setDevices([
            (1, .init(inputRunning: nil, globalRunning: true)),
            (2, .init(inputRunning: nil, globalRunning: false)),
        ])
        let (monitor, flips, diags) = makeMonitor(hal)
        monitor.start()
        // Fallback used the global reading (device 1 hot) and diagnosed ONCE
        // even though two devices needed it across the same sweep.
        XCTAssertEqual(flips(), [true])
        XCTAssertEqual(diags(), ["capture: mic_scope_fallback"])
        // Further recomputes never repeat the diagnostic.
        hal.fireRunState(2)
        XCTAssertEqual(diags(), ["capture: mic_scope_fallback"])
        XCTAssertTrue(monitor.micHot)
    }

    // MARK: device churn (listener lifecycle)

    func testDeviceChurnRemovesListenersBeforeReenumerating() {
        let hal = FakeMicHAL()
        hal.setDevices([(1, .init()), (2, .init())])
        let (monitor, _, _) = makeMonitor(hal)
        monitor.start()
        XCTAssertEqual(hal.journal, ["add:1", "add:2"])

        // Device 1 departs, device 3 arrives (AirPods swap).
        hal.setDevices([(2, .init()), (3, .init())])
        hal.fireDeviceListChanged()
        // Leak guard: EVERY old listener removed BEFORE any new registration.
        XCTAssertEqual(
            hal.journal,
            ["add:1", "add:2", "remove:1", "remove:2", "add:2", "add:3"])
        XCTAssertNil(hal.runHandlers[1])  // nothing left on the departed device
    }

    func testHotDeviceDepartureFlipsAggregateOff() {
        let hal = FakeMicHAL()
        hal.setDevices([(1, .init(inputRunning: true))])
        let (monitor, flips, _) = makeMonitor(hal)
        monitor.start()
        XCTAssertEqual(flips(), [true])

        hal.setDevices([])  // the hot mic unplugs mid-call
        hal.fireDeviceListChanged()
        XCTAssertEqual(flips(), [true, false])
        XCTAssertFalse(monitor.micHot)
    }

    func testArrivingHotDeviceFlipsAggregateOn() {
        let hal = FakeMicHAL()
        hal.setDevices([(1, .init(inputRunning: false))])
        let (monitor, flips, _) = makeMonitor(hal)
        monitor.start()
        XCTAssertEqual(flips(), [])

        hal.setDevices([(1, .init(inputRunning: false)), (9, .init(inputRunning: true))])
        hal.fireDeviceListChanged()
        XCTAssertEqual(flips(), [true])
    }
}
