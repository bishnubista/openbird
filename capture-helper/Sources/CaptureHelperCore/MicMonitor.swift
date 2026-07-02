// MicMonitor — pure mic run-state aggregation for stream mode (Phase C1).
//
// Watches every input-capable audio device's "is running somewhere" state and
// reports ONE aggregate boolean: is any microphone hot? The StreamEngine turns
// flips into `system` events (kind=mic_started / mic_stopped) on the private
// pipe. Privacy: the signal is a run-state METADATA BIT — no audio is read, no
// TCC prompt fires, and no new event type carries text/titles/URLs.
//
// The type is deliberately a pure state machine over an injectable HAL seam
// (`MicHAL`) so the stateful listener lifecycle — the risky part — is unit-
// testable with a fake (`swift test`), like the Scheduler. Consensus rules:
//
//   * Run state is queried on the INPUT-scope property address: the global-
//     scope reading of DeviceIsRunningSomewhere means "running in at least one
//     process" and goes true for OUTPUT playback on duplex devices (AirPods
//     music must NOT read as mic-hot). If the input-scope query errors at
//     runtime (some HALs ignore scope for this selector on some devices), that
//     device falls back to the GLOBAL-scope reading with a one-time
//     `mic_scope_fallback` diagnostic — honest degradation, never silent.
//   * Per-device run-state listeners are REMOVED before every re-enumeration
//     (device churn like AirPods connect/disconnect must not leak listeners).
//   * Emission is aggregate-flip-only (OR across input devices) — a second
//     device going hot while one is already hot emits nothing — plus an
//     initial emission at start when the mic is ALREADY hot.
//   * ALL input devices are watched, not just the default input (Zoom can use
//     a non-default mic).
//
// Threading: main-confined by contract — the CoreAudio HAL delivers listener
// blocks on the MAIN dispatch queue and the engine constructs/starts the
// monitor on the main run loop, so this state needs no locking. Stream mode
// only: the one-shot helper has no persistent process to host listeners (the
// same limitation as AFK detection).

public typealias MicDeviceID = UInt32

/// The audio-HAL surface MicMonitor needs (CoreAudio in production, a fake in
/// tests). Implementations must deliver listener callbacks on the same thread
/// the monitor was started on (the main queue in the helper).
public protocol MicHAL: AnyObject {
    /// All audio devices currently present.
    func listDevices() -> [MicDeviceID]
    /// Whether the device has any INPUT streams (mic-capable).
    func hasInputStreams(_ device: MicDeviceID) -> Bool
    /// Input-scope DeviceIsRunningSomewhere; `nil` when the query fails.
    func isRunningInputScope(_ device: MicDeviceID) -> Bool?
    /// Global-scope reading (fallback ONLY — true during output playback too).
    func isRunningGlobalScope(_ device: MicDeviceID) -> Bool
    /// Install the device-list-changed listener (called once per monitor).
    func addDeviceListListener(_ handler: @escaping () -> Void)
    /// Install a run-state listener for one device.
    func addRunStateListener(_ device: MicDeviceID, handler: @escaping () -> Void)
    /// Remove a previously-installed run-state listener for one device.
    func removeRunStateListener(_ device: MicDeviceID)
}

public final class MicMonitor {
    private let hal: MicHAL
    private let onFlip: (Bool) -> Void
    private let diag: (String) -> Void
    /// Devices we currently hold a run-state listener on (input-capable only).
    private var monitored: [MicDeviceID] = []
    /// One-time guard for the input-scope -> global-scope fallback diagnostic.
    private var scopeFallbackDiagnosed = false
    public private(set) var micHot = false

    /// - Parameters:
    ///   - onFlip: called on every AGGREGATE flip (and once at start when the
    ///     mic is already hot) with the new state. Reason-code side effects
    ///     only — the value is a bit, never content.
    ///   - diag: non-content diagnostic sink (stderr reason codes).
    public init(
        hal: MicHAL,
        onFlip: @escaping (Bool) -> Void,
        diag: @escaping (String) -> Void
    ) {
        self.hal = hal
        self.onFlip = onFlip
        self.diag = diag
    }

    /// Enumerate devices, install listeners, read the initial state, and emit
    /// mic_started when the mic is ALREADY hot at startup (the daemon may have
    /// restarted mid-call).
    public func start() {
        hal.addDeviceListListener { [weak self] in self?.deviceListChanged() }
        attachToInputDevices()
        micHot = anyInputRunning()
        if micHot { onFlip(true) }
    }

    private func deviceListChanged() {
        // Leak guard: remove EVERY per-device listener BEFORE re-enumerating —
        // a departed device's listener must never outlive its device entry,
        // and a surviving device gets a fresh (single) registration.
        for device in monitored { hal.removeRunStateListener(device) }
        monitored = []
        attachToInputDevices()
        recompute()
    }

    private func attachToInputDevices() {
        for device in hal.listDevices() where hal.hasInputStreams(device) {
            hal.addRunStateListener(device) { [weak self] in self?.recompute() }
            monitored.append(device)
        }
    }

    /// One device's mic-run bit: input scope, with the per-device global-scope
    /// fallback (one-time diagnostic; see the header contract).
    private func runState(_ device: MicDeviceID) -> Bool {
        if let running = hal.isRunningInputScope(device) { return running }
        if !scopeFallbackDiagnosed {
            scopeFallbackDiagnosed = true
            diag("capture: mic_scope_fallback")
        }
        return hal.isRunningGlobalScope(device)
    }

    private func anyInputRunning() -> Bool {
        monitored.contains { runState($0) }
    }

    private func recompute() {
        let hot = anyInputRunning()
        guard hot != micHot else { return }  // aggregate-flip-only emission
        micHot = hot
        onFlip(hot)
    }
}
