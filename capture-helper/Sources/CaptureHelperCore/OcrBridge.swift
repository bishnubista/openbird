// OcrBridge — the cancellable async-in-sync bridge for the OCR fallback
// (Phase C2), behind the `OcrHAL` protocol seam (the MicHAL pattern).
//
// The real HAL (ScreenCaptureKit + Vision, in the executable target) is async
// and NOT unit-testable; this bridge is where the risky timing/ownership logic
// lives, so it sits in CaptureHelperCore where `swift test` drives it with a
// fake HAL. Contract (plan consensus):
//
//   * The HAL work runs in a RETAINED, cancellable `Task`. The caller blocks
//     on a semaphore for at most `timeout` seconds (the OCR share of the
//     combined AX+OCR <= 4s per-capture wall budget), so one capture can never
//     occupy the walk queue indefinitely and starve triggers while heartbeats
//     keep liveness looking healthy.
//   * On semaphore timeout the bridge calls `task.cancel()` (the SCK/Vision
//     awaits observe cancellation) and returns `.timeout`.
//   * LATE-COMPLETION OWNERSHIP: each call owns a generation box that is
//     CLOSED under a lock before `.timeout` returns. A task that completes
//     after the deadline finds the box closed and its result is dropped on the
//     floor — a late completion can never mutate state or be emitted after
//     the capture returned, and can never bleed into a later call (every call
//     gets a fresh box).
//
// Throttle interaction: OcrGate consumed the throttle slot at DECISION time,
// so a `.timeout` here cannot cause an immediate retry storm.

import Dispatch
import Foundation

/// What one HAL attempt produced. Reason codes are a CLOSED vocabulary (they
/// reach stderr diagnostics; free text — which could embed window titles —
/// must never flow through here): ocr_no_window, ocr_unavailable, ocr_empty,
/// ocr_error.
public enum OcrHALOutcome: Equatable, Sendable {
    /// Recognized window text (the ONLY content-bearing case; it flows solely
    /// to the private stdout pipe, exactly like AX text).
    case text(String)
    /// No text, with a closed reason code.
    case unavailable(reason: String)
}

/// The OCR mechanism seam. Production: ScreenCaptureKit window still +
/// on-device Vision OCR (executable target, macOS 14+); tests: fakes.
/// Implementations must observe `Task` cancellation at their await points.
public protocol OcrHAL: Sendable {
    func recognizeFrontWindow(pid: Int32, axTitle: String?) async -> OcrHALOutcome
}

/// The bridge's verdict for one bounded OCR attempt.
public enum OcrBridgeOutcome: Equatable, Sendable {
    case text(String)
    /// The HAL reported a closed-vocabulary reason (no window, unavailable, …).
    case skipped(reason: String)
    /// The wall budget elapsed first; the underlying task was cancelled and
    /// any late result is owned (dropped) by the closed generation box.
    case timeout
}

public final class OcrBridge: @unchecked Sendable {
    private let hal: OcrHAL

    public init(hal: OcrHAL) {
        self.hal = hal
    }

    /// Per-call generation token: results are accepted only while open.
    /// Locking is wrapped in synchronous methods (NSLock is not directly
    /// usable from async contexts).
    private final class Box: @unchecked Sendable {
        private let lock = NSLock()
        private var open = true
        private var outcome: OcrHALOutcome?

        /// Store the HAL result iff the box is still open (drop it otherwise).
        func deliver(_ value: OcrHALOutcome) {
            lock.lock()
            defer { lock.unlock() }
            if open { outcome = value }
        }

        /// Close the box forever and return whatever was delivered in time.
        func close() -> OcrHALOutcome? {
            lock.lock()
            defer { lock.unlock() }
            open = false
            return outcome
        }
    }

    /// Run one HAL attempt, blocking the calling (walk-queue) thread for at
    /// most `timeout` seconds. Never throws, never delivers late results.
    public func recognize(pid: Int32, axTitle: String?, timeout: Double) -> OcrBridgeOutcome {
        guard timeout > 0 else { return .timeout }
        let box = Box()
        let semaphore = DispatchSemaphore(value: 0)
        let hal = self.hal
        // RETAINED handle: cancellation on timeout needs it. Utility QoS —
        // OCR is a background fallback and must not preempt UI work.
        let task = Task.detached(priority: .utility) {
            box.deliver(await hal.recognizeFrontWindow(pid: pid, axTitle: axTitle))
            semaphore.signal()
        }
        if semaphore.wait(timeout: .now() + timeout) == .timedOut {
            task.cancel()
            // Close the box BEFORE returning: a completion racing this close
            // either landed (and we lost the race harmlessly — the result is
            // still discarded by the timeout return) or finds the box closed
            // and writes nothing. Either way nothing is emitted later.
            _ = box.close()
            return .timeout
        }
        switch box.close() {
        case .text(let text):
            return .text(text)
        case .unavailable(let reason):
            return .skipped(reason: reason)
        case nil:
            // Defensive: signaled without a stored outcome (cannot happen with
            // the write-then-signal order above) — treat as a bounded failure.
            return .timeout
        }
    }
}
