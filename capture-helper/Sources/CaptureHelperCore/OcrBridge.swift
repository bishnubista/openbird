// OcrBridge — the cancellable async-in-sync bridge for the OCR fallback
// (Phase C2), behind the `OcrHAL` protocol seam (the MicHAL pattern).
//
// The real HAL (ScreenCaptureKit + Vision, in the executable target) is NOT
// unit-testable; this bridge is where the risky timing/ownership logic lives,
// so it sits in CaptureHelperCore where `swift test` drives it with fakes.
//
// TWO-PHASE contract (plan consensus + adversarial-review revision):
//
//   Phase 1 — ACQUIRE (async, cancellable, deadline-raced). Window lookup +
//   the window-scoped screenshot run in a RETAINED `Task`; the caller blocks
//   on a semaphore for at most `timeout` seconds. On timeout the bridge
//   cancels the task (the SCK awaits observe cancellation) and returns
//   `.timeout`. LATE-COMPLETION OWNERSHIP: each call owns a generation box
//   that is CLOSED under a lock before `.timeout` returns — an image acquired
//   after the deadline finds the box closed, is dropped on the floor, and is
//   released immediately; it is never recognized, never emitted, and never
//   bleeds into a later call (every call gets a fresh box).
//
//   Phase 2 — RECOGNIZE (synchronous, NEVER abandoned). Once a pixel buffer
//   exists, OCR runs on the CALLING (walk-queue) thread and the bridge WAITS
//   for it to finish before returning: cancellation cannot interrupt the
//   synchronous `VNImageRequestHandler.perform`, so racing the deadline past
//   this point would let the image outlive the capture return and overlap
//   later captures — violating the scope-bound pixel invariant. Pixels
//   therefore live only between phase 1 delivering the handle and phase 2
//   returning, strictly inside this call's stack frame.
//
// Budget note (documented worst case): the `timeout` deadline bounds ONLY the
// cancellable acquire phase. The recognize phase adds a bounded synchronous
// tail — `.accurate` Vision over ONE window still is typically sub-second —
// so worst-case walk-queue occupancy is `timeout + one Vision pass`
// (~2.5s + ~1s on pathological content). That keeps the combined AX+OCR
// budget's intent (a capture can never occupy the walk queue indefinitely)
// while never trading it against pixel lifetime.
//
// Throttle interaction: OcrGate consumed the throttle slot at DECISION time,
// so a `.timeout` here cannot cause an immediate retry storm.

import Dispatch
import Foundation

/// Opaque pixel-buffer handle crossing the HAL seam. Production wraps a
/// `CGImage`; tests wrap a marker. Reference semantics on purpose: the last
/// strong reference dying is what releases the pixels, and the bridge/HAL
/// contract confines every reference to one `recognize(pid:axTitle:timeout:)`
/// stack frame (or an immediately-dropped late delivery).
public final class OcrImageHandle: @unchecked Sendable {
    public let value: Any

    public init(_ value: Any) {
        self.value = value
    }
}

/// Phase-1 (acquire) result: pixels, or a closed-vocabulary reason.
public enum OcrAcquireOutcome: Sendable {
    case image(OcrImageHandle)
    case unavailable(reason: String)
}

/// What one recognize pass produced. Reason codes are a CLOSED vocabulary
/// (they reach stderr diagnostics; free text — which could embed window
/// titles — must never flow through here): ocr_no_window, ocr_unavailable,
/// ocr_empty, ocr_error.
public enum OcrHALOutcome: Equatable, Sendable {
    /// Recognized window text (the ONLY content-bearing case; it flows solely
    /// to the private stdout pipe, exactly like AX text).
    case text(String)
    /// No text, with a closed reason code.
    case unavailable(reason: String)
}

/// The OCR mechanism seam, split at the PIXEL boundary so the bridge can race
/// its deadline against only the pre-image phase. Production: ScreenCaptureKit
/// still (acquire) + on-device Vision OCR (recognize); tests: fakes.
public protocol OcrHAL: Sendable {
    /// Phase 1 — async, cancellation-observing: window enumeration + the
    /// window-scoped screenshot. May be abandoned by the bridge on timeout.
    func acquireImage(pid: Int32, axTitle: String?) async -> OcrAcquireOutcome
    /// Phase 2 — SYNCHRONOUS and never abandoned: OCR over the acquired
    /// pixels. The bridge always waits for this to return; implementations
    /// must not retain `image` (or any derived pixels) past the call.
    func recognize(image: OcrImageHandle) -> OcrHALOutcome
}

/// The bridge's verdict for one bounded OCR attempt.
public enum OcrBridgeOutcome: Equatable, Sendable {
    case text(String)
    /// The HAL reported a closed-vocabulary reason (no window, unavailable, …).
    case skipped(reason: String)
    /// The acquire deadline elapsed before any pixels existed; the underlying
    /// task was cancelled and a late image is dropped by the closed box.
    case timeout
}

public final class OcrBridge: @unchecked Sendable {
    private let hal: OcrHAL

    public init(hal: OcrHAL) {
        self.hal = hal
    }

    /// Per-call generation token: acquire results are accepted only while
    /// open. Locking is wrapped in synchronous methods (NSLock is not
    /// directly usable from async contexts).
    private final class Box: @unchecked Sendable {
        private let lock = NSLock()
        private var open = true
        private var outcome: OcrAcquireOutcome?

        /// Store the acquire result iff the box is still open. A dropped
        /// (late) `.image` handle dies right here — it is never recognized.
        func deliver(_ value: OcrAcquireOutcome) {
            lock.lock()
            defer { lock.unlock() }
            if open { outcome = value }
        }

        /// Close the box forever and return whatever was delivered in time.
        func close() -> OcrAcquireOutcome? {
            lock.lock()
            defer { lock.unlock() }
            open = false
            return outcome
        }
    }

    /// Run one two-phase HAL attempt. `timeout` bounds ONLY the acquire
    /// phase; once pixels exist the recognize phase always runs to completion
    /// on this thread (scope-bound pixel invariant — see the header, which
    /// also documents the worst-case occupancy). Never throws, never
    /// recognizes or delivers a late-acquired image.
    public func recognize(pid: Int32, axTitle: String?, timeout: Double) -> OcrBridgeOutcome {
        guard timeout > 0 else { return .timeout }
        let box = Box()
        let semaphore = DispatchSemaphore(value: 0)
        let hal = self.hal
        // RETAINED handle: cancellation on timeout needs it. Utility QoS —
        // OCR is a background fallback and must not preempt UI work.
        let task = Task.detached(priority: .utility) {
            box.deliver(await hal.acquireImage(pid: pid, axTitle: axTitle))
            semaphore.signal()
        }
        if semaphore.wait(timeout: .now() + timeout) == .timedOut {
            task.cancel()
            // Close the box BEFORE returning: an acquire racing this close
            // either landed (and its handle is discarded unread by the
            // timeout return, releasing the pixels immediately) or finds the
            // box closed and stores nothing. Either way no late image is ever
            // recognized or emitted.
            _ = box.close()
            return .timeout
        }
        switch box.close() {
        case .image(let handle):
            // Phase 2 — pixels exist NOW, so the deadline no longer applies:
            // run OCR synchronously and wait for the image scope to close
            // before returning. Returning early here would leak the pixel
            // lifetime past the capture return (the reviewed HIGH finding).
            switch hal.recognize(image: handle) {
            case .text(let text):
                return .text(text)
            case .unavailable(let reason):
                return .skipped(reason: reason)
            }
        case .unavailable(let reason):
            return .skipped(reason: reason)
        case nil:
            // Defensive: signaled without a stored outcome (cannot happen with
            // the deliver-then-signal order above) — treat as a bounded failure.
            return .timeout
        }
    }
}
