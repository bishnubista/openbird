// OcrGate — the pure per-capture OCR fallback gate for stream mode (Phase C2).
//
// Decides whether ONE capture whose AX text came back empty may fall back to a
// window-scoped screenshot + on-device OCR. Pure state machine over an injected
// monotonic clock (`now`), exactly like Scheduler: no AX, no ScreenCaptureKit,
// no Vision, no I/O — so every gate and the per-app throttle are unit-testable
// with `swift test` and a fake clock.
//
// Gate order (each producing a DISTINCT reason code, checked first-to-last):
//
//   1. axTextEmpty  — OCR is a *fallback*: an AX walk that produced text never
//                     triggers a screenshot (`ax_text_present`).
//   2. optedIn      — per-app opt-in (`--ocr-apps`); anything else skips
//                     (`not_opted_in`). The allowlist gate already ran upstream
//                     in captureFrontmost, so opted-in ⊆ allowlisted holds by
//                     construction.
//   3. tccGranted   — CGPreflightScreenCaptureAccess only; the helper NEVER
//                     prompts (`tcc_denied`).
//   4. !micHot      — never OCR while a mic is hot: SCK/Vision GPU work must
//                     not contend with a live call (`mic_hot` — screenpipe's
//                     Vision-vs-Zoom lesson).
//   5. throttle     — per-bundle-id min-interval (`throttled`).
//
// Throttle-slot semantics: the slot is consumed at DECISION time — the moment
// `.run` is returned — NOT at OCR completion. A timed-out or failed OCR attempt
// therefore still counts against the interval, so a persistently-failing app
// can never create an immediate retry storm. Skips never consume the slot.
//
// Reason codes are a CLOSED vocabulary (they reach stderr diagnostics):
// ax_text_present, not_opted_in, tcc_denied, mic_hot, throttled.

/// The gate's verdict for one AX-empty capture.
public enum OcrDecision: Equatable, Sendable {
    /// Run the OCR fallback now (the throttle slot has been consumed).
    case run
    /// Skip, with a closed-vocabulary reason code (safe for diagnostics).
    case skip(reason: String)
}

public struct OcrGate: Sendable {
    /// Minimum seconds between OCR *attempts* per bundle id. Arrives
    /// pre-clamped from the Python config (`capture_ocr_min_interval_seconds`,
    /// lo=10); re-clamped defensively here because argv is operator-editable
    /// and a typo'd flag must not defeat the CPU/power budget.
    private let minInterval: Double
    /// Monotonic time of the last `.run` decision per bundle id.
    private var lastAttemptAt: [String: Double] = [:]

    public init(minInterval: Double = 30.0) {
        self.minInterval = minInterval.isFinite ? max(10.0, minInterval) : 30.0
    }

    /// Decide whether this capture may fall back to OCR. `now` is the injected
    /// MONOTONIC clock (wall time must never create or destroy throttle
    /// deadlines — same clock discipline as Scheduler).
    public mutating func decide(
        bundleId: String,
        axTextEmpty: Bool,
        optedIn: Bool,
        tccGranted: Bool,
        micHot: Bool,
        now: Double
    ) -> OcrDecision {
        if !axTextEmpty { return .skip(reason: "ax_text_present") }
        if !optedIn { return .skip(reason: "not_opted_in") }
        if !tccGranted { return .skip(reason: "tcc_denied") }
        if micHot { return .skip(reason: "mic_hot") }
        if let last = lastAttemptAt[bundleId], now - last < minInterval {
            return .skip(reason: "throttled")
        }
        // Consume the throttle slot at DECISION time (see header): a timeout
        // or HAL failure downstream must not permit an immediate retry.
        lastAttemptAt[bundleId] = now
        return .run
    }
}
