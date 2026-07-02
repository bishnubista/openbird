// CaptureEvent — the one-JSON-line-per-capture record (moved from main.swift
// so `swift test` can pin the raw wire shape; no I/O lives here).
//
// Wire-shape contract with the Python daemon (`openbird/capture/daemon.py`):
//
//   * one-shot mode: NO `type`, `trigger`, or `ocr` keys — byte-compatible
//     with pre-stream daemons (a line without `type` is a capture frame);
//   * stream AX frames: `type` + `trigger` present, `ocr` absent;
//   * stream OCR-fallback frames (Phase C2): `ocr: true` additionally present.
//
// The optional-key omission is enforced by an EXPLICIT `encode(to:)` using
// `encodeIfPresent`, making the contract deliberate rather than an artifact of
// synthesis (Swift's synthesized Encodable also omits nil optionals — this
// encoder pins the same behavior against future field/tooling drift, and the
// raw-JSON tests in CaptureEventTests.swift fail if any key leaks or goes
// missing). Old daemons ignore unknown keys, so `ocr` is back-compat free.
public struct CaptureEvent: Encodable {
    public let type: String?
    public let trigger: String?
    public let app: String?
    public let window: String?
    public let url: String?
    public let text: String
    public let ts: Double
    public let incognito: Bool
    /// True ONLY on a frame whose text came from the opt-in OCR fallback
    /// (Phase C2); nil (omitted) on every AX frame and in one-shot mode.
    public let ocr: Bool?

    public init(
        app: String?, window: String?, url: String?, text: String,
        ts: Double, incognito: Bool, trigger: String? = nil, ocr: Bool? = nil
    ) {
        // `type`/`trigger` are stream-mode additions: nil in one-shot mode so
        // the encoder omits them and old-daemon output stays byte-compatible.
        self.type = trigger == nil ? nil : "capture"
        self.trigger = trigger
        self.app = app
        self.window = window
        self.url = url
        self.text = text
        self.ts = ts
        self.incognito = incognito
        self.ocr = ocr
    }

    private enum CodingKeys: String, CodingKey {
        case type, trigger, app, window, url, text, ts, incognito, ocr
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        // Every optional uses encodeIfPresent (key OMITTED when nil, never
        // `null`) — the same shape synthesis produced, now stated explicitly.
        try container.encodeIfPresent(type, forKey: .type)
        try container.encodeIfPresent(trigger, forKey: .trigger)
        try container.encodeIfPresent(app, forKey: .app)
        try container.encodeIfPresent(window, forKey: .window)
        try container.encodeIfPresent(url, forKey: .url)
        try container.encode(text, forKey: .text)
        try container.encode(ts, forKey: .ts)
        try container.encode(incognito, forKey: .incognito)
        try container.encodeIfPresent(ocr, forKey: .ocr)
    }
}
