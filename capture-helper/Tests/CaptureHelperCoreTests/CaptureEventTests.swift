// CaptureEventTests — raw-JSON wire-shape tests for the capture record.
//
// The Python daemon's back-compat contract is keyed on which KEYS exist:
// a line without `type` is a one-shot capture frame; `ocr` marks the Phase C2
// OCR fallback. These tests decode the actual encoder output with
// JSONSerialization and assert the exact key sets, so a field addition or an
// encoder change that leaks a `null` (or drops a key) fails the build.

import Foundation
import XCTest
@testable import CaptureHelperCore

final class CaptureEventTests: XCTestCase {

    private func jsonObject(_ event: CaptureEvent) throws -> [String: Any] {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        let data = try encoder.encode(event)
        let raw = try JSONSerialization.jsonObject(with: data)
        return try XCTUnwrap(raw as? [String: Any])
    }

    func testOneShotFrameHasNoTypeTriggerOrOcrKeys() throws {
        let event = CaptureEvent(
            app: "com.example.app", window: "Title", url: nil,
            text: "hello", ts: 1000.0, incognito: false)
        let obj = try jsonObject(event)
        // Exact byte-shape contract with pre-stream daemons: the optional
        // stream/OCR keys are OMITTED (never `null`).
        XCTAssertEqual(
            Set(obj.keys), ["app", "window", "text", "ts", "incognito"])
        XCTAssertNil(obj["type"])
        XCTAssertNil(obj["trigger"])
        XCTAssertNil(obj["ocr"])
    }

    func testNilAppWindowUrlAreOmittedNotNull() throws {
        // Metadata-only early-return frames (not-allowlisted / dangerous /
        // incognito) carry nil window/url — those keys must be absent.
        let event = CaptureEvent(
            app: nil, window: nil, url: nil,
            text: "", ts: 1000.0, incognito: true)
        let obj = try jsonObject(event)
        XCTAssertEqual(Set(obj.keys), ["text", "ts", "incognito"])
    }

    func testStreamAxFrameHasTypeAndTriggerButNoOcr() throws {
        let event = CaptureEvent(
            app: "com.example.app", window: "Title", url: nil,
            text: "ax text", ts: 1000.0, incognito: false,
            trigger: "app_activated")
        let obj = try jsonObject(event)
        XCTAssertEqual(
            Set(obj.keys),
            ["type", "trigger", "app", "window", "text", "ts", "incognito"])
        XCTAssertEqual(obj["type"] as? String, "capture")
        XCTAssertEqual(obj["trigger"] as? String, "app_activated")
        XCTAssertNil(obj["ocr"])
    }

    func testOcrFallbackFrameCarriesOcrTrue() throws {
        let event = CaptureEvent(
            app: "com.example.app", window: "Title", url: nil,
            text: "recognized text", ts: 1000.0, incognito: false,
            trigger: "idle_tick", ocr: true)
        let obj = try jsonObject(event)
        XCTAssertEqual(
            Set(obj.keys),
            ["type", "trigger", "app", "window", "text", "ts", "incognito", "ocr"])
        XCTAssertEqual(obj["ocr"] as? Bool, true)
    }
}
