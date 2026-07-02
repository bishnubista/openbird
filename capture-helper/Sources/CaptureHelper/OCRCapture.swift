// OCRCapture — the ScreenCaptureKit + Vision mechanism behind the two-phase
// `OcrHAL` seam (Phase C2). Mechanism only: all gating (opt-in, TCC, mic-hot,
// throttle) happened upstream in OcrGate, and the timeout/cancellation/
// late-completion ownership lives in OcrBridge (CaptureHelperCore) where it
// is unit-tested.
//
// FILE-HEADER INVARIANT — pixels are transient: the captured `CGImage` is
// handed to the bridge as an `OcrImageHandle` whose only strong references
// live inside ONE `OcrBridge.recognize(pid:axTitle:timeout:)` stack frame
// (the bridge WAITS for `recognize(image:)` to finish before returning, and
// drops a late-acquired handle unread). It is never written to disk, stderr,
// argv, env, or any log, never retained by this type, and is released when
// that frame unwinds. Only the RECOGNIZED TEXT crosses out of this file, and
// it flows solely to the private stdout pipe through the exact path AX text
// takes (scrubbed daemon-side).
//
// Phase split (review revision): `acquireImage` is the async, cancellable
// pre-pixel phase the bridge races against its deadline; `recognize(image:)`
// is the synchronous Vision phase the bridge never abandons — cancellation
// cannot interrupt `VNImageRequestHandler.perform`, so abandoning it would
// let pixels outlive the capture return.
//
// TCC honesty: this file only ever runs after `CGPreflightScreenCaptureAccess`
// returned true (OcrGate's tcc gate). Nothing here prompts; if the grant was
// revoked mid-run, SCShareableContent throws and we report a reason code.
//
// Availability: `SCScreenshotManager` requires macOS 14, but the package floor
// stays `.v13` — `makeOcrHAL()` is the boundary, handing macOS 13 a stub that
// reports `ocr_unavailable`.
//
// Vision engine: `VNRecognizeTextRequest` ONLY in C2. The macOS-26 SDK
// `RecognizeDocumentsRequest` branch (structured paragraphs/tables) is
// explicitly DEFERRED: a compiler-gated macOS-26 path has no CI coverage here;
// revisit when the toolchain gate can actually be tested.

import CaptureHelperCore
import CoreGraphics
import Foundation
import ScreenCaptureKit
import Vision

/// The macOS-14 availability boundary: the real SCK HAL when the OS supports
/// `SCScreenshotManager`, otherwise a stub reporting `ocr_unavailable`.
func makeOcrHAL() -> OcrHAL {
    if #available(macOS 14.0, *) {
        return ScreenCaptureKitOcrHAL()
    }
    return UnavailableOcrHAL()
}

/// macOS 13 stub (package floor is .v13; SCScreenshotManager needs 14).
final class UnavailableOcrHAL: OcrHAL {
    func acquireImage(pid: Int32, axTitle: String?) async -> OcrAcquireOutcome {
        .unavailable(reason: "ocr_unavailable")
    }

    func recognize(image: OcrImageHandle) -> OcrHALOutcome {
        // Unreachable (acquire never yields an image on this OS); keep the
        // fail-closed reason code anyway.
        .unavailable(reason: "ocr_unavailable")
    }
}

@available(macOS 14.0, *)
final class ScreenCaptureKitOcrHAL: OcrHAL {

    /// Phase 1 — async and cancellation-observing (the ONLY phase the bridge
    /// may abandon on timeout): window enumeration + one window-scoped still.
    func acquireImage(pid: Int32, axTitle: String?) async -> OcrAcquireOutcome {
        do {
            // Enumerate on-screen windows and pick the target app's front
            // window: same pid, standard layer (0), preferring an exact
            // AX-title match, else the largest area. This avoids the private
            // `_AXUIElementGetWindow` API entirely.
            let content = try await SCShareableContent.excludingDesktopWindows(
                false, onScreenWindowsOnly: true)
            try Task.checkCancellation()
            let candidates = content.windows.filter {
                $0.owningApplication?.processID == pid && $0.windowLayer == 0
            }
            guard !candidates.isEmpty else {
                return .unavailable(reason: "ocr_no_window")
            }
            let window = candidates.first { axTitle != nil && $0.title == axTitle }
                ?? candidates.max(by: {
                    $0.frame.width * $0.frame.height < $1.frame.width * $1.frame.height
                })!

            // One WINDOW-SCOPED still (never full-display), cursor excluded.
            let filter = SCContentFilter(desktopIndependentWindow: window)
            let config = SCStreamConfiguration()
            config.showsCursor = false
            config.captureResolution = .best
            // 2x backing scale keeps small UI text legible for the recognizer.
            config.width = max(1, Int(window.frame.width) * 2)
            config.height = max(1, Int(window.frame.height) * 2)
            let image = try await SCScreenshotManager.captureImage(
                contentFilter: filter, configuration: config)
            // The handle's lifetime is owned by the bridge from here: either
            // recognized synchronously within the same capture, or — if this
            // completion lost the deadline race — dropped unread and released
            // immediately by the closed generation box.
            return .image(OcrImageHandle(image))
        } catch is CancellationError {
            return .unavailable(reason: "ocr_error")
        } catch {
            // Reason code only — the error description could embed a window
            // title. Any transient pixels died with this scope.
            return .unavailable(reason: "ocr_error")
        }
    }

    /// Phase 2 — on-device Vision OCR over the transient window still.
    /// Synchronous by Vision's design and NEVER abandoned by the bridge, so
    /// the image cannot outlive the capture return (worst case documented in
    /// OcrBridge.swift: one sub-second-typical `.accurate` pass).
    func recognize(image: OcrImageHandle) -> OcrHALOutcome {
        // CF-type check (a plain `as?` on a CoreFoundation type is rejected
        // by the compiler as always-succeeding): fail closed on a handle
        // that does not actually wrap a CGImage.
        guard CFGetTypeID(image.value as CFTypeRef) == CGImage.typeID else {
            return .unavailable(reason: "ocr_error")
        }
        let cgImage = image.value as! CGImage
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.automaticallyDetectsLanguage = true
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return .unavailable(reason: "ocr_error")
        }
        let lines = (request.results ?? []).compactMap {
            $0.topCandidates(1).first?.string
        }
        let text = lines.joined(separator: "\n")
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return .unavailable(reason: "ocr_empty")
        }
        // The byte cap (Limits.maxTextBytes) is applied by the caller, on the
        // same truncation path AX text takes.
        return .text(text)
    }
}
