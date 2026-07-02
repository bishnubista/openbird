// OCRCapture — the ScreenCaptureKit + Vision mechanism behind the `OcrHAL`
// seam (Phase C2). Mechanism only: all gating (opt-in, TCC, mic-hot, throttle)
// happened upstream in OcrGate, and the timeout/cancellation/late-completion
// ownership lives in OcrBridge (CaptureHelperCore) where it is unit-tested.
//
// FILE-HEADER INVARIANT — pixels are transient: the captured `CGImage` never
// leaves `recognizeFrontWindow`'s scope, is never written to disk, stderr,
// argv, env, or any log, and is released when the function returns. Only the
// RECOGNIZED TEXT crosses out of this file, and it flows solely to the private
// stdout pipe through the exact path AX text takes (scrubbed daemon-side).
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
    func recognizeFrontWindow(pid: Int32, axTitle: String?) async -> OcrHALOutcome {
        .unavailable(reason: "ocr_unavailable")
    }
}

@available(macOS 14.0, *)
final class ScreenCaptureKitOcrHAL: OcrHAL {

    func recognizeFrontWindow(pid: Int32, axTitle: String?) async -> OcrHALOutcome {
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
            // Last cancellation point before the synchronous Vision pass (a
            // cancelled task's eventual result is dropped by OcrBridge's
            // closed generation box either way).
            try Task.checkCancellation()
            return recognizeText(in: image)
        } catch is CancellationError {
            return .unavailable(reason: "ocr_error")
        } catch {
            // Reason code only — the error description could embed a window
            // title. The CGImage (if any) died with this scope.
            return .unavailable(reason: "ocr_error")
        }
    }

    /// On-device Vision OCR over the transient window still. Synchronous by
    /// Vision's design; bounded by OcrBridge's wall budget at the call site.
    private func recognizeText(in image: CGImage) -> OcrHALOutcome {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.automaticallyDetectsLanguage = true
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
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
