import XCTest
@testable import OpenBirdApp

/// Deep-capture (OCR) opt-in persistence (Phase C2): the saved list is always
/// a subset of the allowlist BY CONSTRUCTION — both writers filter.
@MainActor
final class OcrSettingsTests: XCTestCase {
    private let allowlistKey = "openbird.captureAllowlist"
    private let ocrAppsKey = "openbird.captureOcrApps"
    private let detailedCaptureAppsKey = "openbird.detailedCaptureApps"

    /// Save/restore both UserDefaults keys so tests never leak into (or read
    /// from) the developer's real app state.
    private func withRestoredDefaults<T>(_ body: () throws -> T) rethrows -> T {
        let defaults = UserDefaults.standard
        let oldAllow = defaults.stringArray(forKey: allowlistKey)
        let oldOcr = defaults.stringArray(forKey: ocrAppsKey)
        let oldDetailed = defaults.stringArray(forKey: detailedCaptureAppsKey)
        defer {
            for (key, old) in [
                (allowlistKey, oldAllow),
                (ocrAppsKey, oldOcr),
                (detailedCaptureAppsKey, oldDetailed),
            ] {
                if let old {
                    defaults.set(old, forKey: key)
                } else {
                    defaults.removeObject(forKey: key)
                }
            }
        }
        defaults.removeObject(forKey: allowlistKey)
        defaults.removeObject(forKey: ocrAppsKey)
        defaults.removeObject(forKey: detailedCaptureAppsKey)
        return try body()
    }

    private func service() -> OpenBirdService {
        OpenBirdService(
            externalLoopDaemonProbe: { false },
            captureHelperRunningProbe: { false }
        )
    }

    // MARK: pure filter

    func testFilteredOcrAppsKeepsOnlyAllowlistedTrimmedUnique() {
        let out = OpenBirdService.filteredOcrApps(
            [" com.a ", "com.a", "com.b", "", "com.c"],
            allowlist: ["com.a", "com.c"]
        )
        XCTAssertEqual(out, ["com.a", "com.c"])  // com.b not allowlisted; deduped/trimmed
    }

    func testFilteredOcrAppsEmptyAllowlistYieldsEmpty() {
        XCTAssertEqual(
            OpenBirdService.filteredOcrApps(["com.a"], allowlist: []), [])
    }

    // MARK: persisted invariants

    func testSetOcrAppsFiltersToCurrentAllowlist() {
        withRestoredDefaults {
            let svc = service()
            svc.setAllowlist(["com.example.editor", "com.example.mail"])
            svc.setOcrApps(["com.example.editor", "com.example.rogue"])
            XCTAssertEqual(svc.ocrApps(), ["com.example.editor"])
        }
    }

    func testAllowlistRemovalPrunesOcrApps() {
        withRestoredDefaults {
            let svc = service()
            svc.setAllowlist(["com.example.editor", "com.example.mail"])
            svc.setOcrApps(["com.example.editor", "com.example.mail"])
            XCTAssertEqual(svc.ocrApps(), ["com.example.editor", "com.example.mail"])
            // Removing an app from the allowlist must also drop its OCR opt-in
            // (a de-allowlisted app cannot keep a deep-capture grant).
            svc.setAllowlist(["com.example.mail"])
            XCTAssertEqual(svc.ocrApps(), ["com.example.mail"])
        }
    }

    func testSetDetailedCaptureAppsFiltersToCurrentAllowlist() {
        withRestoredDefaults {
            let svc = service()
            svc.setAllowlist(["com.mitchellh.ghostty"])
            svc.setDetailedCaptureApps([
                " com.mitchellh.ghostty ",
                "com.mitchellh.ghostty",
                "com.example.rogue",
            ])
            XCTAssertEqual(svc.detailedCaptureApps(), ["com.mitchellh.ghostty"])
        }
    }

    func testAllowlistRemovalPrunesDetailedCaptureApps() {
        withRestoredDefaults {
            let svc = service()
            svc.setAllowlist(["com.mitchellh.ghostty", "com.apple.Terminal"])
            svc.setDetailedCaptureApps(["com.mitchellh.ghostty", "com.apple.Terminal"])

            svc.setAllowlist(["com.apple.Terminal"])

            XCTAssertEqual(svc.detailedCaptureApps(), ["com.apple.Terminal"])
        }
    }
}
