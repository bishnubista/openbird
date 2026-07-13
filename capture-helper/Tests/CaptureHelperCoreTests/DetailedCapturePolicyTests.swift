import XCTest
@testable import CaptureHelperCore

final class DetailedCapturePolicyTests: XCTestCase {
    func testExactGrantMatchesCaseInsensitively() {
        XCTAssertTrue(hasExactDetailedCaptureGrant(
            bundleId: "com.mitchellh.ghostty",
            entries: ["Com.Mitchellh.Ghostty"]))
    }

    func testMissingAndUnrelatedGrantsDoNotMatch() {
        XCTAssertFalse(hasExactDetailedCaptureGrant(
            bundleId: "com.mitchellh.ghostty", entries: []))
        XCTAssertFalse(hasExactDetailedCaptureGrant(
            bundleId: "com.mitchellh.ghostty", entries: ["com.apple.Terminal"]))
    }

    func testPatternLookingGrantsAreInert() {
        XCTAssertFalse(hasExactDetailedCaptureGrant(
            bundleId: "com.mitchellh.ghostty", entries: ["glob:com.mitchellh.*"]))
        XCTAssertFalse(hasExactDetailedCaptureGrant(
            bundleId: "com.mitchellh.ghostty", entries: ["re:com\\.mitchellh\\..*"]))
    }
}
