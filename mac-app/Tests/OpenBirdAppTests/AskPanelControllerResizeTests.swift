import XCTest
@testable import OpenBirdApp

/// Convergence of the compact Ask panel's measure→resize→re-measure loop (#169). The
/// fixed point is what stops the Core-Animation display cycle from resizing every frame.
final class AskPanelControllerResizeTests: XCTestCase {
    private let screen: CGFloat = 900   // a roomy display; clamp is a no-op here

    func testResizesToCeiledReportWhenContentGrew() {
        // Panel is 140 tall, content reports 312.3 → grow to ceil = 313.
        XCTAssertEqual(
            AskPanelController.compactResizeTarget(
                reportedHeight: 312.3, currentHeight: 140, maxHeight: screen),
            313)
    }

    func testSettlesAtFixedPointOnSecondMeasure() {
        // After applying 313, the same content reports 312.3 again. ceil(312.3)=313 and the
        // panel is already 313 → nil (no resize), so the loop terminates.
        XCTAssertNil(
            AskPanelController.compactResizeTarget(
                reportedHeight: 312.3, currentHeight: 313, maxHeight: screen))
    }

    func testSubEpsilonDifferenceDoesNotResize() {
        // ceil(312.6)=313, current 313 → 0 diff, below 0.5 epsilon → nil.
        XCTAssertNil(
            AskPanelController.compactResizeTarget(
                reportedHeight: 312.6, currentHeight: 313, maxHeight: screen))
    }

    func testClampsToScreenHeightSoTargetIsReachable() {
        // Content wants 5000 but the screen can only show 900 → target floors to 900.
        // This is the key #169 guard: an unclamped target the window server won't grant
        // would keep abs(current - target) > 0.5 true forever.
        XCTAssertEqual(
            AskPanelController.compactResizeTarget(
                reportedHeight: 5000, currentHeight: 140, maxHeight: 900),
            900)
    }

    func testClampedPanelSettlesAtMaxHeight() {
        // Once clamped to 900, an even larger report keeps target at 900 → already there → nil.
        XCTAssertNil(
            AskPanelController.compactResizeTarget(
                reportedHeight: 5000, currentHeight: 900, maxHeight: 900))
    }

    func testNonFiniteReportIsIgnored() {
        XCTAssertNil(
            AskPanelController.compactResizeTarget(
                reportedHeight: .nan, currentHeight: 140, maxHeight: screen))
        XCTAssertNil(
            AskPanelController.compactResizeTarget(
                reportedHeight: .infinity, currentHeight: 140, maxHeight: screen))
    }

    func testZeroOrNegativeMaxHeightIsIgnored() {
        XCTAssertNil(
            AskPanelController.compactResizeTarget(
                reportedHeight: 300, currentHeight: 140, maxHeight: 0))
    }

    func testZeroReportDoesNotResize() {
        // A 0-height report (e.g. a transient pre-layout measure) must not collapse the panel.
        XCTAssertNil(
            AskPanelController.compactResizeTarget(
                reportedHeight: 0, currentHeight: 140, maxHeight: screen))
    }
}
