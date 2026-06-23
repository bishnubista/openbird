import XCTest
@testable import OpenBirdApp

final class AppDestinationTests: XCTestCase {
    // Pins the destination SET. `AppSidebar` iterates `allCases` (auto-syncs) but
    // `MenuBarView` hand-lists `.ask/.today/.setup`, so a new case would silently miss
    // the menu bar. This test fails the moment the set changes, forcing whoever adds a
    // destination to update the menu bar (and this expectation) on purpose.
    func testDestinationSetIsStableAndOrdered() {
        XCTAssertEqual(AppDestination.allCases, [.ask, .today, .setup])
    }

    // Both renderers read display data straight off the enum; empty title/symbol would
    // render a blank menu item or sidebar row.
    func testEveryDestinationHasTitleAndSymbol() {
        for dest in AppDestination.allCases {
            XCTAssertFalse(dest.title.isEmpty, "\(dest) has an empty title")
            XCTAssertFalse(dest.systemImage.isEmpty, "\(dest) has an empty systemImage")
        }
    }

    // `id` backs `ForEach(AppDestination.allCases)` in the sidebar — duplicates would
    // collapse rows / break selection highlighting.
    func testIdentifiersAreUnique() {
        let ids = AppDestination.allCases.map(\.id)
        XCTAssertEqual(Set(ids).count, ids.count, "AppDestination ids must be unique")
    }
}
