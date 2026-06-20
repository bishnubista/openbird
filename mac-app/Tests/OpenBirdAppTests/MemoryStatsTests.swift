import XCTest
@testable import OpenBirdApp

final class MemoryStatsTests: XCTestCase {
    func testParseMemoryStatsDecodesCliJson() {
        let stats = OpenBirdService.parseMemoryStats("""
        {
          "observations": 7,
          "blobs": 3,
          "chunks": 11,
          "vectors": 11,
          "embed_dim": 768,
          "cohort_key": "ollama:ollama/nomic-embed-text:768:test",
          "encryption_enabled": false
        }
        """)

        XCTAssertEqual(stats?.observations, 7)
        XCTAssertEqual(stats?.blobs, 3)
        XCTAssertEqual(stats?.chunks, 11)
        XCTAssertEqual(stats?.vectors, 11)
    }

    func testParseMemoryStatsRejectsNonJsonOutput() {
        XCTAssertNil(OpenBirdService.parseMemoryStats("not json"))
    }
}
