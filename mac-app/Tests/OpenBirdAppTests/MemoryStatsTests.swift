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

    func testParsePreflightDecodesLocalModelRoute() {
        let report = OpenBirdService.parsePreflight("""
        {
          "runtime_ok": true,
          "release_gate_ok": false,
          "ollama": {
            "reachable": true,
            "host": "http://localhost:11434",
            "required_models": ["llama3.2", "nomic-embed-text"],
            "missing_models": []
          },
          "cloud": {
            "llm_model": "ollama/llama3.2",
            "embed_model": "ollama/nomic-embed-text",
            "remote_models": [],
            "uses_local_ollama": true,
            "blocked": false
          }
        }
        """)

        XCTAssertTrue(report.ollamaReachable == true)
        XCTAssertEqual(report.ollamaHost, "http://localhost:11434")
        XCTAssertEqual(report.requiredModels, ["llama3.2", "nomic-embed-text"])
        XCTAssertEqual(report.llmModel, "ollama/llama3.2")
        XCTAssertEqual(report.embedModel, "ollama/nomic-embed-text")
        XCTAssertTrue(report.usesLocalOllama)
        XCTAssertFalse(report.cloudBlocked)
    }

    func testChatFailureSummaryClassifiesMissingModel() {
        XCTAssertEqual(
            OpenBirdService.chatFailureSummary(exitCode: 1, stderr: "model missing: llama3.2"),
            "Chat failed because a required local model is missing."
        )
    }

    func testChatFailureSummaryClassifiesCloudBlock() {
        XCTAssertEqual(
            OpenBirdService.chatFailureSummary(exitCode: 2, stderr: "Set OPENBIRD_ALLOW_CLOUD=1"),
            "Chat blocked because a cloud model is configured without opt-in."
        )
    }

    func testChatFailureSummaryClassifiesOllamaFailure() {
        XCTAssertEqual(
            OpenBirdService.chatFailureSummary(exitCode: 3, stderr: "connection refused to Ollama"),
            "Chat failed because the local Ollama model request did not complete."
        )
    }

    func testChatFailureSummaryPrefersMissingModelOverGenericOllama() {
        XCTAssertEqual(
            OpenBirdService.chatFailureSummary(exitCode: 5, stderr: "ollama error: model not found"),
            "Chat failed because a required local model is missing."
        )
    }

    func testChatFailureSummaryAvoidsRawUnknownStderr() {
        XCTAssertEqual(
            OpenBirdService.chatFailureSummary(exitCode: 4, stderr: "private traceback with captured text"),
            "Chat failed (exit 4). Run openbird doctor for details."
        )
    }
}
