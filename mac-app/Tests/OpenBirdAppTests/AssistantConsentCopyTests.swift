import XCTest

@testable import OpenBirdApp

/// Pins the native consent disclosures. The connect flows call the CLI with
/// `--yes`, so this copy is the only warning the user sees — a copy edit that
/// drops a disclosed data category must fail here, not ship silently.
final class AssistantConsentCopyTests: XCTestCase {
    private let requiredDisclosures = [
        "redacted",             // redacted apps are named…
        "reason code",          // …with their reason codes
        "timestamps",           // observation/window timestamps
        "host label",           // machine identity leaves the Mac
        "usage durations",      // activity metadata
        "excerpts",             // captured text (content tools)
        "memory-store totals",  // status tool: store-lifetime counts
        "encryption state",     // status tool: encryption flag
        "exclusion counts",     // status tool: exclusion configuration
        "timezone",             // resolved query window + system timezone
        "cannot be recalled",   // irreversibility
    ]

    func testClaudeConsentDisclosesEveryEgressCategory() {
        let copy = AssistantConsentCopy.claudeConnectMessage
        for phrase in requiredDisclosures {
            XCTAssertTrue(copy.contains(phrase), "Claude consent copy lost disclosure: \(phrase)")
        }
        XCTAssertTrue(copy.contains("never sends data in the background"))
    }

    func testChatGPTConsentDisclosesEveryEgressCategory() {
        let copy = AssistantConsentCopy.chatGPTConnectMessage
        for phrase in requiredDisclosures {
            XCTAssertTrue(copy.contains(phrase), "ChatGPT consent copy lost disclosure: \(phrase)")
        }
        XCTAssertTrue(copy.contains("never uploads capture in the background"))
    }
}
