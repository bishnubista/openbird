import Foundation
import XCTest
@testable import OpenBirdApp

final class AssistantSettingsTests: XCTestCase {
    func testFailedTunnelStartRemainsRetryableButAppTerminationDoesNot() {
        var lifecycle = ChatGPTTunnelLifecycleState()

        lifecycle.recordStop(.retryable)
        XCTAssertTrue(lifecycle.canLaunch)

        lifecycle.recordStop(.appTermination)
        XCTAssertFalse(lifecycle.canLaunch)
    }

    func testParsesClaudeAssistantStatus() {
        let output = #"{"configured":true,"config_path":"/tmp/claude.json","command":"/Applications/OpenBird.app/Contents/MacOS/openbird-cli"}"#

        let status = OpenBirdService.parseClaudeAssistantStatus(output)

        XCTAssertEqual(
            status,
            ClaudeAssistantStatus(
                configured: true,
                configPath: "/tmp/claude.json",
                command: "/Applications/OpenBird.app/Contents/MacOS/openbird-cli"
            )
        )
    }

    func testRejectsMalformedClaudeAssistantStatus() {
        XCTAssertNil(OpenBirdService.parseClaudeAssistantStatus("not json"))
        XCTAssertNil(OpenBirdService.parseClaudeAssistantStatus(#"{"configured":true}"#))
    }

    func testConnectClaudeUsesConfirmedCLIShape() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let script = root.appendingPathComponent("openbird")
        let body = "#!/bin/sh\n[ \"$1\" = assistant ] && [ \"$2\" = install-claude ] && [ \"$3\" = --yes ] && [ \"$4\" = --executable ] && [ \"$5\" = \"$0\" ]\n"
        try body.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: script.path)
        let service = OpenBirdService(openBirdCLIResolver: { script.path })

        let connected = await service.connectClaudeAssistant()

        XCTAssertTrue(connected)
    }

    func testClaudeStatusUsesMetadataOnlyCLIShape() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let script = root.appendingPathComponent("openbird")
        let body = """
        #!/bin/sh
        [ "$1" = assistant ] && [ "$2" = status ] && [ "$3" = --json ] || exit 9
        printf '%s\\n' '{"configured":false,"config_path":"/tmp/claude.json","command":null}'
        """
        try body.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: script.path)
        let service = OpenBirdService(openBirdCLIResolver: { script.path })

        let status = await service.claudeAssistantStatus()

        XCTAssertEqual(status?.configured, false)
        XCTAssertEqual(status?.configPath, "/tmp/claude.json")
        XCTAssertNil(status?.command)
    }

    func testParsesChatGPTMetadataWithoutCredentials() {
        let output = #"{"configured":true,"helper_available":true}"#

        let status = OpenBirdService.parseChatGPTAssistantStatus(output)

        XCTAssertEqual(
            status,
            ChatGPTAssistantStatus(
                configured: true,
                helperAvailable: true,
                running: false,
                ready: false
            )
        )
        XCTAssertFalse(output.contains("runtime_key"))
    }

    func testChatGPTStatusUsesMetadataOnlyCLIShape() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let script = root.appendingPathComponent("openbird")
        let body = """
        #!/bin/sh
        [ "$1" = assistant ] && [ "$2" = chatgpt-status ] && [ "$3" = --json ] && [ "$4" = --executable ] && [ "$5" = "$0" ] || exit 9
        printf '%s\\n' '{"configured":false,"helper_available":true}'
        """
        try body.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: script.path)
        let service = OpenBirdService(
            openBirdCLIResolver: { script.path },
            chatGPTCredentialLoader: { nil },
            chatGPTCredentialSaver: { _ in false },
            chatGPTCredentialDeleter: { true }
        )

        let status = await service.chatGPTAssistantStatus()

        XCTAssertEqual(status?.configured, false)
        XCTAssertEqual(status?.helperAvailable, true)
        XCTAssertEqual(status?.running, false)
        XCTAssertEqual(status?.ready, false)
    }
}
