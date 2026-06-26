import Foundation
import XCTest
@testable import OpenBirdApp

final class PromptSettingsTests: XCTestCase {
    private func withEnvironment<T>(
        _ updates: [String: String?],
        _ body: () throws -> T
    ) rethrows -> T {
        var old: [String: String?] = [:]
        for key in updates.keys {
            old[key] = ProcessInfo.processInfo.environment[key]
        }
        defer {
            for (key, value) in old {
                if let value {
                    setenv(key, value, 1)
                } else {
                    unsetenv(key)
                }
            }
        }
        for (key, value) in updates {
            if let value {
                setenv(key, value, 1)
            } else {
                unsetenv(key)
            }
        }
        return try body()
    }

    func testPromptDirectoryPrefersExplicitOverrideWithoutExpandingTilde() {
        let env = [
            "OPENBIRD_PROMPTS_DIR": "~/personas",
            "OPENBIRD_DATA_DIR": "~/ignored-data",
        ]

        XCTAssertEqual(OpenBirdService.promptDirectoryPath(environment: env), "~/personas")
        XCTAssertTrue(OpenBirdService.promptDirectoryURL(environment: env).path.hasSuffix("/~/personas"))
        XCTAssertNotEqual(
            OpenBirdService.promptDirectoryURL(environment: env).path,
            "\(NSHomeDirectory())/personas"
        )
    }

    func testPromptDirectoryFallsBackToDataDirPrompts() {
        let env = [
            "OPENBIRD_PROMPTS_DIR": "",
            "OPENBIRD_DATA_DIR": "~/OpenBirdData",
        ]

        XCTAssertEqual(
            OpenBirdService.promptDirectoryPath(environment: env),
            "\(NSHomeDirectory())/OpenBirdData/prompts"
        )
    }

    func testOpenPromptsFolderCreatesDirectoryAndOpensIt() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let prompts = root.appendingPathComponent("personas", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let recorder = PromptFolderRecorder()
        let service = OpenBirdService(
            promptFolderOpener: { recorder.open($0) },
            openBirdCLIResolver: { "/tmp/openbird" }
        )

        withEnvironment([
            "OPENBIRD_PROMPTS_DIR": prompts.path,
            "OPENBIRD_DATA_DIR": nil,
        ]) {
            service.openPromptsFolder()
        }

        XCTAssertEqual(recorder.openedURL?.path, prompts.path)
        var isDirectory: ObjCBool = false
        XCTAssertTrue(FileManager.default.fileExists(atPath: prompts.path, isDirectory: &isDirectory))
        XCTAssertTrue(isDirectory.boolValue)
        let attrs = try FileManager.default.attributesOfItem(atPath: prompts.path)
        XCTAssertEqual(attrs[.posixPermissions] as? Int, 0o700)
    }

    func testEditPromptPersonaLaunchesCliWithKeyAndGuiEditor() async {
        let recorder = PromptEditRecorder()
        let service = OpenBirdService(
            promptEditRunner: { path, arguments, environment in
                recorder.record(path: path, arguments: arguments, environment: environment)
                return ProcessResult(exitCode: 0, stdout: "", stderr: "")
            },
            openBirdCLIResolver: { "/tmp/openbird" }
        )

        let outcome = await service.editPromptPersona(.rag)

        XCTAssertEqual(outcome, .launched)
        XCTAssertEqual(recorder.path, "/tmp/openbird")
        XCTAssertEqual(recorder.arguments, ["prompts", "edit", "rag"])
        XCTAssertEqual(recorder.environment?["EDITOR"], "/usr/bin/open")
        XCTAssertEqual(recorder.environment?["PYTHONDONTWRITEBYTECODE"], "1")
    }

    func testEditPromptPersonaReportsMissingCli() async {
        let recorder = PromptEditRecorder()
        let service = OpenBirdService(
            promptEditRunner: { path, arguments, environment in
                recorder.record(path: path, arguments: arguments, environment: environment)
                return ProcessResult(exitCode: 0, stdout: "", stderr: "")
            },
            openBirdCLIResolver: { nil }
        )

        let outcome = await service.editPromptPersona(.meeting)

        XCTAssertEqual(outcome, .cliMissing)
        XCTAssertNil(recorder.path)
    }

    func testEditPromptPersonaReportsNonZeroExit() async {
        let service = OpenBirdService(
            promptEditRunner: { _, _, _ in
                ProcessResult(exitCode: 3, stdout: "", stderr: "failed")
            },
            openBirdCLIResolver: { "/tmp/openbird" }
        )

        let outcome = await service.editPromptPersona(.signal)

        XCTAssertEqual(outcome, .failed(3))
    }
}

private final class PromptFolderRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var openedURL: URL?

    func open(_ url: URL) {
        lock.lock()
        openedURL = url
        lock.unlock()
    }
}

private final class PromptEditRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var path: String?
    private(set) var arguments: [String]?
    private(set) var environment: [String: String]?

    func record(path: String, arguments: [String], environment: [String: String]) {
        lock.lock()
        self.path = path
        self.arguments = arguments
        self.environment = environment
        lock.unlock()
    }
}
