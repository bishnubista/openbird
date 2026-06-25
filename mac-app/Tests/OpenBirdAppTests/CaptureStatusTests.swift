import XCTest
@testable import OpenBirdApp

/// Unit coverage for the pure external-capture detection used by
/// `OpenBirdService.isCaptureRunning()`. The decision logic is factored out of
/// the process-spawning path so it can be exercised against realistic
/// `pgrep -fl` lines without launching anything.
final class CaptureStatusTests: XCTestCase {
    private let ownPID: Int32 = 4242

    // MARK: - Positive matches (an external `openbird capture --loop` daemon)

    func testDetectsRealPythonConsoleScriptDaemon() {
        // The actual argv observed live: interpreter + console-script + subcommand.
        let line = "31196 /Users/me/.venv/bin/python3 /Users/me/.venv/bin/openbird capture --loop"
        XCTAssertTrue(OpenBirdService.externalCaptureRunning(pgrepOutput: line, ownPID: ownPID))
    }

    func testDetectsHomebrewDaemon() {
        let line = "5001 /opt/homebrew/bin/openbird capture --loop"
        XCTAssertTrue(OpenBirdService.externalCaptureRunning(pgrepOutput: line, ownPID: ownPID))
    }

    func testDetectsBundledCLIDaemon() {
        let line = "5002 /Applications/OpenBird.app/Contents/MacOS/openbird-cli capture --loop"
        XCTAssertTrue(OpenBirdService.externalCaptureRunning(pgrepOutput: line, ownPID: ownPID))
    }

    func testDetectsDaemonAmongMultipleLines() {
        let output = """
        700 /usr/libexec/cameracaptured
        5003 /opt/homebrew/bin/openbird capture --loop
        701 /Applications/Google Chrome.app/Contents/MacOS/Google Chrome Helper --type=utility --utility-sub-type=video_capture.mojom.VideoCaptureService
        """
        XCTAssertTrue(OpenBirdService.externalCaptureRunning(pgrepOutput: output, ownPID: ownPID))
    }

    // MARK: - Negative matches (must NOT report capturing)

    func testRejectsBoundedOncePass() {
        // `--once` is a single bounded pass, not a long-running daemon.
        let line = "5004 /opt/homebrew/bin/openbird capture --once"
        XCTAssertFalse(OpenBirdService.externalCaptureRunning(pgrepOutput: line, ownPID: ownPID))
    }

    func testRejectsUnrelatedOpenBirdSubcommand() {
        let line = "5005 /opt/homebrew/bin/openbird doctor"
        XCTAssertFalse(OpenBirdService.externalCaptureRunning(pgrepOutput: line, ownPID: ownPID))
    }

    func testRejectsChromeVideoCapture() {
        let line = "701 /Applications/Google Chrome.app/Contents/MacOS/Google Chrome Helper --type=utility --utility-sub-type=video_capture.mojom.VideoCaptureService"
        XCTAssertFalse(OpenBirdService.externalCaptureRunning(pgrepOutput: line, ownPID: ownPID))
    }

    func testRejectsSystemCameraCaptured() {
        let line = "700 /usr/libexec/cameracaptured"
        XCTAssertFalse(OpenBirdService.externalCaptureRunning(pgrepOutput: line, ownPID: ownPID))
    }

    func testRejectsOwnProcess() {
        let line = "\(ownPID) /Applications/OpenBird.app/Contents/MacOS/openbird capture --loop"
        XCTAssertFalse(OpenBirdService.externalCaptureRunning(pgrepOutput: line, ownPID: ownPID))
    }

    func testRejectsPgrepSelfMatch() {
        // The very `pgrep -fl` query whose argv contains the search pattern.
        let line = "5006 /usr/bin/pgrep -fl openbird(-cli)? capture --loop"
        XCTAssertFalse(OpenBirdService.externalCaptureRunning(pgrepOutput: line, ownPID: ownPID))
    }

    func testRejectsEditorWithMatchingFilenameButNotTheCLI() {
        // An editor opened on a doc whose name resembles the pattern. No argv token
        // basename is the CLI, and there are no `capture`/`--loop` tokens.
        let line = "5007 /usr/bin/vim openbird-capture-loop.md"
        XCTAssertFalse(OpenBirdService.externalCaptureRunning(pgrepOutput: line, ownPID: ownPID))
    }

    func testRejectsSubstringOnlyBinaryName() {
        // A hypothetical binary whose name merely contains "openbird" must not pass
        // the exact-basename rule.
        let line = "5008 /usr/local/bin/openbird-tray capture --loop"
        XCTAssertFalse(OpenBirdService.externalCaptureRunning(pgrepOutput: line, ownPID: ownPID))
    }

    func testRejectsNonAdjacentCaptureToken() {
        // A lone `openbird` token NOT immediately followed by `capture` must not
        // match even if `capture`/`--loop` appear elsewhere on the line.
        let line = "5009 /opt/homebrew/bin/openbird timeline --note capture --loop"
        XCTAssertFalse(OpenBirdService.externalCaptureRunning(pgrepOutput: line, ownPID: ownPID))
    }

    func testRejectsLoopBeforeCaptureSubcommand() {
        // `--loop` must appear AFTER the `capture` subcommand (the daemon shape),
        // not before an unrelated openbird invocation.
        let line = "5010 /opt/homebrew/bin/openbird --loop-something capture --once"
        XCTAssertFalse(OpenBirdService.externalCaptureRunning(pgrepOutput: line, ownPID: ownPID))
    }

    func testHandlesEmptyOutput() {
        XCTAssertFalse(OpenBirdService.externalCaptureRunning(pgrepOutput: "", ownPID: ownPID))
    }

    func testHandlesMalformedLineWithoutPID() {
        XCTAssertFalse(OpenBirdService.externalCaptureRunning(
            pgrepOutput: "not-a-pid openbird capture --loop", ownPID: ownPID
        ))
    }
}
