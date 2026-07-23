import AppKit
import XCTest
@testable import OpenBirdApp

final class MeetingRecordingTests: XCTestCase {
    func testMeetingCLIEventDecodesMetadataOnlyContract() throws {
        let data = Data(#"{"event":"result","meeting_id":"m1","reason":"completed","backend":"parakeet","observation_id":"o1","dropped_windows":0,"failed_windows":0,"window_count":2,"system_frame_count":12,"mic_frame_count":8,"partial":false}"#.utf8)
        let event = try JSONDecoder().decode(MeetingCLIEvent.self, from: data)

        XCTAssertEqual(event.event, "result")
        XCTAssertEqual(event.meetingId, "m1")
        XCTAssertEqual(event.observationId, "o1")
        XCTAssertEqual(event.backend, "parakeet")
        XCTAssertEqual(event.partial, false)
        XCTAssertEqual(event.windowCount, 2)
        XCTAssertEqual(event.systemFrameCount, 12)
        XCTAssertEqual(event.micFrameCount, 8)
        XCTAssertFalse(String(describing: event).contains("transcript"))
    }

    func testMeetingStatePinsBusyAndRecordingSemantics() {
        XCTAssertFalse(MeetingRecordingState.idle.isBusy)
        XCTAssertFalse(MeetingRecordingState.consent.isBusy)
        XCTAssertTrue(MeetingRecordingState.preparing(downloaded: 1, total: 2).isBusy)
        XCTAssertTrue(MeetingRecordingState.recording(startedAt: Date()).isRecording)
        XCTAssertFalse(
            MeetingRecordingState.finalizing(
                completed: 1, remaining: 1, dropped: 0, failed: 0
            ).isRecording
        )
    }

    func testMeetingRecordingWinsMenuBarSymbol() {
        XCTAssertEqual(
            AppModel.menuBarSymbol(
                captureRunning: true,
                capturePaused: false,
                meetingRecording: true
            ),
            "waveform.circle.fill"
        )
        XCTAssertEqual(
            AppModel.menuBarSymbol(captureRunning: true, capturePaused: false),
            "bird.fill"
        )
    }

    func testAppDelegateUsesNarrowTerminationHook() {
        let delegate = AppDelegate()
        delegate.terminationHandler = { .terminateLater }
        XCTAssertEqual(delegate.applicationShouldTerminate(NSApplication.shared), .terminateLater)
        delegate.terminationHandler = nil
        XCTAssertEqual(delegate.applicationShouldTerminate(NSApplication.shared), .terminateNow)
    }

    func testPreparationStreamsJSONLAndReleasesProcessOwnership() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let cli = directory.appendingPathComponent("openbird-test")
        try "#!/bin/sh\nprintf '%s\\n' '{\"event\":\"result\",\"reason\":\"prepared\",\"backend\":\"parakeet\"}'\n"
            .write(to: cli, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o700], ofItemAtPath: cli.path
        )

        let eventExpectation = expectation(description: "JSONL event")
        let exitExpectation = expectation(description: "process exit")
        let service = OpenBirdService(
            openBirdCLIResolver: { cli.path },
            externalLoopDaemonProbe: { false },
            captureHelperRunningProbe: { false }
        )
        XCTAssertTrue(service.prepareMeetingModel(
            onEvent: { event in
                if event.reason == "prepared" { eventExpectation.fulfill() }
            },
            onExit: { status in
                XCTAssertEqual(status, 0)
                exitExpectation.fulfill()
            }
        ))
        wait(for: [eventExpectation, exitExpectation], timeout: 3)
        XCTAssertFalse(service.meetingProcessIsRunning())
    }
}
