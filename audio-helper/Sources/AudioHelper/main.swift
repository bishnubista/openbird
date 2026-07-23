// OpenBird audio-helper — ScreenCaptureKit system audio + microphone capture.
//
// Design:
//   * System OUTPUT audio is captured via ScreenCaptureKit's `SCStream` audio
//     path. macOS does not expose the system output mix as a normal input
//     device, so `ffmpeg -f avfoundation` is explicitly NOT a fallback.
//   * The microphone is a SEPARATE Core Audio stream (AVAudioEngine input) run
//     CONCURRENTLY and kept as a distinct, synchronized track (trackId "mic") all
//     the way through transcription, so "me (mic) vs others (system)" attribution
//     stays meaningful.
//   * Mic and SCK audio have independent sample clocks, so every emitted frame
//     carries a shared host timestamp (`hostTs`, seconds) for downstream
//     clock-alignment / drift detection.
//
// IPC frame contract (binary, little-endian) — the Python pipeline
// (`openbird/meetings/audio.py`) consumes a stream of records, each a fixed
// header followed by a sampleCount-prefixed float32 sample array:
//
//     uint8   track        (0 = system, 1 = mic)
//     float64 hostTs       (seconds, shared host clock)
//     float64 sampleRate   (Hz)
//     uint32  sampleCount  (number of float32 samples that follow)
//     float32 samples[sampleCount]
//
// Privacy by prevention: PCM samples are written ONLY to the binary IPC sink
// (a 0600 file/pipe via `--out`, else stdout for a parent-owned pipe). They never
// appear in argv/env/stderr; stderr carries only NON-content diagnostics.
//
// Runtime needs the signed bundle + Screen-Recording/Microphone TCC. The helper
// runs until manual stop (SIGINT/SIGTERM); `--smoke`/`--max-seconds` bound a run
// so CI/`swift run` terminates.

import AVFoundation
import CoreGraphics
import Foundation
import ScreenCaptureKit

// MARK: - Non-content diagnostics (stderr only)

private func diag(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private struct GrantReport: Encodable {
    let screen_recording: String
    let microphone: String
    let system_audio: String
}

private func screenRecordingGrantState() -> String {
    if CGPreflightScreenCaptureAccess() {
        return "passed"
    }
    // CoreGraphics exposes a non-prompting boolean here, not the mic-style
    // notDetermined/denied split. False proves only "not currently granted".
    return "unknown"
}

private func microphoneGrantState() -> String {
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized:
        return "passed"
    case .denied, .restricted:
        return "failed"
    case .notDetermined:
        return "unknown"
    @unknown default:
        return "unknown"
    }
}

private func emitGrantReport() {
    let screenRecording: String
    let systemAudio: String
    if #available(macOS 13.0, *) {
        screenRecording = screenRecordingGrantState()
        systemAudio = screenRecording
    } else {
        screenRecording = "unknown"
        systemAudio = "failed"
    }
    let report = GrantReport(
        screen_recording: screenRecording,
        microphone: microphoneGrantState(),
        // OpenBird's current system-audio path is a ScreenCaptureKit display
        // stream with capturesAudio=true, so this grant follows the
        // screen/system-audio TCC gate for the packaged helper.
        system_audio: systemAudio
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.withoutEscapingSlashes]
    guard let data = try? encoder.encode(report) else {
        diag("audio: preflight_encode_failed")
        exit(2)
    }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
}

// MARK: - Frame model (the contract the Python pipeline consumes)

/// One mono float32 PCM frame tagged with its track and a shared host clock.
struct AudioFrame {
    let trackId: String   // "system" or "mic"
    let hostTs: Double
    let sampleRate: Double
    let samples: [Float]
}

/// Convert the current mach absolute time (host clock) into seconds.
private func hostSeconds() -> Double {
    var timebase = mach_timebase_info_data_t()
    mach_timebase_info(&timebase)
    let now = mach_absolute_time()
    let nanos = Double(now) * Double(timebase.numer) / Double(timebase.denom)
    return nanos / 1_000_000_000.0
}

// MARK: - Binary IPC writer

/// Serializes :type:`AudioFrame`s as length-prefixed little-endian records to a
/// single output handle. Thread-safe: the SCK sample-handler queue and the mic
/// tap callback both write concurrently.
final class FrameWriter {
    private let handle: FileHandle
    private let lock = NSLock()
    private(set) var framesWritten = 0
    private(set) var systemFrames = 0
    private(set) var micFrames = 0

    init(handle: FileHandle) {
        self.handle = handle
    }

    private func appendLE<T>(_ value: T, to data: inout Data) {
        var v = value
        withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
    }

    func write(_ frame: AudioFrame) {
        var data = Data()
        let track: UInt8 = frame.trackId == "mic" ? 1 : 0
        appendLE(track, to: &data)
        appendLE(frame.hostTs, to: &data)
        appendLE(frame.sampleRate, to: &data)
        appendLE(UInt32(frame.samples.count), to: &data)
        frame.samples.withUnsafeBytes { raw in data.append(contentsOf: raw) }

        lock.lock()
        defer { lock.unlock() }
        handle.write(data)
        framesWritten += 1
        if track == 1 { micFrames += 1 } else { systemFrames += 1 }
    }
}

// MARK: - ScreenCaptureKit system-audio output sink

/// Receives ScreenCaptureKit sample buffers and forwards mono float32 PCM frames.
@available(macOS 13.0, *)
final class AudioSink: NSObject, SCStreamOutput, SCStreamDelegate {
    private let onFrame: (AudioFrame) -> Void

    init(onFrame: @escaping (AudioFrame) -> Void) {
        self.onFrame = onFrame
    }

    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of type: SCStreamOutputType
    ) {
        guard type == .audio else { return }
        guard CMSampleBufferDataIsReady(sampleBuffer) else { return }
        if let frame = Self.extractMonoFloat(from: sampleBuffer) {
            onFrame(frame)
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        diag("audio: stream_stopped error=\(type(of: error))")
    }

    /// Extract mono float32 samples (first channel) from a CMSampleBuffer.
    static func extractMonoFloat(from sampleBuffer: CMSampleBuffer) -> AudioFrame? {
        guard var description = sampleBuffer.formatDescription?.audioStreamBasicDescription
        else { return nil }
        let requiredFlags = AudioFormatFlags(
            kAudioFormatFlagIsFloat | kAudioFormatFlagIsNonInterleaved
        )
        guard description.mFormatID == kAudioFormatLinearPCM,
              description.mFormatFlags & requiredFlags == requiredFlags,
              description.mBitsPerChannel == 32,
              let format = AVAudioFormat(streamDescription: &description),
              format.commonFormat == .pcmFormatFloat32,
              !format.isInterleaved
        else { return nil }

        do {
            return try sampleBuffer.withAudioBufferList { audioBufferList, _ in
                // Follow Apple's ScreenCaptureKit sample path instead of assuming
                // the first AudioBuffer's bytes are already a mono Float array.
                // AVAudioPCMBuffer respects the ASBD's channel layout/stride.
                guard let pcm = AVAudioPCMBuffer(
                    pcmFormat: format,
                    bufferListNoCopy: audioBufferList.unsafePointer
                ),
                let channels = pcm.floatChannelData,
                pcm.frameLength > 0
                else { return nil }
                let frameCount = Int(pcm.frameLength)
                let samples = Array(
                    UnsafeBufferPointer(start: channels[0], count: frameCount)
                )

                // Use the buffer's PRESENTATION timestamp (the actual capture
                // time of the first sample), not callback wall-clock time.
                let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
                let hostTs = pts.flags.contains(.valid)
                    ? CMTimeGetSeconds(pts)
                    : hostSeconds()
                return AudioFrame(
                    trackId: "system",
                    hostTs: hostTs,
                    sampleRate: format.sampleRate,
                    samples: samples
                )
            }
        } catch {
            return nil
        }
    }
}

// MARK: - Microphone capture (separate, synchronized track)

/// Captures the microphone via AVAudioEngine and emits `mic`-tagged frames with
/// the shared host clock. Runs concurrently with the SCK system stream.
final class MicCapture {
    private let engine = AVAudioEngine()
    private let onFrame: (AudioFrame) -> Void
    private var tapped = false

    init(onFrame: @escaping (AudioFrame) -> Void) {
        self.onFrame = onFrame
    }

    func start() throws {
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, when in
            guard let self = self, let channels = buffer.floatChannelData else { return }
            let n = Int(buffer.frameLength)
            let samples = Array(UnsafeBufferPointer(start: channels[0], count: n))
            // Stamp with the tap's host time (the buffer's actual capture time on
            // the shared mach clock), so mic frames align with SCK PTS rather than
            // drifting by callback latency.
            let hostTs = when.hostTime != 0
                ? AVAudioTime.seconds(forHostTime: when.hostTime)
                : hostSeconds()
            self.onFrame(
                AudioFrame(
                    trackId: "mic",
                    hostTs: hostTs,
                    sampleRate: format.sampleRate,
                    samples: samples
                )
            )
        }
        tapped = true
        engine.prepare()
        try engine.start()
    }

    func stop() {
        if tapped {
            engine.inputNode.removeTap(onBus: 0)
            tapped = false
        }
        if engine.isRunning {
            engine.stop()
        }
    }
}

// MARK: - Stop signaling

/// Block until SIGINT/SIGTERM (manual stop) or, if positive, ``maxSeconds``.
private func waitForStop(maxSeconds: Double) async {
    let sem = DispatchSemaphore(value: 0)
    let sigint = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
    let sigterm = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .global())
    sigint.setEventHandler { sem.signal() }
    sigterm.setEventHandler { sem.signal() }
    signal(SIGINT, SIG_IGN)
    signal(SIGTERM, SIG_IGN)
    sigint.resume()
    sigterm.resume()
    if maxSeconds > 0 {
        DispatchQueue.global().asyncAfter(deadline: .now() + maxSeconds) { sem.signal() }
    }
    await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
        DispatchQueue.global().async {
            sem.wait()
            cont.resume()
        }
    }
}

// MARK: - Capture orchestration

@available(macOS 13.0, *)
private func runAsync(maxSeconds: Double, writer: FrameWriter) async {
    // 1) Microphone track (separate, concurrent). Failure is non-fatal: we can
    //    still capture system audio if the mic grant is missing.
    let mic = MicCapture { writer.write($0) }
    do {
        try mic.start()
        diag("audio: mic_started")
    } catch {
        diag("audio: mic_failed error=\(type(of: error))")
    }

    // 2) System-audio track via ScreenCaptureKit.
    let sink = AudioSink { writer.write($0) }
    var startedStream: SCStream?
    do {
        let content = try await SCShareableContent.excludingDesktopWindows(
            false, onScreenWindowsOnly: true)
        guard let display = content.displays.first else {
            diag("audio: no_display")
            mic.stop()
            return
        }
        let filter = SCContentFilter(display: display, excludingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = 48_000
        config.channelCount = 1
        config.width = 2
        config.height = 2

        let stream = SCStream(filter: filter, configuration: config, delegate: sink)
        try stream.addStreamOutput(
            sink, type: .audio, sampleHandlerQueue: DispatchQueue(label: "openbird.audio"))
        try await stream.startCapture()
        startedStream = stream
        diag("audio: system_started")
    } catch {
        diag("audio: system_failed error=\(type(of: error))")
    }

    diag("audio: running max_seconds=\(maxSeconds)")
    // 3) Stream PCM until manual stop (or the bounded smoke deadline).
    await waitForStop(maxSeconds: maxSeconds)

    if let stream = startedStream {
        try? await stream.stopCapture()
    }
    mic.stop()
    diag("audio: stopped frames=\(writer.framesWritten) "
        + "system=\(writer.systemFrames) mic=\(writer.micFrames)")
}

// MARK: - Entry point

/// Whether ``path`` is an existing **private** FIFO: a named pipe owned by the
/// current user with no group/other permission bits (so PCM can't be read by
/// another user via a world/group-readable pipe).
private func isPrivateFifo(_ path: String) -> Bool {
    var st = stat()
    guard stat(path, &st) == 0 else { return false }
    guard (st.st_mode & S_IFMT) == S_IFIFO else { return false }
    guard st.st_uid == getuid() else { return false }
    return (st.st_mode & (mode_t(S_IRWXG) | mode_t(S_IRWXO))) == 0
}

/// Whether stdout is a private pipe/socket, not a TTY or a redirected regular file.
private func stdoutIsPrivatePipe() -> Bool {
    var st = stat()
    guard fstat(FileHandle.standardOutput.fileDescriptor, &st) == 0 else { return false }
    let mode = st.st_mode & S_IFMT
    return mode == S_IFIFO || mode == S_IFSOCK
}

private func openOutput(_ path: String?) -> FileHandle {
    if let path = path {
        // `--out` MUST be a caller-created FIFO/pipe (e.g. `mkfifo`). If it is
        // missing/invalid/not-a-FIFO we FAIL CLOSED (exit non-zero) rather than
        // fall back to stdout — silently dumping raw meeting PCM to a terminal or
        // log would defeat the whole privacy boundary.
        guard isPrivateFifo(path), let handle = FileHandle(forWritingAtPath: path) else {
            diag("audio: --out must be an owner-only (0600) FIFO (mkfifo); refusing (fail-closed)")
            exit(3)
        }
        return handle
    }
    // No --out: only use stdout when it is a genuine parent-owned pipe/socket —
    // NOT a TTY and NOT a redirected regular file/launchd log (which would persist
    // raw PCM). Fail closed otherwise.
    if !stdoutIsPrivatePipe() {
        diag("audio: stdout is not a private pipe and no --out FIFO given; refusing (fail-closed)")
        exit(3)
    }
    return .standardOutput
}

private func run() {
    if CommandLine.arguments.contains("--preflight-grants") {
        emitGrantReport()
        return
    }

    // Trigger the Screen-Recording authorization prompt for THIS helper binary so
    // macOS registers it in System Settings > Privacy > Screen & System Audio
    // Recording (it never appears there until it requests the grant once).
    if CommandLine.arguments.contains("--request-screen") {
        if #available(macOS 11.0, *) {
            _ = CGRequestScreenCaptureAccess()
        }
        return
    }

    // Trigger the Microphone authorization prompt (registers the helper in the
    // Microphone list). Needs NSMicrophoneUsageDescription, which is embedded in
    // this binary's __info_plist section.
    if CommandLine.arguments.contains("--request-microphone") {
        let sem = DispatchSemaphore(value: 0)
        AVCaptureDevice.requestAccess(for: .audio) { _ in sem.signal() }
        sem.wait()
        return
    }

    guard #available(macOS 13.0, *) else {
        diag("audio: requires_macos_13+")
        exit(2)
    }

    let args = CommandLine.arguments
    var maxSeconds = 0.0   // 0 = run until SIGINT/SIGTERM
    var outPath: String?
    var i = 1
    while i < args.count {
        switch args[i] {
        case "--max-seconds":
            if i + 1 < args.count { maxSeconds = Double(args[i + 1]) ?? 0; i += 1 }
        case "--smoke":
            maxSeconds = 1.0
        case "--out":
            if i + 1 < args.count { outPath = args[i + 1]; i += 1 }
        default:
            break
        }
        i += 1
    }

    let writer = FrameWriter(handle: openOutput(outPath))
    let group = DispatchGroup()
    group.enter()
    Task {
        await runAsync(maxSeconds: maxSeconds, writer: writer)
        group.leave()
    }
    group.wait()
}

run()
