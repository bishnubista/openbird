import AppKit
import Foundation

final class OpenBirdService {
    private let fileManager = FileManager.default

    private var dataDirectory: URL {
        if let override = ProcessInfo.processInfo.environment["OPENBIRD_DATA_DIR"],
           !override.isEmpty {
            return URL(fileURLWithPath: NSString(string: override).expandingTildeInPath)
        }
        return fileManager.homeDirectoryForCurrentUser.appendingPathComponent(".openbird")
    }

    private var pauseFile: URL {
        dataDirectory.appendingPathComponent("capture.paused")
    }

    func isCapturePaused() -> Bool {
        fileManager.fileExists(atPath: pauseFile.path)
    }

    @discardableResult
    func setCapturePaused(_ paused: Bool) throws -> Bool {
        try fileManager.createDirectory(
            at: dataDirectory,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
        if paused {
            try Data().write(to: pauseFile, options: .atomic)
            try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: pauseFile.path)
        } else if fileManager.fileExists(atPath: pauseFile.path) {
            try fileManager.removeItem(at: pauseFile)
        }
        return isCapturePaused()
    }

    func helperStatuses() -> [HelperStatus] {
        [
            helperStatus(id: "capture", label: "Capture helper", executable: "capture-helper"),
            helperStatus(id: "audio", label: "Audio helper", executable: "audio-helper")
        ]
    }

    func stopHelperProcesses() -> Bool {
        let processNames = [
            "capture-helper",
            "audio-helper",
            "CaptureHelper",
            "AudioHelper"
        ]
        return processNames
            .map { Self.run("/usr/bin/pkill", arguments: ["-x", $0]).exitCode == 0 }
            .contains(true)
    }

    func openDataFolder() {
        try? fileManager.createDirectory(
            at: dataDirectory,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
        NSWorkspace.shared.open(dataDirectory)
    }

    func openBundleFolder() {
        NSWorkspace.shared.activateFileViewerSelecting([Bundle.main.bundleURL])
    }

    func preflightSummary() async -> PreflightSummary {
        guard let cli = resolveOpenBirdCLI() else {
            return PreflightSummary(
                status: "CLI Missing",
                detail: "Install the openbird command or use the bundled app wrapper."
            )
        }

        let result = await runAsync(
            cli,
            arguments: ["preflight", "--json", "--no-ollama"],
            timeout: 20
        )
        guard result.exitCode == 0 || result.exitCode == 1 else {
            return PreflightSummary(
                status: "Preflight Error",
                detail: result.stderr.isEmpty ? "openbird exited with \(result.exitCode)." : result.stderr
            )
        }
        return parsePreflight(result.stdout)
    }

    private func helperStatus(id: String, label: String, executable: String) -> HelperStatus {
        let url = Bundle.main.bundleURL
            .appendingPathComponent("Contents")
            .appendingPathComponent("MacOS")
            .appendingPathComponent(executable)
        return HelperStatus(
            id: id,
            label: label,
            isBundled: fileManager.isExecutableFile(atPath: url.path),
            path: url.path
        )
    }

    private func resolveOpenBirdCLI() -> String? {
        let bundled = Bundle.main.bundleURL
            .appendingPathComponent("Contents")
            .appendingPathComponent("MacOS")
            .appendingPathComponent("openbird-cli")
        let candidates = [
            bundled.path,
            "/opt/homebrew/bin/openbird",
            "/usr/local/bin/openbird"
        ]
        return candidates.first { fileManager.isExecutableFile(atPath: $0) }
    }

    private func parsePreflight(_ output: String) -> PreflightSummary {
        guard let data = output.data(using: .utf8),
              let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return PreflightSummary(status: "Unknown", detail: "Could not parse preflight JSON.")
        }

        let runtimeOK = payload["runtime_ok"] as? Bool ?? false
        let releaseOK = payload["release_gate_ok"] as? Bool ?? false
        let encryption = (payload["encryption"] as? [String: Any])?["status"] as? String ?? "unknown"
        let macos = payload["macos"] as? [String: Any]
        let accessibility = macos?["accessibility"] as? String ?? "unknown"
        let systemAudio = macos?["system_audio"] as? String ?? "unknown"

        let status = runtimeOK ? "Runtime Ready" : "Needs Setup"
        let release = releaseOK ? "release gate OK" : "release gate pending"
        let detail = "encryption=\(encryption), ax=\(accessibility), system-audio=\(systemAudio), \(release)"
        return PreflightSummary(status: status, detail: detail)
    }

    private static func run(_ path: String, arguments: [String], timeout: TimeInterval = 4) -> ProcessResult {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = arguments

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr

        do {
            try process.run()
        } catch {
            return ProcessResult(exitCode: 127, stdout: "", stderr: error.localizedDescription)
        }

        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        if process.isRunning {
            process.terminate()
        }
        process.waitUntilExit()

        let out = String(data: stdout.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let err = String(data: stderr.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        return ProcessResult(
            exitCode: Int(process.terminationStatus),
            stdout: out.trimmingCharacters(in: .whitespacesAndNewlines),
            stderr: err.trimmingCharacters(in: .whitespacesAndNewlines)
        )
    }

    private func runAsync(
        _ path: String,
        arguments: [String],
        timeout: TimeInterval
    ) async -> ProcessResult {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                continuation.resume(
                    returning: Self.run(path, arguments: arguments, timeout: timeout)
                )
            }
        }
    }
}

private struct ProcessResult {
    let exitCode: Int
    let stdout: String
    let stderr: String
}
