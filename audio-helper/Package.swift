// swift-tools-version:5.9
import PackageDescription

// OpenBird audio-helper.
//
// A minimal ScreenCaptureKit skeleton that captures system output audio (and is
// designed to mix in the microphone as a SEPARATE synchronized track) and emits
// float32 PCM frames + a shared host timestamp for the Python meetings pipeline
// (`openbird/meetings/`). This target is the dev build; the shipping artifact
// must be a signed `.app` / LaunchAgent holding Screen-Recording + Microphone
// TCC (PLAN.md system-audio gate).
//
// NOTE: ScreenCaptureKit audio capture requires macOS 13+ and Screen-Recording
// permission at runtime; this skeleton COMPILES without those grants — actual
// capture needs the signed bundle + TCC.
let package = Package(
    name: "AudioHelper",
    platforms: [
        .macOS(.v13)
    ],
    targets: [
        .executableTarget(
            name: "AudioHelper",
            path: "Sources/AudioHelper"
        )
    ]
)
