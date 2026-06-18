// swift-tools-version:5.9
import PackageDescription

// OpenBird capture-helper.
//
// A minimal macOS Accessibility (AX) helper that prints the frontmost
// application's active-window text as a single JSON object per capture, on
// stdout, for the Python capture daemon (`openbird/capture/daemon.py`) to parse.
//
// TCC / signing note: Accessibility grants are
// bound to a specific *signed* binary path. This SPM target is the dev build;
// the shipping artifact must be a signed `.app` / LaunchAgent with a stable
// bundle id so the grant persists across rebuilds. `swift run` will NOT carry a
// stable grant — it is for development only.
let package = Package(
    name: "CaptureHelper",
    platforms: [
        .macOS(.v13)
    ],
    targets: [
        .executableTarget(
            name: "CaptureHelper",
            path: "Sources/CaptureHelper"
        )
    ]
)
