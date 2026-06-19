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
            path: "Sources/CaptureHelper",
            // NOTE: dangerous_apps.json is the CANONICAL dangerous-app list and a
            // committed source file, but it is intentionally NOT bundled as a
            // runtime resource. The shipped helper is a bare executable (no
            // SwiftPM resource bundle is copied into OpenBird.app), so reading it
            // via `Bundle.module` would `fatalError`. The list is baked into
            // `main.swift` instead; a Python parity test keeps the JSON, the
            // Swift literal, and the Python tuple in lockstep. See main.swift.
            // Info.plist is embedded into __TEXT,__info_plist (see linkerSettings)
            // to give the bare helper a stable CFBundleIdentifier for TCC; exclude
            // it (and the canonical JSON) from the source/resource list.
            exclude: ["dangerous_apps.json", "Info.plist"],
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", "Sources/CaptureHelper/Info.plist",
                ])
            ]
        )
    ]
)
