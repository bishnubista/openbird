// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "OpenBirdMac",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "OpenBird", targets: ["OpenBirdApp"])
    ],
    targets: [
        .executableTarget(
            name: "OpenBirdApp",
            path: "Sources/OpenBirdApp"
        )
    ]
)
