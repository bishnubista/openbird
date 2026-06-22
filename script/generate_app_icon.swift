#!/usr/bin/env swift
import AppKit
import Foundation

guard CommandLine.arguments.count == 2 else {
    FileHandle.standardError.write(Data("usage: generate_app_icon.swift /path/to/AppIcon.icns\n".utf8))
    exit(2)
}

let output = URL(fileURLWithPath: CommandLine.arguments[1])
try FileManager.default.createDirectory(
    at: output.deletingLastPathComponent(),
    withIntermediateDirectories: true
)

let iconset = FileManager.default.temporaryDirectory
    .appendingPathComponent("OpenBird-\(UUID().uuidString).iconset")
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
defer { try? FileManager.default.removeItem(at: iconset) }

let specs: [(points: Int, scale: Int)] = [
    (16, 1), (16, 2),
    (32, 1), (32, 2),
    (128, 1), (128, 2),
    (256, 1), (256, 2),
    (512, 1), (512, 2),
]

for spec in specs {
    let suffix = spec.scale == 1 ? "" : "@\(spec.scale)x"
    let filename = "icon_\(spec.points)x\(spec.points)\(suffix).png"
    try renderIcon(
        pixels: spec.points * spec.scale,
        to: iconset.appendingPathComponent(filename)
    )
}

guard let iconutilURL = findExecutable("iconutil") else {
    FileHandle.standardError.write(Data("iconutil not found in PATH\n".utf8))
    exit(127)
}

let process = Process()
process.executableURL = iconutilURL
process.arguments = ["-c", "icns", "-o", output.path, iconset.path]
try process.run()
process.waitUntilExit()
guard process.terminationStatus == 0 else {
    FileHandle.standardError.write(Data("iconutil failed with \(process.terminationStatus)\n".utf8))
    exit(process.terminationStatus)
}

func findExecutable(_ name: String) -> URL? {
    let path = ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin"
    for directory in path.split(separator: ":") {
        let candidate = URL(fileURLWithPath: String(directory)).appendingPathComponent(name)
        if FileManager.default.isExecutableFile(atPath: candidate.path) {
            return candidate
        }
    }
    return nil
}

func renderIcon(pixels: Int, to url: URL) throws {
    guard let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: pixels,
        pixelsHigh: pixels,
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .deviceRGB,
        bytesPerRow: 0,
        bitsPerPixel: 0
    ) else {
        throw NSError(domain: "OpenBirdIcon", code: 1)
    }

    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)

    let rect = CGRect(x: 0, y: 0, width: pixels, height: pixels)
    let corner = CGFloat(pixels) * 0.22
    let bg = NSBezierPath(roundedRect: rect, xRadius: corner, yRadius: corner)
    NSColor(calibratedRed: 0.05, green: 0.07, blue: 0.12, alpha: 1).setFill()
    bg.fill()

    let inset = CGFloat(pixels) * 0.12
    let inner = rect.insetBy(dx: inset, dy: inset)
    let glow = NSBezierPath(roundedRect: inner, xRadius: corner * 0.75, yRadius: corner * 0.75)
    NSColor(calibratedRed: 0.18, green: 0.50, blue: 0.95, alpha: 0.18).setFill()
    glow.fill()

    if let context = NSGraphicsContext.current?.cgContext {
        drawBird(in: context, rect: rect.insetBy(dx: CGFloat(pixels) * 0.15, dy: CGFloat(pixels) * 0.22))
    }

    NSGraphicsContext.restoreGraphicsState()

    guard let data = rep.representation(using: .png, properties: [:]) else {
        throw NSError(domain: "OpenBirdIcon", code: 2)
    }
    try data.write(to: url)
}

func drawBird(in context: CGContext, rect: CGRect) {
    let scale = min(rect.width, rect.height) / 48.0
    context.saveGState()
    context.translateBy(x: rect.minX, y: rect.maxY)
    context.scaleBy(x: scale, y: -scale)

    let path = CGMutablePath()
    func p(_ x: CGFloat, _ y: CGFloat) -> CGPoint { CGPoint(x: x, y: y) }

    // Beak
    path.move(to: p(15, 16)); path.addLine(to: p(2, 11)); path.addLine(to: p(15, 19)); path.closeSubpath()

    // Head
    path.addEllipse(in: CGRect(x: 16.5 - 4.8, y: 17 - 4.8, width: 9.6, height: 9.6))

    // Body, rotated 30 degrees around center.
    let body = CGMutablePath()
    body.addEllipse(in: CGRect(x: 24 - 10, y: 26 - 6.2, width: 20, height: 12.4))
    let transform = CGAffineTransform(translationX: 24, y: 26)
        .rotated(by: .pi / 6)
        .translatedBy(x: -24, y: -26)
    path.addPath(body, transform: transform)

    // Tail
    path.move(to: p(30, 30)); path.addLine(to: p(45, 36)); path.addLine(to: p(37, 30)); path.addLine(to: p(44, 25)); path.closeSubpath()

    // Wing
    path.move(to: p(19, 20))
    path.addCurve(to: p(45, 9), control1: p(26, 8), control2: p(36, 5))
    path.addCurve(to: p(30, 25), control1: p(37, 11), control2: p(32, 16))
    path.addCurve(to: p(19, 20), control1: p(27, 20), control2: p(22, 19))
    path.closeSubpath()

    context.setFillColor(CGColor(red: 0.18, green: 0.50, blue: 0.95, alpha: 1))
    context.addPath(path)
    context.fillPath()
    context.restoreGState()
}
