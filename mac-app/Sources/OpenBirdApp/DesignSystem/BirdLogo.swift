import SwiftUI

/// The OpenBird hummingbird mark, recreated as a vector `Shape` from the handoff's
/// inline SVG (`viewBox 0 0 48 48`): a beak triangle, head circle, a body ellipse
/// rotated 30°, tail feathers, and a Bézier wing. Fill it with a color via
/// `.foregroundStyle` (mirrors the SVG `fill:currentColor`). Used at 13–54pt.
struct BirdLogo: Shape {
    func path(in rect: CGRect) -> Path {
        let s = min(rect.width, rect.height) / 48.0
        func p(_ x: CGFloat, _ y: CGFloat) -> CGPoint {
            CGPoint(x: rect.minX + x * s, y: rect.minY + y * s)
        }
        // Scale-then-translate matrix for sub-paths authored in 48pt space.
        let toRect = CGAffineTransform(translationX: rect.minX, y: rect.minY).scaledBy(x: s, y: s)

        var path = Path()

        // Beak (M15 16 L2 11 L15 19 Z)
        path.move(to: p(15, 16)); path.addLine(to: p(2, 11)); path.addLine(to: p(15, 19)); path.closeSubpath()

        // Head (circle cx16.5 cy17 r4.8)
        path.addEllipse(in: CGRect(x: 16.5 - 4.8, y: 17 - 4.8, width: 9.6, height: 9.6).applying(toRect))

        // Body (ellipse cx24 cy26 rx10 ry6.2, rotate 30° about its center)
        var body = Path(ellipseIn: CGRect(x: 24 - 10, y: 26 - 6.2, width: 20, height: 12.4))
        let c = CGPoint(x: 24, y: 26)
        let rot = CGAffineTransform(translationX: c.x, y: c.y)
            .rotated(by: .pi / 6)
            .translatedBy(x: -c.x, y: -c.y)
        body = body.applying(rot)
        path.addPath(body, transform: toRect)

        // Tail (M30 30 L45 36 L37 30 L44 25 Z)
        path.move(to: p(30, 30)); path.addLine(to: p(45, 36)); path.addLine(to: p(37, 30)); path.addLine(to: p(44, 25)); path.closeSubpath()

        // Wing (M19 20 C 26 8 36 5 45 9 C 37 11 32 16 30 25 C 27 20 22 19 19 20 Z)
        path.move(to: p(19, 20))
        path.addCurve(to: p(45, 9), control1: p(26, 8), control2: p(36, 5))
        path.addCurve(to: p(30, 25), control1: p(37, 11), control2: p(32, 16))
        path.addCurve(to: p(19, 20), control1: p(27, 20), control2: p(22, 19))
        path.closeSubpath()

        return path
    }
}
