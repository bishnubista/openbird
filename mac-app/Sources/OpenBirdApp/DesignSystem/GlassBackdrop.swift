import SwiftUI

/// Opaque-window backdrop: a soft base plus the handoff's blurred color "orbs" so
/// glass cards have something colorful to refract (a window has no desktop behind
/// it, unlike the floating Ask panel — so here the orbs are intentional). Kept
/// low-opacity for legibility in both appearances. Shared by the main window and
/// the Today/day view.
struct GlassBackdrop: View {
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        ZStack {
            (scheme == .dark ? Color(hex: 0x1B1B20) : Color(hex: 0xF2F2F5))
            orb(Color(hex: 0x785AFF), x: 0.12, y: 0.08, size: 460)   // purple
            orb(Color(hex: 0x2F7FF2), x: 0.92, y: 0.30, size: 520)   // blue
            orb(Color(hex: 0xFF78AA), x: 0.70, y: 0.95, size: 420)   // pink
        }
        .ignoresSafeArea()
    }

    private func orb(_ color: Color, x: CGFloat, y: CGFloat, size: CGFloat) -> some View {
        GeometryReader { geo in
            RadialGradient(
                gradient: Gradient(colors: [color.opacity(scheme == .dark ? 0.30 : 0.18), .clear]),
                center: .center, startRadius: 0, endRadius: size / 2
            )
            .frame(width: size, height: size)
            .position(x: geo.size.width * x, y: geo.size.height * y)
            .blur(radius: 20)
        }
    }
}
