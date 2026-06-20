import SwiftUI

extension View {
    /// Apply the shared Liquid Glass material to a floating surface.
    ///
    /// On macOS 26 (Tahoe) this uses the system `.glassEffect` so the surface gets
    /// real specular highlights + lensing. On every earlier macOS (the app targets
    /// 13+) it falls back to `.ultraThinMaterial` plus the handoff's gradient sheen,
    /// hairline border, and float/contact shadow stack — a faithful match without
    /// the native API.
    ///
    /// IMPORTANT: the native path is behind a *compile-time* gate (`#if
    /// compiler(>=6.2)`), not just the runtime `#available`. `.glassEffect` is a
    /// symbol that only exists in the macOS 26 SDK; referencing it under a toolchain
    /// whose SDK predates it would fail to *compile*, long before any runtime check
    /// runs. The material path is therefore the unconditional default.
    @ViewBuilder
    func glassSurface(cornerRadius r: CGFloat) -> some View {
        #if compiler(>=6.2)
        if #available(macOS 26, *) {
            self.glassEffect(.regular, in: RoundedRectangle(cornerRadius: r, style: .continuous))
        } else {
            self.modifier(MaterialGlassSurface(cornerRadius: r))
        }
        #else
        self.modifier(MaterialGlassSurface(cornerRadius: r))
        #endif
    }
}

/// macOS 13+ fallback that approximates Liquid Glass with system materials plus the
/// handoff's documented gradient/border/shadow recipe.
private struct MaterialGlassSurface: ViewModifier {
    @Environment(\.colorScheme) private var scheme
    let cornerRadius: CGFloat

    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
        return content
            // Material + sheen sit BELOW the content (background, not overlay) so
            // text is never tinted by the top-down gradient.
            .background {
                shape.fill(.ultraThinMaterial)
                shape.fill(sheen)
            }
            // Hairline rim on top — only touches the edges.
            .overlay(shape.strokeBorder(border, lineWidth: 0.5))
            // Float + contact shadows (handoff "glass shadow stack").
            .shadow(color: .black.opacity(scheme == .dark ? 0.55 : 0.18), radius: 35, y: 30)
            .shadow(color: .black.opacity(scheme == .dark ? 0.30 : 0.12), radius: 7, y: 4)
    }

    /// Top-down specular sheen baked into the fill (handoff material-fill gradients).
    private var sheen: LinearGradient {
        let stops: [Gradient.Stop] = scheme == .dark
            ? [
                .init(color: .white.opacity(0.14), location: 0.0),
                .init(color: .white.opacity(0.02), location: 0.36),
                .init(color: .white.opacity(0.0), location: 0.62),
            ]
            : [
                .init(color: .white.opacity(0.85), location: 0.0),
                .init(color: .white.opacity(0.30), location: 0.40),
                .init(color: .white.opacity(0.0), location: 0.62),
            ]
        return LinearGradient(gradient: Gradient(stops: stops), startPoint: .top, endPoint: .bottom)
    }

    private var border: Color {
        scheme == .dark ? .white.opacity(0.10) : .black.opacity(0.08)
    }
}
