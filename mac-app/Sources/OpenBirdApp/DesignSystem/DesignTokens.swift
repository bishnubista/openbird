import SwiftUI

/// Design tokens transcribed verbatim from the Claude Design handoff
/// (`docs/design/handoff/.../README.md` §"Design Tokens" + §"Liquid Glass").
/// Centralized so every Liquid Glass surface (this PR ships the Spotlight Ask
/// panel; later PRs reskin the menu/Today/Onboarding) reads one source of truth.
enum OB {

    // MARK: Accent / status colors

    /// Primary accent (`#2f7ff2`). Alternates exist in the handoff but we ship one.
    static let accent = Color(hex: 0x2F7FF2)
    static func ok(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(hex: 0x32D74B) : Color(hex: 0x1E9E3A)
    }
    /// The capturing/recording dot (`#ff453a`).
    static let capturingDot = Color(hex: 0xFF453A)

    // MARK: Corner radii (handoff §"Radii")

    enum Radius {
        static let spotlight: CGFloat = 17
        static let setup: CGFloat = 14
        static let window: CGFloat = 13
        static let dropdown: CGFloat = 12
        static let card: CGFloat = 11
        static let control: CGFloat = 8
        static let pill: CGFloat = 999
    }

    // MARK: Spacing rhythm (handoff §"Spacing": 4/6/8/12/14/18/22)

    enum Space {
        static let xs: CGFloat = 4
        static let s: CGFloat = 6
        static let sm: CGFloat = 8
        static let m: CGFloat = 12
        static let ml: CGFloat = 14
        static let l: CGFloat = 18
        static let xl: CGFloat = 22
    }

    // MARK: Text opacities (handoff §"Text")

    static func textPrimary(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color.white.opacity(0.92) : Color.black.opacity(0.85)
    }
    static func textSecondary(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color.white.opacity(0.56) : Color.black.opacity(0.5)
    }
    static func textTertiary(_ scheme: ColorScheme) -> Color {
        Color.primary.opacity(0.34)
    }
    static func separator(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color.white.opacity(0.12) : Color.black.opacity(0.1)
    }
    /// Field / card fills (handoff §"Field/Card fills").
    static func fieldFill(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color.white.opacity(0.08) : Color.black.opacity(0.05)
    }
}

/// The visual identity of a captured source app — a glyph + brand color — used to
/// render citation chips. Values from the handoff §2 ("Source identity").
struct SourceIdentity: Equatable {
    let glyph: String
    let color: Color

    /// Map a captured app name to its identity. Matching is substring/case-insensitive
    /// so "Visual Studio Code", "Code", "com.microsoft.VSCode" all resolve. Unknown
    /// apps fall back to their first initial (or a neutral dot) on a neutral tile.
    static func forApp(_ app: String?) -> SourceIdentity {
        let name = (app ?? "").lowercased()
        if name.contains("code") || name.contains("vscode") {
            return SourceIdentity(glyph: "{}", color: Color(hex: 0x2F6BE0))
        }
        if name.contains("zoom") {
            return SourceIdentity(glyph: "Z", color: Color(hex: 0x2D8CFF))
        }
        if name.contains("linear") {
            return SourceIdentity(glyph: "L", color: Color(hex: 0x5E6AD2))
        }
        if name.contains("notion") {
            return SourceIdentity(glyph: "N", color: Color(hex: 0x4B4B52))
        }
        if name.contains("slack") {
            return SourceIdentity(glyph: "#", color: Color(hex: 0x4A154B))
        }
        if name.contains("chrome") {
            return SourceIdentity(glyph: "◉", color: Color(hex: 0x4285F4))
        }
        let initial = (app?.first).map { String($0).uppercased() } ?? "•"
        return SourceIdentity(glyph: initial, color: Color(hex: 0x6B6B73))
    }
}

/// Shared formatting for citation rows, so the Spotlight panel and the window
/// `ChatView` render sources identically (single source of truth — avoids drift).
enum CitationFormatting {
    static func sourceLabel(app: String?, window: String?) -> String {
        let parts = [app, window].compactMap { $0 }.filter { !$0.isEmpty }
        return parts.isEmpty ? "unknown" : parts.joined(separator: " / ")
    }

    static func timeLabel(_ ts: Double) -> String {
        Date(timeIntervalSince1970: ts).formatted(date: .abbreviated, time: .shortened)
    }

    /// Time-only label for compact source chips (e.g. "9:12 AM").
    static func shortTime(_ ts: Double) -> String {
        Date(timeIntervalSince1970: ts).formatted(date: .omitted, time: .shortened)
    }
}

extension Color {
    /// Construct from a 0xRRGGBB literal (handoff tokens are written this way).
    init(hex: UInt32) {
        let r = Double((hex >> 16) & 0xFF) / 255.0
        let g = Double((hex >> 8) & 0xFF) / 255.0
        let b = Double(hex & 0xFF) / 255.0
        self.init(.sRGB, red: r, green: g, blue: b, opacity: 1)
    }
}
