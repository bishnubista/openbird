import SwiftUI

/// The 262px "Sources" panel from the handoff (Direction B): a header count and a
/// scroll of numbered source cards — glyph tile + index + title + metadata +
/// a one-line excerpt. Bound to the latest completed answer's sources; shows an
/// empty state before the first answer.
struct SourcesRail: View {
    let sources: [ChatSource]
    /// Invoked when a source card is clicked — navigates to that citation's source.
    /// Defaults to a no-op so previews/tests can render without wiring navigation.
    var onSelectCitation: (ChatCitation) -> Void = { _ in }

    @Environment(\.colorScheme) private var scheme

    init(sources: [ChatSource], onSelectCitation: @escaping (ChatCitation) -> Void = { _ in }) {
        self.sources = sources
        self.onSelectCitation = onSelectCitation
    }

    init(citations: [ChatCitation], onSelectCitation: @escaping (ChatCitation) -> Void = { _ in }) {
        self.sources = citations.map(ChatSource.occurrence)
        self.onSelectCitation = onSelectCitation
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("SOURCES · \(sources.count)")
                .font(.system(size: 10.5, weight: .bold))
                .tracking(0.6)
                .foregroundStyle(OB.textTertiary(scheme))
                .padding(.horizontal, OB.Space.ml)
                .padding(.top, OB.Space.ml)
                .padding(.bottom, OB.Space.sm)

            if sources.isEmpty {
                Text("Ask a question to see its sources.")
                    .font(.system(size: 12))
                    .foregroundStyle(OB.textTertiary(scheme))
                    .padding(.horizontal, OB.Space.ml)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: OB.Space.sm) {
                        ForEach(sources) { card($0) }
                    }
                    .padding(.horizontal, OB.Space.m)
                    .padding(.top, OB.Space.xs)
                    .padding(.bottom, OB.Space.ml)
                }
            }
        }
        .frame(width: 262)
        .frame(maxHeight: .infinity)
    }

    /// Card title: the captured window title, falling back to the app name, then a
    /// generic label — so a card never renders blank. Trim first so whitespace-only
    /// metadata still falls through instead of rendering blank-looking text (CodeRabbit).
    static func cardTitle(_ c: ChatCitation) -> String {
        if let window = c.window?.trimmingCharacters(in: .whitespacesAndNewlines), !window.isEmpty {
            return window
        }
        if let app = c.app?.trimmingCharacters(in: .whitespacesAndNewlines), !app.isEmpty {
            return app
        }
        return "Source"
    }

    static func derivedCardTitle(_ c: DerivedChatCitation) -> String {
        c.displayLabel
    }

    static func derivedCountText(_ c: DerivedChatCitation) -> String {
        let count = max(0, c.derivedFromTotal)
        return "\(count) source observation\(count == 1 ? "" : "s")"
    }

    @ViewBuilder
    private func card(_ source: ChatSource) -> some View {
        switch source {
        case .occurrence(let citation):
            Button { onSelectCitation(citation) } label: {
                occurrenceCardLabel(citation)
            }
            .buttonStyle(.plain)
            .help("Open this source in Today")
        case .derived(let citation):
            derivedCardLabel(citation)
                .help(Self.derivedCountText(citation))
        }
    }

    private func occurrenceCardLabel(_ c: ChatCitation) -> some View {
        let identity = SourceIdentity.forApp(c.app)
        let title = Self.cardTitle(c)
        return HStack(alignment: .top, spacing: 10) {
            Text(identity.glyph)
                .font(.system(size: 11, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: 24, height: 24)
                .background(identity.color, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
            VStack(alignment: .leading, spacing: 0) {
                HStack(alignment: .firstTextBaseline, spacing: OB.Space.s) {
                    Text("\(c.index)")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(OB.accent)
                    Text(title)
                        .font(.system(size: 12.5, weight: .semibold))
                        .foregroundStyle(OB.textPrimary(scheme))
                        .lineLimit(1)
                }
                Text("\(c.app ?? "unknown") · \(CitationFormatting.shortTime(c.ts))")
                    .font(.system(size: 11))
                    .foregroundStyle(OB.textSecondary(scheme))
                if !c.snippet.isEmpty {
                    Text(c.snippet)
                        .font(.system(size: 11))
                        .lineSpacing(1)
                        .foregroundStyle(OB.textTertiary(scheme))
                        .lineLimit(2)
                        .padding(.top, 3)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(OB.fieldFill(scheme), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(OB.separator(scheme), lineWidth: 0.5)
        )
        .contentShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    private func derivedCardLabel(_ c: DerivedChatCitation) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Text("D")
                .font(.system(size: 11, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: 24, height: 24)
                .background(OB.accent, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
            VStack(alignment: .leading, spacing: 0) {
                HStack(alignment: .firstTextBaseline, spacing: OB.Space.s) {
                    Text("\(c.index)")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(OB.accent)
                    Text(Self.derivedCardTitle(c))
                        .font(.system(size: 12.5, weight: .semibold))
                        .foregroundStyle(OB.textPrimary(scheme))
                        .lineLimit(1)
                }
                Text(Self.derivedCountText(c))
                    .font(.system(size: 11))
                    .foregroundStyle(OB.textSecondary(scheme))
                if !c.snippet.isEmpty {
                    Text(c.snippet)
                        .font(.system(size: 11))
                        .lineSpacing(1)
                        .foregroundStyle(OB.textTertiary(scheme))
                        .lineLimit(2)
                        .padding(.top, 3)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(OB.fieldFill(scheme), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(OB.separator(scheme), lineWidth: 0.5)
        )
        .contentShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }
}
