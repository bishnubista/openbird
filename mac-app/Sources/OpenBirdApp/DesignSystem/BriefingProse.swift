import SwiftUI

/// Renders a daily-briefing string as clean prose (TodayView §"Daily briefing card").
///
/// The briefing prompt is constrained to emit a short prose paragraph with inline
/// `**bold**` only (no headings/lists/dividers/chain-of-thought). But a model can
/// still misbehave, and the card must NEVER show raw Markdown symbols the way a bare
/// `Text(briefing)` does. So rendering is two-stage and defensive:
///
///  1. `BriefingProse.paragraphs(from:)` — a pure, testable normaliser that strips
///     stray block syntax (heading `#`, horizontal rules `---`, list bullets/numbers)
///     and reflows the remainder into paragraphs. Worst case degrades to clean prose,
///     never literal `###`/`---`.
///  2. Each paragraph is rendered with INLINE-only Markdown so surviving `**bold**` /
///     `*italic*` format correctly while block syntax we already removed can't reflow
///     the text into something lossy.
enum BriefingProse {
    /// Normalise a raw briefing into display paragraphs, stripping block-level Markdown.
    ///
    /// Pure (no SwiftUI) so it can be unit-tested directly. Inline emphasis markers
    /// (`**`, `*`, `` ` ``) are preserved for the inline renderer to interpret.
    static func paragraphs(from raw: String) -> [String] {
        var paragraphs: [String] = []
        var current: [String] = []

        func flush() {
            if !current.isEmpty {
                paragraphs.append(current.joined(separator: " "))
                current = []
            }
        }

        for rawLine in raw.components(separatedBy: "\n") {
            // Trim newlines too, so a CRLF tail (`\r`) can't defeat the symbol/heading
            // checks below (e.g. "####\r" would otherwise render literal `####`).
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)

            // A blank line, or a line that is ENTIRELY Markdown punctuation, closes the
            // current paragraph and is dropped. This covers horizontal rules (`---`,
            // `***`, `___`), bare/empty markers left after trimming (`-`, `####`, `**`),
            // and code fences (```) — none of which carry prose.
            if line.isEmpty || isSymbolNoise(line) {
                flush()
                continue
            }

            // A fenced-code opener with an info string (```swift, ~~~json) isn't all
            // punctuation, so drop it by prefix; its content survives as plain prose.
            if line.hasPrefix("```") || line.hasPrefix("~~~") {
                flush()
                continue
            }

            // Heading (`## Foo`) and list items (`- foo`, `1. foo`) lose their markers
            // and each stand as their own paragraph, so a degraded list still reads as
            // separate short lines instead of one run-on sentence. A marker with no text
            // (`####`, bare `- `) strips to "" and is dropped, never an empty paragraph.
            if let heading = strippedHeading(line) {
                flush()
                if !heading.isEmpty { paragraphs.append(heading) }
            } else if let item = strippedListItem(line) {
                flush()
                if !item.isEmpty { paragraphs.append(item) }
            } else {
                // Plain prose lines reflow together into one paragraph.
                current.append(line)
            }
        }
        flush()
        return paragraphs
    }

    /// True when every character is Markdown structural punctuation, so the line is
    /// noise (a rule, a stray/empty marker, or a code fence) with no prose to keep.
    private static let symbolNoise: Set<Character> = ["-", "*", "+", "_", "#", "=", "~", "`"]
    private static func isSymbolNoise(_ line: String) -> Bool {
        !line.isEmpty && line.allSatisfy { symbolNoise.contains($0) }
    }

    /// Strip a leading ATX heading marker (`#`..`######` + spaces). Returns nil if the
    /// line is not a heading.
    private static func strippedHeading(_ line: String) -> String? {
        guard line.first == "#" else { return nil }
        let body = line.drop(while: { $0 == "#" })
        guard body.first == " " || body.isEmpty else { return nil }  // `#tag` is not a heading
        return body.trimmingCharacters(in: .whitespaces)
    }

    /// Strip a leading list marker (`- `, `* `, `+ `, or `1. `). Returns nil otherwise.
    private static func strippedListItem(_ line: String) -> String? {
        if let first = line.first, first == "-" || first == "*" || first == "+" {
            let rest = line.dropFirst()
            if rest.first == " " {
                return rest.trimmingCharacters(in: .whitespaces)
            }
        }
        // Ordered list: digits then `. ` or `) `.
        let digits = line.prefix(while: { $0.isNumber })
        if !digits.isEmpty {
            let after = line.dropFirst(digits.count)
            if let sep = after.first, sep == "." || sep == ")", after.dropFirst().first == " " {
                return after.dropFirst().trimmingCharacters(in: .whitespaces)
            }
        }
        return nil
    }

    /// Parse one paragraph's inline Markdown (`**bold**`, `*italic*`, `` `code` ``).
    /// Inline-only so block syntax cannot reflow/collapse the text; falls back to the
    /// plain string if parsing fails.
    static func inlineAttributed(_ paragraph: String) -> AttributedString {
        (try? AttributedString(
            markdown: paragraph,
            options: .init(
                interpretedSyntax: .inlineOnlyPreservingWhitespace,
                failurePolicy: .returnPartiallyParsedIfPossible
            )
        )) ?? AttributedString(paragraph)
    }
}

/// SwiftUI view that renders a briefing string as clean prose paragraphs with inline
/// emphasis. Drop-in replacement for `Text(briefing)` in the Today card body.
struct BriefingText: View {
    let briefing: String
    var font: Font = .system(size: 14)
    var lineSpacing: CGFloat = 4
    var paragraphSpacing: CGFloat = OB.Space.sm

    /// Shown when normalisation removes everything (whitespace- or symbol-only model
    /// output). A fixed plain line — NEVER the raw symbols we just stripped, and never
    /// a blank card even though the caller only checked the raw string was non-empty.
    static let emptyFallback = "No briefing available for this day."

    var body: some View {
        let parsed = BriefingProse.paragraphs(from: briefing)
        return VStack(alignment: .leading, spacing: paragraphSpacing) {
            if parsed.isEmpty {
                Text(Self.emptyFallback)
                    .font(font)
                    .lineSpacing(lineSpacing)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                ForEach(Array(parsed.enumerated()), id: \.offset) { _, para in
                    Text(BriefingProse.inlineAttributed(para))
                        .font(font)
                        .lineSpacing(lineSpacing)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
    }
}
