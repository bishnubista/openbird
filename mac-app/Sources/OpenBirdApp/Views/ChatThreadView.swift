import SwiftUI

/// Windowed chat rendering shared by the Direction B (Sources) and Direction C
/// (Timeline) Ask surfaces: a right-aligned accent user bubble, then the grounded
/// assistant answer (status dot + prose + horizontal source chips), then staggered
/// thinking dots while busy. The Spotlight panel keeps its own (label-style)
/// rendering; these windows use the prototype's bubble style.
struct ChatThreadView: View {
    let turns: [AskPanelModel.Turn]
    let busy: Bool

    @Environment(\.colorScheme) private var scheme

    var body: some View {
        VStack(alignment: .leading, spacing: OB.Space.ml) {
            ForEach(turns) { turn in
                userBubble(turn.question)
                if let error = turn.error {
                    Label(error, systemImage: "exclamationmark.triangle")
                        .font(.system(size: 13))
                        .foregroundStyle(.orange)
                } else if let result = turn.result {
                    assistantAnswer(result)
                }
            }
            if busy { ThinkingDots() }
        }
    }

    private func userBubble(_ text: String) -> some View {
        HStack {
            Spacer(minLength: 40)
            Text(text)
                .font(.system(size: 13.5))
                .foregroundStyle(.white)
                .padding(.horizontal, 13)
                .padding(.vertical, 9)
                .background(OB.accent, in: RoundedRectangle(cornerRadius: 15, style: .continuous))
        }
    }

    @ViewBuilder
    private func assistantAnswer(_ result: ChatResult) -> some View {
        VStack(alignment: .leading, spacing: OB.Space.m) {
            HStack(spacing: OB.Space.s) {
                Circle()
                    .fill(result.grounded ? OB.ok(scheme) : Color.orange)
                    .frame(width: 6, height: 6)
                Text(groundedLabel(result))
                    .font(.system(size: 11))
                    .foregroundStyle(OB.textTertiary(scheme))
            }
            Text(result.answer.isEmpty ? "(no answer)" : result.answer)
                .font(.system(size: 14))
                .lineSpacing(4)
                .foregroundStyle(OB.textPrimary(scheme))
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            if !result.citations.isEmpty {
                HStack(spacing: OB.Space.sm) {
                    ForEach(result.citations) { sourceChip($0) }
                }
            }
        }
    }

    private func sourceChip(_ c: ChatCitation) -> some View {
        let identity = SourceIdentity.forApp(c.app)
        return HStack(spacing: OB.Space.s) {
            Text(identity.glyph)
                .font(.system(size: 9, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: 18, height: 18)
                .background(identity.color, in: RoundedRectangle(cornerRadius: 5, style: .continuous))
            Text("\(c.app ?? "unknown") · \(CitationFormatting.shortTime(c.ts))")
                .font(.system(size: 11.5))
                .foregroundStyle(OB.textSecondary(scheme))
        }
        .padding(.leading, 5)
        .padding(.trailing, 9)
        .padding(.vertical, 5)
        .background(OB.fieldFill(scheme), in: RoundedRectangle(cornerRadius: OB.Radius.control, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: OB.Radius.control, style: .continuous)
                .strokeBorder(OB.separator(scheme), lineWidth: 0.5)
        )
    }

    private func groundedLabel(_ result: ChatResult) -> String {
        guard result.grounded else { return "ungrounded — no verified source" }
        let n = result.citations.count
        return "grounded in \(n) source\(n == 1 ? "" : "s")"
    }
}

/// A row of follow-up input: a pill text field + a round accent send button
/// (handoff B/C bottom bar). Submits on Enter and on the button.
struct AskFollowUpBar: View {
    @Binding var draft: String
    var onSubmit: () -> Void

    @Environment(\.colorScheme) private var scheme

    var body: some View {
        HStack(spacing: OB.Space.sm) {
            TextField("Ask about your work…", text: $draft)
                .textFieldStyle(.plain)
                .font(.system(size: 13.5))
                .foregroundStyle(OB.textPrimary(scheme))
                .padding(.horizontal, 13)
                .padding(.vertical, 9)
                .background(OB.fieldFill(scheme), in: Capsule())
                .overlay(Capsule().strokeBorder(OB.separator(scheme), lineWidth: 0.5))
                .onSubmit(onSubmit)
            Button(action: onSubmit) {
                Image(systemName: "arrow.up")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 34, height: 34)
                    .background(OB.accent, in: RoundedRectangle(cornerRadius: 9, style: .continuous))
            }
            .buttonStyle(.plain)
            .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
        .padding(OB.Space.m)
        .overlay(alignment: .top) {
            Rectangle().fill(OB.separator(scheme)).frame(height: 0.5)
        }
    }
}

/// Staggered three-dot "thinking" indicator (handoff `obDot`). Shared by the
/// windowed Ask surfaces (the Spotlight panel has its own private copy).
struct ThinkingDots: View {
    @State private var animating = false
    var body: some View {
        HStack(spacing: 5) {
            ForEach(0..<3, id: \.self) { i in
                Circle()
                    .frame(width: 6, height: 6)
                    .scaleEffect(animating ? 1.0 : 0.5)
                    .opacity(animating ? 1.0 : 0.4)
                    .animation(
                        .easeInOut(duration: 0.5).repeatForever().delay(Double(i) * 0.15),
                        value: animating
                    )
            }
        }
        .foregroundStyle(.secondary)
        .onAppear { animating = true }
    }
}
