import SwiftUI

/// Direction A ("Spotlight") from the Claude Design handoff: a single centered glass
/// card with a big input row, a grounded answer with source chips, a follow-up
/// thread, and suggestion pills. Asks run through `AskPanelModel` (panel-owned, see
/// its docs); read-only status comes from the shared `AppModel`.
struct AskPanelView: View {
    @ObservedObject var askModel: AskPanelModel
    @ObservedObject var appModel: AppModel
    var onEscape: () -> Void
    /// Promote the compact panel to the expanded window (chat + Sources/Timeline rails).
    var onExpand: () -> Void
    var onSizeChange: (CGSize) -> Void

    @Environment(\.colorScheme) private var scheme
    @State private var draft = ""
    @FocusState private var inputFocused: Bool

    private let suggestions = [
        "Summarize the Memory sync",
        "What's left on OB-142",
        "Draft my standup",
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            inputRow
            if let reason = appModel.askUnavailableReason, askModel.thread.isEmpty {
                unavailableRow(reason)
            }
            if showsAnswerArea {
                Divider().overlay(OB.separator(scheme))
                answerArea
            }
            suggestionRow
        }
        .frame(width: 620, alignment: .leading)
        .glassSurface(cornerRadius: OB.Radius.spotlight)
        .padding(44)                                  // room for the cast glass shadow
        .background(sizeReader)
        .onPreferenceChange(PanelSizeKey.self, perform: onSizeChange)
        .onExitCommand(perform: onEscape)
        .onAppear { inputFocused = true }
    }

    private var showsAnswerArea: Bool { !askModel.thread.isEmpty || askModel.busy }

    // MARK: Input

    private var inputRow: some View {
        HStack(spacing: OB.Space.m) {
            BirdLogo()
                .fill(OB.accent)
                .frame(width: 22, height: 22)
            TextField("Ask about your work…", text: $draft)
                .textFieldStyle(.plain)
                .font(.system(size: 18))
                .foregroundStyle(OB.textPrimary(scheme))
                .focused($inputFocused)
                .onSubmit(submit)
            expandButton
            escChip
        }
        .padding(.horizontal, OB.Space.l)
        .padding(.vertical, OB.Space.ml)
    }

    private var expandButton: some View {
        Button(action: onExpand) {
            Image(systemName: "arrow.up.left.and.arrow.down.right")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(OB.textSecondary(scheme))
                .frame(width: 22, height: 22)
                .background(OB.fieldFill(scheme), in: RoundedRectangle(cornerRadius: OB.Radius.control))
        }
        .buttonStyle(.plain)
        .help("Expand to show sources and timeline")
    }

    private var escChip: some View {
        Text("esc")
            .font(.system(size: 11, weight: .medium))
            .foregroundStyle(OB.textSecondary(scheme))
            .padding(.horizontal, OB.Space.sm)
            .padding(.vertical, OB.Space.xs)
            .background(OB.fieldFill(scheme), in: RoundedRectangle(cornerRadius: OB.Radius.control))
    }

    private func unavailableRow(_ reason: String) -> some View {
        Label(reason, systemImage: "info.circle")
            .font(.system(size: 12.5))
            .foregroundStyle(OB.textSecondary(scheme))
            .padding(.horizontal, OB.Space.l)
            .padding(.bottom, OB.Space.m)
    }

    // MARK: Answer thread

    private var answerArea: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: OB.Space.l) {
                    ForEach(askModel.thread) { turn in
                        turnView(turn).id(turn.id)
                    }
                    if askModel.busy {
                        ThinkingDots().id("thinking")
                    }
                }
                .padding(.horizontal, OB.Space.l)
                .padding(.vertical, OB.Space.ml)
            }
            .frame(maxHeight: 420)
            .onChange(of: askModel.thread.count) { _ in
                if let last = askModel.thread.last {
                    withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }
        }
    }

    @ViewBuilder
    private func turnView(_ turn: AskPanelModel.Turn) -> some View {
        VStack(alignment: .leading, spacing: OB.Space.sm) {
            Text(turn.question)
                .font(.system(size: 13.5, weight: .semibold))
                .foregroundStyle(OB.textSecondary(scheme))
            if let error = turn.error {
                Label(error, systemImage: "exclamationmark.triangle")
                    .font(.callout)
                    .foregroundStyle(.orange)
            } else if let result = turn.result {
                answerBlock(result)
            }
        }
    }

    @ViewBuilder
    private func answerBlock(_ result: ChatResult) -> some View {
        HStack(spacing: OB.Space.s) {
            Circle()
                .fill(result.grounded ? OB.ok(scheme) : Color.orange)
                .frame(width: 7, height: 7)
            Text(groundedLabel(result))
                .font(.system(size: 11.5))
                .foregroundStyle(OB.textSecondary(scheme))
        }
        Text(result.answer.isEmpty ? "(no answer)" : result.answer)
            .font(.system(size: 14))
            .foregroundStyle(OB.textPrimary(scheme))
            .textSelection(.enabled)
            .fixedSize(horizontal: false, vertical: true)
        if !result.citations.isEmpty {
            ScrollView(.horizontal, showsIndicators: false) {
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
                .font(.system(size: 10, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: 18, height: 18)
                .background(identity.color, in: RoundedRectangle(cornerRadius: 5))
            Text("\(c.app ?? "unknown") · \(CitationFormatting.shortTime(c.ts))")
                .font(.system(size: 11.5))
                .foregroundStyle(OB.textSecondary(scheme))
        }
        .padding(.horizontal, OB.Space.sm)
        .padding(.vertical, OB.Space.s)
        .background(OB.fieldFill(scheme), in: Capsule())
    }

    private func groundedLabel(_ result: ChatResult) -> String {
        guard result.grounded else { return "ungrounded — no verified source" }
        let n = result.citations.count
        return "Answer · grounded in \(n) source\(n == 1 ? "" : "s")"
    }

    // MARK: Suggestions

    private var suggestionRow: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: OB.Space.sm) {
                ForEach(suggestions, id: \.self) { suggestion in
                    Button {
                        draft = suggestion
                        submit()
                    } label: {
                        Text(suggestion)
                            .font(.system(size: 12.5))
                            .padding(.horizontal, OB.Space.m)
                            .padding(.vertical, OB.Space.sm)
                            .background(OB.fieldFill(scheme), in: Capsule())
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(OB.textSecondary(scheme))
                }
            }
            .padding(.horizontal, OB.Space.l)
        }
        .padding(.top, showsAnswerArea ? OB.Space.s : 0)
        .padding(.bottom, OB.Space.ml)
    }

    // MARK: Actions

    private func submit() {
        // Clear the draft only if the ask was accepted — a submit while busy is rejected,
        // and the text must survive so it isn't silently dropped (Codex review).
        if askModel.ask(draft) { draft = "" }
    }

    private var sizeReader: some View {
        GeometryReader { geo in
            Color.clear.preference(key: PanelSizeKey.self, value: geo.size)
        }
    }
}

/// Reports the panel's full rendered size up to `AskPanelController`, which resizes
/// the hosting `NSPanel` so the card grows downward as the thread fills.
private struct PanelSizeKey: PreferenceKey {
    static var defaultValue: CGSize = .zero
    static func reduce(value: inout CGSize, nextValue: () -> CGSize) { value = nextValue() }
}

// ThinkingDots now lives in ChatThreadView.swift (shared by the Spotlight panel and
// the windowed Ask surfaces) — see that file.
