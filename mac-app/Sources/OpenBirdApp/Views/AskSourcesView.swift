import SwiftUI

/// The Sources window (handoff Direction B — "Sources rail"): a two-pane Ask surface
/// with the chat on the left and a 262px Sources panel on the right. The rail and the
/// titlebar "grounded" indicator track the latest COMPLETED successful answer (never a
/// pending or error turn), so they never imply a not-yet-existing answer is grounded
/// or that older turns share the current rail (Codex review). Per-answer attribution
/// stays visible via the inline source chips `ChatThreadView` already renders.
struct AskSourcesView: View {
    /// Window-local chat thread (its own `AskPanelModel`; no history shared with the
    /// Spotlight panel or the Timeline window).
    @ObservedObject var chat: AskPanelModel

    @Environment(\.colorScheme) private var scheme
    @State private var draft = ""

    private let suggestions = [
        "Summarize the Memory sync",
        "What's left on OB-142",
        "Draft my standup",
    ]

    enum GroundedState: Equatable { case hidden, thinking, grounded, ungrounded }

    struct SourcesDisplay: Equatable {
        var indicator: GroundedState
        var citations: [ChatCitation]
    }

    private var display: SourcesDisplay {
        Self.makeDisplay(thread: chat.thread, busy: chat.busy)
    }

    /// Rail + indicator state, keyed to the MOST RECENT turn so neither ever reflects a
    /// turn that errored or never answered (Codex review): after a failed follow-up the
    /// titlebar must not still read "grounded" beside the error. While busy → "thinking"
    /// and the rail keeps the last successful answer's sources as explicit prior context;
    /// an idle last turn that errored or produced nothing → neutral indicator + empty rail.
    static func makeDisplay(thread: [AskPanelModel.Turn], busy: Bool) -> SourcesDisplay {
        if busy {
            let prior = thread.last { $0.result != nil }?.result?.citations ?? []
            return SourcesDisplay(indicator: .thinking, citations: prior)
        }
        if let result = thread.last?.result {
            return SourcesDisplay(indicator: result.grounded ? .grounded : .ungrounded,
                                  citations: result.citations)
        }
        return SourcesDisplay(indicator: .hidden, citations: [])
    }

    var body: some View {
        VStack(spacing: 0) {
            titleBar
            Divider().overlay(OB.separator(scheme))
            HStack(spacing: 0) {
                chatColumn
                Rectangle().fill(OB.separator(scheme)).frame(width: 0.5)
                SourcesRail(citations: display.citations)
            }
        }
        .frame(minWidth: 800, minHeight: 540)
        .background(GlassBackdrop())
    }

    // MARK: Title bar

    private var titleBar: some View {
        ZStack {
            Text("Ask OpenBird")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(OB.textSecondary(scheme))
            HStack {
                Spacer()
                groundedIndicator
            }
            .padding(.trailing, OB.Space.ml)
        }
        .frame(height: 40)
    }

    @ViewBuilder
    private var groundedIndicator: some View {
        switch display.indicator {
        case .hidden: EmptyView()
        case .thinking: indicator(color: OB.textTertiary(scheme), label: "Thinking…")
        case .grounded: indicator(color: OB.ok(scheme), label: "grounded")
        case .ungrounded: indicator(color: .orange, label: "ungrounded")
        }
    }

    private func indicator(color: Color, label: String) -> some View {
        HStack(spacing: OB.Space.s) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text(label)
                .font(.system(size: 11))
                .foregroundStyle(OB.textTertiary(scheme))
        }
    }

    // MARK: Chat column

    private var chatColumn: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    Group {
                        if chat.thread.isEmpty && !chat.busy {
                            emptyPrompt
                        } else {
                            ChatThreadView(turns: chat.thread, busy: chat.busy)
                                .id("chat-tail")
                        }
                    }
                    .padding(18)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .onChange(of: chat.thread.count) { _ in
                    withAnimation { proxy.scrollTo("chat-tail", anchor: .bottom) }
                }
            }
            AskFollowUpBar(draft: $draft, isBusy: chat.busy, onSubmit: submit)
        }
        .frame(maxWidth: .infinity)
    }

    private var emptyPrompt: some View {
        VStack(alignment: .leading, spacing: OB.Space.ml) {
            Text("Ask about your work to get a grounded, cited answer.")
                .font(.system(size: 14))
                .foregroundStyle(OB.textSecondary(scheme))
            HStack(spacing: OB.Space.sm) {
                ForEach(suggestions, id: \.self) { suggestion in
                    Button {
                        chat.ask(suggestion)
                    } label: {
                        Text(suggestion)
                            .font(.system(size: 12.5))
                            .padding(.horizontal, OB.Space.m)
                            .padding(.vertical, OB.Space.sm)
                            .background(OB.fieldFill(scheme), in: Capsule())
                            .overlay(Capsule().strokeBorder(OB.separator(scheme), lineWidth: 0.5))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(OB.textSecondary(scheme))
                }
            }
        }
    }

    private func submit() {
        let q = draft
        draft = ""
        chat.ask(q)
    }
}
