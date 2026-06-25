import SwiftUI

/// The chrome-free Ask surface shared by BOTH the borderless `ExpandedAskView` (the
/// ⌥Space overlay window) and the in-window `AskPaneView` (the sidebar's Ask pane):
/// the SAME conversation (`askModel`), as a single clean chat column. Sources are surfaced
/// as inline citation chips under each answer (the "Spotlight" direction) — there are no
/// side rails, so the empty state is just the prompt + suggestions, never empty panels.
///
/// This view deliberately owns NO window-only chrome — no collapse/close buttons (they come
/// from `onCollapse`/`onClose`, which the pane leaves nil), no cast-shadow padding, no
/// `WindowConfigurator`, no Escape handling. Each wrapper layers those on as its role
/// requires (Codex review: a `mode` flag would entangle window-only behavior into one body
/// — a shared content view + thin wrappers keeps them separate).
struct AskContentView: View {
    @ObservedObject var askModel: AskPanelModel
    @ObservedObject var appModel: AppModel
    /// Window-only actions. Both nil in pane mode, which hides the buttons entirely.
    var onCollapse: (() -> Void)?
    var onClose: (() -> Void)?
    /// Invoked when a citation chip is clicked — navigates to that source. Defaults to a
    /// no-op so previews/tests render without wiring it.
    var onSelectCitation: (ChatCitation) -> Void = { _ in }

    @Environment(\.colorScheme) private var scheme
    @State private var draft = ""
    @FocusState private var inputFocused: Bool

    /// Readable single-column width. Without the old side rails, both the wide in-window
    /// pane and the expanded overlay would stretch messages edge-to-edge (user bubbles far
    /// right, answers far left); capping + centering keeps the conversation legible.
    private let contentMaxWidth: CGFloat = 680

    /// Drives the header's grounded/thinking indicator from the current thread.
    private var display: SourcesDisplay {
        SourcesDisplay.make(thread: askModel.thread, busy: askModel.busy)
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(OB.separator(scheme))
            chatColumn
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear { inputFocused = true }
    }

    // MARK: Header

    private var header: some View {
        HStack(spacing: OB.Space.sm) {
            BirdLogo().fill(OB.accent).frame(width: 18, height: 18)
            Text("Ask OpenBird")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(OB.textSecondary(scheme))
            groundedIndicator
            Spacer()
            // Window-only controls — absent in the in-window pane (no window to
            // collapse/close).
            if let onCollapse {
                iconButton("arrow.down.right.and.arrow.up.left", help: "Collapse", action: onCollapse)
            }
            if let onClose {
                iconButton("xmark", help: "Close", action: onClose)
            }
        }
        .padding(.horizontal, OB.Space.ml)
        .frame(height: 44)
    }

    @ViewBuilder
    private var groundedIndicator: some View {
        switch display.indicator {
        case .hidden: EmptyView()
        case .thinking: indicatorChip(color: OB.textTertiary(scheme), label: "Thinking…")
        case .grounded: indicatorChip(color: OB.ok(scheme), label: "grounded")
        case .ungrounded: indicatorChip(color: .orange, label: "ungrounded")
        }
    }

    private func indicatorChip(color: Color, label: String) -> some View {
        HStack(spacing: OB.Space.s) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text(label).font(.system(size: 11)).foregroundStyle(OB.textTertiary(scheme))
        }
    }

    private func iconButton(_ systemImage: String, help: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(OB.textSecondary(scheme))
                .frame(width: 22, height: 22)
        }
        .buttonStyle(.plain)
        .help(help)
    }

    // MARK: Chat column

    private var chatColumn: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    Group {
                        if askModel.thread.isEmpty && !askModel.busy {
                            emptyPrompt
                        } else {
                            ChatThreadView(
                                turns: askModel.thread,
                                busy: askModel.busy,
                                onSelectCitation: onSelectCitation
                            )
                            .id("chat-tail")
                        }
                    }
                    .padding(18)
                    // Cap + center the conversation so it stays legible in wide containers
                    // (the expanded overlay and the in-window pane) rather than stretching
                    // user bubbles and answers to opposite edges.
                    .frame(maxWidth: contentMaxWidth, alignment: .leading)
                    .frame(maxWidth: .infinity)
                }
                .onChange(of: askModel.thread.count) { _ in
                    withAnimation { proxy.scrollTo("chat-tail", anchor: .bottom) }
                }
            }
            AskFollowUpBar(draft: $draft, isBusy: askModel.busy, focused: $inputFocused, onSubmit: submit)
                .frame(maxWidth: contentMaxWidth)
                .frame(maxWidth: .infinity)
        }
        .frame(maxWidth: .infinity)
    }

    private var emptyPrompt: some View {
        VStack(alignment: .leading, spacing: OB.Space.ml) {
            Text(appModel.askEmptyPrompt)
                .font(.system(size: 14))
                .foregroundStyle(OB.textSecondary(scheme))
            if appModel.askUnavailableReason == nil {
                HStack(spacing: OB.Space.sm) {
                    ForEach(appModel.askSuggestions, id: \.self) { suggestion in
                        Button { askModel.ask(suggestion) } label: {
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
    }

    private func submit() {
        if askModel.ask(draft) { draft = "" }
    }
}
