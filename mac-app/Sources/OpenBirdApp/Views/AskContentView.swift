import SwiftUI

/// The chrome-free Ask surface shared by BOTH the borderless `ExpandedAskView` (the
/// ⌥Space overlay window) and the in-window `AskPaneView` (the sidebar's Ask pane):
/// the SAME conversation (`askModel`), with optional Sources and Timeline rails. This
/// view deliberately owns NO window-only chrome — no collapse/close buttons (they come
/// from `onCollapse`/`onClose`, which the pane leaves nil), no cast-shadow padding, no
/// `WindowConfigurator`, no Escape handling. Each wrapper layers those on as its role
/// requires (Codex review: a `mode` flag would entangle window-only behavior into one
/// body — a shared content view + thin wrappers keeps them separate).
///
/// The Timeline rail is PASSIVE context — its day stepper reloads only `TimelineModel`
/// and never touches the shared Ask thread. The rails collapse responsively so a narrow
/// window never clips the chat column.
struct AskContentView: View {
    @ObservedObject var askModel: AskPanelModel
    @ObservedObject var appModel: AppModel
    @ObservedObject var timelineModel: TimelineModel
    /// Window-only actions. Both nil in pane mode, which hides the buttons entirely.
    var onCollapse: (() -> Void)?
    var onClose: (() -> Void)?

    @Environment(\.colorScheme) private var scheme
    @State private var draft = ""
    @FocusState private var inputFocused: Bool
    @AppStorage("openbird.ask.showSources") private var showSources = false
    @AppStorage("openbird.ask.showTimeline") private var showTimeline = false

    // Rail widths and the chat column's floor. When a requested rail won't fit
    // alongside a usable chat column, it's dropped until the window widens — the chat
    // is never clipped (Codex review: reconcile min size incl. Ask rails).
    private let timelineRailWidth: CGFloat = 312
    private let sourcesRailWidth: CGFloat = 262
    // Chat floor below which a rail is dropped. Kept at 320 — under the ~337pt the
    // 960pt expanded overlay already gives the chat with BOTH rails on (960 − 48pt
    // window padding − 312 − 262 − 1pt separators) — so this responsive collapse never
    // hides Sources in the expanded overlay; it only trims rails for a narrow in-window
    // pane (Codex review caught the regression a 380 floor would cause).
    private let chatMinWidth: CGFloat = 320
    private let separatorWidth: CGFloat = 0.5   // each visible rail adds one divider

    private let suggestions = [
        "Summarize the Memory sync",
        "What's left on OB-142",
        "Draft my standup",
    ]

    private var display: SourcesDisplay {
        SourcesDisplay.make(thread: askModel.thread, busy: askModel.busy)
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(OB.separator(scheme))
            GeometryReader { geo in
                railsAndChat(availableWidth: geo.size.width)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear { inputFocused = true }
        .task {
            if timelineModel.timeline == nil { await timelineModel.load() }
        }
    }

    /// Greedily lay out the requested rails around the chat column for the available
    /// width, preferring the Timeline rail over Sources when only one fits.
    @ViewBuilder
    private func railsAndChat(availableWidth: CGFloat) -> some View {
        // Each shown rail also costs its separator, so fold that into the fit test —
        // otherwise the chat column is short by 0.5pt at the exact threshold (Codex).
        let showT = showTimeline && availableWidth >= chatMinWidth + timelineRailWidth + separatorWidth
        let widthAfterTimeline = availableWidth - (showT ? timelineRailWidth + separatorWidth : 0)
        let showS = showSources && widthAfterTimeline >= chatMinWidth + sourcesRailWidth + separatorWidth
        HStack(spacing: 0) {
            if showT {
                timelineRail
                verticalSeparator
            }
            chatColumn
            if showS {
                verticalSeparator
                SourcesRail(citations: display.citations)
            }
        }
    }

    private var verticalSeparator: some View {
        Rectangle().fill(OB.separator(scheme)).frame(width: 0.5)
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
            toggle("sidebar.left", label: "Timeline", isOn: $showTimeline)
            toggle("sidebar.right", label: "Sources", isOn: $showSources)
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

    private func toggle(_ systemImage: String, label: String, isOn: Binding<Bool>) -> some View {
        Button { isOn.wrappedValue.toggle() } label: {
            Image(systemName: systemImage)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(isOn.wrappedValue ? OB.accent : OB.textSecondary(scheme))
                .frame(width: 24, height: 22)
                .background(
                    RoundedRectangle(cornerRadius: OB.Radius.control)
                        .fill(isOn.wrappedValue ? OB.accent.opacity(0.14) : .clear)
                )
        }
        .buttonStyle(.plain)
        .help("\(isOn.wrappedValue ? "Hide" : "Show") \(label)")
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

    // MARK: Timeline rail (passive context)

    private var timelineRail: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(timelineModel.dayHeading)
                    .font(.system(size: 10.5, weight: .bold))
                    .tracking(0.6)
                    .foregroundStyle(OB.textTertiary(scheme))
                Spacer()
                dayStepper
            }
            .padding(.horizontal, 16)
            .padding(.top, OB.Space.ml)
            .padding(.bottom, OB.Space.sm)
            ScrollView {
                timelineBody
                    .padding(.horizontal, 16)
                    .padding(.bottom, 16)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .frame(width: timelineRailWidth)
    }

    private var dayStepper: some View {
        HStack(spacing: OB.Space.xs) {
            stepButton("chevron.left", help: "Previous day") { changeDay(by: 1) }
            stepButton("chevron.right", help: "Next day", disabled: timelineModel.dayOffset == 0) { changeDay(by: -1) }
        }
    }

    private func stepButton(_ systemImage: String, help: String, disabled: Bool = false, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(disabled ? OB.textTertiary(scheme) : OB.textSecondary(scheme))
                .frame(width: 20, height: 20)
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .help(help)
    }

    /// Reloads ONLY the timeline rail. The shared Ask thread is never cleared by day
    /// navigation (Codex review — the conversation is global, not day-scoped).
    private func changeDay(by delta: Int) {
        let target = timelineModel.dayOffset + delta
        guard target >= 0 else { return }
        Task { await timelineModel.setDay(target) }
    }

    @ViewBuilder
    private var timelineBody: some View {
        if timelineModel.loading && timelineModel.timeline == nil {
            HStack(spacing: OB.Space.sm) {
                ProgressView().controlSize(.small)
                Text("Loading…").font(.system(size: 12)).foregroundStyle(OB.textSecondary(scheme))
            }
            .padding(.top, OB.Space.sm)
        } else if let timeline = timelineModel.timeline, !timeline.sessions.isEmpty {
            SessionTimeline(sessions: timeline.sessions, displayName: timelineModel.displayName)
        } else {
            Text("No capture sessions for this day.")
                .font(.system(size: 12))
                .foregroundStyle(OB.textSecondary(scheme))
                .padding(.top, OB.Space.sm)
        }
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
                            ChatThreadView(turns: askModel.thread, busy: askModel.busy)
                                .id("chat-tail")
                        }
                    }
                    .padding(18)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .onChange(of: askModel.thread.count) { _ in
                    withAnimation { proxy.scrollTo("chat-tail", anchor: .bottom) }
                }
            }
            AskFollowUpBar(draft: $draft, isBusy: askModel.busy, focused: $inputFocused, onSubmit: submit)
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
                    ForEach(suggestions, id: \.self) { suggestion in
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
