import SwiftUI

/// The expanded Ask surface (hosted in the persistent `AskExpandedWindow`): the SAME
/// conversation as the compact panel (`askModel`), with optional Sources and Timeline
/// rails. A glass card filling its borderless window. The Timeline rail is PASSIVE
/// context — its day stepper reloads only `TimelineModel` and never touches the shared
/// Ask thread (Codex review). Compact and expanded are separate views sharing
/// `ChatThreadView` / `AskFollowUpBar` / `SourcesRail` / `SessionTimeline` / `SourcesDisplay`.
struct ExpandedAskView: View {
    @ObservedObject var askModel: AskPanelModel
    @ObservedObject var appModel: AppModel
    @ObservedObject var timelineModel: TimelineModel
    var onCollapse: () -> Void
    var onClose: () -> Void

    @Environment(\.colorScheme) private var scheme
    @State private var draft = ""
    @FocusState private var inputFocused: Bool
    @AppStorage("openbird.ask.showSources") private var showSources = true
    @AppStorage("openbird.ask.showTimeline") private var showTimeline = false

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
            HStack(spacing: 0) {
                if showTimeline {
                    timelineRail
                    verticalSeparator
                }
                chatColumn
                if showSources {
                    verticalSeparator
                    SourcesRail(citations: display.citations)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .glassSurface(cornerRadius: OB.Radius.window)
        .padding(24)                       // room for the cast glass shadow
        .background(WindowConfigurator())  // draggable by background
        .onExitCommand(perform: onClose)
        .onAppear { inputFocused = true }
        .task {
            if timelineModel.timeline == nil { await timelineModel.load() }
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
            iconButton("arrow.down.right.and.arrow.up.left", help: "Collapse", action: onCollapse)
            iconButton("xmark", help: "Close", action: onClose)
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
        .frame(width: 312)
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
            Text("Ask about your work to get a grounded, cited answer.")
                .font(.system(size: 14))
                .foregroundStyle(OB.textSecondary(scheme))
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

    private func submit() {
        if askModel.ask(draft) { draft = "" }
    }
}
