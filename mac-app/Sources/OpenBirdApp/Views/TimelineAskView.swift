import AppKit
import SwiftUI

/// The standalone Timeline window (handoff Direction C — "Timeline-grounded"):
/// a left timeline rail of the day's capture sessions, and a right column with a
/// deterministic, clearly-labelled "day recap" (rendered from the SAME real session
/// data — never a fabricated LLM narrative) followed by a real follow-up chat thread.
/// The recap and the chat are visually separate so a follow-up never looks like it
/// continues an assistant answer that never happened (Codex review).
struct TimelineAskView: View {
    @ObservedObject var model: TimelineModel
    /// Per-window chat thread (a shared `AskPanelModel`). Follow-ups are general Ask —
    /// the copy does not claim they are scoped to the selected day.
    @ObservedObject var chat: AskPanelModel

    @Environment(\.colorScheme) private var scheme
    @State private var draft = ""

    private let recapLimit = 8

    var body: some View {
        VStack(spacing: 0) {
            titleBar
            Divider().overlay(OB.separator(scheme))
            HStack(spacing: 0) {
                railColumn.frame(width: 312)
                Rectangle().fill(OB.separator(scheme)).frame(width: 0.5)
                chatColumn
            }
        }
        .frame(minWidth: 840, minHeight: 560)
        .background(GlassBackdrop())
        .task {
            if model.timeline == nil { await model.load() }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            // Refresh the day's captures when the app re-activates with this window
            // open. Generation-guarded in the model, so a stale load can't overwrite.
            Task { await model.refresh() }
        }
    }

    // MARK: Title bar (traffic lights overlaid by .hiddenTitleBar)

    private var titleBar: some View {
        ZStack {
            Text("\(model.dayTitle) · Ask")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(OB.textSecondary(scheme))
            HStack {
                Spacer()
                dayStepper
            }
            .padding(.trailing, OB.Space.m)
        }
        .frame(height: 40)
    }

    private var dayStepper: some View {
        HStack(spacing: OB.Space.xs) {
            stepButton(systemName: "chevron.left", help: "Previous day") {
                changeDay(by: 1)   // older
            }
            stepButton(systemName: "chevron.right", help: "Next day", disabled: model.dayOffset == 0) {
                changeDay(by: -1)  // newer
            }
        }
    }

    private func stepButton(systemName: String, help: String, disabled: Bool = false, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(disabled ? OB.textTertiary(scheme) : OB.textSecondary(scheme))
                .frame(width: 22, height: 22)
                .background(OB.fieldFill(scheme), in: RoundedRectangle(cornerRadius: OB.Radius.control, style: .continuous))
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .help(help)
    }

    private func changeDay(by delta: Int) {
        let target = model.dayOffset + delta
        guard target >= 0 else { return }
        // The recap is day-scoped; switching days makes the existing chat thread
        // dangle out of context, so reset it (Codex review: explicit reset semantics).
        chat.clear()
        Task { await model.setDay(target) }
    }

    // MARK: Left rail

    private var railColumn: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(model.dayHeading)
                .font(.system(size: 10.5, weight: .bold))
                .tracking(0.6)
                .foregroundStyle(OB.textTertiary(scheme))
                .padding(.horizontal, 16)
                .padding(.top, OB.Space.ml)
                .padding(.bottom, OB.Space.sm)
            ScrollView {
                railBody
                    .padding(.horizontal, 16)
                    .padding(.bottom, 16)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    @ViewBuilder
    private var railBody: some View {
        if model.loading && model.timeline == nil {
            HStack(spacing: OB.Space.sm) {
                ProgressView().controlSize(.small)
                Text("Loading timeline…").font(.system(size: 12)).foregroundStyle(OB.textSecondary(scheme))
            }
            .padding(.top, OB.Space.sm)
        } else if let error = model.error {
            Label(error, systemImage: "exclamationmark.triangle")
                .font(.system(size: 12))
                .foregroundStyle(.orange)
                .padding(.top, OB.Space.sm)
        } else if let timeline = model.timeline, !timeline.sessions.isEmpty {
            SessionTimeline(sessions: timeline.sessions, displayName: model.displayName)
        } else {
            Text("No capture sessions for this day.")
                .font(.system(size: 12))
                .foregroundStyle(OB.textSecondary(scheme))
                .padding(.top, OB.Space.sm)
        }
    }

    // MARK: Right column (deterministic recap + chat)

    private var chatColumn: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: OB.Space.l) {
                        dayRecap
                        if !chat.thread.isEmpty || chat.busy {
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
    }

    @ViewBuilder
    private var dayRecap: some View {
        if let timeline = model.timeline, !timeline.sessions.isEmpty {
            VStack(alignment: .leading, spacing: OB.Space.m) {
                Text("DAY RECAP")
                    .font(.system(size: 10.5, weight: .bold))
                    .tracking(0.6)
                    .foregroundStyle(OB.textTertiary(scheme))
                Text("Reading down your day — \(model.sessionSummary):")
                    .font(.system(size: 14))
                    .lineSpacing(4)
                    .foregroundStyle(OB.textPrimary(scheme))
                    .fixedSize(horizontal: false, vertical: true)
                VStack(alignment: .leading, spacing: 9) {
                    ForEach(Array(recapSessions.enumerated()), id: \.offset) { _, session in
                        recapBullet(session)
                    }
                    if recapOverflow > 0 {
                        Text("+\(recapOverflow) more in the timeline")
                            .font(.system(size: 12))
                            .foregroundStyle(OB.textTertiary(scheme))
                            .padding(.leading, 18)
                    }
                }
            }
        }
    }

    private func recapBullet(_ session: TimelineSession) -> some View {
        let identity = SourceIdentity.forApp(session.app)
        let name = model.displayName(session.app)
        let title = (session.window?.isEmpty == false) ? session.window! : name
        let suffix = title == name ? "" : " · \(name)"
        return HStack(alignment: .top, spacing: 10) {
            Circle()
                .fill(identity.color)
                .frame(width: 8, height: 8)
                .padding(.top, 5)
            (Text(CitationFormatting.shortTime(session.start)).fontWeight(.semibold)
                + Text(" — \(title)")
                + Text(suffix).foregroundColor(OB.textSecondary(scheme)))
                .font(.system(size: 13.5))
                .lineSpacing(2)
                .foregroundStyle(OB.textPrimary(scheme))
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
    }

    private var recapSessions: [TimelineSession] {
        let sorted = (model.timeline?.sessions ?? []).sorted { $0.start < $1.start }
        return Array(sorted.prefix(recapLimit))
    }

    private var recapOverflow: Int {
        max(0, (model.timeline?.sessions.count ?? 0) - recapLimit)
    }

    private func submit() {
        let q = draft
        draft = ""
        chat.ask(q)
    }
}
