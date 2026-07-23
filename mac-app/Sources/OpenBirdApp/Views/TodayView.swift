import SwiftUI

/// The Today/day detail pane (handoff §3): a main column with a grounded briefing card,
/// stat chips, and a session timeline on a connector rail — rendered over the
/// `timeline`/`briefing` CLI JSON via `TodayModel`. The shared nav sidebar and the
/// window chrome are owned by the `AppShellView`; this view is just the day column.
struct TodayView: View {
    @ObservedObject var model: TodayModel
    @ObservedObject var appModel: AppModel
    /// Summon the Spotlight Ask panel hard-scoped to a day offset (0=today,
    /// 1=yesterday, ...). The "Ask about this day" buttons pass the currently
    /// viewed day so answers are confined to it.
    var onAsk: (Int) -> Void
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        mainColumn
            // Reload every time the pane opens (not just once) so returning to Today after
            // capture has run reflects new activity. `load()` keeps the current timeline
            // visible while re-fetching and reuses the cached briefing, so there's no flash.
            .task { await model.load() }
    }

    private var mainColumn: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: OB.Space.l) {
                header
                meetingCard
                briefingCard
                productivityCard
                timelineSection
            }
            .padding(OB.Space.xl)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: Header

    private var header: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.dayTitle)
                    .font(.system(size: 21, weight: .bold))
                    .foregroundStyle(OB.textPrimary(scheme))
                Text(model.daySubtitle)
                    .font(.subheadline)
                    .foregroundStyle(OB.textSecondary(scheme))
            }
            Spacer()
            askButton
        }
    }

    private var askButton: some View {
        Button(action: { onAsk(model.dayOffset) }) {
            HStack(spacing: OB.Space.s) {
                BirdLogo().fill(OB.accent).frame(width: 15, height: 15)
                Text("Ask about this day")
                    .font(.system(size: 13, weight: .medium))
            }
            .foregroundStyle(OB.textPrimary(scheme))
            .padding(.horizontal, OB.Space.ml)
            .padding(.vertical, OB.Space.sm)
            .background(OB.fieldFill(scheme),
                        in: RoundedRectangle(cornerRadius: OB.Radius.control, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: OB.Radius.control, style: .continuous)
                    .strokeBorder(OB.separator(scheme), lineWidth: 0.5)
            )
        }
        .buttonStyle(.plain)
    }

    // MARK: Meeting recording

    private var meetingCard: some View {
        VStack(alignment: .leading, spacing: OB.Space.m) {
            HStack {
                Label("MEETING RECORDING", systemImage: meetingIcon)
                    .font(.system(size: 10.5, weight: .bold))
                    .tracking(0.6)
                    .foregroundStyle(meetingAccent)
                Spacer()
                meetingStatusBadge
            }
            meetingBody
            if !appModel.meetingLastMessage.isEmpty {
                Text(appModel.meetingLastMessage)
                    .font(.caption)
                    .foregroundStyle(OB.textSecondary(scheme))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(OB.Space.l)
        .glassSurface(cornerRadius: OB.Radius.card)
    }

    @ViewBuilder
    private var meetingBody: some View {
        switch appModel.meetingState {
        case .idle:
            Text("Manually record system audio and your microphone into a local, searchable transcript. Start only after everyone agrees to be recorded.")
                .font(.callout)
                .foregroundStyle(OB.textPrimary(scheme))
            if let reason = appModel.meetingReadinessMessage {
                Label(reason, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            Button("Start Meeting Recording") { appModel.requestStartMeeting() }
                .buttonStyle(.borderedProminent)
                .disabled(!appModel.meetingCanStart)

        case .consent:
            Text("Download the approximately 2.51 GB Parakeet model from Hugging Face to ~/.openbird/models/huggingface?")
                .font(.callout.weight(.semibold))
            Text("The request contains only the public model id and normal HTTP metadata—never your audio or transcript. Partial downloads can resume. \(appModel.privacyTransmissionSummary)")
                .font(.caption)
                .foregroundStyle(OB.textSecondary(scheme))
            HStack {
                Button("Download & Continue") { appModel.confirmMeetingModelDownload() }
                    .buttonStyle(.borderedProminent)
                Button("Cancel") { appModel.cancelMeetingAction() }
            }

        case .preparing(let downloaded, let total):
            ProgressView(value: Double(downloaded), total: Double(max(total, 1)))
            Text("Preparing local transcription · \(Self.byteProgress(downloaded, total))")
                .font(.caption)
                .monospacedDigit()
                .foregroundStyle(OB.textSecondary(scheme))
            Button("Cancel") { appModel.cancelMeetingAction() }

        case .recording(let startedAt):
            TimelineView(.periodic(from: .now, by: 1)) { context in
                Text("● Recording · \(Self.elapsed(from: startedAt, to: context.date))")
                    .font(.system(.title3, design: .monospaced).weight(.semibold))
                    .foregroundStyle(.red)
            }
            Text("Audio remains in memory/private pipes; only the encrypted transcript is stored.")
                .font(.caption)
                .foregroundStyle(OB.textSecondary(scheme))
            HStack {
                Button("Stop & Save") { appModel.stopMeetingRecording() }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                Button("Force Stop Audio") { appModel.forceStopMeetingAudio() }
                    .buttonStyle(.bordered)
            }

        case .finalizing(let completed, let remaining, let dropped, let failed):
            ProgressView()
            Text("Finalizing locally · \(completed) complete · \(remaining) remaining")
                .font(.callout)
                .monospacedDigit()
            if dropped > 0 || failed > 0 {
                Text("Partial transcript: \(dropped) dropped, \(failed) failed window(s).")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            Button("Force Stop Audio") { appModel.forceStopMeetingAudio() }
                .buttonStyle(.bordered)
        }
    }

    private var meetingIcon: String {
        appModel.meetingState.isRecording ? "waveform.circle.fill" : "waveform.circle"
    }

    private var meetingAccent: Color {
        appModel.meetingState.isRecording ? .red : OB.accent
    }

    @ViewBuilder
    private var meetingStatusBadge: some View {
        let label: String = {
            switch appModel.meetingState {
            case .idle: return appModel.meetingCanStart ? "Ready" : "Needs setup"
            case .consent: return "Consent"
            case .preparing: return "Preparing"
            case .recording: return "Recording"
            case .finalizing: return "Finalizing"
            }
        }()
        Text(label)
            .font(.caption.weight(.semibold))
            .foregroundStyle(appModel.meetingState.isRecording ? .red : OB.textSecondary(scheme))
    }

    private static func byteProgress(_ downloaded: Int64, _ total: Int64) -> String {
        String(format: "%.2f / %.2f GB", Double(downloaded) / 1_000_000_000, Double(total) / 1_000_000_000)
    }

    private static func elapsed(from start: Date, to end: Date) -> String {
        let seconds = max(0, Int(end.timeIntervalSince(start)))
        return String(format: "%02d:%02d:%02d", seconds / 3600, (seconds % 3600) / 60, seconds % 60)
    }

    // MARK: Daily briefing card

    private var briefingCard: some View {
        VStack(alignment: .leading, spacing: OB.Space.m) {
            HStack(alignment: .firstTextBaseline) {
                Text("DAILY BRIEFING")
                    .font(.system(size: 10.5, weight: .bold))
                    .tracking(0.6)
                    .foregroundStyle(OB.accent)
                Spacer()
                if let route = model.briefingRouteLabel {
                    Text(route)
                        .font(.system(size: 12))
                        .foregroundStyle(OB.textTertiary(scheme))
                        .lineLimit(1)
                }
                if let at = model.briefingGeneratedAt {
                    Text("generated \(at.formatted(date: .omitted, time: .shortened))")
                        .font(.system(size: 12))
                        .monospacedDigit()
                        .foregroundStyle(OB.textTertiary(scheme))
                }
            }
            briefingBody
            briefingSourceTrail
            if let timeline = model.timeline {
                statChips(timeline)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(OB.Space.l)
        .glassSurface(cornerRadius: OB.Radius.card)
    }

    // MARK: Source trail (what the briefing is based on)

    /// A clickable trail of the occurrences the briefing prose was grounded in.
    /// Each row reuses the SAME citation navigation as chat sources
    /// (`appModel.navigateToCitation`), focusing that observation in this day. Only
    /// shown when the day actually has grounding sources (an empty day shows none,
    /// matching the deterministic no-activity prose).
    @ViewBuilder
    private var briefingSourceTrail: some View {
        if !model.briefingSources.isEmpty {
            VStack(alignment: .leading, spacing: OB.Space.s) {
                Text(sourceTrailHeader)
                    .font(.system(size: 10.5, weight: .bold))
                    .tracking(0.6)
                    .foregroundStyle(OB.textTertiary(scheme))
                ForEach(Array(model.briefingSources.enumerated()), id: \.element.id) { index, source in
                    sourceRow(source, index: index + 1)
                }
            }
            .padding(.top, OB.Space.xs)
        }
    }

    /// "BASED ON · 5" or "BASED ON · 12 of 47" when the CLI capped the trail — the
    /// total is surfaced so truncation is never silent.
    private var sourceTrailHeader: String {
        let shown = model.briefingSources.count
        let total = model.briefingSourcesTotal
        return total > shown ? "BASED ON · \(shown) of \(total)" : "BASED ON · \(shown)"
    }

    private func sourceRow(_ source: BriefingSource, index: Int) -> some View {
        let identity = SourceIdentity.forApp(source.app)
        let title = SourcesRail.cardTitle(source.asCitation(index: index))
        return Button {
            appModel.navigateToCitation(source.asCitation(index: index))
        } label: {
            HStack(alignment: .top, spacing: OB.Space.sm) {
                Text(identity.glyph)
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(width: 20, height: 20)
                    .background(identity.color,
                                in: RoundedRectangle(cornerRadius: 5, style: .continuous))
                VStack(alignment: .leading, spacing: 1) {
                    Text(title)
                        .font(.system(size: 12.5, weight: .semibold))
                        .foregroundStyle(OB.textPrimary(scheme))
                        .lineLimit(1)
                    Text("\(source.app ?? "unknown") · \(CitationFormatting.shortTime(source.ts))")
                        .font(.system(size: 11))
                        .foregroundStyle(OB.textSecondary(scheme))
                    if !source.snippet.isEmpty {
                        Text(source.snippet)
                            .font(.system(size: 11))
                            .foregroundStyle(OB.textTertiary(scheme))
                            .lineLimit(1)
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(OB.Space.sm)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(OB.fieldFill(scheme),
                        in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(OB.separator(scheme), lineWidth: 0.5)
            )
            .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        .buttonStyle(.plain)
        .help("Open this source in \(model.dayTitle)")
    }

    @ViewBuilder
    private var briefingBody: some View {
        if model.loadingBriefing {
            HStack(spacing: OB.Space.sm) {
                ProgressView().controlSize(.small)
                Text("Generating briefing…")
                    .font(.callout)
                    .foregroundStyle(OB.textSecondary(scheme))
            }
        } else if let briefing = model.briefing, !briefing.isEmpty {
            BriefingText(briefing: briefing)
                .foregroundStyle(OB.textPrimary(scheme))
                .textSelection(.enabled)
        } else {
            Text("No briefing available for this day.")
                .font(.callout)
                .foregroundStyle(OB.textSecondary(scheme))
        }
    }

    // MARK: Stat chips (pill, bordered, tabular)

    private func statChips(_ timeline: DayTimeline) -> some View {
        // Three HONEST chips: apps, captures, active time. The handoff's "1 meeting"
        // chip is intentionally omitted — there is no real meeting-evidence source
        // yet (meetings is a stub), so claiming a meeting count would be fabricated.
        HStack(spacing: OB.Space.sm) {
            statChip("\(timeline.distinctApps) app\(timeline.distinctApps == 1 ? "" : "s")")
            statChip("\(Self.grouped(timeline.totalObservations)) captures")
            statChip("\(Self.durationLabel(timeline.activeSeconds)) active")
        }
    }

    private func statChip(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 12))
            .monospacedDigit()
            .foregroundStyle(OB.textSecondary(scheme))
            .padding(.horizontal, OB.Space.m)
            .padding(.vertical, OB.Space.s)
            .background(OB.fieldFill(scheme), in: Capsule())
            .overlay(Capsule().strokeBorder(OB.separator(scheme), lineWidth: 0.5))
    }

    // MARK: Productivity facts

    private var productivityCard: some View {
        VStack(alignment: .leading, spacing: OB.Space.m) {
            HStack(alignment: .firstTextBaseline) {
                Text("PRODUCTIVITY")
                    .font(.system(size: 10.5, weight: .bold))
                    .tracking(0.6)
                    .foregroundStyle(OB.accent)
                Spacer()
                if let route = model.productivity?.routeLabel {
                    Text(route)
                        .font(.system(size: 12))
                        .foregroundStyle(OB.textTertiary(scheme))
                }
            }
            productivityBody
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(OB.Space.l)
        .glassSurface(cornerRadius: OB.Radius.card)
    }

    @ViewBuilder
    private var productivityBody: some View {
        if model.loadingProductivity && model.productivity == nil {
            HStack(spacing: OB.Space.sm) {
                ProgressView().controlSize(.small)
                Text("Loading local facts…")
                    .font(.callout)
                    .foregroundStyle(OB.textSecondary(scheme))
            }
        } else if let productivity = model.productivity {
            if productivity.hasActivityFacts {
                productivityFacts(productivity)
            } else {
                Text("No productivity facts were recorded for this day.")
                    .font(.callout)
                    .foregroundStyle(OB.textSecondary(scheme))
            }
        } else if let error = model.productivityError {
            Label(error, systemImage: "exclamationmark.triangle")
                .font(.callout)
                .foregroundStyle(.orange)
        } else {
            Text("No productivity facts available for this day.")
                .font(.callout)
                .foregroundStyle(OB.textSecondary(scheme))
        }
    }

    private func productivityFacts(_ productivity: DayProductivity) -> some View {
        let facts = productivity.facts
        return VStack(alignment: .leading, spacing: OB.Space.s) {
            productivityStatChips(facts)
            if let top = facts.topCategory {
                productivityRow(
                    label: "Top category",
                    value: "\(Self.categoryLabel(top.category)) · \(Self.minutesLabel(top.minutes))",
                    detail: "\(top.sourceCount) source\(top.sourceCount == 1 ? "" : "s")"
                )
            }
            if let hour = facts.topHour {
                productivityRow(
                    label: "Strongest hour",
                    value: "\(hour.hour) · \(Self.minutesLabel(hour.minutes))",
                    detail: "\(hour.sourceCount) source\(hour.sourceCount == 1 ? "" : "s")"
                )
            }
            if let block = facts.longestFocusBlock {
                productivityRow(
                    label: "Longest block",
                    value: "\(Self.categoryLabel(block.category ?? "unknown")) · \(Self.durationLabel(block.seconds))",
                    detail: "\(block.sessionCount) session\(block.sessionCount == 1 ? "" : "s")"
                )
            }
        }
    }

    private func productivityStatChips(_ facts: ProductivityFacts) -> some View {
        let active = "\(Self.minutesLabel(facts.activeMinutes)) active"
        let switches = "\(facts.contextSwitchCount) switch\(facts.contextSwitchCount == 1 ? "" : "es")"
        let rate = "\(Self.rateLabel(facts.contextSwitchesPerActiveHour))/active hour"

        return ViewThatFits(in: .horizontal) {
            HStack(spacing: OB.Space.sm) {
                statChip(active)
                statChip(switches)
                statChip(rate)
            }
            VStack(alignment: .leading, spacing: OB.Space.sm) {
                statChip(active)
                statChip(switches)
                statChip(rate)
            }
        }
    }

    private func productivityRow(label: String, value: String, detail: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .font(.system(size: 12))
                .foregroundStyle(OB.textTertiary(scheme))
                .frame(width: 96, alignment: .leading)
            Text(value)
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(OB.textPrimary(scheme))
                .lineLimit(1)
            Spacer(minLength: OB.Space.s)
            Text(detail)
                .font(.system(size: 12))
                .monospacedDigit()
                .foregroundStyle(OB.textSecondary(scheme))
        }
    }

    // MARK: Session timeline

    @ViewBuilder
    private var timelineSection: some View {
        VStack(alignment: .leading, spacing: OB.Space.m) {
            Text("TIMELINE")
                .font(.system(size: 10.5, weight: .bold))
                .tracking(0.6)
                .foregroundStyle(OB.textSecondary(scheme))
            timelineBody
        }
    }

    @ViewBuilder
    private var timelineBody: some View {
        if model.loadingTimeline && model.timeline == nil {
            HStack(spacing: OB.Space.sm) {
                ProgressView().controlSize(.small)
                Text("Loading timeline…").foregroundStyle(OB.textSecondary(scheme))
            }
        } else if let error = model.timelineError {
            Label(error, systemImage: "exclamationmark.triangle")
                .foregroundStyle(.orange)
        } else if let timeline = model.timeline, !timeline.sessions.isEmpty {
            sessionList(timeline)
        } else {
            emptyTimelinePanel
        }
    }

    private var emptyTimelinePanel: some View {
        // The Today window has no day navigation today, so this current-readiness
        // guidance cannot imply that a historical empty day can be fixed retroactively.
        VStack(alignment: .leading, spacing: OB.Space.m) {
            Label("No activity captured for this day.", systemImage: "sparkle.magnifyingglass")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(OB.textPrimary(scheme))
            Text(appModel.nextStepSummary)
                .font(.system(size: 13))
                .foregroundStyle(OB.textSecondary(scheme))
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: OB.Space.sm) {
                Button("Open Setup") {
                    appModel.selection = .setup
                }
                .buttonStyle(.bordered)

                if appModel.canStartCaptureNow {
                    Button("Start Capture") {
                        appModel.startCapture()
                    }
                    .buttonStyle(.borderedProminent)
                }

                Button("Ask about this day") {
                    onAsk(model.dayOffset)
                }
                .buttonStyle(.bordered)
            }
        }
        .padding(OB.Space.l)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassSurface(cornerRadius: OB.Radius.card)
    }

    private func sessionList(_ timeline: DayTimeline) -> some View {
        // Index-keyed: NULL-session rows can share app/ts, so element identity isn't
        // guaranteed unique; the list is render-only, so the index is a safe id.
        VStack(alignment: .leading, spacing: OB.Space.sm) {
            ForEach(Array(timeline.sessions.enumerated()), id: \.offset) { _, session in
                sessionRow(session)
            }
        }
        // Continuous connector rail behind the nodes. Positioned under the node
        // column (time column width + spacing + node center).
        .background(alignment: .topLeading) {
            Rectangle()
                .fill(OB.separator(scheme))
                .frame(width: 1.5)
                .padding(.leading, Self.timeColumnWidth + OB.Space.m + 5 - 0.75)
                .padding(.vertical, 20)
        }
    }

    private func sessionRow(_ session: TimelineSession) -> some View {
        let identity = SourceIdentity.forApp(session.app)
        let appName = model.displayName(session.app)
        let title = (session.window?.isEmpty == false) ? session.window! : appName
        return HStack(alignment: .top, spacing: OB.Space.m) {
            // Left time column: start over end.
            VStack(alignment: .trailing, spacing: 0) {
                Text(CitationFormatting.shortTime(session.start))
                    .foregroundStyle(OB.textSecondary(scheme))
                Text(CitationFormatting.shortTime(session.end))
                    .foregroundStyle(OB.textTertiary(scheme))
            }
            .font(.system(size: 12))
            .monospacedDigit()
            .lineLimit(1)                       // never wrap the " AM"/" PM" onto a second line
            .frame(width: Self.timeColumnWidth, alignment: .trailing)
            .padding(.top, 14)

            // Node on the connector rail.
            Circle().fill(identity.color).frame(width: 10, height: 10).padding(.top, 18)

            // Session card.
            HStack(alignment: .top, spacing: OB.Space.m) {
                Text(identity.glyph)
                    .font(.system(size: 13, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(width: 34, height: 34)
                    .background(identity.color,
                                in: RoundedRectangle(cornerRadius: OB.Radius.control, style: .continuous))
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(OB.textPrimary(scheme))
                        .lineLimit(1)
                    Text("\(appName) · \(Self.durationLabel(session.end - session.start))")
                        .font(.system(size: 12))
                        .foregroundStyle(OB.textSecondary(scheme))
                }
                Spacer()
                Text("\(Self.grouped(session.count)) captures")
                    .font(.system(size: 12))
                    .monospacedDigit()
                    .foregroundStyle(OB.textTertiary(scheme))
            }
            .padding(OB.Space.m)
            .frame(maxWidth: .infinity, alignment: .leading)
            .glassSurface(cornerRadius: OB.Radius.card)
        }
    }

    // MARK: Formatting

    // Wide enough for a two-digit 12-hour time with day period ("10:52 PM") at 12pt
    // monospaced-digit — the gutter must not wrap (the connector rail inset derives
    // from this constant, so it follows automatically).
    static let timeColumnWidth: CGFloat = 62

    /// Group-separated integer ("1,284"). Forwards to the shared formatter so the
    /// Today card rail and the compact Direction-C rail never disagree on a value.
    static func grouped(_ n: Int) -> String { TimelineFormatting.grouped(n) }

    static func durationLabel(_ seconds: Double) -> String { TimelineFormatting.durationLabel(seconds) }

    static func minutesLabel(_ minutes: Double) -> String {
        durationLabel(max(0, minutes) * 60.0)
    }

    static func rateLabel(_ value: Double) -> String {
        if value.rounded() == value {
            return String(Int(value))
        }
        return String(format: "%.1f", value)
    }

    static func categoryLabel(_ value: String) -> String {
        value
            .split(separator: "_")
            .map { part in
                part.prefix(1).uppercased() + part.dropFirst()
            }
            .joined(separator: " ")
    }
}
